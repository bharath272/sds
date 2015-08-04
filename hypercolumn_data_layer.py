import cv2
import caffe
import numpy as np
import numpy.random as npr
import argparse, pprint
import pickle
from prepare_blobs import get_blobs
import sbd
import os
import sds_config as cfg
def get_box_overlap(box_1, box_2):
  box1 = box_1.copy().astype(np.float32)
  box2 = box_2.copy().astype(np.float32)
  xmin = np.maximum(box1[:,0].reshape((-1,1)),box2[:,0].reshape((1,-1)))
  ymin = np.maximum(box1[:,1].reshape((-1,1)),box2[:,1].reshape((1,-1)))
  xmax = np.minimum(box1[:,2].reshape((-1,1)),box2[:,2].reshape((1,-1)))
  ymax = np.minimum(box1[:,3].reshape((-1,1)),box2[:,3].reshape((1,-1)))
  iw = np.maximum(xmax-xmin+1.,0.)
  ih = np.maximum(ymax-ymin+1.,0.)
  inters = iw*ih
  area1 = (box1[:,3]-box1[:,1]+1.)*(box1[:,2]-box1[:,0]+1.)  
  area2 = (box2[:,3]-box2[:,1]+1.)*(box2[:,2]-box2[:,0]+1.)  
  uni = area1.reshape((-1,1))+area2.reshape((1,-1))-inters
  iu = inters/uni
  return iu

class HypercolumnDataLayer(caffe.Layer):
  def _parse_args(self, str_arg):
    parser = argparse.ArgumentParser(description='Hypercolumn Data Layer Parameters')
    parser.add_argument('--proposalfile', default='train_mcg_boxes.pkl', type=str)
    parser.add_argument('--gtfile', default=os.path.join(sbd.sbddir, 'gttrain.pkl'), type=str)
    parser.add_argument('--trainset', default='train.txt', type=str)
    parser.add_argument('--ovthresh', default=0.7, type=float)
    parser.add_argument('--imgpath', default='/data1/shubhtuls/cachedir/VOCdevkit/VOC2012/JPEGImages/{}.jpg', type=str)
    args = parser.parse_args(str_arg.split())
    print('Using config:')
    pprint.pprint(args)
    return args




  def setup(self, bottom, top):
    self._params = self._parse_args(self.param_str_)
    with open(self._params.proposalfile,'r') as f:
      o = pickle.load(f)
    self.boxes=o['boxes']
    with open(self._params.gtfile, 'r') as f:
      self.gt = pickle.load(f)
    #categories are from 1 to 20 so subtract 1
    self.gt['classes'] = [x-1 for x in self.gt['classes']]


    with open(self._params.trainset) as f:
      names = f.readlines()
    self.names = [x[:-1] for x in names]
    
    #how many categories are there?
    maxcateg = np.max(np.array([np.max(x) for x in self.gt['classes']]))
    self.numclasses=maxcateg+1
    
    #initialize
    self.data_percateg=[]
    for i in range(self.numclasses):
      self.data_percateg.append({'boxids':[],'imids':[],'instids':[], 'im_end_index':[-1]})

    #compute all overlaps and pick boxes that have greater than threshold overlap
    for i in range(len(self.names)):
      ov = get_box_overlap(self.boxes[i], self.gt['boxes'][i])
      
      #this maintains the last index for each image for each category
      for classlabel in range(self.numclasses):
        self.data_percateg[classlabel]['im_end_index'].append(self.data_percateg[classlabel]['im_end_index'][-1])
      
      #for every gt
      for j in range(len(self.gt['classes'][i])):
        idx = ov[:,j]>=self._params.ovthresh
        if not np.any(idx):
          continue
        #save the boxes
        classlabel = self.gt['classes'][i][j]
        self.data_percateg[classlabel]['boxids'].extend(np.where(idx)[0].tolist())
        self.data_percateg[classlabel]['imids'].extend([i]*np.sum(idx))
        self.data_percateg[classlabel]['instids'].extend([j]*np.sum(idx))
        self.data_percateg[classlabel]['im_end_index'][-1] += np.sum(idx)
    #convert everything to a np array because python is an ass
    for j in range(self.numclasses):
      self.data_percateg[j]['boxids']=np.array(self.data_percateg[j]['boxids'])
      self.data_percateg[j]['imids']=np.array(self.data_percateg[j]['imids'])
      self.data_percateg[j]['instids']=np.array(self.data_percateg[j]['instids'])


    #also save a dictionary of where each blob goes to
    self.blob_names = ['image','normalizedboxes','sppboxes','categids','labels', 'instance_wts']
    blobs=dict()
    self.myblobs=blobs


  def reshape(self, bottom, top):
    #sample a category
    categid = np.random.choice(self.numclasses)
    #sample an image for this category
    imid = self.data_percateg[categid]['imids'][np.random.choice(len(self.data_percateg[categid]['imids']))]
    
    img = cv2.imread(self._params.imgpath.format(self.names[imid]))

    #get all possibilities for this category
    start = self.data_percateg[categid]['im_end_index'][imid]+1
    stop = self.data_percateg[categid]['im_end_index'][imid+1]
    #pick a box
    idx = np.random.choice(np.arange(start,stop+1), cfg.TRAIN_SAMPLES_PER_IMG)
    boxid = self.data_percateg[categid]['boxids'][idx]
    boxes = self.boxes[imid][boxid,:]-1    

    instid = self.data_percateg[categid]['instids'][idx]
    #load the gt
    [inst, categories] = sbd.load_gt(self.names[imid])
    masks = np.zeros((idx.size, 1,inst.shape[0],inst.shape[1]))
    for k in range(idx.size):
      masks[k,0,:,:] = (inst==instid[k]+1).astype(np.float32)
    categids = categid*np.ones(idx.size)

    #get the blobs
    im_new, spp_boxes, normalized_boxes, categids, masksblob, instance_wts = get_blobs(img, boxes.astype(np.float32), categids, masks)

    #save blobs in private dict
    self.myblobs['image']=im_new.astype(np.float32)
    self.myblobs['normalizedboxes']=normalized_boxes.astype(np.float32)
    self.myblobs['sppboxes']=spp_boxes.astype(np.float32)
    self.myblobs['categids']=categids.astype(np.float32)
    self.myblobs['labels']=masksblob.astype(np.float32)
    self.myblobs['instance_wts']=instance_wts.astype(np.float32)

    #and reshape
    for i in range(len(top)):
      top[i].reshape(*(self.myblobs[self.blob_names[i]].shape))

  def forward(self, bottom, top):
    for i in range(len(top)):
      top[i].data[...] = self.myblobs[self.blob_names[i]]

  def backward(self, top, propagate_down, bottom):
    pass

    
 

