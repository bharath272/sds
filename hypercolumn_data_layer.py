import caffe
import numpy as np
import numpy.random as npr
import argparse, pprint
import pickle
from prepare_blobs import get_blobs
import sbd
import os

def get_box_overlap(box1, box2)


class HypercolumnDataLayer(caffe.Layer):
  def _parse_args(self, str_arg):
    parser = argparse.ArgumentParser(description='Hypercolumn Data Layer Parameters')
    parser.add_argument('--proposalfile', default='train_mcg_boxes.pkl', type=str)
    parser.add_argument('--gtfile', default=os.path.join(sbd.sbddir, 'gttrain.pkl'), type=str)
    parser.add_argument('--trainset', default='train.txt', type=str)
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
    with open(self._params.trainset) as f:
      names = f.readlines()
    self.names = [x[:-1] for x in names]
    
    #how many categories are there?
    maxcateg = np.max(np.array([np.max(x) for x in self.gt['classes']]))
    self.numclasses=maxcateg+1
    
    #initialize
    data_percateg=[]
    for i in range(self.numclasses):
      data_percateg.append({'boxids':[],'imids':[],'instids':[], 'im_end_index':[-1]})

    #compute all overlaps and pick boxes that have greater than threshold overlap
    for i in range(len(self.names)):
      ov = get_box_overlap(self.boxes[i], self.gt['boxes'][i])
      
      #this maintains the last index for each image for each category
      for classlabel in range(self.numclasses):
        data_percateg[classlabel]['im_end_index'].append(data_percateg[classlabel]['im_end_index'][-1])
      
      #for every gt
      for j in range(len(self.gt['classes'][i])):
        idx = ov[:,j]>=self._params.ovthresh
        if not np.any(idx):
          continue
        #save the boxes
        classlabel = gt[i]['classes'][j]
        data_percateg[classlabel]['boxids'].extend(np.where(idx).tolist())
        data_percateg[classlabel]['imids'].extend([i]*np.sum(idx))
        data_percateg[classlabel]['instids'].extend([j]*np.sum(idx))
        data_percateg[classlabel]['im_end_index'][-1] += np.sum(idx)
    #also save a dictionary of where each blob goes to
    self.blob_names = ['image','normalizedboxes','sppboxes','categids','labels', 'instance_wts']


  def reshape(self, bottom, top):
    #sample a category
    categid = np.random.choice(self.numclasses)
    #sample an image for this category
    imid = data_percateg[categid]['imids'][np.random.choice(len(data_percateg[categid]['imids']))]
    #get all possibilities for this category
    start = data_percateg[categid]['im_end_index'][imid]+1
    stop = data_percateg[categid]['im_end_index'][imid+1]
    #pick a box
    idx = np.random.choice(np.arange(start,stop+1), cfg.TRAIN_SAMPLES_PER_IMG)
    boxid = data_percateg[categid]['boxids'][idx]
    instid = data_percateg[categid]['instids'][idx]
    #load the gt
    [inst, categories] = sbd.load_gt(self.names[imid])
    masks = np.zeros((idx.size, inst.shape[0],inst.shape[1]))
    for k in range(idx.size):
      masks[k,:,:] = (inst==instid[k]+1).astype(np.float32)
    categids = categid*np.ones(idx.size)

    #get the blobs
    im_new, spp_boxes, normalized_boxes, categids, masksblob, instance_wts = get_blobs(img, boxes, categids, masks)

    #save blobs in private dict
    self.blobs['image']=im_new.astype(np.float32)
    self.blobs['normalizedboxes']=normalized_boxes.astype(np.float32)
    self.blobs['sppboxes']=spp_boxes.astype(np.float32)
    self.blobs['categids']=categids.astype(np.float32)
    self.blobs['labels']=maskblob.astype(np.float32)
    self.blobs['instance_wts']=instance_wts.astype(np.float32)

    #and reshape
    for i in range(top.size):
      top[i].reshape(*(self.blobs[self.blob_names[i]].shape))

  def forward(self, bottom, top):
    for i in range(top.size):
      top[i].data[...] = self.blobs[self.blob_names[i]]

  def backward(self, top, propagate_down, bottom):
    pass

    
 

