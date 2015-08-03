import numpy as np
import sys
#sys.path.insert(0, '/home/eecs/bharath2/final_sds/caffe/python')
import caffe_paths
import caffe
import sds_config as cfg
import cv2
import scipy.misc as scipymisc
import time
from prepare_blobs import get_blobs
import sbd
import superpixel_representation as sprep
import evaluation
import math
import my_accumarray as accum
def get_hypercolumn_prediction(net, img, boxes, categids):

  #clip boxes to image
  boxes = clip_boxes(boxes, img.shape) 

  im_new, spp_boxes, normalized_boxes, categids = get_blobs(img, boxes, categids)
 
  #reshape the network
  net.blobs['image'].reshape(*(im_new.shape))
  net.blobs['normalizedboxes'].reshape(*(normalized_boxes.shape))
  net.blobs['sppboxes'].reshape(*(spp_boxes.shape))
  net.blobs['categids'].reshape(*(categids.shape))

  #forward
  output_blobs = net.forward(image=im_new, normalizedboxes=normalized_boxes,sppboxes=spp_boxes, categids=categids)

  output =output_blobs['loss']
  return output

def clip_boxes(boxes, im_shape):
  clipped_boxes = boxes.copy()
  clipped_boxes[:,[0,2]] = np.maximum(0, np.minimum(im_shape[1]-1, clipped_boxes[:,[0,2]]))
  clipped_boxes[:,[1,3]] = np.maximum(0, np.minimum(im_shape[0]-1, clipped_boxes[:,[1,3]]))
  return clipped_boxes


def paste_output(output, boxes, im_shape, target_output_size = [50, 50]):
  pasted_output =np.zeros((output.shape[0], output.shape[1], im_shape[0], im_shape[1]))
  #clip and round
  new_boxes = np.hstack((np.ceil(boxes[:,0:2]),np.ceil(boxes[:,2:4])))
  new_boxes = clip_boxes(new_boxes, im_shape)
  xmin = new_boxes[:,0]
  ymin = new_boxes[:,1]
  w = new_boxes[:,2]-xmin+1
  h = new_boxes[:,3]-ymin+1
  for item in range(output.shape[0]):
    xmin_ = xmin[item]
    ymin_ = ymin[item]
    w_ = w[item]
    h_ = h[item]
    y_all = np.arange(ymin_, ymin_+h_)
    Y_all = (y_all-ymin_)*target_output_size[0]/h_
    Y_all = np.maximum(0, np.minimum(target_output_size[0]-1, np.floor(Y_all)))

    x_all = np.arange(xmin_, xmin_+w_)
    X_all = (x_all-xmin_)*target_output_size[1]/w_
    X_all = np.maximum(0, np.minimum(target_output_size[1]-1, np.floor(X_all)))


    

    for channel in range(output.shape[1]):
      pasted_output_this = pasted_output[item,channel]
      output_this = output[item, channel]
      for i in range(len(Y_all)):
        pasted_output_this_row = pasted_output_this[y_all[i]]
        output_this_row = output_this[Y_all[i]]
        for j in range(len(X_all)):
          #X = (x-xmin_)*target_output_size[1]/w_
          #X = math.floor(X)
          #X = max(0, min(target_output_size[1]-1,X))
          pasted_output_this_row[x_all[j]] = output_this_row[X_all[j]]

  return pasted_output
   
def paste_output_sp(output, boxes, im_shape, sp, target_output_size = [50, 50]):
  pasted_output =np.zeros((output.shape[0], output.shape[1], np.max(sp)))
  #clip and round
  new_boxes = np.hstack((np.ceil(boxes[:,0:2]),np.ceil(boxes[:,2:4])))
  new_boxes = clip_boxes(new_boxes, im_shape)
  xmin = new_boxes[:,0]
  ymin = new_boxes[:,1]
  w = new_boxes[:,2]-xmin+1
  h = new_boxes[:,3]-ymin+1
  for item in range(output.shape[0]):
    xmin_ = xmin[item]
    ymin_ = ymin[item]
    w_ = w[item]
    h_ = h[item]
    y_all = np.arange(ymin_, ymin_+h_)
    Y_all = (y_all-ymin_)*target_output_size[0]/h_
    Y_all = np.maximum(0, np.minimum(target_output_size[0]-1, np.floor(Y_all)))

    x_all = np.arange(xmin_, xmin_+w_)
    X_all = (x_all-xmin_)*target_output_size[1]/w_
    X_all = np.maximum(0, np.minimum(target_output_size[1]-1, np.floor(X_all)))


    

    for channel in range(output.shape[1]):
      pasted_output_this = pasted_output[item,channel]
      output_this = output[item, channel]
      for i in range(len(Y_all)):
        sp_this_row = sp[y_all[i]]
        output_this_row = output_this[Y_all[i]]
        for j in range(len(X_all)):
          #X = (x-xmin_)*target_output_size[1]/w_
          #X = math.floor(X)
          #X = max(0, min(target_output_size[1]-1,X))
          pasted_output_this[sp_this_row[x_all[j]]-1] = output_this_row[X_all[j]]

  return pasted_output
 
def get_all_outputs(net,names, dets, imgpath, sppath, regsppath,thresh=0.4,outpath=None):
  numcategs=dets['boxes'].size
  all_ov=[]
  for j in range(numcategs):
    all_ov.append([])

  for i in range(len(names)):
    t1=time.time()
    print 'Doing'+str(i)
    img = cv2.imread(imgpath.format(names[i]))
    #get all boxes for this image
    boxes_img = np.zeros((0,4))
    cids_img = np.zeros((0,1))
    for j in range(numcategs):
      boxes_img = np.vstack((boxes_img, dets['boxes'][j][i]))
      cids_img = np.vstack((cids_img, j*np.ones((dets['boxes'][j][i].shape[0],1))))
    #get the predictions
    print boxes_img.shape
    t2=time.time()
    output = get_hypercolumn_prediction(net, img, boxes_img.astype(np.float32), cids_img)
    t2pt5 = time.time()
    #pasted_output = paste_output(output, boxes_img.astype(np.float32), img.shape)
    t3=time.time()
    #project to sp
    (sp, reg2sp) = sprep.read_sprep(sppath.format(names[i]), regsppath.format(names[i]))
    #newreg2sp_all = np.zeros((np.max(sp),cids_img.size))
    #for j in range(cids_img.size):
    #  newreg2sp = sprep.project_to_sp(sp, pasted_output[j,0])
    #  newreg2sp_all[:,j] = newreg2sp
    counts_all = paste_output_sp(output, boxes_img.astype(np.float32), img.shape, sp)
    counts_all =np.squeeze(counts_all)
    counts = accum.my_accumarray(sp.reshape(-1)-1,1,np.max(sp))
    newreg2sp_all = counts_all/counts
    newreg2sp_all = newreg2sp_all>=thresh
    newreg2sp_all = newreg2sp_all.T
    t4=time.time()
    #evaluate
    inst, categories = sbd.load_gt(names[i])
    ov = evaluation.compute_overlap_sprep(sp, newreg2sp_all, inst)
    #separate according to categories
    for j in range(numcategs):
      all_ov[j].append(ov[:,cids_img.reshape(-1)==j])
    t5=time.time()
    print 'Get boxes:{:f}, get pred:{:f},{:f}, get sp:{:f}, get ov:{:f}'.format(t2-t1,t2pt5-t2, t3-t2pt5,t4-t3,t5-t4)
    #save if needed
    if outpath is not None:
      cv2.imwrite(outpath.format(names[i]), newreg2sp_all)
  return all_ov 
