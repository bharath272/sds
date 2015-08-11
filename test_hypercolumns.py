import numpy as np
import sys
import _init_paths
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
from scipy.io import savemat, loadmat
def get_hypercolumn_prediction(net, img, boxes, categids):
  boxes = boxes.copy()
  #clip boxes to image
  boxes = clip_boxes(boxes-1, img.shape) 

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
  w = new_boxes[:,2]-xmin+1.
  h = new_boxes[:,3]-ymin+1.
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


    x_all = x_all.astype(np.intp)
    y_all = y_all.astype(np.intp)
    X_all = X_all.astype(np.intp)
    Y_all = Y_all.astype(np.intp)

    spind = sp[np.ix_(y_all,x_all)]-1
       

    for channel in range(output.shape[1]):
      pasted_output_this = pasted_output[item,channel]
      output_this = output[item, channel]
      vals = output_this[np.ix_(Y_all,X_all)]
      np.add.at(pasted_output_this, spind, vals)


  #finally divide by sum
  counts = accum.my_accumarray(sp.reshape(-1)-1,1,np.max(sp))
  pasted_output = pasted_output/counts.reshape((1,1,-1))
  return pasted_output
 
def get_all_outputs(net,names, dets, imgpath, sppath, regsppath,thresh=0.4,outpath=None, do_eval = True, eval_thresh = [0.5, 0.7]):
  numcategs=dets['boxes'].size

  if do_eval:
    #we will accumulate the overlaps and the classes of the gt
    all_ov=[]
    gt = []
    for j in range(numcategs):
      all_ov.append([])


  #a dictionary of times
  times = {}
  times['boxes']=0.
  times['pred']=0.
  times['sp']=0.
  times['ov']=0.
  times['total']=0.
  for i in range(len(names)):
    t1=time.time()
    img = cv2.imread(imgpath.format(names[i]))
    #get all boxes for this image
    boxes_img = np.zeros((0,4))
    cids_img = np.zeros((0,1))
    for j in range(numcategs):
      boxes_img = np.vstack((boxes_img, dets['boxes'][j][i]))
      cids_img = np.vstack((cids_img, j*np.ones((dets['boxes'][j][i].shape[0],1))))
    t2=time.time()
    times['boxes']=times['boxes']+t2-t1


    #get the predictions
    output = get_hypercolumn_prediction(net, img, boxes_img.astype(np.float32), cids_img)
    t3=time.time()
    times['pred']=times['pred']+t3-t2
    #project to sp
    (sp, reg2sp) = sprep.read_sprep(sppath.format(names[i]), regsppath.format(names[i]))
    newreg2sp_all = paste_output_sp(output, boxes_img.astype(np.float32)-1., sp.shape, sp)
    newreg2sp_all = np.squeeze(newreg2sp_all)
    newreg2sp_all = newreg2sp_all>=thresh
    newreg2sp_all = newreg2sp_all.T
    t4=time.time()
    times['sp'] = times['sp']+t4-t3
    #save if needed
    if outpath is not None:
      savemat(outpath.format(names[i]), {'output':output})
    #evaluate
    if do_eval:
      inst, categories = sbd.load_gt(names[i])
      ov = evaluation.compute_overlap_sprep(sp, newreg2sp_all, inst)
      #separate according to categories
      for j in range(numcategs):
        all_ov[j].append(ov[:,cids_img.reshape(-1)==j])
      #append categories
      gt.append(np.squeeze(categories-1))
    t5=time.time()
    times['ov'] = times['ov']+t5-t4
    if i % 100 == 0:
      total = float(i+1)
      print 'Doing : {:d}, get boxes:{:.2f} s, get pred:{:.2f} s, get sp:{:.2f} s, get ov:{:.2f} s'.format(i, times['boxes']/total,
                        times['pred']/total, times['sp']/total, times['ov']/total)

  ap = [[] for _ in eval_thresh]
  prec = [[] for _ in eval_thresh]
  rec = [[] for _ in eval_thresh]
  for i in range(numcategs):
    print 'Evaluating :{:d}'.format(i)

    for t,thr in enumerate(eval_thresh):
      ap_, prec_, rec_ = evaluation.generalized_det_eval_simple(dets['scores'][i].tolist()[0:len(names)], all_ov[i], gt, i, thr)
      ap[t].append(ap_)
      prec[t].append(prec_)
      rec[t].append(rec_)
  return ap, prec, rec, all_ov, gt
