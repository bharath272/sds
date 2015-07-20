import numpy as np
import sys
sys.path.insert(0, '/home/eecs/bharath2/final_sds/caffe/python')
import caffe
import sds_config as cfg
import cv2
import scipy.misc as scipymisc
import time
class SDSNet:
  def __init__(self, conv5def, scoredef, regressdef, refinedef, netfile):
    self.conv5_net = caffe.Net(conv5def, netfile, caffe.TEST)
    self.regress_net = caffe.Net(regressdef, netfile, caffe.TEST)
    self.refine_net = caffe.Net(refinedef, netfile, caffe.TEST)
    self.score_net = caffe.Net(scoredef, netfile, caffe.TEST)
    
 
  def get_conv5(self, image):
    #first get the image blob
    imshape=image.shape
    im_blob = np.zeros(1,imshape[2], imshape[0], imshape[1])
    im_blob[0,:,:,:] = image.transpose(2,0,1)
    #then reshape
    self.conv5_net.blobs['data'].reshape(*(im_blob.shape))
    out = self.conv5_net.forward(data=im_blob.astype(np.float32, copy=False))
    conv5 = out['conv5_3']
    return conv5
  
  def get_scores(self, conv5, spp_boxes_in, masks_in):
    #get the sppboxes reshaped
    spp_boxes = np.reshape(spp_boxes_in,(spp_boxes_in.shape[0], spp_boxes_in.shape[1], 1, 1))
    #get the masks reshaped
    masks = np.reshape(masks_in,(masks_in.shape[0],1,masks_in.shape[1], masks_in.shape[2]))
    
    #batch the boxes, to avoid running out of memory
    num_boxes = spp_boxes.shape[0]
    num_batches = int(num_boxes)/int(cfg.BATCH_SIZE_TEST)
    if num_batches*cfg.BATCH_SIZE_TEST<num_boxes:
      num_batches+= 1
    scores = np.zeros(num_boxes, 20)
    for i in range(num_batches):
      start = i*cfg.BATCH_SIZE_TEST
      stop = min(start+cfg.BATCH_SIZE_TEST-1,num_boxes-1)
      spp_boxes_batch = spp_boxes[start:stop,:,:,:]
      masks_batch = masks[start:stop,:,:,:]
      #reshape the net
      self.score_net.blobs['conv5_3'].reshape(*(conv5.shape))
      self.score_net.blobs['sppboxes'].reshape(*(spp_boxes_batch.shape))
      self.score_net.blobs['masks'].reshape(*(masks_batch.shape))
      #forward
      out = self.score_net.forward(conv5_3=conv5.astype(np.float32, copy=False),
                                 sppboxes=spp_boxes_batch.astype(np.float32, copy=False),
                                 masks=masks_batch.astype(np.float32, copy=False))
      scores_batch = out['cls_prob']
      #remove background
      scores_batch = scores_batch[:,1:]
      scores[start:stop,:] = scores_batch
    return scores 
  
  def get_regressed_boxes(self, conv5, spp_boxes_in, masks_in, cids_in):
    #get the sppboxes reshaped
    spp_boxes = np.reshape(spp_boxes_in,(spp_boxes_in.shape[0], spp_boxes_in.shape[1], 1, 1))
    #get the masks reshaped
    masks = np.reshape(masks_in,(masks_in.shape[0],1,masks_in.shape[1], masks_in.shape[2]))
    #get the cids reshaped   
    cids = np.reshape(cids_in,(cids_in.size, 1, 1, 1)) 	
    #batch the boxes, to avoid running out of memory
    num_boxes = spp_boxes.shape[0]
    num_batches = int(num_boxes)/int(cfg.BATCH_SIZE_TEST)
    if num_batches*cfg.BATCH_SIZE_TEST<num_boxes:
      num_batches+= 1
    scores = np.zeros(num_boxes, 20)
    for i in range(num_batches):
      start = i*cfg.BATCH_SIZE_TEST
      stop = min(start+cfg.BATCH_SIZE_TEST-1,num_boxes-1)
      spp_boxes_batch = spp_boxes[start:stop,:,:,:]
      masks_batch = masks[start:stop,:,:,:]
      #reshape the net
      self.score_net.blobs['conv5_3'].reshape(*(conv5.shape))
      self.score_net.blobs['sppboxes'].reshape(*(spp_boxes_batch.shape))
      self.score_net.blobs['masks'].reshape(*(masks_batch.shape))
      #forward
      out = self.score_net.forward(conv5_3=conv5.astype(np.float32, copy=False),
                                 sppboxes=spp_boxes_batch.astype(np.float32, copy=False),
                                 masks=masks_batch.astype(np.float32, copy=False))
      scores_batch = out['cls_prob']
      #remove background
      scores_batch = scores_batch[:,1:]
      scores[start:stop,:] = scores_batch
    return scores

 
def get_resized_image(image):
  im = image.astype(np.float32, copy=True)
  im -= cfg.PIXEL_MEANS
  im_shape = im.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  target_size = cfg.TARGET_SIZE
  target_scale = float(target_size)/float(im_size_min)
  max_size_after_resize = np.round(im_size_max*target_scale)
  if max_size_after_resize>cfg.MAX_SIZE:
    target_scale = float(cfg.MAX_SIZE)/float(im_size_max)
  im_new = cv2.resize(im, None, None, fx=target_scale, fy=target_scale, 
                         interpolation=cv2.INTER_LINEAR)
  new_im_shape = im_new.shape
  final_scale_factors = np.divide(np.array(new_im_shape, dtype=np.float32), 
                               np.array(im_shape, dtype=np.float32))
  return im_new, final_scale_factors


def get_boxes_for_spp(boxes, scale_factors):
  num_boxes = boxes.shape[0]
  #the boxes snapped to the feature map
  spp_boxes = np.round(np.divide(np.multiply(boxes, scale_factors), cfg.FEAT_SCALE))
  spp_boxes = np.concatenate((np.zeros(num_boxes,1), spp_boxes), 1)
  return spp_boxes

def get_normalized_boxes(boxes, categids, im_shape):
  num_boxes = boxes.shape[0]
  #the normalized boxes
  normalized_boxes = np.divide(boxes+np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float32), 
                                np.array(im_shape-1, dtype=np.float32))
  normalized_boxes = np.concatenate((np.zeros(num_boxes,1), categids,normalized_boxes), 1) 
  return normalized_boxes

def get_clipped_resized_masks(boxes, sp, reg2sp):
  target_size = cfg.MASK_SIZE
  num_boxes = boxes.shape[0]
  #masks = cymask.clip_resize_mask(sp, reg2sp, boxes, target_size)
  masks = np.zeros((num_boxes, target_size, target_size))
  totaltime=0.
  yv = np.arange(target_size).reshape(-1,1)
  xv = np.arange(target_size).reshape(1,-1)
  yv = np.true_divide(yv, float(target_size)-1.)
  xv = np.true_divide(xv, float(target_size)-1.)
  
  for k in range(num_boxes):
    box = boxes[k,:]
    xmin = box[0]
    ymin = box[1]
    w = box[2]-box[0]
    h = box[3]-box[1]
    xv1 = np.round(xmin + xv*w).astype(int)
    yv1 = np.round(ymin + yv*h).astype(int)
    S = sp[yv1, xv1]
    R = reg2sp[:,k]
    masks[k,:,:]=R[S-1]
    #S = sp[box[1]:box[3]+1,box[0]:box[2]+1]
    #M = reg2sp[:,k]
    #M = M[S-1]
    #start = time.clock()
    #masks[k,:,:] = scipymisc.imresize(M, (target_size, target_size), 'nearest')
    #stop = time.clock()
    #totaltime+=stop-start
  print totaltime
  return masks
  
def do_region_nms(sp, reg2sp, scores, ov_thresh):
  #compute overlap
  spareas = np.bincount(np.ravel(sp-1))
  spareas = np.reshape(spareas,(-1,1))
  s = str(reg2sp.shape[1])
  print s
  inters = np.dot(reg2sp.T,np.multiply(spareas, reg2sp))
  print s
  totalareas = np.dot(spareas.T, reg2sp)
  uni = np.add(totalareas.T, totalareas)-inters
  overlap = np.divide(inters,uni)
  print np.max(overlap)
  #get sorted indices
  sorted_idx = np.argsort(scores, 0)
  sorted_idx = sorted_idx[::-1,:]
  #next do nms
  chosen = []
  for categ in range(scores.shape[1]):
    regions = sorted_idx[:,categ]
    to_consider = np.ones(len(regions), dtype=bool)
    picked = np.zeros(len(regions), dtype=bool)
    for i in range(len(regions)):
      if not to_consider[i]:
        continue
      picked[i] = True
      to_consider = np.bitwise_and(to_consider,overlap[regions,regions[i]]<ov_thresh)
    chosen.insert(categ,regions[picked])
    print str(len(chosen))
  return chosen

def test_full_sds(net, img, sp, reg2sp, boxes, nms_ov_thresh):
  #first resize the image
  im_new, final_scale_factors = get_resized_image(img)
  # and get conv features
  conv5 = net.get_conv5(im_new)
  #next get the boxes for spp
  spp_boxes = get_boxes_for_spp(boxes, final_scale_factors)
  # and the masks
  masks = get_clipped_resized_masks(boxes, sp, reg2sp)
  #next score the network
  scores = net.get_scores(conv5, spp_boxes, masks)
  #and regress boxes 
