import numpy as np
import sys
sys.path.insert(0, '/home/eecs/bharath2/final_sds/caffe/python')
import caffe
import sds_config as cfg
import cv2
import scipy.misc as scipymisc
import time
from prepare_blobs import get_blobs

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

    for channel in range(output.shape[1]):
      for y in np.arange(ymin_, ymin_+h_):
        Y = (y - ymin_)*target_output_size[0]/h_
        Y = np.floor(Y)
        Y = np.maximum(0, np.minimum(target_output_size[0]-1,Y))
        for x in np.arange(xmin_, xmin_+w_):
          X = (x-xmin_)*target_output_size[1]/w_
          X = np.floor(X)
          X = np.maximum(0, np.minimum(target_output_size[1]-1,X))
          pasted_output[item,channel,y,x] = output[item, channel, Y, X]

  return pasted_output
   

