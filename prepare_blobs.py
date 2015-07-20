import numpy as np
import cv2
import sds_config as cfg
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
  spp_boxes = np.round(np.divide(np.multiply(boxes, scale_factors[[1,0,1,0]]), cfg.FEAT_SCALE))
  spp_boxes = np.hstack((np.zeros((num_boxes,1), spp_boxes.dtype), spp_boxes))
  return spp_boxes

def get_normalized_boxes(boxes, categids, im_shape):
  im_shape_1 = np.array(im_shape, dtype=np.float32)
  num_boxes = boxes.shape[0]
  #the normalized boxes
  normalized_boxes = np.divide(boxes+np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float32), 
                                im_shape_1[[1,0,1,0]]-1)
  normalized_boxes = np.hstack((np.zeros((num_boxes,1),normalized_boxes.dtype), categids.reshape((1,-1)),normalized_boxes)) 
  return normalized_boxes

def get_clipped_resized_masks(boxes, input_masks):
  target_size = cfg.MASK_SIZE
  num_boxes = boxes.shape[0]
  num_channels = input_masks.shape[1]
  #masks = cymask.clip_resize_mask(sp, reg2sp, boxes, target_size)
  masks = np.zeros((num_boxes, num_channels, target_size, target_size))
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
    for l in range(num_channels):
      M = input_masks[k,l,yv1, xv1]
      masks[k,l,:,:]=M
    #S = sp[box[1]:box[3]+1,box[0]:box[2]+1]
    #M = reg2sp[:,k]
    #M = M[S-1]
    #start = time.clock()
    #masks[k,:,:] = scipymisc.imresize(M, (target_size, target_size), 'nearest')
    #stop = time.clock()
    #totaltime+=stop-start
 
  return masks
 


def get_blobs(img, boxes, categids, labelmasks=None):
  boxes = boxes.astype(np.float32)
  im_shape = img.shape
  #resize
  im_new, scale_factors = get_resized_image(img)
  
  #use boxes to produce two things: spp_boxes, which are boxes snapped to the pool5 feat map, and normalized boxes which can be used anywhere
  spp_boxes = get_boxes_for_spp(boxes, scale_factors)
  normalized_boxes = get_normalized_boxes(boxes, categids, im_shape)
  
  #if labelmasks are available, crop and resize them
  #also compute trainign loss weights based on box size
  if labelmasks is not None:
    masks = get_clipped_resized_masks(boxes, labelmasks)
    box_sizes = np.prod(boxes[:,[2,3]]-boxes[:,[0,1]]+1,1)
    instance_wts = np.minimum(box_sizes/(224.*224.), 1.)
  else:
    masks = None
    instance_wts =  None


  #reshape these things to match the network
  #the image needs a channel swap
  channel_swap = (2, 0, 1)
  im_new = im_new.transpose(channel_swap)
  im_new = im_new.reshape((1,im_new.shape[0], im_new.shape[1], im_new.shape[2]))
  spp_boxes = spp_boxes.reshape((spp_boxes.shape[0], spp_boxes.shape[1],1,1))
  normalized_boxes = normalized_boxes.reshape((normalized_boxes.shape[0], normalized_boxes.shape[1],1,1))
  categids = categids.reshape((-1,1,1,1))
 
  if masks is not None:
    instance_wts = instance_wts.reshape((-1,1,1,1))
    return im_new, spp_boxes, normalized_boxes, categids, masks, instance_wts
  else:
    return im_new, spp_boxes, normalized_boxes, categids
 
