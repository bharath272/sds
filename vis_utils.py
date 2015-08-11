import numpy as np
def color_seg(mask, img):
  #convert to single
  img2 = img.copy().astype(np.float32)
  mask2 = mask.copy().astype(np.float32)[:,:,np.newaxis]
  color = np.array([255., 0., 0.])
  color = color[np.newaxis, np.newaxis,:]
  colorbg = np.array([255., 255., 255.])
  colorbg = colorbg[np.newaxis, np.newaxis,:]
  img2 = 0.5*img2 + 0.5*mask2*color + 0.5*(1.-mask2)*colorbg
  return img2.astype(np.uint8)
