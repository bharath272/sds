import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import my_accumarray as accum

def read_sp(filename):
  with open(filename, 'r') as f:
    s = f.read()
  f.close()
  values = [int(x) for x in s.split()]
  spimg = np.asarray(values[2:])
  spimg = np.reshape(spimg, (values[1], values[0]))
  spimg = np.transpose(spimg)
  return spimg

def read_sprep(spfilename, reg2spfilename):
  spimg = read_sp(spfilename)
  reg2sp = imread(reg2spfilename)
  reg2sp=reg2sp.astype(float, copy=False)
  return (spimg, reg2sp)

def write_reg2sp(reg2spfilename, reg2sp):
  imsave(reg2spfilename, reg2sp)

def write_sp(filename, sp):
  sp = np.transpose(sp)
  vals = sp.ravel().tolist()
  vals = [sp.shape[1], sp.shape[0]] + vals
  strs = [str(x) for x in vals]
  with open(filename, 'w') as f:
    for item in strs:
      f.write(item+' ')
  f.close()

def get_region_boxes(sp, reg2sp):
  x = np.arange(0, sp.shape[1])
  y = np.arange(0, sp.shape[0])
  xv, yv = np.meshgrid(x, y)
  maxsp = np.max(sp)
  sp1=sp.reshape(-1)-1
  xv = xv.reshape(-1)
  yv = yv.reshape(-1)
  spxmin = accum.my_accumarray(sp1,xv, maxsp, 'min')
  spymin = accum.my_accumarray(sp1,yv, maxsp, 'min')
  spxmax = accum.my_accumarray(sp1,xv, maxsp, 'max')
  spymax = accum.my_accumarray(sp1,yv, maxsp, 'max')
  
  Z = reg2sp.astype(float, copy=True)
  Z[reg2sp==0] = np.inf
  xmin = np.nanmin(np.multiply(spxmin.reshape(-1,1), Z),0)
  ymin = np.nanmin(np.multiply(spymin.reshape(-1,1), Z),0)
  xmax = np.amax(np.multiply(spxmax.reshape(-1,1), reg2sp),0)
  ymax = np.amax(np.multiply(spymax.reshape(-1,1), reg2sp), 0)
  xmin[np.isinf(xmin)]=0
  ymin[np.isinf(ymin)]=0
  

  boxes = np.hstack((xmin.reshape(-1,1), ymin.reshape(-1,1), xmax.reshape(-1,1), ymax.reshape(-1,1)))
  return boxes 
  

def project_to_sp(sp, mask):
  sp1=sp.reshape(-1)-1
  mask1 = mask.reshape(-1)
  maxsp = np.max(sp)
  reg2sp = accum.my_accumarray(sp1,mask1,maxsp)
  areas = accum.my_accumarray(sp1,1,maxsp)
  reg2sp = reg2sp/areas
  projected_mask = reg2sp[sp-1]
  return reg2sp, projected_mask
