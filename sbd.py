from scipy.io import loadmat
import os
import my_accumarray as accum
import numpy as np


sbddir = '/work5/bharath2/SBD_check/benchmark_RELEASE/dataset'

def load_gt(name):
  filename = os.path.join(sbddir, 'cls', name+'.mat')
  output = loadmat(filename)
  filename = os.path.join(sbddir, 'inst', name+'.mat')
  output = loadmat(filename)
  inst = output['GTinst'][0,0]['Segmentation']
  categories = output['GTinst'][0,0]['Categories']
  return inst, categories


def get_bboxes(inst):
  x = np.arange(0, inst.shape[1])
  y = np.arange(0, inst.shape[0])
  xv, yv = np.meshgrid(x, y)
  maxinst = np.max(inst)
  
  inst1 = inst.reshape(-1)
  xv = xv.reshape(-1)
  yv = yv.reshape(-1)
  
  idx = inst1>0
  inst1=inst1[idx]
  xv=xv[idx]
  yv = yv[idx]
  instxmin = accum.my_accumarray(inst1-1, xv, maxinst, 'min')
  instymin = accum.my_accumarray(inst1-1, yv, maxinst, 'min')
  instxmax = accum.my_accumarray(inst1-1, xv, maxinst, 'max')
  instymax = accum.my_accumarray(inst1-1, yv, maxinst, 'max')
  boxes = np.hstack((instxmin.reshape(-1,1),instymin.reshape(-1,1),instxmax.reshape(-1,1), instymax.reshape(-1,1)))
  return boxes

def get_all_sbdboxes(names):
  boxes=[]
  for i, name in enumerate(names):
    inst, categories = load_gt(name)
    boxes.append(get_bboxes(inst.astype(int)))
    if i%10 == 0:
      print str(i)+':'+name
  return boxes




  
