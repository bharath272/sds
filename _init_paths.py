import os.path as osp
import sys
this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir,'caffe-sds', 'python')
sys.path.insert(0,caffe_path)






