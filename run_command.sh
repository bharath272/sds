#download training and gt and initial model
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/hypercolumn/train_mcg_boxes.pkl
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/hypercolumn/gttrain.pkl
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/hypercolumn/vgg16_fast_rcnn_MCGdedup5K_voc_2012_train_iter_40000.caffemodel
PYTHONUNBUFFERED="True" PYTHONPATH=:caffe/python/ GLOG_logtostderr=1 caffe/build/tools/caffe.bin train   -gpu 2   -solver model_defs/solver.prototxt -weights vgg16_fast_rcnn_MCGdedup5K_voc_2012_train_iter_40000.caffemodel 
