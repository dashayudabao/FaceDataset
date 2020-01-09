# -*- coding: utf-8 -*-
import caffe
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
#import Image
import sys
import os
from  math import pow
from PIL import Image, ImageDraw, ImageFont
import cv2
import math
import random
caffe_root = '/home/cy/CodeDemo/caffe/'

sys.path.insert(0, caffe_root + 'python')
os.environ['GLOG_minloglevel'] = '2'

caffe.set_mode_cpu()
#####################################################################################################
# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('/home/cy/CodeDemo/faceDetect/MY_MN_Deploy.prototxt',
                '/home/cy/CodeDemo/faceDetect/model/MN_train_solver_iter_20000.caffemodel',
                caffe.TEST)
params = ['fc1']
# fc_params = {name: (weights, biases)}
fc_params = {pr: net.params[pr][0].data for pr in params}

for fc in params:
    print '{} weights are {} dimensional '.format(fc, fc_params[fc][0].shape)
#######################################################################################################
# Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net('/home/cy/CodeDemo/faceDetect/MY_MN_full_deploy.prototxt',
                          '/home/cy/CodeDemo/faceDetect/model/MN_train_solver_iter_20000.caffemodel',
                          caffe.TEST)
params_full_conv = ['fc1-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: net_full_conv.params[pr][0].data for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional '.format(conv, conv_params[conv][0].shape)
#############################################################################################################
for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    # conv_params[pr_conv][1][...] = fc_params[pr][1]
##############################################################################################################
net_full_conv.save('/home/cy/CodeDemo/faceDetect/model/MY_MN_full_conv.caffemodel')