from __future__ import division
import os
import time
from glob import glob
import numpy as np
from six.moves import xrange
from collections import namedtuple
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage import  color
import scipy.misc
import scipy.io as sio
import imageio
import argparse

#### wMAE, wRMSE
parser = argparse.ArgumentParser(description='')
parser.add_argument('--result_path',required=True, help='Pls set your result path')
args = parser.parse_args()

temp=sio.loadmat('weight_matrix.mat')
weight_matrix=temp['weight_matrix']

test_total_se=0
test_total_ae=0
test_total_count=0
start=time.time()
for i in range(401,500+1):
    ####  change image result path here
    im_out = imageio.imread('%s/%.4d.png' % (args.result_path,i))
    
    h,w,channel=im_out.shape
    im_groundtruth1 = imageio.imread('gt_images/%.4d_1.png' % (i))
    im_groundtruth1 = np.float64(im_groundtruth1)
    im_groundtruth2 = imageio.imread('gt_images/%.4d_2.png' % (i))
    im_groundtruth2 = np.float64(im_groundtruth2)
    im_groundtruth3 = imageio.imread('gt_images/%.4d_3.png' % (i))
    im_groundtruth3 = np.float64(im_groundtruth3)
    im_groundtruth4 = imageio.imread('gt_images/%.4d_4.png' % (i))
    im_groundtruth4 = np.float64(im_groundtruth4)
    im_groundtruth5 = imageio.imread('gt_images/%.4d_5.png' % (i))
    im_groundtruth5 = np.float64(im_groundtruth5)
    image_ae = np.sum(np.abs(im_out-im_groundtruth1))*weight_matrix[i-1,0] \
                +np.sum(np.abs(im_out-im_groundtruth2))*weight_matrix[i-1,1] \
                +np.sum(np.abs(im_out-im_groundtruth3))*weight_matrix[i-1,2] \
                +np.sum(np.abs(im_out-im_groundtruth4))*weight_matrix[i-1,3] \
                +np.sum(np.abs(im_out-im_groundtruth5))*weight_matrix[i-1,4] 
    image_se = np.sum(np.square(im_out-im_groundtruth1))*weight_matrix[i-1,0] \
                +np.sum(np.square(im_out-im_groundtruth2))*weight_matrix[i-1,1] \
                +np.sum(np.square(im_out-im_groundtruth3))*weight_matrix[i-1,2] \
                +np.sum(np.square(im_out-im_groundtruth4))*weight_matrix[i-1,3] \
                +np.sum(np.square(im_out-im_groundtruth5))*weight_matrix[i-1,4]
    test_total_ae=test_total_ae + image_ae
    test_total_se=test_total_se + image_se
    test_total_count=test_total_count+h*w*channel

print ('test WMAE: %.3f' % (test_total_ae/test_total_count))
print ((time.time()-start)/100)
print ('test WRMSE: %.3f' % (np.sqrt(test_total_se/test_total_count)))






