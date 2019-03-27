from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage import  color
#import matplotlib.pyplot  as plt
#matplotlib inline
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import scipy.misc
import scipy.io as sio

from module import SRGAN_delta
tf.reset_default_graph()
test_input = tf.placeholder(tf.float32,[1, None, None,3], name='test')
test_input_scale=test_input/127.5-1
test_output=SRGAN_delta(test_input_scale,is_train=False,reuse=False)
saver = tf.train.Saver()
sess=tf.Session()

saver.restore(sess, 'checkpoint_delta/model-345002') #mae:
#saver.restore(sess, 'checkpoint_delta/model-191002') #mae:


#MAE

temp=sio.loadmat('../dataset/weight_matrix.mat')
weight_matrix=temp['weight_matrix']

test_total_ae=0
test_total_se=0
test_total_count=0
start=time.time()
for i in range(401,500+1):
    #print(i,time.time()-start)
    #start=time.time()
    print(i)
    im = scipy.misc.imread('../dataset/origin_images/%.4d.png' % (i))
    im = np.float32(im)
    batch_images=[im]         
    test_output_eval= sess.run(test_output,feed_dict={test_input: batch_images}) 
    im_out=test_output_eval[0]
    h,w,channel=im_out.shape
    im_out=(np.float64(im_out)+1)/2*255
    im_out[im_out>255]=255
    im_out[im_out<0]=0
    scipy.misc.imsave('result_ResNet/%.4d.png'%(i),im_out/255)
    im_groundtruth1 = scipy.misc.imread('../dataset/gt_images/%.4d_1.png' % (i))
    im_groundtruth1 = np.float64(im_groundtruth1)
    im_groundtruth2 = scipy.misc.imread('../dataset/gt_images/%.4d_2.png' % (i))
    im_groundtruth2 = np.float64(im_groundtruth2)
    im_groundtruth3 = scipy.misc.imread('../dataset/gt_images/%.4d_3.png' % (i))
    im_groundtruth3 = np.float64(im_groundtruth3)
    im_groundtruth4 = scipy.misc.imread('../dataset/gt_images/%.4d_4.png' % (i))
    im_groundtruth4 = np.float64(im_groundtruth4)
    im_groundtruth5 = scipy.misc.imread('../dataset/gt_images/%.4d_5.png' % (i))
    im_groundtruth5 = np.float64(im_groundtruth5)
    image_ae = np.sum(np.abs(im_out-im_groundtruth1))*weight_matrix[i-1,0] \
                +np.sum(np.abs(im_out-im_groundtruth2))*weight_matrix[i-1,1] \
                +np.sum(np.abs(im_out-im_groundtruth3))*weight_matrix[i-1,2] \
                +np.sum(np.abs(im_out-im_groundtruth4))*weight_matrix[i-1,3] \
                +np.sum(np.abs(im_out-im_groundtruth5))*weight_matrix[i-1,4] 
    test_total_ae=test_total_ae + image_ae

    image_se = np.sum(np.square(im_out-im_groundtruth1))*weight_matrix[i-1,0] \
                +np.sum(np.square(im_out-im_groundtruth2))*weight_matrix[i-1,1] \
                +np.sum(np.square(im_out-im_groundtruth3))*weight_matrix[i-1,2] \
                +np.sum(np.square(im_out-im_groundtruth4))*weight_matrix[i-1,3] \
                +np.sum(np.square(im_out-im_groundtruth5))*weight_matrix[i-1,4] 
    test_total_se=test_total_se + image_se

    test_total_count=test_total_count+h*w*channel
    
    ### save output
print ('test wMAE: %.3f' % (test_total_ae/test_total_count))
print ((time.time()-start)/100)
print ('test wRMSE: %.3f' % (np.sqrt(test_total_se/test_total_count)))



    
