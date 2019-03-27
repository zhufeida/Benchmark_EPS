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
from module import model_vdsr
import random
import scipy.misc
import utils2 as util
import scipy.io as sio

class mymodel(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.stride=21
        self.index_in_epoch=-1
        self.epochs_completed=0
        self.batch_input_images = self.batch_size * [None]
        self.batch_true_images1 = self.batch_size * [None]
        self.batch_true_images2 = self.batch_size * [None]
        self.batch_true_images3 = self.batch_size * [None]
        self.batch_true_images4 = self.batch_size * [None]
        self.batch_true_images5 = self.batch_size * [None]
        self.batch_true_weights1 = self.batch_size * [None]
        self.batch_true_weights2 = self.batch_size * [None]
        self.batch_true_weights3 = self.batch_size * [None]
        self.batch_true_weights4 = self.batch_size * [None]
        self.batch_true_weights5 = self.batch_size * [None]
        #self.loss_beta = 0.0001 
    def build_model(self,args):
        self.global_step = tf.Variable(0,trainable=False)
        self.global_step_add1= tf.assign_add(self.global_step,1)

        ### embedding
        self.x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="X")
        self.y1 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="Y1")
        self.y2 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="Y2")
        self.y3 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="Y3")
        self.y4 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="Y4")
        self.y5 = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="Y5")
        self.w1 = tf.placeholder(tf.float32, shape=[None,1,1,1], name="w1")
        self.w2 = tf.placeholder(tf.float32, shape=[None,1,1,1], name="w2")
        self.w3 = tf.placeholder(tf.float32, shape=[None,1,1,1], name="w3")
        self.w4 = tf.placeholder(tf.float32, shape=[None,1,1,1], name="w4")
        self.w5 = tf.placeholder(tf.float32, shape=[None,1,1,1], name="w5")

        self.y1_scale=self.y1
        self.y2_scale=self.y2
        self.y3_scale=self.y3 
        self.y4_scale=self.y4 
        self.y5_scale=self.y5
        self.net_g, weights   = model_vdsr(self.x,reuse=False)
        ####### [0,255]
        self.loss =   tf.reduce_mean(tf.abs(self.net_g - self.y1_scale)*self.w1 \
                    + tf.abs(self.net_g - self.y2_scale)*self.w2 \
                    + tf.abs(self.net_g - self.y3_scale)*self.w3 \
                    + tf.abs(self.net_g - self.y4_scale)*self.w4 \
                    + tf.abs(self.net_g - self.y5_scale)*self.w5)
        
        self.loss_ReguTerm = self.calculate_ReguTerm()
        for w in weights:
            self.loss += tf.nn.l2_loss(w)*1e-4

        self.lr_input = tf.placeholder(tf.float32, shape=[], name="LearningRate")
        self.train_step = tf.train.AdamOptimizer(self.lr_input).minimize(self.loss+self.loss_ReguTerm)
        self.saver=tf.train.Saver()

        ########## test
        self.test_input = tf.placeholder(tf.float32,[1, None, None,3], name='test')
        #self.test_input_scale = self.test_input/127.5-1 [-1,1]
        self.test_input_scale = self.test_input ###[0,255]
        self.test_output, temp_weights = model_vdsr(self.test_input_scale,reuse=True)

    def train(self, args):
        """Train cyclegan"""
        
        self.sess.run(tf.global_variables_initializer())

        counter = 0
        start_time = time.time()

        if self.load(args.checkpoint_path):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.prepare_train_patch()

        start_step = self.sess.run(self.global_step)

        for iter_count in range(start_step,int(args.iteration_num)): 
            self.build_current_batch()
            
            self.sess.run(self.global_step_add1)

            _ = self.sess.run([self.train_step], feed_dict={self.x: self.batch_input_images,
                                      self.y1: self.batch_true_images1, 
                                      self.y2: self.batch_true_images2,
                                      self.y3: self.batch_true_images3,
                                      self.y4: self.batch_true_images4,
                                      self.y5: self.batch_true_images5,
                                      self.w1: np.array(self.batch_true_weights1)[:,np.newaxis,np.newaxis,np.newaxis],
                                      self.w2: np.array(self.batch_true_weights2)[:,np.newaxis,np.newaxis,np.newaxis],
                                      self.w3: np.array(self.batch_true_weights3)[:,np.newaxis,np.newaxis,np.newaxis],
                                      self.w4: np.array(self.batch_true_weights4)[:,np.newaxis,np.newaxis,np.newaxis],
                                      self.w5: np.array(self.batch_true_weights5)[:,np.newaxis,np.newaxis,np.newaxis],
                                      self.lr_input: 0.0001})             
            
            if np.mod(iter_count,100) == 1:
                print("iter: [%.7d/%.7d] epochs: %.3d time: %4.4f"%\
                    ( iter_count,args.iteration_num,self.epochs_completed,time.time()-start_time))
                start_time=time.time()

            if np.mod(iter_count, 1000) == 2:
                self.save(args.checkpoint_path, iter_count)
                self.test()
   

    def prepare_train_patch(self):
        temp=sio.loadmat('../dataset/weight_matrix.mat')
        weight_matrix=temp['weight_matrix']

        count=400
        origin_images=count * [None]
        gt_images1=count * [None]
        gt_images2=count * [None]
        gt_images3=count * [None]
        gt_images4=count * [None]
        gt_images5=count * [None]
        for i in range(1,count+1):
            origin_images[i-1] = scipy.misc.imread('../dataset/origin_images/%.4d.png'% i)
            gt_images1[i-1] = scipy.misc.imread('../dataset/gt_images/%.4d_1.png'% i)
            gt_images2[i-1] = scipy.misc.imread('../dataset/gt_images/%.4d_2.png'% i)
            gt_images3[i-1] = scipy.misc.imread('../dataset/gt_images/%.4d_3.png'% i)
            gt_images4[i-1] = scipy.misc.imread('../dataset/gt_images/%.4d_4.png'% i)
            gt_images5[i-1] = scipy.misc.imread('../dataset/gt_images/%.4d_5.png'% i)

        batch_images_orig = count * [None]
        batch_images_gt1 = count * [None]
        batch_images_gt2 = count * [None]
        batch_images_gt3 = count * [None]
        batch_images_gt4 = count * [None]
        batch_images_gt5 = count * [None]
        batch_images_count = 0

        for i in range(count):
            batch_images_orig[i] = util.get_split_images(origin_images[i], self.patch_size, self.stride)
            batch_images_gt1[i] = util.get_split_images(gt_images1[i], self.patch_size, self.stride)
            batch_images_gt2[i] = util.get_split_images(gt_images2[i], self.patch_size, self.stride)
            batch_images_gt3[i] = util.get_split_images(gt_images3[i], self.patch_size, self.stride)
            batch_images_gt4[i] = util.get_split_images(gt_images4[i], self.patch_size, self.stride)
            batch_images_gt5[i] = util.get_split_images(gt_images5[i], self.patch_size, self.stride)
            batch_images_count += batch_images_orig[i].shape[0]

        self.patches_orig = batch_images_count * [None]
        self.patches_gt1 = batch_images_count * [None]
        self.patches_gt2 = batch_images_count * [None]
        self.patches_gt3 = batch_images_count * [None]
        self.patches_gt4 = batch_images_count * [None]
        self.patches_gt5 = batch_images_count * [None]
        self.weight_gt1 = batch_images_count * [None]
        self.weight_gt2 = batch_images_count * [None]
        self.weight_gt3 = batch_images_count * [None]
        self.weight_gt4 = batch_images_count * [None]
        self.weight_gt5 = batch_images_count * [None]
        no = 0
        for i in range(count):
            for j in range(batch_images_orig[i].shape[0]):
                self.patches_orig[no] = batch_images_orig[i][j]
                self.patches_gt1[no] = batch_images_gt1[i][j]
                self.patches_gt2[no] = batch_images_gt2[i][j]
                self.patches_gt3[no] = batch_images_gt3[i][j]
                self.patches_gt4[no] = batch_images_gt4[i][j]
                self.patches_gt5[no] = batch_images_gt5[i][j]
                self.weight_gt1[no]=weight_matrix[i,0]
                self.weight_gt2[no]=weight_matrix[i,1]
                self.weight_gt3[no]=weight_matrix[i,2]
                self.weight_gt4[no]=weight_matrix[i,3]
                self.weight_gt5[no]=weight_matrix[i,4]
                no += 1
        self.patch_count = batch_images_count
         
    def build_current_batch(self):
        if self.index_in_epoch < 0:
            self.batch_index = random.sample(range(0, self.patch_count), self.patch_count)
            self.index_in_epoch = 0
    
        for i in range(self.batch_size):
            if self.index_in_epoch >= self.patch_count:
                self.batch_index = random.sample(range(0, self.patch_count), self.patch_count)
                self.epochs_completed += 1
                self.index_in_epoch = 0

            self.batch_input_images[i] = self.patches_orig[self.batch_index[self.index_in_epoch] ]
            self.batch_true_images1[i] = self.patches_gt1[self.batch_index[self.index_in_epoch] ]
            self.batch_true_images2[i] = self.patches_gt2[self.batch_index[self.index_in_epoch] ]
            self.batch_true_images3[i] = self.patches_gt3[self.batch_index[self.index_in_epoch] ]
            self.batch_true_images4[i] = self.patches_gt4[self.batch_index[self.index_in_epoch] ]
            self.batch_true_images5[i] = self.patches_gt5[self.batch_index[self.index_in_epoch] ]
            self.batch_true_weights1[i] = self.weight_gt1[self.batch_index[self.index_in_epoch]]
            self.batch_true_weights2[i] = self.weight_gt2[self.batch_index[self.index_in_epoch]]
            self.batch_true_weights3[i] = self.weight_gt3[self.batch_index[self.index_in_epoch]]
            self.batch_true_weights4[i] = self.weight_gt4[self.batch_index[self.index_in_epoch]]
            self.batch_true_weights5[i] = self.weight_gt5[self.batch_index[self.index_in_epoch]]
            self.index_in_epoch += 1




    def save(self, checkpoint_path, step):
        # self.saver.save(self.sess,os.path.join(checkpoint_path, 'model'),global_step=step)
        self.saver.save(self.sess,os.path.join(checkpoint_path, 'model'),global_step=step)

    def load(self, checkpoint_path):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def test(self):
        temp=sio.loadmat('../dataset/weight_matrix.mat')
        weight_matrix=temp['weight_matrix']
        ########################################### WMAE
        test_total_ae=0
        test_total_se=0
        test_total_count=0
        start=time.time()
        for i in range(401,500+1):
            #print(i,time.time()-start)
            #start=time.time()
            #print(i)
            im = scipy.misc.imread('../dataset/origin_images/%.4d.png' % (i))
            im = np.float32(im)
            batch_images=[im]         
            test_output_eval= self.sess.run(self.test_output,feed_dict={self.test_input: batch_images}) 
            im_out=test_output_eval[0]
            h,w,channel=im_out.shape
            #im_out=(np.float64(im_out)+1)/2*255
            im_out[im_out>255]=255
            im_out[im_out<0]=0
            #scipy.misc.imsave('model_result_ResNet/%.4d.png'%(i),im_out/255)
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


    def calculate_ReguTerm(self):

        pixel_grad1= (self.net_g[:,1: , : ,:]-self.net_g[:, :-1, : ,:])-(self.y1_scale[:, 1: , : ,:]-self.y1_scale[:, :-1, : ,:])
        pixel_grad2= (self.net_g[:, :-1, : ,:]-self.net_g[:, 1: , : ,:])-(self.y1_scale[:, :-1, : ,:]-self.y1_scale[:,1:, : ,:])
        pixel_grad3= (self.net_g[:, : ,1: ,:]-self.net_g[:, : , :-1,:])-(self.y1_scale[:, : ,1: ,:]-self.y1_scale[:, :, :-1,:])
        pixel_grad4= (self.net_g[:, : , :-1,:]-self.net_g[:, : , 1: ,:])-(self.y1_scale[:, : , :-1,:]-self.y1_scale[:, : ,1: ,:])
        pixel_grad5= (self.net_g[:, :-1, :-1,:]-self.net_g[:, 1:, 1: ,:])-(self.y1_scale[:, :-1, :-1,:]-self.y1_scale[:,1:,1: ,:])
        pixel_grad6= (self.net_g[:, 1: , 1: ,:]-self.net_g[:, :-1, :-1,:])-(self.y1_scale[:, 1: , 1: ,:]-self.y1_scale[:, :-1, :-1,:])
        pixel_grad7= (self.net_g[:, 1: , :-1,:]-self.net_g[:, :-1, 1: ,:])-(self.y1_scale[:, 1: , :-1,:]-self.y1_scale[:, :-1, 1: ,:])
        pixel_grad8= (self.net_g[:, :-1 , 1: ,:]-self.net_g[:, 1:, :-1,:])-(self.y1_scale[:, :-1 , 1: ,:]-self.y1_scale[:, 1:, :-1 ,:])
        pixel_grad9= (self.net_g[:,2: , : ,:]-self.net_g[:, :-2, : ,:])-(self.y1_scale[:, 2: , : ,:]-self.y1_scale[:, :-2, : ,:])
        pixel_grad10= (self.net_g[:, :-2, : ,:]-self.net_g[:, 2: , : ,:])-(self.y1_scale[:, :-2, : ,:]-self.y1_scale[:,2:, : ,:])
        pixel_grad11= (self.net_g[:, : ,2: ,:]-self.net_g[:, : , :-2,:])-(self.y1_scale[:, : ,2: ,:]-self.y1_scale[:, :, :-2,:])
        pixel_grad12= (self.net_g[:, : , :-2,:]-self.net_g[:, : , 2: ,:])-(self.y1_scale[:, : , :-2,:]-self.y1_scale[:, : ,2: ,:])
        ReguTerm1 = tf.reduce_mean(tf.abs(pixel_grad1)) \
                      + tf.reduce_mean(tf.abs(pixel_grad2)) \
                      + tf.reduce_mean(tf.abs(pixel_grad3)) \
                      + tf.reduce_mean(tf.abs(pixel_grad4)) \
                      + tf.reduce_mean(tf.abs(pixel_grad5)) \
                      + tf.reduce_mean(tf.abs(pixel_grad6)) \
                      + tf.reduce_mean(tf.abs(pixel_grad7)) \
                      + tf.reduce_mean(tf.abs(pixel_grad8)) \
                      + tf.reduce_mean(tf.abs(pixel_grad9)) \
                      + tf.reduce_mean(tf.abs(pixel_grad10)) \
                      + tf.reduce_mean(tf.abs(pixel_grad11)) \
                      + tf.reduce_mean(tf.abs(pixel_grad12))

        pixel_grad1= (self.net_g[:,1: , : ,:]-self.net_g[:, :-1, : ,:])-(self.y2_scale[:, 1: , : ,:]-self.y2_scale[:, :-1, : ,:])
        pixel_grad2= (self.net_g[:, :-1, : ,:]-self.net_g[:, 1: , : ,:])-(self.y2_scale[:, :-1, : ,:]-self.y2_scale[:,1:, : ,:])
        pixel_grad3= (self.net_g[:, : ,1: ,:]-self.net_g[:, : , :-1,:])-(self.y2_scale[:, : ,1: ,:]-self.y2_scale[:, :, :-1,:])
        pixel_grad4= (self.net_g[:, : , :-1,:]-self.net_g[:, : , 1: ,:])-(self.y2_scale[:, : , :-1,:]-self.y2_scale[:, : ,1: ,:])
        pixel_grad5= (self.net_g[:, :-1, :-1,:]-self.net_g[:, 1:, 1: ,:])-(self.y2_scale[:, :-1, :-1,:]-self.y2_scale[:,1:,1: ,:])
        pixel_grad6= (self.net_g[:, 1: , 1: ,:]-self.net_g[:, :-1, :-1,:])-(self.y2_scale[:, 1: , 1: ,:]-self.y2_scale[:, :-1, :-1,:])
        pixel_grad7= (self.net_g[:, 1: , :-1,:]-self.net_g[:, :-1, 1: ,:])-(self.y2_scale[:, 1: , :-1,:]-self.y2_scale[:, :-1, 1: ,:])
        pixel_grad8= (self.net_g[:, :-1 , 1: ,:]-self.net_g[:, 1:, :-1,:])-(self.y2_scale[:, :-1 , 1: ,:]-self.y2_scale[:, 1:, :-1 ,:])
        pixel_grad9= (self.net_g[:,2: , : ,:]-self.net_g[:, :-2, : ,:])-(self.y2_scale[:, 2: , : ,:]-self.y2_scale[:, :-2, : ,:])
        pixel_grad10= (self.net_g[:, :-2, : ,:]-self.net_g[:, 2: , : ,:])-(self.y2_scale[:, :-2, : ,:]-self.y2_scale[:,2:, : ,:])
        pixel_grad11= (self.net_g[:, : ,2: ,:]-self.net_g[:, : , :-2,:])-(self.y2_scale[:, : ,2: ,:]-self.y2_scale[:, :, :-2,:])
        pixel_grad12= (self.net_g[:, : , :-2,:]-self.net_g[:, : , 2: ,:])-(self.y2_scale[:, : , :-2,:]-self.y2_scale[:, : ,2: ,:])
        ReguTerm2 = tf.reduce_mean(tf.abs(pixel_grad1)) \
                      + tf.reduce_mean(tf.abs(pixel_grad2)) \
                      + tf.reduce_mean(tf.abs(pixel_grad3)) \
                      + tf.reduce_mean(tf.abs(pixel_grad4)) \
                      + tf.reduce_mean(tf.abs(pixel_grad5)) \
                      + tf.reduce_mean(tf.abs(pixel_grad6)) \
                      + tf.reduce_mean(tf.abs(pixel_grad7)) \
                      + tf.reduce_mean(tf.abs(pixel_grad8)) \
                      + tf.reduce_mean(tf.abs(pixel_grad9)) \
                      + tf.reduce_mean(tf.abs(pixel_grad10)) \
                      + tf.reduce_mean(tf.abs(pixel_grad11)) \
                      + tf.reduce_mean(tf.abs(pixel_grad12))

        pixel_grad1= (self.net_g[:,1: , : ,:]-self.net_g[:, :-1, : ,:])-(self.y3_scale[:, 1: , : ,:]-self.y3_scale[:, :-1, : ,:])
        pixel_grad2= (self.net_g[:, :-1, : ,:]-self.net_g[:, 1: , : ,:])-(self.y3_scale[:, :-1, : ,:]-self.y3_scale[:,1:, : ,:])
        pixel_grad3= (self.net_g[:, : ,1: ,:]-self.net_g[:, : , :-1,:])-(self.y3_scale[:, : ,1: ,:]-self.y3_scale[:, :, :-1,:])
        pixel_grad4= (self.net_g[:, : , :-1,:]-self.net_g[:, : , 1: ,:])-(self.y3_scale[:, : , :-1,:]-self.y3_scale[:, : ,1: ,:])
        pixel_grad5= (self.net_g[:, :-1, :-1,:]-self.net_g[:, 1:, 1: ,:])-(self.y3_scale[:, :-1, :-1,:]-self.y3_scale[:,1:,1: ,:])
        pixel_grad6= (self.net_g[:, 1: , 1: ,:]-self.net_g[:, :-1, :-1,:])-(self.y3_scale[:, 1: , 1: ,:]-self.y3_scale[:, :-1, :-1,:])
        pixel_grad7= (self.net_g[:, 1: , :-1,:]-self.net_g[:, :-1, 1: ,:])-(self.y3_scale[:, 1: , :-1,:]-self.y3_scale[:, :-1, 1: ,:])
        pixel_grad8= (self.net_g[:, :-1 , 1: ,:]-self.net_g[:, 1:, :-1,:])-(self.y3_scale[:, :-1 , 1: ,:]-self.y3_scale[:, 1:, :-1 ,:])
        pixel_grad9= (self.net_g[:,2: , : ,:]-self.net_g[:, :-2, : ,:])-(self.y3_scale[:, 2: , : ,:]-self.y3_scale[:, :-2, : ,:])
        pixel_grad10= (self.net_g[:, :-2, : ,:]-self.net_g[:, 2: , : ,:])-(self.y3_scale[:, :-2, : ,:]-self.y3_scale[:,2:, : ,:])
        pixel_grad11= (self.net_g[:, : ,2: ,:]-self.net_g[:, : , :-2,:])-(self.y3_scale[:, : ,2: ,:]-self.y3_scale[:, :, :-2,:])
        pixel_grad12= (self.net_g[:, : , :-2,:]-self.net_g[:, : , 2: ,:])-(self.y3_scale[:, : , :-2,:]-self.y3_scale[:, : ,2: ,:])
        ReguTerm3 = tf.reduce_mean(tf.abs(pixel_grad1)) \
                      + tf.reduce_mean(tf.abs(pixel_grad2)) \
                      + tf.reduce_mean(tf.abs(pixel_grad3)) \
                      + tf.reduce_mean(tf.abs(pixel_grad4)) \
                      + tf.reduce_mean(tf.abs(pixel_grad5)) \
                      + tf.reduce_mean(tf.abs(pixel_grad6)) \
                      + tf.reduce_mean(tf.abs(pixel_grad7)) \
                      + tf.reduce_mean(tf.abs(pixel_grad8)) \
                      + tf.reduce_mean(tf.abs(pixel_grad9)) \
                      + tf.reduce_mean(tf.abs(pixel_grad10)) \
                      + tf.reduce_mean(tf.abs(pixel_grad11)) \
                      + tf.reduce_mean(tf.abs(pixel_grad12))               

        pixel_grad1= (self.net_g[:,1: , : ,:]-self.net_g[:, :-1, : ,:])-(self.y4_scale[:, 1: , : ,:]-self.y4_scale[:, :-1, : ,:])
        pixel_grad2= (self.net_g[:, :-1, : ,:]-self.net_g[:, 1: , : ,:])-(self.y4_scale[:, :-1, : ,:]-self.y4_scale[:,1:, : ,:])
        pixel_grad3= (self.net_g[:, : ,1: ,:]-self.net_g[:, : , :-1,:])-(self.y4_scale[:, : ,1: ,:]-self.y4_scale[:, :, :-1,:])
        pixel_grad4= (self.net_g[:, : , :-1,:]-self.net_g[:, : , 1: ,:])-(self.y4_scale[:, : , :-1,:]-self.y4_scale[:, : ,1: ,:])
        pixel_grad5= (self.net_g[:, :-1, :-1,:]-self.net_g[:, 1:, 1: ,:])-(self.y4_scale[:, :-1, :-1,:]-self.y4_scale[:,1:,1: ,:])
        pixel_grad6= (self.net_g[:, 1: , 1: ,:]-self.net_g[:, :-1, :-1,:])-(self.y4_scale[:, 1: , 1: ,:]-self.y4_scale[:, :-1, :-1,:])
        pixel_grad7= (self.net_g[:, 1: , :-1,:]-self.net_g[:, :-1, 1: ,:])-(self.y4_scale[:, 1: , :-1,:]-self.y4_scale[:, :-1, 1: ,:])
        pixel_grad8= (self.net_g[:, :-1 , 1: ,:]-self.net_g[:, 1:, :-1,:])-(self.y4_scale[:, :-1 , 1: ,:]-self.y4_scale[:, 1:, :-1 ,:])
        pixel_grad9= (self.net_g[:,2: , : ,:]-self.net_g[:, :-2, : ,:])-(self.y4_scale[:, 2: , : ,:]-self.y4_scale[:, :-2, : ,:])
        pixel_grad10= (self.net_g[:, :-2, : ,:]-self.net_g[:, 2: , : ,:])-(self.y4_scale[:, :-2, : ,:]-self.y4_scale[:,2:, : ,:])
        pixel_grad11= (self.net_g[:, : ,2: ,:]-self.net_g[:, : , :-2,:])-(self.y4_scale[:, : ,2: ,:]-self.y4_scale[:, :, :-2,:])
        pixel_grad12= (self.net_g[:, : , :-2,:]-self.net_g[:, : , 2: ,:])-(self.y4_scale[:, : , :-2,:]-self.y4_scale[:, : ,2: ,:])
        ReguTerm4 = tf.reduce_mean(tf.abs(pixel_grad1)) \
                      + tf.reduce_mean(tf.abs(pixel_grad2)) \
                      + tf.reduce_mean(tf.abs(pixel_grad3)) \
                      + tf.reduce_mean(tf.abs(pixel_grad4)) \
                      + tf.reduce_mean(tf.abs(pixel_grad5)) \
                      + tf.reduce_mean(tf.abs(pixel_grad6)) \
                      + tf.reduce_mean(tf.abs(pixel_grad7)) \
                      + tf.reduce_mean(tf.abs(pixel_grad8)) \
                      + tf.reduce_mean(tf.abs(pixel_grad9)) \
                      + tf.reduce_mean(tf.abs(pixel_grad10)) \
                      + tf.reduce_mean(tf.abs(pixel_grad11)) \
                      + tf.reduce_mean(tf.abs(pixel_grad12))

        pixel_grad1= (self.net_g[:,1: , : ,:]-self.net_g[:, :-1, : ,:])-(self.y5_scale[:, 1: , : ,:]-self.y5_scale[:, :-1, : ,:])
        pixel_grad2= (self.net_g[:, :-1, : ,:]-self.net_g[:, 1: , : ,:])-(self.y5_scale[:, :-1, : ,:]-self.y5_scale[:,1:, : ,:])
        pixel_grad3= (self.net_g[:, : ,1: ,:]-self.net_g[:, : , :-1,:])-(self.y5_scale[:, : ,1: ,:]-self.y5_scale[:, :, :-1,:])
        pixel_grad4= (self.net_g[:, : , :-1,:]-self.net_g[:, : , 1: ,:])-(self.y5_scale[:, : , :-1,:]-self.y5_scale[:, : ,1: ,:])
        pixel_grad5= (self.net_g[:, :-1, :-1,:]-self.net_g[:, 1:, 1: ,:])-(self.y5_scale[:, :-1, :-1,:]-self.y5_scale[:,1:,1: ,:])
        pixel_grad6= (self.net_g[:, 1: , 1: ,:]-self.net_g[:, :-1, :-1,:])-(self.y5_scale[:, 1: , 1: ,:]-self.y5_scale[:, :-1, :-1,:])
        pixel_grad7= (self.net_g[:, 1: , :-1,:]-self.net_g[:, :-1, 1: ,:])-(self.y5_scale[:, 1: , :-1,:]-self.y5_scale[:, :-1, 1: ,:])
        pixel_grad8= (self.net_g[:, :-1 , 1: ,:]-self.net_g[:, 1:, :-1,:])-(self.y5_scale[:, :-1 , 1: ,:]-self.y5_scale[:, 1:, :-1 ,:])
        pixel_grad9= (self.net_g[:,2: , : ,:]-self.net_g[:, :-2, : ,:])-(self.y5_scale[:, 2: , : ,:]-self.y5_scale[:, :-2, : ,:])
        pixel_grad10= (self.net_g[:, :-2, : ,:]-self.net_g[:, 2: , : ,:])-(self.y5_scale[:, :-2, : ,:]-self.y5_scale[:,2:, : ,:])
        pixel_grad11= (self.net_g[:, : ,2: ,:]-self.net_g[:, : , :-2,:])-(self.y5_scale[:, : ,2: ,:]-self.y5_scale[:, :, :-2,:])
        pixel_grad12= (self.net_g[:, : , :-2,:]-self.net_g[:, : , 2: ,:])-(self.y5_scale[:, : , :-2,:]-self.y5_scale[:, : ,2: ,:])
        ReguTerm5 = tf.reduce_mean(tf.abs(pixel_grad1)) \
                      + tf.reduce_mean(tf.abs(pixel_grad2)) \
                      + tf.reduce_mean(tf.abs(pixel_grad3)) \
                      + tf.reduce_mean(tf.abs(pixel_grad4)) \
                      + tf.reduce_mean(tf.abs(pixel_grad5)) \
                      + tf.reduce_mean(tf.abs(pixel_grad6)) \
                      + tf.reduce_mean(tf.abs(pixel_grad7)) \
                      + tf.reduce_mean(tf.abs(pixel_grad8)) \
                      + tf.reduce_mean(tf.abs(pixel_grad9)) \
                      + tf.reduce_mean(tf.abs(pixel_grad10)) \
                      + tf.reduce_mean(tf.abs(pixel_grad11)) \
                      + tf.reduce_mean(tf.abs(pixel_grad12))

        total_term=ReguTerm1*self.w1 + ReguTerm2*self.w2 + ReguTerm3*self.w3 + ReguTerm4*self.w4 + ReguTerm5*self.w5
        return total_term
    
