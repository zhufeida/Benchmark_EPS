import argparse
import os
import scipy.misc
import numpy as np
import tensorflow as tf
from model_resnet_delta import mymodel

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser(description='')
parser.add_argument('--iteration_num', type=int, default=1e6, help='# of epoch')
parser.add_argument('--batch_size', type=int, default=4, help='# images in batch')
parser.add_argument('--patch_size', type=int, default=96, help='then crop to this size')
parser.add_argument('--stride', type=int, default=40, help='stride between training patch ')

#parser.add_argument('--feature_num', type=int, default=256, help='feature')

parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--phase', default='train', help='train, test')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint_delta', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample_delta', help='sample are saved here')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    #if not os.path.exists(args.sample_dir):
    #    os.makedirs(args.sample_dir)
    args.checkpoint_path=args.checkpoint_dir
    args.sample_path=args.sample_dir
   
    
    tfconfig = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
    #with tf.Session() as sess:
        model = mymodel(sess, args)
        model.build_model(args)
        model.train(args) 

if __name__ == '__main__':
    tf.app.run()
