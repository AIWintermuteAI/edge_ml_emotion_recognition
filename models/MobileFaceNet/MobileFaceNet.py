import os
import warnings
import sys 

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

sys.path.append('models/MobileFaceNet')
from __init__ import get_submodules_from_kwargs
from imagenet_utils import preprocess_input

backend = None
layers = None
models = None
keras_utils = None

def conv_block(inputs, filters, kernel_size, strides, padding):
    
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    
    Z = layers.Conv2D(filters, kernel_size, strides = strides, padding = padding)(inputs)
    Z = layers.BatchNormalization(axis = channel_axis)(Z)
    
    return layers.LeakyReLU(name='conv_pw_%d_relu' % filters)(Z)

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.LeakyReLU(name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((1, 1), (1, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.LeakyReLU(name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.LeakyReLU(name='conv_pw_%d_relu' % block_id)(x)


def linear_GD_conv_block(inputs, kernel_size, strides):
    
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    
    Z = layers.DepthwiseConv2D(kernel_size, strides = strides, padding = 'valid', depth_multiplier = 1)(inputs)
    Z = layers.BatchNormalization(axis = channel_axis)(Z)
    
    return Z

def preprocess(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return preprocess_input(x, mode='tf', **kwargs)

def mobile_face_base(input_shape=(128, 128, 3),
              input_tensor = None,
              alpha=1.0,
              depth_multiplier=1,
              weights=None,
              variant=None,
              **kwargs):
              
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if (weights == 'imagenet'):
        raise ValueError('There is no imagenet weights for MobileFaceNet')

    x = _conv_block(input_tensor, 64, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=0)
    
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier,
                              strides=(2, 2), block_id=1)    
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=2)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=3)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=4)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=10)                              
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=11) 
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=12) 
        
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=13)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=14)                              
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=15)  
    
    x = linear_GD_conv_block(x, input_shape[0]//16, 1)

    return x

