#-*- coding:utf-8 -*-

## 对抗网络 两部分组成 Generator 学习者 尝试用各种方法生成图片 蒙骗打分者
##  Discriminator 打分者 为学习者的学习成果打分 找出学习者的错误



import numpy as np
import tensorflow as tf
from ops import conv2d, conv2d_transpose, pixelwise_accuracy



## 学习者采用UNET
## UNET 由 加码器  与 译码器组成
## tf.get_collection
## 该函数可以用来获取key集合中的所有元素，返回一个列表。列表的顺序依变量放入集合中的先后而定。
## scope为可选参数，表示的是名称空间（名称域），如果指定，就返回名称域中所有放入‘key’的变量的列表，不指定则返回所有变量。
class Generator(object):
    def __init__(self, name, encoder_kernels, decoder_kernels, output_channels=3, training=True):
        self.name = name
        self.encoder_kernels = encoder_kernels
        self.decoder_kernels = decoder_kernels
        self.output_channels = output_channels
        self.training = training
        self.var_list = []

    def create(self, inputs, kernel_size=None, seed=None, reuse_variables=None):
        output = inputs

        with tf.variable_scope(self.name, reuse=reuse_variables):

            layers = []

            # encoder branch
            for index, kernel in enumerate(self.encoder_kernels):

                name = 'conv' + str(index)
                output = conv2d(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    activation=tf.nn.leaky_relu,
                    seed=seed
                )

                # save contracting path layers to be used for skip connections
                layers.append(output)
                
                if kernel[2] > 0:
                    keep_prob = 1.0 - kernel[2] if self.training else 1.0
                    output = tf.nn.dropout(output, keep_prob=keep_prob, name='dropout_' + name, seed=seed)

            # decoder branch
            for index, kernel in enumerate(self.decoder_kernels):

                name = 'deconv' + str(index)
                output = conv2d_transpose(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    activation=tf.nn.relu,
                    seed=seed
                )

                if kernel[2] > 0:
                    keep_prob = 1.0 - kernel[2] if self.training else 1.0
                    output = tf.nn.dropout(output, keep_prob=keep_prob, name='dropout_' + name, seed=seed)

                # concat the layer from the contracting path with the output of the current layer
                # concat only the channels (axis=3)
                #将紧缩路径中的图层与当前图层的输出连接
                #仅连接通道（轴= 3）
                output = tf.concat([layers[len(layers) - index - 2], output], axis=3)

            output = conv2d(
                inputs=output,
                name='conv_last',
                filters=self.output_channels,   # number of output chanels
                kernel_size=1,                  # last layer kernel size = 1
                strides=1,                      # last layer stride = 1
                bnorm=False,                    # do not use batch-norm for the last layer
                activation=tf.nn.tanh,          # tanh activation function for the output
                seed=seed
            )

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return output



#打分者
#对打分者的成绩进行打分
class Discriminator(object):
    def __init__(self, name, kernels, training=True):
        self.name = name
        self.kernels = kernels
        self.training = training
        self.var_list = []

    def create(self, inputs, kernel_size=None, seed=None, reuse_variables=None):
        output = inputs
        with tf.variable_scope(self.name, reuse=reuse_variables):
            for index, kernel in enumerate(self.kernels):

                # not use batch-norm in the first layer
                bnorm = False if index == 0 else True
                name = 'conv' + str(index)
                output = conv2d(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    bnorm=bnorm,
                    activation=tf.nn.leaky_relu,
                    seed=seed
                )

                if kernel[2] > 0:
                    keep_prob = 1.0 - kernel[2] if self.training else 1.0
                    output = tf.nn.dropout(output, keep_prob=keep_prob, name='dropout_' + name, seed=seed)

            output = conv2d(
                inputs=output,
                name='conv_last',
                filters=1,
                kernel_size=4,                  # last layer kernel size = 4
                strides=1,                      # last layer stride = 1
                bnorm=False,                    # do not use batch-norm for the last layer
                seed=seed
            )

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return output

