#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : common.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 09:56:29
#   Description :
#
# ================================================================

import tensorflow as tf

slim = tf.contrib.slim


def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        if downsample:
            # pad_h, pad_w = filters_shape[0]-1, filters_shape[1]-1
            # pad_h_begin, pad_w_begin = pad_h//2, pad_w//2
            # pad_h_end, pad_w_end = pad_h - pad_h_begin, pad_w-pad_w_begin
            # paddings = tf.constant([[0, 0], [pad_h_begin, pad_h_end], [pad_w_begin, pad_w_end], [0, 0]])
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=trainable,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=trainable,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True:
            conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv

# def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#         if downsample:
#             pad_h, pad_w = filters_shape[0] - 1, filters_shape[1] - 1
#             pad_h_begin, pad_w_begin = pad_h // 2, pad_w // 2
#             pad_h_end, pad_w_end = pad_h - pad_h_begin, pad_w - pad_w_begin
#             paddings = tf.constant([[0, 0], [pad_h_begin, pad_h_end], [pad_w_begin, pad_w_end], [0, 0]])
#             # pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
#             # paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
#             input_data = tf.pad(input_data, paddings, 'CONSTANT')
#             strides = 2
#             padding = 'VALID'
#         else:
#             strides = 1
#             padding = "SAME"
#
#         if bn:
#             conv = slim.conv2d(input_data, filters_shape[-1], filters_shape[0], stride=strides,
#                                padding=padding, normalizer_fn=slim.batch_norm,
#                                normalizer_params={
#                                    'decay': 0.9,
#                                    'epsilon': 1e-05,
#                                    'scale': True,
#                                    'is_training': trainable,
#                                    'fused': None,  # Use fused batch norm if possible.
#                                },
#                                biases_initializer=None,
#                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1))
#         else:
#             conv = slim.conv2d(input_data, filters_shape[-1], filters_shape[0], stride=1,
#                                padding=padding, normalizer_fn=None,
#                                biases_initializer=tf.zeros_initializer(),
#                                activation_fn=None)
#
#     return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):
    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1, filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output


def route(name, previous_output, current_output):
    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="resize"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2, 2), kernel_initializer=tf.random_normal_initializer())

    return output
