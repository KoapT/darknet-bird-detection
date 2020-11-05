#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolov3.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 10:47:03
#   Description :
#
# ================================================================

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""

    def __init__(self, input_data, trainable, tiny=False):

        self.trainable = trainable
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class = len(self.classes)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD

        if tiny:
            try:
                self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_tiny_3l(input_data)
            except:
                raise NotImplementedError("Can not build up yolov3-tiny-3l network!")

            with tf.variable_scope('pred_sbbox'):
                self.pred_sbbox, self.sdetections = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

            with tf.variable_scope('pred_mbbox'):
                self.pred_mbbox, self.mdetections = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

            with tf.variable_scope('pred_lbbox'):
                self.pred_lbbox, self.ldetections = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

            self.detection_boxes = self.__detection_boxes(
                tf.concat([self.sdetections, self.mdetections, self.ldetections], axis=1))

        else:
            try:
                self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_nework(input_data)
            except:
                raise NotImplementedError("Can not build up yolov3 network!")

            with tf.variable_scope('pred_sbbox'):
                self.pred_sbbox, self.sdetections = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

            with tf.variable_scope('pred_mbbox'):
                self.pred_mbbox, self.mdetections = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

            with tf.variable_scope('pred_lbbox'):
                self.pred_lbbox, self.ldetections = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

            self.detection_boxes = self.__detection_boxes(
                tf.concat([self.sdetections, self.mdetections, self.ldetections], axis=1))

    def __build_nework(self, input_data):
        input_data = input_data / 255.
        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)

        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv52')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv54')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, name='conv_mobj_branch')
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox

    def __build_tiny_3l(self, input_data):
        input_data = input_data / 255.
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 3, 16), trainable=self.trainable,
                                          name='conv0')
        input_data = tf.nn.max_pool(input_data, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME',
                                    name='max_pool0')
        for i in range(5):
            input_data = common.convolutional(input_data, filters_shape=(3, 3, 16 * pow(2, i), 16 * pow(2, i + 1)),
                                              trainable=self.trainable,
                                              name='conv{}'.format(i + 1))
            if i == 2:
                route_0 = input_data
            if i == 3:
                route_1 = input_data
            if i == 4:
                input_data = tf.nn.max_pool(input_data, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding='SAME',
                                            name='max_pool{}'.format(i + 1))
            else:
                input_data = tf.nn.max_pool(input_data, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME',
                                            name='max_pool{}'.format(i + 1))
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024), trainable=self.trainable,
                                          name='conv6')
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 1024, 256), trainable=self.trainable,
                                          name='conv7')
        conv_lobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, filters_shape=(1, 1, 256, 128), trainable=self.trainable,
                                          name='conv8')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)
        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_1], axis=-1)
        conv_mobj_branch = common.convolutional(input_data, (3, 3, 384, 256), self.trainable, name='conv_mobj_branch')
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = common.convolutional(conv_mobj_branch, filters_shape=(1, 1, 256, 128), trainable=self.trainable,
                                          name='conv9')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)
        with tf.variable_scope('route_0'):
            input_data = tf.concat([input_data, route_0], axis=-1)
        conv_sobj_branch = common.convolutional(input_data, (3, 3, 256, 128), self.trainable, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 128, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)
        return conv_lbbox, conv_mbbox, conv_sbbox

    def decode(self, conv_output, anchors, stride):
        """
        :return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output,
                                 (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        result = tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
        detections = tf.reshape(result, (-1, output_size * output_size * anchor_per_scale, 5 + self.num_class))

        return result, detections

    def __detection_boxes(self, detections):
        """
        Converts center x, center y, width and height values to coordinates of top left and bottom right points.

        :param detections: outputs of YOLO v3 detector of shape (?, 10647, (num_classes + 5))
        :return: converted detections of same shape as input
        """
        center_x, center_y, width, height, attrs = tf.split(
            detections, [1, 1, 1, 1, -1], axis=-1)
        w2 = width / 2
        h2 = height / 2
        x0 = center_x - w2
        y0 = center_y - h2
        x1 = center_x + w2
        y1 = center_y + h2

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        detections = tf.concat([boxes, attrs], axis=-1, name="output_boxes")
        return detections

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        # boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
        #                     boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        # boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
        #                     boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
        #
        # boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
        #                     tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        # boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
        #                     tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)
        #
        # boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        # boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        """
        conv: 神经网络的输出结果：[batch_size,output_size,output_size,3*(5+num_class)],
        pred: 经过decode之后的结果：[batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]， 其中最后一维的x,y,w,h是在input size上的。
        label: shape同pred,每个pred对应一个label。
        bboxes : [batch, num_of_bboxes, xywh]   对于s，m，l三种scale的bboxes，有不同的number.
        """
        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
        # 加入了bbox_loss_scale，对于小物体，bbox_loss_scale>1， giou_loss增大。 通过该手段可以增大小物体对于定位损失的影响，从而使小物体的定位更准确。

        # 如果某pred_box不匹配该点的某一个label，该label的respond_bbox被置为0，表示它是一个背景。 但是其他的anchor的label可能与这个pred匹配，
        # 所以要将该pred_bbox与所有的gt_bboxes进行求iou，如果他与每个gt_bbox的iou都小于阈值，那么才可以确认它真的没有物体，是一个背景。 
        # 否则，在计算conf_loss时会产生错误。
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)  # 匹配与ground truth 的bbox的iou最大的结果。

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh,
                                                     tf.float32)  # bgd=background ， iou小于某个阈值的，进一步确认为背景。

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                giou_loss*.2 +   # 对于置信度加入定位的影响
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
                                                                       logits=conv_raw_conf)  # 前景（有物体）的置信度损失
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            # 背景的置信度损失
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))  # 每个batch的所有anchor的loss求和之后，再对batch求平均。

        return giou_loss, conf_loss, prob_loss

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss
