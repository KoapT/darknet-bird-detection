#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : dataset.py
#   Author      : YunYang1994
#   Created date: 2019-03-15 18:05:03
#   Description :
#
# ================================================================

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
import imgaug as ia
from imgaug import augmenters as iaa

COLOR = {'probability': .5, 'multiply': (0.7, 1.6), 'add_to_hue_value': (-20, 20), 'gamma': (0.5, 2.0),
         'per_channel': 0.3}
BLUR = {'probability': .5, 'gaussian_sigma': (0.0, 1.5), 'average_k': (2, 5), 'median_k': (1, 11)}
NOISE = {'probability': .5, 'gaussian_scale': (0.0, 1.5), 'salt_p': 0.005, 'drop_out_p': (0, 0.01), 'per_channel': 0.3}
CROP = {'probability': 0.5}
PAD = {'probability': 0.5, 'size': (0.05, 0.2)}
FLIPUD = {'probability': .0}
FLIPLR = {'probability': .5}
PIECEWISEAFFINE = {'probability': .0}


class Dataset(object):
    """implement Dataset here"""

    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.annot_path = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else cfg.TEST.DATA_AUG
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 100

        self.annotations = self.load_annotations()  # ->list
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt]
        if self.dataset_type == 'train':
            np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):
            self.input_size = random.choice(self.input_sizes) \
                if self.dataset_type == 'train' else self.input_sizes
            self.output_sizes = self.input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3))
            batch_image_with_bboxes = np.copy(batch_image)

            batch_label_sbbox = np.zeros((self.batch_size, self.output_sizes[0], self.output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, self.output_sizes[1], self.output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, self.output_sizes[2], self.output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes))

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    image_with_bboxes = self.image_with_bboxes(np.copy(image), bboxes)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
                        bboxes)

                    batch_image[num, :, :, :] = image
                    batch_image_with_bboxes[num, :, :, :] = image_with_bboxes
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes, batch_image_with_bboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def img_augment(self, image, bboxes=None, p=.7):
        """
        使用imgaug库进行的图像增强。
        :param image: np.array, images.
        :param bboxes: np.array, bboxes of object detection.
        :param n: max number of augmenters.
        :return: image and bboxes after augmenting.
        """
        if random.random() > p:
            return image, bboxes

        h, w, _ = image.shape
        if bboxes is not None:
            bboxes_list = bboxes.tolist()
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            top = max_bbox[1]
            left = max_bbox[0]
            bottom = h - max_bbox[3]
            right = w - max_bbox[2]
        else:
            top = int(h * 0.25)
            left = int(w * 0.25)
            bottom = int(h * 0.25)
            right = int(w * 0.25)

        while True:
            new_bndbox_list = []
            seq = iaa.Sequential(
                children=[
                    # color
                    iaa.Sometimes(
                        COLOR['probability'],
                        iaa.SomeOf(2, [
                            iaa.Multiply(COLOR['multiply'], per_channel=COLOR['per_channel']),
                            iaa.AddToHueAndSaturation(COLOR['add_to_hue_value'], per_channel=COLOR['per_channel']),
                            iaa.GammaContrast(COLOR['gamma'], per_channel=COLOR['per_channel']),
                            iaa.ChannelShuffle()
                        ])
                    ),

                    # blur
                    iaa.Sometimes(
                        BLUR['probability'],
                        iaa.OneOf([
                            iaa.GaussianBlur(sigma=BLUR['gaussian_sigma']),
                            iaa.AverageBlur(k=BLUR['average_k']),
                            iaa.MedianBlur(k=BLUR['median_k'])
                        ])
                    ),

                    # noise
                    iaa.Sometimes(
                        NOISE['probability'],
                        iaa.OneOf([
                            iaa.AdditiveGaussianNoise(scale=NOISE['gaussian_scale'], per_channel=NOISE['per_channel']),
                            iaa.SaltAndPepper(p=NOISE['salt_p'], per_channel=NOISE['per_channel']),
                            iaa.Dropout(p=NOISE['drop_out_p'], per_channel=NOISE['per_channel']),
                            iaa.CoarseDropout(p=NOISE['drop_out_p'], size_percent=(0.05, 0.1),
                                              per_channel=NOISE['per_channel'])
                        ])
                    ),

                    # crop and pad
                    iaa.Sometimes(CROP['probability'], iaa.Crop(px=(
                        random.randint(0, top), random.randint(0, right),
                        random.randint(0, bottom), random.randint(0, left)),
                        keep_size=False)),
                    iaa.Sometimes(PAD['probability'], iaa.Pad(
                        percent=PAD['size'],
                        # pad_mode=ia.ALL,
                        pad_mode=["constant", "edge", "linear_ramp", "maximum", "mean", "median",
                                  "minimum"] if bboxes is not None else ia.ALL,
                        pad_cval=(0, 255)
                    )),

                    # flip
                    iaa.Flipud(FLIPUD['probability']),
                    iaa.Fliplr(FLIPLR['probability']),

                    iaa.Sometimes(PIECEWISEAFFINE['probability'],
                                  iaa.PiecewiseAffine(scale=(0.01, 0.04)))
                ])
            seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变
            # 读取图片
            image_aug = seq_det.augment_images([image])[0]
            n_h, n_w, _ = image_aug.shape
            if bboxes is not None:
                for box in bboxes_list:
                    x1, y1, x2, y2, c = tuple(box)
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    ], shape=image.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    n_x1 = int(max(1, min(image_aug.shape[1], bbs_aug.bounding_boxes[0].x1)))
                    n_y1 = int(max(1, min(image_aug.shape[0], bbs_aug.bounding_boxes[0].y1)))
                    n_x2 = int(max(1, min(image_aug.shape[1], bbs_aug.bounding_boxes[0].x2)))
                    n_y2 = int(max(1, min(image_aug.shape[0], bbs_aug.bounding_boxes[0].y2)))
                    new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2, c])
                bboxes_aug = np.array(new_bndbox_list)
            else:
                bboxes_aug = bboxes
            # 长宽比太大的图片不要，产生新的image和bboxes
            if 1 / 3 <= image_aug.shape[0] / image_aug.shape[1] <= 3:
                break
        return image_aug, bboxes_aug

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            if bboxes is not None:
                bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            if bboxes is not None:
                max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            else:
                max_bbox = np.array([w * .25, h * .25, w * .75, h * .75])
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = min(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = min(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            if bboxes is not None:
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        '''
        仿射变换（不旋转）
        '''
        if random.random() < 0.5:
            h, w, _ = image.shape
            if bboxes is not None:
                max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            else:
                max_bbox = np.array([w * .25, h * .25, w * .75, h * .75])

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]  # 到图片4个边的最大距离

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            if bboxes is not None:
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = np.array(cv2.imread(image_path))
        if line[1:]:
            bboxes = np.array(
                [list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])  # shape:[n,5]  n表示bbox的个数
        else:
            bboxes = None
        if self.data_aug:
            # image, bboxes = self.random_horizontal_flip(np.copy(image), bboxes)
            # image, bboxes = self.random_crop(np.copy(image), bboxes)
            # image, bboxes = self.random_translate(np.copy(image), bboxes)
            image, bboxes = self.img_augment(np.copy(image), bboxes)
        image, bboxes = utils.image_preporcess(np.copy(image), [self.input_size, self.input_size],
                                               bboxes)
        return image, bboxes

    def image_with_bboxes(self, image, bboxes):
        '''
        Draw bboxes on image to show if the img_augment get the right bboxes, on tensorboard.
        '''
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        if bboxes is not None:
            bboxes_list = bboxes.tolist()
            for bbox in bboxes_list:
                box = bbox[:4]
                cls = bbox[4]
                color = colors[cls % 6]
                left_top, right_bottom = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(image, left_top, right_bottom, color, 2)
        return image

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):
        '''
        处理某张图片中的所有gt_bboxes。
        :return: label_sbbox, label_mbbox, label_lbbox  ->  shape:[ouputsize,outputsize,anchors_per_scale，5+num_classes]
                 sbboxes, mbboxes, lbboxes              ->  shape:[num_of_bboxes，4]   4:x,y,w,h
        '''
        label = [np.zeros((self.output_sizes[i], self.output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        if bboxes is not None:
            for bbox in bboxes:
                bbox_coor = bbox[:4]
                bbox_class_ind = bbox[4]

                onehot = np.zeros(self.num_classes, dtype=np.float)
                onehot[bbox_class_ind] = 1.0

                uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)  # 均匀分布
                deta = 0.01
                smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution  # 为什么要用smooth_onehot ，效果更好吗？？

                bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]],
                                           axis=-1)
                bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:,
                                                                    np.newaxis]  # 在ouput_size上，bbox的x,y,w,h -> shape:[3,4],  3种strides

                iou = []
                exist_positive = False
                for i in range(3):  # 3种scale
                    anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                    anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # anchor 的中心位置设置
                    anchors_xywh[:, 2:4] = self.anchors[i]  # 在ouput_size上，anchor的位置和大小x,y,w,h -> shape:[3,4]

                    iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                    iou.append(iou_scale)
                    iou_mask = iou_scale > 0.3

                    if np.any(iou_mask):
                        xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)  # 在output map上面的横、纵坐标。

                        label[i][yind, xind, iou_mask, :] = 0
                        label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                        label[i][yind, xind, iou_mask, 4:5] = 1.0
                        label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                        bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                        bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                        bbox_count[i] += 1

                        exist_positive = True

                if not exist_positive:  # 如果bbox与三种scale的每个anchor都没有IOU>0.3的，选IOU最大的进行匹配。
                    best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                    best_detect = int(best_anchor_ind / self.anchor_per_scale)  # scale编号
                    best_anchor = int(best_anchor_ind % self.anchor_per_scale)  # anchor编号
                    xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                    label[best_detect][yind, xind, best_anchor, :] = 0
                    label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                    label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                    label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                    bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                    bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs
