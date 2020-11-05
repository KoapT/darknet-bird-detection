import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import random

import imgaug as ia
from imgaug import augmenters as iaa
import argparse


ia.seed(1)


def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    size = (width,height)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)

    return bndboxlist,size

# print(read_xml_annotation("../Annotations", 'w20190719093725711_28.xml'))

def change_xml_list_annotation(root, image_id, new_target, saveroot, id, h, w):
    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)

    elem = tree.find('filename')
    elem.text = id + '.jpg'

    _path = tree.find('path')
    _path.text = os.path.join(saveroot, id + '.xml')

    size = tree.find('size')
    width = size.find('width')
    width.text = str(w)
    height = size.find('height')
    height.text = str(h)

    xmlroot = tree.getroot()
    index = 0

    for object in xmlroot.findall('object'):  # 找到root节点下的所有boject节点
        bndbox = object.find('bndbox')  # 子节点下节点bndbox的值

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        if new_xmin is not None:
            xmin = bndbox.find('xmin')
            xmin.text = str(new_xmin)
            ymin = bndbox.find('ymin')
            ymin.text = str(new_ymin)
            xmax = bndbox.find('xmax')
            xmax.text = str(new_xmax)
            ymax = bndbox.find('ymax')
            ymax.text = str(new_ymax)
        else:
            xmlroot.remove(object)
        index += 1

    tree.write(os.path.join(saveroot, id + '.xml'))


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False


if __name__ == "__main__":


    IMG_DIR = "../JPEGImages"
    XML_DIR = "../Annotations"

    AUG_XML_DIR = "./Annotations"  # 存储增强后的XML文件夹路径
    try:
        shutil.rmtree(AUG_XML_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_XML_DIR)

    AUG_IMG_DIR = "./JPEGImages"  # 存储增强后的影像文件夹路径
    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_IMG_DIR)

    AUGLOOP = 1  # 每张影像增强的数量

    # boxes_img_aug_list = []
    new_bndbox_list = []

    w_crop = 1024
    h_crop = 900

    for root, sub_folders, files in os.walk(XML_DIR):

        for name in files:
            bndbox,size = read_xml_annotation(XML_DIR, name)
            w_raw = size[0]
            h_raw = size[1]
            if len(bndbox) == 0:
                continue
            arr = np.array(bndbox)
            min_x, min_y = np.min(arr, axis=0)[0], np.min(arr, axis=0)[1]
            max_x, max_y = np.max(arr, axis=0)[2], np.max(arr, axis=0)[3]
            top = max(0, min_y-int(h_crop*0.8))
            bottom = min(h_raw-h_crop-1, max_y+int(h_crop*0.8))
            left = min(w_raw-w_crop-1, w_raw-(min_x-int(w_crop*0.8)))
            right = max(0, w_raw-(max_x+int(w_crop*0.8)))

            # shutil.copy(os.path.join(XML_DIR, name), AUG_XML_DIR)               #复制文件
            # shutil.copy(os.path.join(IMG_DIR, name[:-4] + '.jpg'), AUG_IMG_DIR)

            for epoch in range(AUGLOOP):
                a = random.randint(1, 3) if w_raw < 2000 else random.randint(1, 2)
                b = random.randint(1, 2) if w_raw < 2000 else random.randint(2, 3)
                hlist = list(np.linspace(min(top, h_raw - h_crop -1), bottom, a, dtype=int))
                wlist = list(np.linspace(min(right, w_raw - w_crop -1), left, b, dtype=int))

                for iter_h, h in enumerate(hlist):
                    for iter_w, w in enumerate(wlist):
                        # 影像增强
                        # seq = iaa.Crop(px=(h, w, h_raw-h_crop-h, w_raw-w_crop-w), keep_size=False)
                        seq = iaa.Sequential(
                                         children=[
                                             iaa.Sometimes(0.5, iaa.Multiply((0.5, 1.5), per_channel=0.3)),
                                             # 调整亮度,per_channel=0.3表示有0.3的概率3通道乘以不同的参数
                                             iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0.0, 1.5))),
                                             iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255), per_channel=0.5)),
                                             iaa.Sometimes(0.1, iaa.Dropout(p=(0, 0.01), per_channel=0.3)),
                                             # iaa.Crop(px=(h, w, h_raw - h_crop - h, w_raw - w_crop - w), keep_size=False),
                                             # iaa.Sometimes(0.1, iaa.Flipud()),  # 上下翻转 50%
                                             iaa.Sometimes(0.5, iaa.Fliplr(0.5)),  # 左右翻转 50%
                                             # iaa.Resize(eval(args.resize)),
                                             #iaa.Sometimes(0.5,iaa.Affine(
                                                 # translate_px={"x": 15, "y": 15},   # 在x或y 方向上平移一定的像素
                                                 #scale=(0.9,1.2),
                                                 #rotate=(-10,10),
                                                 #shear=(-10,10),
                                                 #fit_output= True,
                                             #),
                                             #)# 仿射变换
                                         ] if w_raw < 2000 else [
                                             iaa.Sometimes(0.5, iaa.Multiply((0.5, 1.5), per_channel=0.3)),
                                             # 调整亮度,per_channel=0.3表示有0.3的概率3通道乘以不同的参数
                                             iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0.0, 1.5))),
                                             iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255), per_channel=0.5)),
                                             iaa.Sometimes(0.1, iaa.Dropout(p=(0, 0.01), per_channel=0.3)),
                                             iaa.Crop(px=(h, w, h_raw - h_crop - h, w_raw - w_crop - w), keep_size=False),
                                             # iaa.Sometimes(0.1, iaa.Flipud()),  # 上下翻转 50%
                                             iaa.Sometimes(0.5, iaa.Fliplr(0.5)),  # 左右翻转 50%
                                             # iaa.Resize(eval(args.resize)),
                                             # iaa.Sometimes(0.5,iaa.Affine(
                                                 # translate_px={"x": 15, "y": 15},   # 在x或y 方向上平移一定的像素
                                                 # scale=(1.0,1.3),
                                                 # rotate=(-20,20),
                                                 # shear=(-20,20),
                                                 # fit_output= True
                                             # 仿射变换
                                         ]

                        )

                        seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
                        # 读取图片
                        img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                        # sp = img.size
                        img = np.asarray(img)

                        image_aug = seq_det.augment_images([img])[0]
                        # bndbox 坐标增强
                        for i in range(len(bndbox)):
                            x1, y1, x2, y2 = bndbox[i][0], bndbox[i][1], bndbox[i][2], bndbox[i][3]
                            bbs = ia.BoundingBoxesOnImage([
                                ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                            ], shape=img.shape)

                            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                            # boxes_img_aug_list.append(bbs_aug)

                            # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                            n_x1 = int(max(1, min(image_aug.shape[1], bbs_aug.bounding_boxes[0].x1)))
                            n_y1 = int(max(1, min(image_aug.shape[0], bbs_aug.bounding_boxes[0].y1)))
                            n_x2 = int(max(1, min(image_aug.shape[1], bbs_aug.bounding_boxes[0].x2)))
                            n_y2 = int(max(1, min(image_aug.shape[0], bbs_aug.bounding_boxes[0].y2)))
                            if n_x1 == 1 and n_x1 == n_x2:
                                n_x1 = None
                            elif n_x1 >= n_x2:
                                n_x1 = None
                            elif n_x2-n_x1 < 0.3*(x2-x1):
                                n_x1 = None
                            if n_y1 == 1 and n_y2 == n_y1:
                                n_x1 = None
                            elif n_y1 >= n_y2:
                                n_x1 = None
                            elif n_y2-n_y1< 0.3*(y2-y1):
                                n_x1 = None
                            new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])
                            # 存储变化后的图片
                        path = os.path.join(AUG_IMG_DIR, name[:-4] + '_epoch' + '%d' % epoch + '_aug' + '%d' % iter_h + 'x' + '%d' % iter_w + '.jpg')
                        Image.fromarray(image_aug).save(path)
                        h_after,w_after = image_aug.shape[0], image_aug.shape[1]

                        # 存储变化后的XML
                        change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR,
                                                   name[:-4] + '_epoch' + '%d' % epoch + '_aug' + '%d' % iter_h + 'x' + '%d' % iter_w,
                                                   h_after, w_after)
                        print(name[:-4] + '_epoch' + '%d' % epoch + '_aug' + '%d' % iter_h + 'x' + '%d' % iter_w + '.jpg')
                        new_bndbox_list = []
