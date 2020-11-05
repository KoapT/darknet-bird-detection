import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
import shutil
import random
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


ia.seed(1)
DEL_FORMER = False  # 删除之前的输出文件
INPUT_DIR = input("Input the path:")
OUTPUT_DIR = './after_aug'
AUGLOOP = 1  # 每张影像增强的数量

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
    in_file = open(os.path.join(root, str(image_id) + '.xml'))
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


def mkdir(dir):
    if os.path.exists(dir) and DEL_FORMER:
        shutil.rmtree(dir)
    try:
        os.mkdir(dir)
    except FileExistsError as e:
        pass


def img_augment(image, bboxes=None, p=.7):
    """
    使用imgaug库进行的图像增强。
    :param image: np.array, images.
    :param bboxes: np.array, bboxes of object detection.
    :param p: ratio to do augment.
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
        for box in bboxes_list:
            x1, y1, x2, y2 = tuple(box)
            bbs = ia.BoundingBoxesOnImage([
                ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            ], shape=image.shape)

            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
            n_x1 = int(max(1, min(image_aug.shape[1], bbs_aug.bounding_boxes[0].x1)))
            n_y1 = int(max(1, min(image_aug.shape[0], bbs_aug.bounding_boxes[0].y1)))
            n_x2 = int(max(1, min(image_aug.shape[1], bbs_aug.bounding_boxes[0].x2)))
            n_y2 = int(max(1, min(image_aug.shape[0], bbs_aug.bounding_boxes[0].y2)))
            new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])
        # 长宽比太大的图片不要，产生新的image和bboxes
        if 1 / 3 <= image_aug.shape[0] / image_aug.shape[1] <= 3:
            break
    return image_aug, new_bndbox_list

if __name__ == "__main__":
    mkdir(OUTPUT_DIR)
    for root, sub_folders, files in os.walk(INPUT_DIR):
        for name in files:
            if name.endswith('.xml'):
                img = Image.open(os.path.join(INPUT_DIR, name[:-4] + '.jpg'))
                img = np.asarray(img)
                bndbox,size = read_xml_annotation(INPUT_DIR, name)
                w_raw = size[0]
                h_raw = size[1]
                if len(bndbox) == 0:
                    continue
                bndbox = np.array(bndbox)
                for epoch in range(AUGLOOP):
                    image_aug, new_bndbox_list = img_augment(img, bboxes=bndbox, p=1.)
                    path = os.path.join(OUTPUT_DIR, name[:-4] + '_epoch' + '%d' % epoch + '_aug' + '.jpg')
                    Image.fromarray(image_aug).save(path)
                    h_after,w_after = image_aug.shape[0], image_aug.shape[1]

                    # 存储变化后的XML
                    change_xml_list_annotation(INPUT_DIR, name[:-4], new_bndbox_list, OUTPUT_DIR,
                                                   name[:-4] + '_epoch' + '%d' % epoch + '_aug',
                                                   h_after, w_after)
                    print(name[:-4] + '_epoch' + '%d' % epoch + '_aug' + '.jpg')
