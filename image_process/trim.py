import xml.etree.ElementTree as ET
import os
import shutil
import cv2
from tqdm import tqdm

INPUT_DIR = './before_trim'
OUTPUT_DIR = './after_trim'
DEL_FORMER = True  # 删除之前的输出文件
NUM_H = 2
NUM_W = 4
img_form = '.jpg'
xml_form = '.xml'


def mkdir(dir):
    if os.path.exists(dir) and DEL_FORMER:
        shutil.rmtree(dir)
    try:
        os.mkdir(dir)
    except FileExistsError as e:
        pass


def trim(img, num_h, num_w):
    jpg_path = os.path.join(INPUT_DIR, img)
    xml_path = jpg_path.replace(img_form, xml_form)
    if os.path.exists(xml_path):
        assert FileExistsError
    img_arr = cv2.imread(jpg_path)
    h, w, _ = img_arr.shape
    for i in range(num_h):
        for j in range(num_w):
            left, right = int(j / num_w * w), int((j + 1) / num_w * w)
            top, bottom = int(i / num_h * h), int((i + 1) / num_h * h)
            # print(left,right,top,bottom)
            img_trim = img_arr[top:bottom, left:right, :]
            img_name = os.path.splitext(img)[0] + '_trim_{}_{}'.format(j, i) + img_form
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img_trim)
            # cv2.imshow('dd',img_trim)
            # cv2.waitKey(0)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            root.find('filename').text = img_name
            root.find('size').find('width').text = str(w // num_w)
            root.find('size').find('height').text = str(h // num_h)
            objs = root.findall('object')
            for obj in objs:
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                xmax = int(bndbox.find('xmax').text)
                ymin = int(bndbox.find('ymin').text)
                ymax = int(bndbox.find('ymax').text)
                square = (xmax - xmin) * (ymax - ymin)
                if any([xmin >= right, xmax <= left, ymin >= bottom, ymax <= top]):
                    root.remove(obj)  # 范围以外的目标移除
                else:
                    if xmin < left: xmin = left
                    if xmax > right: xmax = right
                    if ymin < top: ymin = top
                    if ymax > bottom: ymax = bottom
                    new_square = (xmax - xmin) * (ymax - ymin)
                    if new_square / square < 0.3:
                        root.remove(obj)  # 跟原物体比太小的目标移除
                    else:
                        bndbox.find('xmin').text = str(xmin - left)
                        bndbox.find('xmax').text = str(xmax - left)
                        bndbox.find('ymin').text = str(ymin - top)
                        bndbox.find('ymax').text = str(ymax - top)
            tree.write(os.path.join(OUTPUT_DIR, img_name.replace(img_form, xml_form)))
    pbar.set_description("%s" % jpg_path)


if __name__ == '__main__':
    mkdir(OUTPUT_DIR)
    jpg_list = [jpg for jpg in os.listdir(INPUT_DIR) if
                os.path.splitext(jpg)[-1] == img_form]
    pbar = tqdm(jpg_list)
    for jpg in pbar:
        trim(jpg, NUM_H, NUM_W)
        pbar.set_description("%s" % jpg)
