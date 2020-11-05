import xml.etree.ElementTree as ET
import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
import random

dir_path = input("Input the path:")
input_size = input("Input size of your module:")
BIRD_THRESH = (15,15)
PERSON_THRESH = (20,20)
BIRD_FREE = False
namelist = [f[:-4] for f in os.listdir(dir_path) if f.endswith('.xml')]


count = 0
count_obj = 0
for name in tqdm(namelist):
    print(name)
    xml = os.path.join(dir_path, name + '.xml')
    jpg = xml.replace('.xml', '.jpg')
    img = Image.open(jpg)
    img = np.array(img)
    tree = ET.parse(xml)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    root = tree.getroot()
    delnum = 0
    for object in root.findall('object'):
        bndbox = object.find('bndbox')
        category = object.find('name').text

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)

        w = (xmax - xmin) / width * int(input_size)
        h = (ymax - ymin) / height * int(input_size)
        if category == 'bird':
            if BIRD_FREE or w < BIRD_THRESH[0] or h < BIRD_THRESH[1]:
                delnum += 1
                root.remove(object)
                left_point, right_point = img[ymin:ymax - 1, max(0, xmin - 5), :], img[ymin: ymax - 1,
                                                                                   min(xmax + 5, width - 1), :]
                for channel in range(3):
                    img[ymin:ymax - 1, xmin:xmax - 1, channel] = np.linspace(left_point[:, channel],
                                                                             right_point[:, channel],
                                                                             xmax - 1 - xmin, dtype=int).transpose() if random.random()>.16 else 0
        elif w < PERSON_THRESH[0] or h < PERSON_THRESH[1]:
            delnum += 1
            root.remove(object)
            left_point, right_point = img[ymin:ymax-1, max(0,xmin - 5), :], img[ymin: ymax-1, min(xmax+5,width-1), :]
            for channel in range(3):
                img[ymin:ymax-1, xmin:xmax-1, channel] = np.linspace(left_point[:, channel], right_point[:, channel],
                                                                 xmax - 1 - xmin, dtype=int).transpose() if random.random()>.16 else 0

    if delnum != 0:
        count += 1
        count_obj += delnum
        tree.write(os.path.join(dir_path,name + '.xml'))
        img = Image.fromarray(img)
        img.save(os.path.join(dir_path,name + '.jpg'))
        print("{} has {} objects to be deleted!".format(xml, delnum))

print(
    "{} pictures has {} very small objects to be deleted.".format(
        count,
        count_obj))
