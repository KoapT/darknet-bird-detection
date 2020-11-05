import os 
import shutil
valid_path = input('input the valid.txt path:')
gt_path = '/home/wootion/TK/g-darknet/results/mAP/input/ground-truth/'
img_path = '/home/wootion/TK/g-darknet/results/mAP/input/images-optional/'
with open(valid_path,'r') as f:
    for jpg in f.readlines():
        jpg = jpg.strip()
        xml = jpg.replace('.jpg','.xml').replace('JPEGImages','Annotations')
        #shutil.copy(jpg,'test_jpg/')
        shutil.copy(xml,gt_path)


