import os 
import shutil
valid_path = '/home/wootion/TK/g-darknet/data/labeled/seg/valid.txt'
gt_path = 'mAP/input/ground-truth/'
img_path = 'mAP/input/images-optional/'
shutil.rmtree(gt_path)
os.mkdir(gt_path)
with open(valid_path,'r') as f:
    for jpg in f.readlines():
        jpg = jpg.strip()
        xml = jpg.replace('.jpg','.xml').replace('JPEGImages','Annotations')
        #shutil.copy(jpg,img_path)
        shutil.copy(xml,gt_path)


