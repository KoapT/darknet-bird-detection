import pandas as pd
import numpy as np
from PIL import ImageDraw, Image
import os,shutil

rs_path = './draw_results'
gt_path = 'mAP/input/ground-truth/'
dr_path = 'mAP/input/detection-results/'
img_path = 'mAP/input/jpg_images/'

shutil.rmtree(rs_path)
os.mkdir(rs_path)

imglist = [txt[:-4] for txt in os.listdir(dr_path) if txt.endswith('.txt')]

for img in imglist:
    jpg = os.path.join(img_path,img+'.jpg')
    dr = os.path.join(dr_path,img+'.txt')
    gt = os.path.join(gt_path,img+'.txt')
    
    pic = Image.open(jpg)
    draw = ImageDraw.Draw(pic)
    colors = [(255,0,0), (0,255,0)]
    with open(gt,'r') as g:
        for i in g.readlines():
            items = i.strip().split(' ')
            bbox = [int(b) for b in items[1:]]
            draw.rectangle(bbox, outline=colors[1])
            center = [(bbox[2]+bbox[0])//2-10, (bbox[3]+bbox[1])//2-5]
            draw.text(center, '{}'.format(items[0]), fill=colors[1])
    with open(dr,'r') as d:
        for n, i in enumerate(d.readlines()):
            items = i.strip().split(' ')
            bbox = [int(b) for b in items[2:]]
            draw.rectangle(bbox, outline=colors[0])
            draw.text(bbox[:2], '{}'.format(items[0]), fill=colors[0])
    pic.save(os.path.join(rs_path,img+'.jpg'))
    print("{} has been drawn and saved in {}".format(img,rs_path))