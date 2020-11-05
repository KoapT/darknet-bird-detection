from PIL import Image
import numpy as np

import os 
import xml.etree.ElementTree as ET

xmllist = [xml for xml in os.listdir('Annotations') if xml.endswith('.xml')]
xmldir = './Annotations/'
jpgdir = './JPEGImages/'
# savedir = './dd/'

for xml in xmllist:
    in_file = open(os.path.join(xmldir, xml))
    tree = ET.parse(in_file)
    jpgname = tree.find('filename')
    if not jpgname.text.endswith('.jpg'):
        jpgname.text += '.jpg'
        print(xml + 'filename chaged!')
#     print(jpgname)
    img = Image.open(os.path.join(jpgdir, jpgname.text))
    img = np.asarray(img)
    h,w = img.shape[0], img.shape[1]
#     print(w,h)
    size = tree.find('size')
    width = size.find('width')
    if int(width.text.strip()) != w:
        print(xml + 'width changed!')
        width.text = str(w)
    height = size.find('height')
    if int(height.text.strip()) != h:
        print(xml + 'height changed!')
        height.text = str(h)
    
    
    tree.write(os.path.join(xmldir, xml))
