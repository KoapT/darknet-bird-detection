import xml.etree.ElementTree as ET
import os
import random
import shutil

xmllist = [i for i in os.listdir('fake_birds/') if i.endswith('.xml')]
# jpglist = [i for i in os.listdir('fake_birds/') if i.endswith('.jpg')]
# no_obj = 'no_obj/'

i = 0
for xml in xmllist:
    xmlfile = 'fake_birds/{}'.format(xml)
    jpgfile = 'fake_birds/{}'.format(xml.replace('.xml','.jpg'))
    tree=ET.parse(xmlfile)
    root = tree.getroot()
    if not root.find('object'):
        #if random.randint(1,10)>3:
        shutil.move(xmlfile,'Annotations/')
        shutil.move(jpgfile,'JPEGImages/')
    if 'bird' in [obj.find('name').text for obj in root.findall('object')]:
        continue
    shutil.move(xmlfile,'Annotations/')
    shutil.move(jpgfile,'JPEGImages/')
