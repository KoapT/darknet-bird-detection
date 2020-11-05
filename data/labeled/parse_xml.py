import os 
import shutil

jpglist = []
xmllist = []
for root, dirs, files in os.walk('JPEGImages'):
    for i in files:
        if i.endswith('.jpg'):
            jpglist.append(os.path.join(root,i))
        if i.endswith('.xml'):
            xmllist.append(os.path.join(root,i))

assert len(xmllist)==len(jpglist)

try:
    os.mkdir("Annotations")
except FileExistsError:
    print("File \'Annotations\' already exist! ")

for xml in xmllist:
    shutil.move(xml,os.path.join('Annotations',xml.split('/')[-1]))


