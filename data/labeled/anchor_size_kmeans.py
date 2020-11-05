import os 
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans

#os.listdir('Annotations')

inputsize = 608

def get_hw(anno_path):
    global inputsize
    hw_list = []
    root = ET.parse(anno_path).getroot()
    size = root.find('size')
    w = int(size.find('width').text.strip())
    h = int(size.find('height').text.strip())
    # l = max(w,h)
    objects = root.findall('object')
    for obj in objects:
        difficult = obj.find('difficult').text.strip()
        if int(difficult) == 1:
            continue
        bbox = obj.find('bndbox')
#           class_ind = classes.index(obj.find('name').text.lower().strip())
        xmin = int(bbox.find('xmin').text.strip())
        xmax = int(bbox.find('xmax').text.strip())
        ymin = int(bbox.find('ymin').text.strip())
        ymax = int(bbox.find('ymax').text.strip())
        x = (xmax-xmin)/(w/inputsize)
        y = (ymax-ymin)/(h/inputsize)
        hw_list.append((x,y))
    return hw_list
# get_hw("Annotations/w20190719093725711_0.xml")

HW_list = []
for xml in os.listdir('Annotations'):
    if xml.endswith('.xml'):
        anno_path = os.path.join('Annotations',xml)
        hw_list = get_hw(anno_path)
        HW_list.extend(hw_list)

n_clusters = int(input('n_clusters:'))
clf = KMeans(n_clusters=n_clusters, max_iter=300, n_init=10)
classes = clf.fit_predict(HW_list)
classes = classes.tolist()
print('The size of the anchors by K-Means:')
print(clf.cluster_centers_, '\n')

class_set = set(classes)
dd = {}
for item in class_set:
    dd.update({item:classes.count(item)})
print("Count of each anchor size:")
print(dd)
