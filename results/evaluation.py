import xml.etree.ElementTree as ET
import pandas as pd
import argparse
import os

delta = 0.00001   # 防止除以0的参数
parser = argparse.ArgumentParser(description="objectiveness thresh:")
parser.add_argument('--thresh', default='0.3')
parser.add_argument('--iouthresh', default='0.3')
parser.add_argument('--valid_path', default='/home/wootion/TK/g-darknet/data/labeled/valid.txt')
args = parser.parse_args()

df_bird = pd.read_csv('comp4_det_test_bird.txt',sep=' ', names=['filename','trust','x1','y1','x2','y2'])
df_person = pd.read_csv('comp4_det_test_person.txt',sep=' ', names=['filename','trust','x1','y1','x2','y2'])
df_bird = df_bird[df_bird.trust>=eval(args.thresh)]
df_person = df_person[df_person.trust>=eval(args.thresh)]
df_bird['category']='bird'
df_person['category'] = 'person'
df = pd.concat([df_bird,df_person],axis=0)
df = df.reset_index(drop=True)
df.insert(1, 'istrue', 0)
categorys = ['bird', 'person']
miss_list = []
P = {'bird':df_bird.shape[0],
     'person':df_person.shape[0],
     'sum':df.shape[0]}
T = {'bird':0,
     'person':0,
     'sum':0}
TP = {'bird':0,
     'person':0,
     'sum':0}
FP = {'bird':0,
     'person':0,
     'sum':0}
FN = {'bird':0,
     'person':0,
     'sum':0}
valid_path = args.valid_path
def IOU(b,b1):
    x1_max = max(b[0],b1[0])
    x1_min = min(b[0],b1[0])
    x2_max = max(b[1],b1[1])
    x2_min = min(b[1],b1[1])
    y1_max = max(b[2],b1[2])
    y1_min = min(b[2],b1[2])
    y2_max = max(b[3],b1[3])
    y2_min = min(b[3],b1[3])
    I = max((x2_min-x1_max),0)*max((y2_min-y1_max),0)
    U = (b[1]-b[0])*(b[3]-b[2]) + (b1[1]-b1[0])*(b1[3]-b1[2]) - I
    IoU = I/U
    return IoU

with open(valid_path,'r') as v:
    n = 0
    for p in v.readlines():
        n+=1
        xml = p.strip().replace('JPEGImages','Annotations').replace('.jpg','.xml')
        #xml = '/home/wootion/TK/g-darknet/data/labeled/Annotations/w20190805083851060_132_epoch0_aug0x0.xml'
        tree=ET.parse(open(xml))
        root = tree.getroot()
        filename = os.path.split(xml)[-1][:-4]
        temp_df = df[df.filename==filename]
        for obj in root.iter('object'):
            T['sum'] += 1
            flag = 0
            difficult = obj.find('difficult').text
            if int(difficult) == 1:
                continue
            cls = obj.find('name').text.strip()
            T[cls] += 1
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), \
                    float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            
            temp1_df = temp_df[temp_df.category==cls]
            for i in range(temp1_df.shape[0]):
                b1 = (temp1_df.iloc[i].x1, temp1_df.iloc[i].x2, temp1_df.iloc[i].y1, temp1_df.iloc[i].y2)
                if IOU(b,b1) >= eval(args.iouthresh):
                    name = temp1_df.iloc[i].name
                    # temp_df.loc[name,'istrue'] = 1
                    df.loc[name,'istrue'] += 1
                    flag = 1
            if flag == 1:
                TP[cls] +=1
                TP['sum'] +=1
            else:
                FN[cls] +=1
                FN['sum'] +=1
                if filename not in miss_list:
                    miss_list.append(filename)
        print("{} pictures have been evaluated!".format(n))
        
dff= df[df.istrue==0]
FP['sum'] = dff.shape[0]
FP['bird'] = dff[dff.category=='bird'].shape[0]
FP['person'] = dff[dff.category=='person'].shape[0]
wrong_list = list(dff.filename)

df.to_csv('bboxes.csv',index=False)

with open('wrong_and_miss.txt','w') as w:
    w.write('#wrong_list({}):'.format(len(wrong_list))+'\n')
    for i in wrong_list:
        w.write(i.strip()+'\n')
    w.write('#'*20+'\n'+'#miss_list({}):'.format(len(miss_list))+'\n')
    for j in miss_list:
        w.write(j.strip()+'\n')

with open('results.txt','w') as f:
    f.write(' '+'——'*30+'\n')
    f.write('|{:4s}|{:>4s}|{:>4s}|{:>4s}|{:>4s}|{:>4s}|{:>4s}|{:>4s}|\n'\
        .format('类别','真实数','检测数','正检数','误检数','漏检数','识别率','误检率'))
    f.write(' '+'——'*30+'\n')
    for cls in P.keys():
        f.write('|{:6s}|{:6d}|{:7d}|{:7d}|{:7d}|{:6d}|{:>6.2f}%|{:>6.2f}%|\n'\
            .format(cls,T[cls],P[cls],TP[cls],FP[cls],FN[cls],TP[cls]/(T[cls]+delta)*100,FP[cls]/(P[cls]+delta)*100))
    f.write(' '+'——'*30+'\n')

print(' '+'——'*31)
print('|{:4s}|{:>4s}|{:>4s}|{:>4s}|{:>4s}|{:>4s}|{:>4s}|{:>4s}|'\
    .format('类别','真实数','检测数','正检数','误检数','漏检数','识别率','误检率'))
print(' '+'——'*31)
for cls in P.keys():
    print('|{:6s}|{:7d}|{:7d}|{:7d}|{:7d}|{:7d}|{:>6.2f}%|{:>6.2f}%|'\
        .format(cls,T[cls],P[cls],TP[cls],FP[cls],FN[cls],TP[cls]/(T[cls]+delta)*100,FP[cls]/(P[cls]+delta)*100))
print(' '+'——'*31)
