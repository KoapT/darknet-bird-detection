import pandas as pd
import os 
import shutil

outfile_path = 'mAP/input/detection-results'
if os.path.exists(outfile_path):
    shutil.rmtree(outfile_path)
os.mkdir(outfile_path)
df_bird = pd.read_csv('comp4_det_test_bird.txt',sep=' ', names=['filename','trust','x1','y1','x2','y2'])
df_person = pd.read_csv('comp4_det_test_person.txt',sep=' ', names=['filename','trust','x1','y1','x2','y2'])

df_bird['category']='bird'
df_person['category'] = 'person'
df = pd.concat([df_bird,df_person],axis=0)
df = df[df.trust>=0.3]

for filename in set(df.filename):
    temp_df = df[df.filename==filename]
    outpath = os.path.join(outfile_path,temp_df.iloc[0].filename+'.txt')
    with open(outpath,'w') as f:
        for i in range(temp_df.shape[0]):
            ss = temp_df.iloc[i]
            f.write("{} {} {} {} {} {}\n".format(ss.category, ss.trust, int(ss.x1), int(ss.y1), int(ss.x2), int(ss.y2)))
