import os 
import shutil

source = '../../input/ground-truth'
no_path = os.path.join(source,'backup_no_matches_found')
dr_path = '../../input/detection-results'

if os.path.exists(no_path):
    for file in os.listdir(no_path):
        with open(os.path.join(no_path, file),'r') as t:
            if t.readlines():
                shutil.copy(os.path.join(no_path, file), source)
                with open(os.path.join(dr_path, file),'w') as f:
                    f.write('')
                print('The empty file:\''+file + '\' has been generated!')
else:
	pass


