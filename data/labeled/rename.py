import os 
import shutil
for files in os.listdir('fake/'):
    file_path = os.path.join('fake',files)
    new_name = files.replace('fake_','')
    new_path = os.path.join('fake',new_name)
    shutil.move(file_path, new_path)

