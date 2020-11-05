import os 
import xml.etree.ElementTree as ET
import random
import shutil

r = .1

input_path = input('Input the to deleted path:')
try:
	os.mkdir(os.path.join(input_path,'../no_obj/'))
except FileExistsError as e:
	pass
output_path = os.path.join(input_path,'../no_obj/')

xml_names = [xml for xml in os.listdir(input_path) if xml.endswith('.xml')]
for xml in xml_names:
	xml_path = os.path.join(input_path,xml)
	jpg_path = os.path.join(input_path,xml.replace('.xml','.jpg'))
	tree = ET.parse(xml_path)
	root = tree.getroot()
	objects = root.find('object')
	if not objects and random.random()>r:
		shutil.move(xml_path,output_path)
		shutil.move(jpg_path,output_path)
		print('Move the file {} ---------to---------{}!'.format(xml_path[:-4],output_path))

