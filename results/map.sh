#!/bin/bash

python gen_results.py
echo "The detection results have been located!"
python gen_gt.py
echo "The ground truth xmls have been located!"
cd mAP/scripts/extra/ && python convert_gt_xml.py
echo "The ground truth xmls have been convert to txts!"
python intersect-gt-and-dr.py
python gen_empty.py
cd ../../ && python main.py


