#!/bin/bash

DATA="cfg/bird.data"
CFG="cfg/bird.cfg"
WEIGHTS="backup/bird9/bird_final.weights"
CONF_THRESH=0.3
IOU_THRESH=0.3

./darknet detector valid ${DATA} ${CFG} ${WEIGHTS}
echo "Validation results has been generated. Now start evaluation..."
cd ./results && python evaluation.py --thresh ${CONF_THRESH} --iouthresh ${IOU_THRESH}
echo "Finished evaluation, you can see the results in results/results.txt"
echo "Start calculating mAP..."
. map.sh
echo "Show the wrong and miss detection results..."
cd .. && python draw_wrongNmiss.py
echo "Finished!"
cd ..
