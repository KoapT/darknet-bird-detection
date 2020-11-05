#!/bin/bash
DATA="cfg/bird.data"
CFG="cfg/bird.cfg"
LAYERS=74
INWEIGHTS="backup/bird9/bird_final.weights"
OUTWEIGHTS="pretrained_weights/bird.weights${LAYERS}"

./darknet partial ${CFG} ${INWEIGHTS} ${OUTWEIGHTS} ${LAYERS}
