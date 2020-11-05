#!/bin/bash

DATA="cfg/bird.data"
CFG="cfg/bird.cfg"
WEIGHTS="pretrained_weights/bird.105"

./darknet detector train ${DATA} ${CFG} ${WEIGHTS}  -gpus 0,1
