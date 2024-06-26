#!/bin/bash

data="a100-data.txt"

python train_pad_mm.py ${data} \
    --heuristic-name PadMMA100 \
    --gpu A100
