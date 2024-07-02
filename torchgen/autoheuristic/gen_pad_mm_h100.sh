#!/bin/bash

data="h100-data.txt"

python train_pad_mm.py ${data} \
    --heuristic-name PadMMH100 \
    --gpu H100
