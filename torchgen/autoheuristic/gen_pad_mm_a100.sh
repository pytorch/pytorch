#!/bin/bash

data="pad_mm_a100_data.txt"

python train_pad_mm.py ${data} --heuristic-name PadMMA100
