#!/bin/bash

data="pad_mm_h100_data.txt"

python train_regression_pad_mm.py ${data} --heuristic-name PadMMH100
