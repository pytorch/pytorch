#!/bin/bash

data="mixedmm_h100_data.txt"

python train_decision_mixedmm.py ${data} --heuristic-name MixedMMH100
