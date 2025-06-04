#!/bin/bash

data="mixedmm_a100_data.txt"

python train_decision_mixedmm.py ${data} --heuristic-name MixedMMA100
