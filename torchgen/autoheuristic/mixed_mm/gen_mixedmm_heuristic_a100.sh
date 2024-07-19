#!/bin/bash

data="a100_mixedmm_data.txt"

python train_decision_mixedmm.py ${data} --heuristic-name MixedMMA100
