#!/bin/bash

dir="a100/"
data="a100_mm.txt"
python train_decision_mm.py ${dir}a100_mm.txt --heuristic-name MMRankingA100 --ranking --data train_timm ${dir}timm_train_mm.txt --data train_hf ${dir}hf_train_mm.txt
