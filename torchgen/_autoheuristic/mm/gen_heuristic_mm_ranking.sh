#!/bin/bash

dir="a100/"
data="a100_mm.txt"
python train_decision_mm.py ${dir}a100_mm.txt --heuristic-name MMRankingA100 --data mm_hf ${dir}mm_hf.txt --data mm_timm ${dir}mm_timm.txt --data mm_torchbench ${dir}mm_torchbench.txt --ranking --save-dot --data train_timm ${dir}timm_train_mm.txt --data train_hf ${dir}hf_train_mm.txt
