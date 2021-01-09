#!/bin/bash

DATASET_ROOT_DIR=$HOME/datasets/

# wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz -P $DATASET_ROOT_DIR
# tar -xvf $DATASET_ROOT_DIR/dlmc.tar.gz 

echo "!! SPARSE SPMS TIME BENCHMARK!! " 

python matmul_dlmc_bench.py --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset random_pruning --operation matmul --output /tmp/matmul_bench.pkl
python matmul_dlmc_bench.py --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset random_pruning --operation backward --output /tmp/backward_bench.pkl

python plot_results.py -i /tmp/matmul_bench.pkl
python plot_results.py -i /tmp/backward_bench.pkl
