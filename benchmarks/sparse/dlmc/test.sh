#!/bin/bash

DATASET_ROOT_DIR=$HOME/datasets/

# wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz -P $DATASET_ROOT_DIR
# tar -xvf $DATASET_ROOT_DIR/dlmc.tar.gz 

echo "!! SPARSE SPMS TIME BENCHMARK!! " 

python -m dlmc.spmm  --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset random_pruning --operation matmul --spmm_type=sparse-dense --output /tmp/sspmm_bench.pkl
python -m dlmc.spmm  --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset random_pruning --operation backward --spmm_type=sparse-dense --output /tmp/sspmm_backward_bench.pkl

python -m dlmc.spmm  --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset random_pruning --operation matmul --spmm_type=sparse-sparse --output /tmp/spmm_bench.pkl
python -m dlmc.spmm  --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset random_pruning --operation backward --spmm_type=sparse-sparse --output /tmp/spmm_backward_bench.pkl

python -m dlmc.spmv  --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset random_pruning --operation matmul --output /tmp/spmv_bench.pkl
