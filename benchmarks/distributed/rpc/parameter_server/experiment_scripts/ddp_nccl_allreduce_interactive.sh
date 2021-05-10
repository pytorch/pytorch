#!/bin/bash

srun --label \
    --job-name=ddp_nccl_allreduce_interactive \
    --partition=q2 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --gpus-per-task=4 \
    --nodes=1 \
    --gpus-per-node=4 \
    --time=1:00:00 \
    ddp_nccl_allreduce.sh
