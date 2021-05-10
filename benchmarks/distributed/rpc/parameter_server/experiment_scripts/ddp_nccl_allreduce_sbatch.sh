#!/bin/bash

#SBATCH --job-name=ddp_nccl_allreduce_sbatch

#SBATCH --partition=q2

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16

#SBATCH --gpus-per-task=4

#SBATCH --nodes=1

#SBATCH --gpus-per-node=4

#SBATCH --time=1:00:00

srun --label ddp_nccl_allreduce.sh
