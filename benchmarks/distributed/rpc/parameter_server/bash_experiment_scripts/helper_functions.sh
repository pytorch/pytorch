#!/bin/sh

run_benchmark_basic() {
    # requires slurm
    gpurun='srun -p q2 --cpus-per-task=16 -t 5:00:00 --gpus-per-node=4'
    $gpurun python launcher.py --benchmark=$1 --data=$2 --model=$3 --trainer=$4 --server=$5
}
