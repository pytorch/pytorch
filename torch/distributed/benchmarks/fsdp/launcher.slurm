#!/bin/bash

#SBATCH --job-name=fsdp-benchmark
#SBATCH --partition=train
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
# Enable EFA for A100
export LOGLEVEL=INFO
export FI_PROVIDER="efa"
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_ALGO=ring
export WORK_DIR=${HOME}
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${WORK_DIR}/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
# srun torchrun --nnodes 12 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" torch/distributed/benchmarks/fsdp/trainer.py --model-name "GPT175B" --batch-size 1 --mode pytorch_fsdp
# srun torchrun --nnodes 1 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" torch/distributed/benchmarks/fsdp/trainer.py --model-name "GPTSmall" --batch-size 2 --mode pytorch_fsdp
# # srun torchrun --nnodes 4 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" torch/distributed/benchmarks/fsdp/trainer.py --model-name "t5-11b" --batch-size 1 --mode pytorch_fsdp
srun torchrun --nnodes 4 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" torch/distributed/benchmarks/fsdp/trainer.py --model-name "t5-small" --batch-size 16 --mode pytorch_fsdp