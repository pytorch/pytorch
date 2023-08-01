#!/usr/bin/env python3
#
# Measure distributed training iteration time.
#
# This program performs a sweep over a) a number of model architectures, and
# b) an increasing number of processes. This produces a 1-GPU baseline,
# an 8-GPU baseline (if applicable), as well as measurements for however
# many processes can participate in training.
#

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributed._tensor import (
    DeviceMesh,
)
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
)
from torch.distributed.nn.functional import (
    _reduce_scatter_base,
    _all_gather_base,
    all_reduce,
)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


class MLPModel(nn.Module):
    def __init__(self, dim_size):
        super(MLPModel, self).__init__()
        self.net1 = nn.Linear(10, dim_size)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(dim_size, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def run_tp(rank, args):
    """
    Main body of the demo of a basic version of tensor parallel by using
    PyTorch native APIs.
    """
    print(f"Running basic Megatron style TP example on rank {rank}.")
    setup(rank, args.world_size)
    tp_degree = args.world_size
    dim_size = 64
    # create a sharding plan based on the given world_size.
    device_mesh = DeviceMesh(
        "cuda",
        torch.arange(args.world_size),
    )
    LR = 0.25

    # Control group to mimic Megatron.
    model = MLPModel(dim_size // tp_degree).cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    t0 = time.perf_counter()
    for _ in range(args.iter_nums):
        inp = torch.rand(20, 10).cuda(rank)
        if args.sequence_parallel:
            output = torch.empty_like(inp)
            inp = _all_gather_base(output, inp)
        output = model(inp)
        if args.sequence_parallel:
            output = all_reduce(output)
        else:
            output_rs = torch.empty(output.size(0) // tp_degree, *output.size()[1:], device=output.device)
            output = _reduce_scatter_base(output_rs, output)
        output.sum().backward()
        optimizer.step()
    torch.distributed.barrier()
    t1 = time.perf_counter()
    control_group = t1 - t0

    # create model and move it to GPU with id rank
    model = MLPModel().cuda(rank)
    # Create a optimizer for the parallelized module.

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    # Parallelize the module based on the given Parallel Style.
    parallel_style = SequenceParallel() if args.sequence_parallel else PairwiseParallel()
    model = parallelize_module(model, device_mesh, parallel_style)

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    t0 = time.perf_counter()
    for _ in range(args.iter_nums):
        inp = torch.rand(20, 10).cuda(rank)
        output = model(inp)
        output.sum().backward()
        optimizer.step()
    torch.distributed.barrier()
    t1 = time.perf_counter()
    test_group = t1 - t0

    if rank == 0:
        print(f"Elapsed time for control group: {control_group:0.4f}")
        print(f"Elapsed time for test group: {test_group:0.4f}")

    cleanup()

def run_tp_mp(fn, args):
    mp.spawn(fn,
             args=(args,),
             nprocs=args.world_size,
             join=True)

def main():
    parser = argparse.ArgumentParser(description="PyTorch distributed TP benchmark")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--iter_nums", type=int, default=10)
    parser.add_argument("--distributed-backend", type=str, default="nccl")
    parser.add_argument(
        "--dump-profiler", type=bool, help="Write file with benchmark results"
    )
    parser.add_argument(
        "--sequence-parallel", type=bool, help="Write file with benchmark results"
    )
    args = parser.parse_args()
    run_tp_mp(run_tp, args)




    # output = allgather_run("nvidia-smi topo -m")
    # if not allequal(output):
    #     print('Output of "nvidia-smi topo -m" differs between machines')
    #     sys.exit(1)

    # if args.rank == 0:
    #     print("-----------------------------------")
    #     print("PyTorch distributed benchmark suite")
    #     print("-----------------------------------")
    #     print("")
    #     print(f"* PyTorch version: {torch.__version__}")
    #     print(f"* CUDA version: {torch.version.cuda}")
    #     print(f"* Distributed backend: {args.distributed_backend}")
    #     print("")
    #     print("--- nvidia-smi topo -m ---")
    #     print("")
    #     print(output[0])
    #     print("--------------------------")
    #     print("")

    # torch.cuda.set_device(dist.get_rank() % 8)
    # device = torch.device("cuda:%d" % (dist.get_rank() % 8))

    # benchmark_results = []
    # for benchmark in benchmarks:
    #     if args.rank == 0:
    #         print(f"\nBenchmark: {str(benchmark)}")
    #     result = sweep(benchmark)
    #     benchmark_results.append(
    #         {
    #             "model": benchmark.model,
    #             "batch_size": benchmark.batch_size,
    #             "result": result,
    #         }
    #     )


if __name__ == "__main__":
    main()
