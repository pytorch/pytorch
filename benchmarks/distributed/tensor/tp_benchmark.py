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
import contextlib
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.nn.functional import (
    _reduce_scatter_base,
    _all_gather_base,
    all_reduce,
)
from torch.utils._pytree import tree_flatten, tree_map

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


class DummyDTensor(torch.Tensor):
    _local_tensor: torch.Tensor

    @staticmethod
    def __new__(
        cls,
        local_tensor: torch.Tensor,
    ) -> "DummyDTensor":
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            local_tensor.size(),
            strides=local_tensor.stride(),
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=local_tensor.requires_grad,
        )
        r._local_tensor = local_tensor
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap_schema(e: object) -> object:
            return e._local_tensor if isinstance(e, DummyDTensor) else e

        def wrap_schema(e: object) -> object:
            return DummyDTensor(e) if isinstance(e, torch.Tensor) else e
        args_unwrap = tree_map(unwrap_schema, args)
        kwargs_unwrap = tree_map(unwrap_schema, kwargs)
        return tree_map(wrap_schema, func(*args_unwrap, **kwargs_unwrap))



class MLPModel(nn.Module):
    def __init__(self, dim_size):
        super(MLPModel, self).__init__()
        self.net1 = nn.Linear(128, dim_size)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(dim_size, 128)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))




@contextlib.contextmanager
def maybe_run_profiler(args, path):
    if args.dump_profiler:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(path),
            profile_memory=True,
            with_stack=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield torch_profiler


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
    inp = torch.rand(128, 128).cuda(rank)
    torch.distributed.barrier()
    with maybe_run_profiler(args, "./control/") as torch_profiler:
        for i in range(args.iter_nums):
            if args.sequence_parallel:
                output = torch.empty(inp.size(0) * tp_degree, *inp.size()[1:], device=inp.device)
                input = _all_gather_base(output, inp)
            output = model(input)
            if args.sequence_parallel:
                output_rs = torch.empty(output.size(0) // tp_degree, *output.size()[1:], device=output.device)
                output = _reduce_scatter_base(output_rs, output)
            else:
                output = all_reduce(output)
            output.sum().backward()
            optimizer.step()
            if args.dump_profiler:
                torch_profiler.step()
            if i == 0:
                t0 = time.perf_counter()
    torch.distributed.barrier()
    t1 = time.perf_counter()
    control_group = t1 - t0
    if rank == 0:
        print(f"Elapsed time for control group: {control_group:.6f}")

    # Control group to mimic Megatron.
    model.net1.weight = torch.nn.Parameter(DummyDTensor(model.net1.weight))
    model.net1.bias = torch.nn.Parameter(DummyDTensor(model.net1.bias))
    model.net2.weight = torch.nn.Parameter(DummyDTensor(model.net2.weight))
    model.net2.bias = torch.nn.Parameter(DummyDTensor(model.net2.bias))
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    inp = torch.rand(128, 128).cuda(rank)
    torch.distributed.barrier()
    with maybe_run_profiler(args, "./control_upper_bound/") as torch_profiler:
        for i in range(args.iter_nums):
            if args.sequence_parallel:
                output = torch.empty(inp.size(0) * tp_degree, *inp.size()[1:], device=inp.device)
                input = _all_gather_base(output, inp)
            output = model(input)
            if args.sequence_parallel:
                output_rs = torch.empty(output.size(0) // tp_degree, *output.size()[1:], device=output.device)
                output = _reduce_scatter_base(output_rs, output)
            else:
                output = all_reduce(output)
            output.sum().backward()
            optimizer.step()
            if args.dump_profiler:
                torch_profiler.step()
            if i == 0:
                t0 = time.perf_counter()
    torch.distributed.barrier()
    t1 = time.perf_counter()
    control_group = t1 - t0
    if rank == 0:
        print(f"Elapsed time for control upperbound group: {control_group:.6f}")

    # create model and move it to GPU with id rank
    model = MLPModel(dim_size).cuda(rank)
    # Create a optimizer for the parallelized module.

    # Parallelize the module based on the given Parallel Style.
    parallel_style = SequenceParallel() if args.sequence_parallel else PairwiseParallel()
    model = parallelize_module(model, device_mesh, parallel_style)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    torch.distributed.barrier()
    with maybe_run_profiler(args, "./test/") as torch_profiler:
        for i in range(args.iter_nums):
            output = model(inp)
            output.sum().backward()
            optimizer.step()
            if args.dump_profiler:
                torch_profiler.step()
            if i == 0:
                t0 = time.perf_counter()
    torch.distributed.barrier()
    t1 = time.perf_counter()
    test_group = t1 - t0

    if rank == 0:
        print(f"Elapsed time for test group: {test_group:.6f}")

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


if __name__ == "__main__":
    main()
