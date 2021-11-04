"""
A simple tool to compare the performance of different impls of
DistributedDataParallel on resnet50, three flavors:

1. DistributedDataParallel, which has a python wrapper and C++ core to do
   gradient distribution and reduction. It's current production version.

2. PythonDDP with async gradient reduction.

3. PythonDDP with synchrous gradient reduction.

Example::
    >>> modify configs in main func
    >>> python compare_ddp.py
    >>> Sample out: compare_ddp_sample.md
"""

import numpy as np
import os
import pickle
import glob
import python_ddp
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from collections import OrderedDict
from enum import Enum
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel as DDP

class DDPOption(Enum):
    DDP_CPP_CORE = 1
    PYTHON_DDP_SYNC_REDUCTION = 2
    PYTHON_DDP_ASYNC_REDUCTION = 3

class LatencyData:
    __slots__ = ["buffer_size_in_M", "ddp_option", "rank", "metrics"]

    def __init__(self, buffer_size_in_M, ddp_option, rank, metrics):
        self.buffer_size_in_M = buffer_size_in_M
        self.ddp_option = ddp_option
        self.rank = rank
        self.metrics = metrics

def serialize(buffer_size_in_M, ddp_option, rank, metrics,
              data_dir="./tmp", ext="ddpraw"):
    if not os.path.exists(data_dir):
        print(f'{data_dir} not exist, mkdir {data_dir}')
        os.mkdir(data_dir)
    file_name = "buffer_size_{}M_rank{}_{}.{}".format(
        buffer_size_in_M, rank, ddp_option, ext)
    file_path = os.path.join(data_dir, file_name)
    print("Writing metrics to file: '{}'".format(file_path))
    data = LatencyData(buffer_size_in_M, ddp_option, rank, metrics)
    with open(file_path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print(f"Wrote metrics to '{file_path}''")

def load_detailed_metrics(data_dir="./tmp", ext="ddpraw"):
    assert os.path.exists(data_dir)
    file_pattern = os.path.join(data_dir, f"*.{ext}")
    files = glob.glob(file_pattern)
    print("load_detailed_metrics found {} files".format(len(files)))
    buffer_size_to_metrics = OrderedDict()
    for file_path in files:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        # Add data to buffer_size_to_metrics
        buffer_size = data.buffer_size_in_M
        if buffer_size not in buffer_size_to_metrics:
            buffer_size_to_metrics[buffer_size] = {}
        metrics = buffer_size_to_metrics.get(buffer_size)
        assert metrics is not None
        metrics[data.ddp_option] = data.metrics
    return buffer_size_to_metrics

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def create_ddp_model(module, rank, pg, ddp_option, buffer_size_in_M):
    """Helper to create DDPModel. """
    if ddp_option == DDPOption.DDP_CPP_CORE:
        ddp_model = DDP(module, device_ids=[rank],
                        process_group=pg,
                        bucket_cap_mb=buffer_size_in_M)
        ddp_model._set_static_graph()
        return ddp_model
    elif ddp_option == DDPOption.PYTHON_DDP_SYNC_REDUCTION:
        M = 2 ** 20
        return python_ddp.PythonDDP(module, pg, False, buffer_size=buffer_size_in_M * M)
    elif ddp_option == DDPOption.PYTHON_DDP_ASYNC_REDUCTION:
        M = 2 ** 20
        return python_ddp.PythonDDP(module, pg, True, buffer_size=buffer_size_in_M * M)
    else:
        raise NotImplementedError

def run_ddp(rank, world_size, epochs, ddp_option, buffer_size_in_M, warmup_iterations=20):
    print(f'Invoked run_ddp rank {rank}')
    assert epochs > warmup_iterations

    # Setup
    print("setting up ... ")
    setup(rank, world_size)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    device = torch.device('cuda:%d' % rank)
    print('setup done')

    # Create ResNet50 module and wrap in DDP module.
    pg = dist.distributed_c10d._get_default_group()
    model = models.resnet50().to(device)
    ddp_model = create_ddp_model(model, rank, pg, ddp_option, buffer_size_in_M)
    assert ddp_model is not None

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Container to hold: event -> list of events in milliseconds
    MODEL_FORWARD = "forward"
    MODEL_BACKWARD = "backward"
    metrics = {MODEL_FORWARD: [], MODEL_BACKWARD: []}

    for epoch in range(epochs):
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs} ...')

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # TODO(bowangbj): Switch to real training set from ImageNet.
        inputs = torch.rand([32, 3, 224, 224], device=device)
        labels = torch.rand([32, 1000], device=device)

        # Forward
        start.record()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)

        end.record()
        torch.cuda.synchronize()
        if epoch >= warmup_iterations:
            metrics[MODEL_FORWARD].append(start.elapsed_time(end))

        # Backward
        start.record()
        loss.backward()
        # Reduce all grad, this is needed for non-DDP_CPP_CORE since the hook
        # for all_reduce does not exist yet.
        if ddp_option != DDPOption.DDP_CPP_CORE:
            ddp_model.all_reduce_grads()
        end.record()
        torch.cuda.synchronize()
        if epoch >= warmup_iterations:
            metrics[MODEL_BACKWARD].append(start.elapsed_time(end))

        # Optimization
        optimizer.step()
        optimizer.zero_grad()

    if rank == 0:
        print(f"\nMetrics for GPU {rank}, ddp_option={ddp_option}, buffer_size={buffer_size_in_M}M")
        print(f"Skipped {warmup_iterations} CUDA warmpup iterations. ")
        for step, elapsed_milliseconds in metrics.items():
            A = np.array(elapsed_milliseconds)
            print(' {N} iterations, {step}, mean={mean} ms, median={median} ms, p90={p90} ms, p99={p99} ms'.format(
                N=len(A), step=step, mean=np.mean(A),
                median=np.percentile(A, 50), p90=np.percentile(A, 90),
                p99=np.percentile(A, 99)))

        # Serialize the raw data to be used to compute summary. Didn't choose to
        # maintain a global object holding the metrics b/c mp.spawn tries to
        # fork all the arguments before spawning new process thus it's infeasible
        # save global states in an object.
        serialize(buffer_size_in_M, ddp_option, rank, metrics)

def append_delta(row_list, base, exp):
    percent = 100 * ((exp - base) / base)
    row_list.append(percent)

def print_summary(buffer_size_to_metrics):
    # metrics: {ddp_option, Metrics}
    # Metrics: step -> [latency]
    for buffer_size, metrics in buffer_size_to_metrics.items():
        assert DDPOption.DDP_CPP_CORE in metrics.keys()
        baseline = metrics.get(DDPOption.DDP_CPP_CORE)
        print(f"=== Summary for buffer_size: {buffer_size}M === ")
        for step in baseline.keys():
            # step takes value from [forward, backward]
            # compute latency for each step into a table, each row is looks like
            # [option, mean, diff, mean, diff, p90, diff, p95, diff, p99, diff]
            data = []
            baseline_latencies = baseline.get(step)
            assert baseline_latencies is not None
            A_baseline = np.array(baseline_latencies)
            for ddp_option, exp_metrics in metrics.items():
                exp_latencies = exp_metrics.get(step)
                assert exp_latencies is not None
                A_exp = np.array(exp_latencies)
                # Yield option, mean, p50, p90, p95, p99 and delta.
                row = [ddp_option]
                row.append(np.mean(A_exp))
                append_delta(row, np.mean(A_baseline), np.mean(A_exp))
                for px in [50, 90, 95, 99]:
                    base = np.percentile(A_baseline, px)
                    exp = np.percentile(A_exp, px)
                    row.append(exp)
                    append_delta(row, base, exp)
                data.append(row)

            # Output buffer_size, step as a table.
            print(tabulate(data,
                           headers=[f"DDP: [{step}]", "Mean", "delta%",
                                    "mean", "delta%", "p90", "delta%",
                                    "p95", "delta%%", "p99", "delta%"]))
            print("\n")

def main():
    world_size = 2
    epochs = 120

    # resnet50 model facts:
    # total_param_count = 161
    # total_elements = 25557032 ~= 24.37M
    # param_max_elements = 2359296 ~= 2.25M
    # Try different bucket sizes.
    buffer_size_in_mbs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    print("buffer_size_in_mbs: " + str(buffer_size_in_mbs))
    for buffer_size_in_M in buffer_size_in_mbs:
        print("\n\n=== NEW EXPERIMENT: buffer_size={}M, {} epochs, world_size={} ===".format(
            buffer_size_in_M, epochs, world_size))
        options = [
            DDPOption.DDP_CPP_CORE,
            DDPOption.PYTHON_DDP_ASYNC_REDUCTION,
            DDPOption.PYTHON_DDP_SYNC_REDUCTION
        ]
        for option in options:
            print("Measuring option: {} ... ".format(option))
            mp.spawn(run_ddp,
                     args=(world_size, epochs, option, buffer_size_in_M),
                     nprocs=world_size,
                     join=True)

    print("\n Generating summaries ... ")
    buffer_size_to_metrics = load_detailed_metrics(data_dir="./tmp", ext="ddpraw")
    print_summary(buffer_size_to_metrics)

if __name__ == "__main__" :
    main()
