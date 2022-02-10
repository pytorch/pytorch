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
import torchvision


def allgather_object(obj):
    out = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(out, obj)
    return out

def allgather_run(cmd):
    proc = subprocess.run(shlex.split(cmd), capture_output=True)
    assert(proc.returncode == 0)
    return allgather_object(proc.stdout.decode("utf-8"))



def allequal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def benchmark_process_group(pg, benchmark, use_ddp_for_single_rank=True):
    torch.manual_seed(pg.rank())
    torch.cuda.manual_seed(pg.rank())

    model = benchmark.create_model()
    data = [(benchmark.generate_inputs(), benchmark.generate_target())]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        0.001,
        momentum=0.9,
        weight_decay=1e-4)
    if use_ddp_for_single_rank or pg.size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            process_group=pg,
            bucket_cap_mb=benchmark.bucket_size)

    measurements = []
    warmup_iterations = 5
    measured_iterations = 10
    for (inputs, target) in (data * (warmup_iterations + measured_iterations)):
        start = time.time()
        output = model(*inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        measurements.append(time.time() - start)

    # Throw away measurements for warmup iterations
    return measurements[warmup_iterations:]


def run_benchmark(benchmark, ranks, opts):
    group = dist.new_group(ranks=ranks, backend=benchmark.distributed_backend)
    measurements = []
    if dist.get_rank() in set(ranks):
        if not opts:
            opts = dict()
        measurements = benchmark_process_group(group, benchmark, **opts)
    dist.destroy_process_group(group)
    dist.barrier()

    # Aggregate measurements for better estimation of percentiles
    return list(itertools.chain(*allgather_object(measurements)))


def sweep(benchmark):
    # Synthesize the set of benchmarks to run.
    # This list contain tuples for ("string prefix", [rank...]).
    benchmarks = []

    def append_benchmark(prefix, ranks, opts=None):
        prefix = "%4d GPUs -- %s" % (len(ranks), prefix)
        benchmarks.append((prefix, ranks, opts))

    def local_print(msg):
        if dist.get_rank() == 0:
            print(msg, end='', flush=True)  # noqa: E999

    def print_header():
        local_print("\n")
        local_print("%22s" % "")
        for p in [50, 75, 90, 95]:
            local_print("%14s%10s" % ("sec/iter", "ex/sec"))
        local_print("\n")

    def print_measurements(prefix, nelem, measurements):
        measurements = sorted(measurements)
        local_print("%8s:" % prefix)
        for p in [50, 75, 90, 95]:
            v = np.percentile(measurements, p)
            local_print("  p%02d:  %1.3fs  %6d/s" % (p, v, nelem / v))
        local_print("\n")

    # Every process runs once by themselves to warm up (CUDA init, etc).
    append_benchmark("  warmup", [dist.get_rank()], {"use_ddp_for_single_rank": False})

    # Single machine baselines
    append_benchmark("  no ddp", range(1), {"use_ddp_for_single_rank": False})
    append_benchmark("   1M/1G", range(1))
    append_benchmark("   1M/2G", range(2))
    append_benchmark("   1M/4G", range(4))

    # Multi-machine benchmarks
    for i in range(1, (dist.get_world_size() // 8) + 1):
        append_benchmark("   %dM/8G" % i, range(i * 8))

    # Run benchmarks in order of increasing number of GPUs
    print_header()
    results = []
    for prefix, ranks, opts in sorted(benchmarks, key=lambda tup: len(tup[1])):
        # Turn range into materialized list.
        ranks = list(ranks)
        measurements = run_benchmark(benchmark, ranks, opts)
        if "warmup" not in prefix:
            print_measurements(prefix, benchmark.batch_size, measurements)
            results.append({"ranks": ranks, "measurements": measurements})

    return results


class Benchmark(object):
    def __init__(self, device, distributed_backend, bucket_size):
        self.device = device
        self.batch_size = 32
        self.distributed_backend = distributed_backend
        self.bucket_size = bucket_size

    def __str__(self):
        raise NotImplementedError

    def create_model(self):
        raise NotImplementedError

    def generate_inputs(self):
        raise NotImplementedError

    def generate_target(self):
        raise NotImplementedError


class TorchvisionBenchmark(Benchmark):
    def __init__(self, device, distributed_backend, bucket_size, model):
        super(TorchvisionBenchmark, self).__init__(
            device,
            distributed_backend,
            bucket_size,
        )
        self.model = model

    def __str__(self):
        return "{} with batch size {}".format(self.model, self.batch_size)

    def create_model(self):
        return torchvision.models.__dict__[self.model]().to(self.device)

    def generate_inputs(self):
        return [torch.rand([self.batch_size, 3, 224, 224], device=self.device)]

    def generate_target(self):
        return torch.tensor([1] * self.batch_size, dtype=torch.long, device=self.device)


def main():
    parser = argparse.ArgumentParser(description='PyTorch distributed benchmark suite')
    parser.add_argument("--rank", type=int, default=os.environ["RANK"])
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--distributed-backend", type=str, default="nccl")
    parser.add_argument("--bucket-size", type=int, default=25)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--master-port", type=str, required=True)
    parser.add_argument("--model", type=str)
    parser.add_argument("--json", type=str, metavar="PATH", help="Write file with benchmark results")
    args = parser.parse_args()

    num_gpus_per_node = torch.cuda.device_count()
    assert num_gpus_per_node == 8, "Expected 8 GPUs per machine"

    # The global process group used only for communicating benchmark
    # metadata, like measurements. Not for benchmarking itself.
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
        rank=args.rank,
        world_size=args.world_size,
    )

    output = allgather_run("nvidia-smi topo -m")
    if not allequal(output):
        print('Output of "nvidia-smi topo -m" differs between machines')
        sys.exit(1)

    if args.rank == 0:
        print("-----------------------------------")
        print("PyTorch distributed benchmark suite")
        print("-----------------------------------")
        print("")
        print("* PyTorch version: {}".format(torch.__version__))
        print("* CUDA version: {}".format(torch.version.cuda))
        print("* Distributed backend: {}".format(args.distributed_backend))
        print("* Maximum bucket size: {}MB".format(args.bucket_size))
        print("")
        print("--- nvidia-smi topo -m ---")
        print("")
        print(output[0])
        print("--------------------------")
        print("")

    torch.cuda.set_device(dist.get_rank() % 8)
    device = torch.device('cuda:%d' % (dist.get_rank() % 8))

    benchmarks = []
    if args.model:
        benchmarks.append(
            TorchvisionBenchmark(
                device=device,
                distributed_backend=args.distributed_backend,
                bucket_size=args.bucket_size,
                model=args.model))
    else:
        for model in ["resnet50", "resnet101", "resnext50_32x4d", "resnext101_32x8d"]:
            benchmarks.append(
                TorchvisionBenchmark(
                    device=device,
                    distributed_backend=args.distributed_backend,
                    bucket_size=args.bucket_size,
                    model=model))

    benchmark_results = []
    for benchmark in benchmarks:
        if args.rank == 0:
            print("\nBenchmark: {}".format(str(benchmark)))
        result = sweep(benchmark)
        benchmark_results.append({
            "model": benchmark.model,
            "batch_size": benchmark.batch_size,
            "result": result,
        })

    # Write file with benchmark results if applicable
    if args.rank == 0 and args.json:
        report = {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "distributed_backend": args.distributed_backend,
            "bucket_size": args.bucket_size,
            "benchmark_results": benchmark_results,
        }
        with open(args.json, 'w') as f:
            json.dump(report, f)


if __name__ == '__main__':
    main()
