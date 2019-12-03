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
import gc
import io
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
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    input_tensor = torch.ByteTensor(list(buffer.getvalue()))
    input_length = torch.IntTensor([input_tensor.size(0)])
    dist.all_reduce(input_length, op=dist.ReduceOp.MAX)
    input_tensor.resize_(input_length[0])
    output_tensors = [
        torch.empty(input_tensor.size(), dtype=torch.uint8)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(output_tensors, input_tensor)
    output = []
    for tensor in output_tensors:
        buffer = io.BytesIO(np.asarray(tensor).tobytes())
        output.append(torch.load(buffer))
    return output


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


def benchmark_process_group(pg, benchmark):
    torch.manual_seed(pg.rank())
    torch.cuda.manual_seed(pg.rank())

    model = benchmark.create_model()
    data = [(benchmark.generate_inputs(), benchmark.generate_target())]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          0.001,
                          momentum=0.9,
                          weight_decay=1e-4)
    if pg.size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            process_group=pg)

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


def run_benchmark(prefix, ranks, benchmark):
    def local_print(msg):
        if dist.get_rank() == 0:
            print(msg, end='', flush=True)

    def print_measurements(prefix, nelem, measurements):
        measurements = sorted(measurements)
        local_print("%s: " % prefix)
        for p in [50, 75, 90, 95]:
            v = np.percentile(measurements, p)
            local_print("p%02d:  %1.3f  %4d/s  " % (p, v, nelem / v))
        local_print("\n")

    group_nccl = dist.new_group(ranks=ranks, backend="nccl")
    group_gloo = dist.new_group(ranks=ranks, backend="gloo")
    group_to_benchmark = group_nccl

    measurements = []
    if dist.get_rank() in set(ranks):
        measurements = benchmark_process_group(group_to_benchmark, benchmark)

    dist.destroy_process_group(group_nccl)
    dist.destroy_process_group(group_gloo)
    dist.barrier()

    # Force destruction of everything that is no longer used.
    # Most notably, this is needed to destroy the NCCL process group
    # if the PyTorch 1.0 DistributedDataParallel class with Python
    # side reduction is used. If we don't explicitly run GC,
    # creation of NCCL process groups later on results in a crash.
    gc.collect()

    # Aggregate measurements for better estimation of percentiles
    measurements = list(itertools.chain(*allgather_object(measurements)))
    print_measurements(prefix, benchmark.batch_size, measurements)
    return measurements


def sweep(benchmark):
    # Synthesize the set of benchmarks to run.
    # This list contain tuples for ("string prefix", [rank...]).
    benchmarks = []

    def append_benchmark(prefix, ranks):
        prefix = "%4d GPUs -- %s" % (len(ranks), prefix)
        benchmarks.append((prefix, ranks))

    # Every process runs once by themselves to warm up (CUDA init, etc).
    append_benchmark("  warmup", [dist.get_rank()])

    # Single machine baselines
    append_benchmark("   1M/1G", range(1))
    append_benchmark("   1M/2G", range(2))
    append_benchmark("   1M/4G", range(4))

    # Multi-machine benchmarks
    for i in range(1, (dist.get_world_size() // 8) + 1):
        append_benchmark("   %dM/8G" % i, range(i * 8))

    # Run benchmarks in order of increasing number of GPUs
    results = []
    for prefix, ranks in sorted(benchmarks, key=lambda tup: len(tup[1])):
        # Turn range into materialized list.
        ranks = list(ranks)
        measurements = run_benchmark(prefix, ranks, benchmark)
        results.append({"ranks": ranks, "measurements": measurements})
    return results


class Benchmark(object):
    def __init__(self, device):
        self.device = device
        self.batch_size = 32

    def __str__(self):
        raise NotImplementedError

    def create_model(self):
        raise NotImplementedError

    def generate_inputs(self):
        raise NotImplementedError

    def generate_target(self):
        raise NotImplementedError


class TorchvisionBenchmark(Benchmark):
    def __init__(self, device, model):
        super(TorchvisionBenchmark, self).__init__(device)
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
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--master-port", type=str, required=True)
    parser.add_argument("--model", type=str)
    parser.add_argument("--json", type=str, metavar="PATH", help="Write file with benchmark results")
    args = parser.parse_args()

    num_gpus_per_node = torch.cuda.device_count()
    assert num_gpus_per_node == 8, "Expected 8 GPUs per machine"

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
        benchmarks.append(TorchvisionBenchmark(device=device, model=args.model))
    else:
        for model in ["resnet50", "resnet101", "resnext50_32x4d", "resnext101_32x8d"]:
            benchmarks.append(TorchvisionBenchmark(device=device, model=model))

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
            "benchmark_results": benchmark_results,
        }
        with open(args.json, 'w') as f:
            json.dump(report, f)


if __name__ == '__main__':
    main()
