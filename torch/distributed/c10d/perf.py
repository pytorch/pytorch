r"""
`torch.distributed.c10d.perf` implements a performance measurement
suite that can be used to evaluate the impact of changes to the c10d
backends, or analyse performance of different systems.

This can be used either by invoking it directly, or as a module where
you provide the process group instance and call the run function. When
used directly, it requires precense of a shared filesystem to perform
rendezvous.
"""

import argparse
import numpy as np
from statistics import median
import sys
import time

import torch
import torch.distributed.c10d as c10d


def _run_iterations(pg, tensors, N):
    start = [None] * len(tensors)
    work = [None] * len(tensors)
    samples = []
    for i in range(N):
        j = i % len(tensors)
        if work[j]:
            work[j].wait()
            samples.append(time.time() - start[j])

        start[j] = time.time()
        opts = c10d.AllreduceOptions()
        opts.reduceOp = c10d.ReduceOp.SUM
        work[j] = pg.allreduce(tensors[j], opts)

    for j in range(len(work)):
        if work[j]:
            work[j].wait()
            samples.append(time.time() - start[j])

    return samples


def _broadcast(pg, value, device):
    t = torch.tensor([int(value)], device=device)
    pg.broadcast(t, root=0).wait()
    return int(t.item())


def _run_for_tensors(pg, tensors):
    samples = _run_iterations(pg, tensors, len(tensors) * 2)
    samples.sort()

    # Determine number of iterations to run for
    T = 2.0
    N = _broadcast(pg, int(T / median(samples)), tensors[0][0].device)

    # Real run
    samples = _run_iterations(pg, tensors, len(tensors) * N)
    samples.sort()
    return samples


def run(
        pg,
        sizes=None,
        parallelism=None,
        devices=None,
):
    # Defaults can't be specified in the argument list because they are mutable
    if not sizes:
        sizes = [1000, 10000, 100000, 1000000]
    if not parallelism:
        parallelism = [1, 2, 4, 8]

    header_format = " ".join([
        "{:>12}",
        "{:>12}",
        "{:>15}",
        "{:>15}",
        "{:>15}",
        "{:>15}",
    ])

    line_format = " ".join([
        "{:>12}",
        "{:>12}",
        "{:>15.6f}",
        "{:>15.6f}",
        "{:>15.6f}",
        "{:>15}",
    ])

    header_entries = [
        'concurrency',
        'elements',
        'p50',
        'p90',
        'p99',
        'p50 xput',
    ]

    if not devices:
        devices = [torch.device('cpu')]

    print(header_format.format(*header_entries))
    for s in sizes:
        for p in parallelism:
            tensors = [[torch.ones([s], device=device) for device in devices]
                       for _ in range(p)]
            samples = _run_for_tensors(pg, tensors)
            nbytes = p * tensors[0][0].numel() * tensors[0][0].element_size()
            nbytes /= (1024 * 1024)
            xput = "{:.3f} MB/s".format(nbytes / np.percentile(samples, 50))
            print(line_format.format(
                p,
                s,
                np.percentile(samples, 50),
                np.percentile(samples, 90),
                np.percentile(samples, 99),
                xput,
            ))
            sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="c10d perf suite")
    parser.add_argument("--backend", type=str, default='gloo',
                        help="The c10d backend to use")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of process group threads (for gloo backend)")
    parser.add_argument("--path", type=str, required=True,
                        help="Path for file:// rendezvous")
    parser.add_argument("--rank", type=int, required=True,
                        help="Rank of this process")
    parser.add_argument("--size", type=int, required=True,
                        help="Number of processes")
    parser.add_argument("--device", type=str, default='cpu',
                        help="Tensor device")
    parser.add_argument("--inputs", type=int, default=1,
                        help="Number of input tensors per call")

    args = parser.parse_args()

    if args.backend == 'gloo':
        def create_process_group(store, rank, size):
            opts = c10d.ProcessGroupGloo.Options()
            opts.devices = [
                c10d.ProcessGroupGloo.create_tcp_device(),
            ]
            opts.threads = args.threads
            opts.cacheNumAlgorithmEntries = args.threads
            return c10d.ProcessGroupGloo(store, rank, size, opts)
    elif args.backend == 'nccl':
        def create_process_group(store, rank, size):
            return c10d.ProcessGroupNCCL(store, rank, size)
    else:
        raise ValueError("Invalid backend: {}".format(args.backend))

    if args.device == 'cpu':
        devices = [torch.device('cpu')] * args.inputs
    elif args.device == 'cuda':
        # Construct list containing every CUDA device such that tensors
        # are distributed across devices for tests with more than 1 input.
        devices = [torch.device('cuda:{}'.format(i % torch.cuda.device_count()))
                   for i in range(args.inputs)]
    else:
        raise ValueError("Invalid device: {}".format(args.device))

    url = "file://%s?rank=%d&size=%d" % (args.path, args.rank, args.size)
    store, rank, size = next(c10d.rendezvous(url))
    run(create_process_group(store, rank, size), devices=devices)
