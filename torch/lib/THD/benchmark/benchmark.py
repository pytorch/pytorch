import argparse
import os
from timeit import default_timer as timer
import torch
import torch.distributed as dist


def print_header(title):
    print(title)
    print("{:>8}\t{:>5}\t{:<{num_tensors_width}}\t{:>11}\t{:>11}".
          format("MB/s", "MB", "#", "s", "ms/op",
                 num_tensors_width=MAX_NUM_TENSORS))


def print_stats(bytes, num_tensors, time):
    print("{:>8.3f}\t{:>5.1f}\t{:<{num_tensors_width}}\t{:>11.3f}\t{:>11.3f}".
          format(bytes * num_tensors / (2**20 * time),
                 bytes / 2**20,
                 num_tensors,
                 time,
                 1000 * time / num_tensors,
                 num_tensors_width=MAX_NUM_TENSORS))


parser = argparse.ArgumentParser(description='Benchmark torch.distributed.')
parser.add_argument('--max-bytes', dest='max_bytes', action='store', default=28,
                    type=int,
                    help='set the inclusive upper limit for tensor size; ' +
                    'default: 22 (2**22 = 4 MB)')
parser.add_argument('--max-num-tensors', dest='max_num_tensors', action='store',
                    default=3, type=int,
                    help='set the inclusive upper limit for the number of ' +
                    'tensors to be sent during one test run; ' +
                    'default: 3 (10**3 = 1000)')
parser.add_argument('--min-bytes', dest='min_bytes', action='store', default=19,
                    type=int,
                    help='set the inclusive lower limit for tensor size; ' +
                    'default: 19 (2**19 = 512 KB)')
parser.add_argument('--min-num-tensors', dest='min_num_tensors', action='store',
                    default=2, type=int,
                    help='set the inclusive lower limit for the number of ' +
                    'tensors to be sent during one test run; ' +
                    'default: 2 (10**2 = 100)')

args = parser.parse_args()

MIN_NUM_TENSORS = args.min_num_tensors
MIN_BYTES = args.min_bytes
MAX_NUM_TENSORS = args.max_num_tensors + 1
MAX_BYTES = args.max_bytes + 1

dist.init_process_group(backend=os.environ['BACKEND'])

rank = dist.get_rank()
dist.barrier()

if rank == 0:
    print_header("broadcast")
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            start = timer()
            for i in range(0, num_tensors):
                dist.broadcast(tensor, 0)
            end = timer()
            print_stats(bytes, num_tensors, end - start)
    print()
else:
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes)
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            for i in range(0, num_tensors):
                dist.broadcast(tensor, 0)
dist.barrier()

if rank == 0:
    print_header("send from 0 to 1")
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            start = timer()
            for i in range(0, num_tensors):
                dist.send(tensor, 1)
            end = timer()
            print_stats(bytes, num_tensors, end - start)
    print()
elif rank == 1:
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            for i in range(0, num_tensors):
                dist.recv(tensor, 0)
dist.barrier()

if rank == 0:
    print_header("reduce")
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            start = timer()
            for i in range(0, num_tensors):
                dist.reduce(tensor, 0)
            end = timer()
            print_stats(bytes, num_tensors, end - start)
    print()
else:
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            for i in range(0, num_tensors):
                dist.reduce(tensor, 0)
dist.barrier()

if rank == 0:
    print_header("all reduce")
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            start = timer()
            for i in range(0, num_tensors):
                dist.all_reduce(tensor)
            end = timer()
            print_stats(bytes, num_tensors, end - start)
    print()
else:
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            for i in range(0, num_tensors):
                dist.all_reduce(tensor)
dist.barrier()

if rank == 0:
    print_header("scatter")
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        tensors = [tensor for n in range(0, dist.get_world_size())]
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            start = timer()
            for i in range(0, num_tensors):
                dist.scatter(tensor, scatter_list=tensors)
            end = timer()
            print_stats(bytes, num_tensors, end - start)
    print()
else:
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            for i in range(0, num_tensors):
                dist.scatter(tensor, src=0)
dist.barrier()

if rank == 0:
    print_header("gather")
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        tensors = [tensor for n in range(0, dist.get_world_size())]
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            start = timer()
            for i in range(0, num_tensors):
                dist.gather(tensor, gather_list=tensors)
            end = timer()
            print_stats(bytes, num_tensors, end - start)
    print()
else:
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            for i in range(0, num_tensors):
                dist.gather(tensor, dst=0)
dist.barrier()

if rank == 0:
    print_header("all gather")
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        tensors = [tensor for n in range(0, dist.get_world_size())]
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            start = timer()
            for i in range(0, num_tensors):
                dist.all_gather(tensors, tensor)
            end = timer()
            print_stats(bytes, num_tensors, end - start)
    print()
else:
    for bytes in [2**n for n in range(MIN_BYTES, MAX_BYTES)]:
        tensor = torch.ByteTensor(bytes).fill_(42)
        tensors = [tensor for n in range(0, dist.get_world_size())]
        for num_tensors in [10**n for n in range(MIN_NUM_TENSORS, MAX_NUM_TENSORS)]:
            for i in range(0, num_tensors):
                dist.all_gather(tensors, tensor)
dist.barrier()
