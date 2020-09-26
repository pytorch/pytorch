import argparse
import sys
import torch
from .utils import gen_sparse_gcs, gen_sparse_coo, Event

def test_sparse_gcs(m, nnz, test_count):
    start_timer = Event(enable_timing=True)
    stop_timer = Event(enable_timing=True)

    gcs = gen_sparse_gcs((m, m), nnz)
    vector = torch.randn(m, dtype=torch.double)

    times = []
    for _ in range(test_count):
        start_timer.record()
        gcs.matmul(vector)
        stop_timer.record()
        times.append(start_timer.elapsed_time(stop_timer))

    return sum(times) / len(times)

def test_sparse_coo(m, nnz, test_count):
    start_timer = Event(enable_timing=True)
    stop_timer = Event(enable_timing=True)

    coo = gen_sparse_coo((m, m), nnz)
    vector = torch.randn(m, dtype=torch.double)

    times = []
    for _ in range(test_count):
        start_timer.record()
        coo.matmul(vector)
        stop_timer.record()
        times.append(start_timer.elapsed_time(stop_timer))

    return sum(times) / len(times)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpMV")

    parser.add_argument("--format", default='gcs', type=str)
    parser.add_argument("--m", default='1000', type=int)
    parser.add_argument("--nnz_ratio", default='0.1', type=float)
    parser.add_argument("--outfile", default='stdout', type=str)
    parser.add_argument("--test_count", default='10', type=int)

    args = parser.parse_args()

    if args.outfile == 'stdout':
        outfile = sys.stdout
    elif args.outfile == 'stderr':
        outfile = sys.stderr
    else:
        outfile = open(args.outfile, "a")

    test_count = args.test_count
    m = args.m
    nnz_ratio = args.nnz_ratio

    nnz = int(nnz_ratio * m * m)
    if args.format == 'gcs':
        time = test_sparse_gcs(m, nnz, test_count)
    elif args.format == 'coo':
        time = test_sparse_coo(m, nnz, test_count)

    print("format=", args.format, " nnz_ratio=", nnz_ratio, " m=", m,
          " time=", time, file=outfile)
