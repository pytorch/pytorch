# Sparse benchmarks

# These benchmarks are for the sparse matrix functionality. 
# They exist for comparing the performance of sparse matrix routines
# torch.sparse.mm(sparse, sparse)` with different backends (CPU/CUDA)
# and with other frameworks such as scipy. 

import sys
from pathlib import Path
import pandas as pd
import argparse
import torch
import torch.utils.benchmark as benchmark_utils
from .utils import load_spmm_dataset, to_coo_scipy, scipy_coo_matmul
import pickle

def torch_backward(a_dense, b_dense):
    a_dense.requires_grad = True
    b_dense.requires_grad = True
    r1 = a_dense.matmul(b_dense)
    f1 = torch.sum(r1)
    f1.backward()


def sparse_torch_backward(a, b):
    a.requires_grad = True
    b.requires_grad = True

    r2 = torch.sparse.mm(a, b)
    if r2.is_sparse:
        f2 = torch.sparse.sum(r2)
    else:
        f2 = torch.sum(r2)
    f2.backward()


if __name__ == '__main__':

    path = Path()
    parser = argparse.ArgumentParser(description='spmm Bench')

    parser.add_argument('--path', type=str, help='dataset path')
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset name (random_pruning, variational_dropout, magnitude_pruning, \
                            extended_magnitude_pruning)',
                        default='random_pruning')
    parser.add_argument('--operation',
                        type=str,
                        help='matmul or backward',
                        default='matmul')
    parser.add_argument('--spmm_type',
                        type=str,
                        help='sparse-sparse or sparse-dense',
                        default='sparse-dense')
    parser.add_argument('--output',
                        type=str,
                        help='dataframe output path',
                        default='/tmp/matmul_bench.pkl')
    parser.add_argument('--device',
                        type=str,
                        help='device (cpu or cuda)',
                        default='cpu')

    args = parser.parse_args()
    print('path     =', args.path)
    print('dataset       =', args.dataset)
    print('spmm_type        =', args.spmm_type)
    print('operation     =', args.operation)
    print('output        =', args.output)
    print('device        =', args.device)

    dataset_path = args.path
    dataset_name = args.dataset
    dataset_path = f"{dataset_path}/{dataset_name}"
    df_output_path = args.output
    spmm_type = args.spmm_type
    device = args.device

    tasks = []
    if args.operation == 'matmul':
        if device == 'cpu':
            tasks = [
                ("matmul", device, "torch", "torch.mm(dx, dy)"),
                ("matmul", device, "torch.sparse", "torch.sparse.mm(x, y)"),
                ("matmul", device, "scipy", "scipy_coo_matmul(sx, sy)"),
            ]
        else:
            tasks = [
                ("matmul", device, "torch", "torch.mm(dx, dy)"),
                ("matmul", device, "torch.sparse", "torch.sparse.mm(x, y)"),
            ]
    else:
        tasks = [
            ("backward", device, "torch", "torch_backward(dx, dy)"),
            ("backward", device, "torch.sparse", "sparse_torch_backward(x, y)"),
        ]
    serialized_results = []
    repeats = 2
    timers = [
        benchmark_utils.Timer(
            stmt=stmt,
            globals={
                "scipy_coo_matmul": scipy_coo_matmul,
                "torch_backward": torch_backward,
                "sparse_torch_backward": sparse_torch_backward,
                "sx": to_coo_scipy(x) if x.is_sparse and device == 'cpu' else x.numpy(),
                "sy": to_coo_scipy(y) if y.is_sparse and device == 'cpu' else y.numpy(),
                "x": x,
                "y": y,
                "dx": x.to_dense() if x.is_sparse else x,
                "dy": y.to_dense() if y.is_sparse else y,
            },
            label=label,
            sub_label=sub_label,
            description=f"{sparsity}",
            env=device,
        ) for hidden_size in [512]
        for sparsity in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
        for label, device, sub_label, stmt in tasks
        for x, y in load_spmm_dataset(dataset_path, hidden_size, sparsity, spmm_type, device)
    ]
    measurements = []

    for i, timer in enumerate(timers * repeats):
        m = timer.blocked_autorange(min_run_time=0.05)
        serialized_results.append(pickle.dumps(m))
        m.metadata = {
            "device": 'cuda' if m.task_spec.env.find("cuda") >= 0 else 'cpu'
        }
        measurements.append(m)
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()
    print()

    comparison = benchmark_utils.Compare(
        [pickle.loads(i) for i in serialized_results])

    print("== Unformatted " + "=" * 80 + "\n" + "/" * 95 + "\n")
    comparison.print()

    print("== Formatted " + "=" * 80 + "\n" + "/" * 93 + "\n")
    comparison.trim_significant_figures()
    comparison.colorize()
    comparison.print()

    table = [(m.task_spec.sub_label, m.task_spec.description,
              m.metadata["device"], m.mean) for m in measurements]
    df = pd.DataFrame(table, columns=['method', 'sparsity', 'device', 'time'])
    df.to_pickle(df_output_path)
