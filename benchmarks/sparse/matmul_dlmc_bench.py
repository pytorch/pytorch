# Sparse benchmarks

# These benchmarks are for the sparse matrix functionality. 
# They exist for comparing the performance of sparse matrix routines
# torch.sparse.mm(sparse, sparse)` with different backends (CPU/CUDA)
# and with other frameworks such as scipy. 

import sys
from scipy import sparse
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
import torch
import torch.utils.benchmark as benchmark_utils

def read_matrix_params(path):
    sys.stdin = open(path)
    nrows, ncols, nnz = map(lambda el: int(el), input().split(', '))
    return (nrows, ncols), nnz


def load_matrix(path):
    sys.stdin = open(path)
    nrows, ncols, nnz = map(lambda el: int(el), input().split(', '))
    index_pointers = map(lambda el: int(el), input().split())
    indices = map(lambda el: int(el), input().split())

    index_pointers = list(index_pointers)
    indices = list(indices)
    data = np.random.rand(nnz)
    coo = sparse.csr_matrix(
        (data, np.array(indices), np.array(index_pointers)),
        shape=(nrows, ncols)).tocoo()
    return torch.sparse_coo_tensor([coo.row, coo.col], coo.data, coo.shape)


def scipy_coo_matmul(mat1, mat2):
    result = mat1.dot(mat2).tocoo()
    return torch.sparse_coo_tensor([result.row, result.col], result.data,
                                   result.shape)


def to_coo_scipy(x):
    indices_1 = x._indices().numpy()
    values_1 = x._values().numpy()
    return sparse.coo_matrix((values_1, (indices_1[0], indices_1[1])),
                             shape=x.shape)


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
    f2 = torch.sparse.sum(r2)
    f2.backward()


def load_dataset(dataset_path, hidden_size, sparsity, n_limit=20):
    current_folder_path = f"{dataset_path}/{sparsity}"
    path = Path(current_folder_path)
    files = path.glob('**/*.smtx')
    xs = []
    ys = []
    print(dataset_path, hidden_size, sparsity)
    index = 0
    for elem in files:
        if index == n_limit:
            break
        print('.', end='')
        size, nnz = read_matrix_params(elem.as_posix())
        if size[1] == hidden_size:
            xs.append(load_matrix(elem.as_posix()))
        if size[0] == hidden_size:
            ys.append(load_matrix(elem.as_posix()))
        index += 1
    print()
    return zip(xs, ys)


if __name__ == '__main__':

    path = Path()
    parser = argparse.ArgumentParser(description='Sparse Matmul Bench')

    parser.add_argument('--path', type=str, help='dataset path')
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset name',
                        default='random_pruning')
    parser.add_argument('--operation',
                        type=str,
                        help='matmul or backward',
                        default='matmul')
    parser.add_argument('--output',
                        type=str,
                        help='dataframe output path',
                        default='/tmp/matmul_bench.pkl')
    args = parser.parse_args()
    print('path     =', args.path)
    print('dataset       =', args.dataset)
    print('operation     =', args.operation)
    print('output        =', args.output)

    dataset_path = args.path
    dataset_name = args.dataset
    dataset_path = f"{dataset_path}/{dataset_name}"
    df_output_path = args.output
    tasks = []
    if args.operation == 'matmul':
        tasks = [
            ("matmul", "cpu", "torch", "torch.mm(dense_x, dense_y)"),
            ("matmul", "cpu", "torch.sparse", "torch.sparse.mm(tx, ty)"),
            ("matmul", "cpu", "scipy",
             "scipy_coo_matmul(scipy_varx, scipy_vary)"),
            ("matmul", "cuda", "torch",
             "torch.mm(dense_cuda_x, dense_cuda_y)"),
            ("matmul", "cuda", "torch.sparse",
             "torch.sparse.mm(tx_cuda, ty_cuda)"),
        ]
    else:
        tasks = [
            ("backward", "cpu", "torch", "torch_backward(dense_x, dense_y)"),
            ("backward", "cpu", "torch.sparse",
             "sparse_torch_backward(tx, ty)"),
            ("backward", "cuda", "torch",
             "torch_backward(dense_cuda_x, dense_cuda_y)"),
            ("backward", "cuda", "torch.sparse",
             "sparse_torch_backward(tx_cuda, ty_cuda)"),
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
                "scipy_varx": to_coo_scipy(x),
                "scipy_vary": to_coo_scipy(y),
                "tx": x,
                "ty": y,
                "tx_cuda": x.cuda(),
                "ty_cuda": y.cuda(),
                "dense_cuda_x": x.to_dense().cuda(),
                "dense_cuda_y": y.to_dense().cuda(),
                "dense_x": x.to_dense(),
                "dense_y": y.to_dense(),
            },
            label=label,
            sub_label=sub_label,
            description=f"{sparsity}",
            env=device,
            # num_threads=num_threads,
        ) for hidden_size in [512]
        for sparsity in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
        for label, device, sub_label, stmt in tasks
        for num_threads in [1, 4, 8, 16]
        for x, y in load_dataset(dataset_path, hidden_size, sparsity)
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
