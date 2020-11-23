import sys
from scipy import sparse
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
import pickle
import sys
import time
import torch
import torch.utils.benchmark as benchmark_utils
import pandas as pd

def read_matrix_params(path):
    sys.stdin = open(path)
    nrows, ncols, nnz = map(lambda el: int(el), input().split(', '))
    return (nrows, ncols), nnz

def gen_matrix(path):
    sys.stdin = open(path)
    nrows, ncols, nnz = map(lambda el: int(el), input().split(', '))
    index_pointers = map(lambda el: int(el), input().split())
    indices = map(lambda el: int(el), input().split())

    index_pointers = list(index_pointers)
    indices = list(indices)
    data = np.random.rand(nnz)
    coo = sparse.csr_matrix( (data, np.array(indices), np.array(index_pointers)), shape=(nrows, ncols)).tocoo()
    st = torch.sparse_coo_tensor([coo.row, coo.col], coo.data, coo.shape)
    return coo, st

def scipy_coo_matmul(mat1, mat2):
    result = mat1.dot(mat2).tocoo()
    return torch.sparse_coo_tensor([result.row, result.col], result.data, result.shape)

def get_sparse_tensors(dataset_path, hidden_size, sparsity):
    current_folder_path = f"{dataset_path}/{sparsity}"
    path = Path(current_folder_path)
    files = path.glob('**/*.smtx')
    xs = []
    ys = []
    print(dataset_path, hidden_size, sparsity)
    for elem in files:
        print('.', end='')
        size, nnz = read_matrix_params(elem.as_posix())
        if size[1] == hidden_size:
            xs.append(gen_matrix(elem.as_posix())) 
        if size[0] == hidden_size:
            ys.append(gen_matrix(elem.as_posix())) 
    print()
    return zip(xs, ys)

path = Path()
parser = argparse.ArgumentParser(description='Sparse Matmul Bench')

parser.add_argument('--path', '-p',action='store', dest='rn50_path',
                    help='rn50 dataset path', default=path.cwd()/'rn50/')
parser.add_argument('--dataset', '-d',action='store', dest='dataset',
                    help='rn50 dataset path', default='random_pruning')
parser.add_argument('--output', '-o',action='store', dest='output',
                    help='dataframe output path', default='/tmp/matmul_bench.pkl')
results = parser.parse_args()
print ('rn50_path     =', results.rn50_path)
print ('dataset       =', results.dataset)
print ('output        =', results.output)

rn50_path = results.rn50_path
dataset_name = results.dataset
dataset_path = f"{rn50_path}/{dataset_name}"
df_output_path = results.output

tasks = [
    ("matmul", "cpu",  "torch", "torch.mm(dense_x, dense_y)"),
    ("matmul", "cuda", "torch",  "torch.mm(dense_cuda_x, dense_cuda_y)"),
    ("matmul", "cpu",  "torch.sparse", "torch.sparse.mm(tx, ty)"),
    ("matmul", "cuda", "torch.sparse", "torch.sparse.mm(tx_cuda, ty_cuda)"),
    ("matmul", "cpu",  "scipy", "scipy_coo_matmul(scipy_varx, scipy_vary)"),
]
serialized_results = []
repeats = 2
timers = [
    benchmark_utils.Timer(
        stmt=stmt,
        globals={
            "scipy_coo_matmul": scipy_coo_matmul,
            "scipy_varx": x[0],
            "scipy_vary": y[0],
            "tx": x[1],
            "ty": y[1],
            "tx_cuda": x[1].cuda(),
            "ty_cuda": y[1].cuda(),
            "dense_cuda_x": x[1].to_dense().cuda(),
            "dense_cuda_y": y[1].to_dense().cuda(),
            "dense_x": x[1].to_dense(),
            "dense_y": y[1].to_dense(),
        },
        label=label,
        sub_label=sub_label,
        description=f"{sparsity}",
        env=device,
        num_threads=num_threads,
    )
    for hidden_size in [512]
    for sparsity in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
    for label, device, sub_label, stmt in tasks
    for num_threads in [1]
    for x, y in get_sparse_tensors(dataset_path, hidden_size, sparsity)
]
measurements = []

for i, timer in enumerate(timers * repeats):
    m = timer.blocked_autorange(min_run_time=0.05)
    serialized_results.append(pickle.dumps(m))
    m.metadata = {"device" : 'cuda' if m.task_spec.env.find("cuda") >= 0 else 'cpu'}
    measurements.append(m)
    print(f"\r{i + 1} / {len(timers) * repeats}", end="")
    sys.stdout.flush()
print()

comparison = benchmark_utils.Compare([
    pickle.loads(i) for i in serialized_results
])

print("== Unformatted " + "=" * 80 + "\n" + "/" * 95 + "\n")
comparison.print()

print("== Formatted " + "=" * 80 + "\n" + "/" * 93 + "\n")
comparison.trim_significant_figures()
comparison.colorize()
comparison.print()

table = [ (m.task_spec.sub_label, m.task_spec.description, m.metadata["device"], m.mean) for m in measurements]
df = pd.DataFrame(table,  columns =['method', 'sparsity', 'device', 'time']) 
df.to_pickle(df_output_path)
