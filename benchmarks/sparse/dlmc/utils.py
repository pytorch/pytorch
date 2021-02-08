import torch
from pathlib import Path
import sys
from scipy import sparse
import math 
from scipy.sparse import isspmatrix

def to_coo_scipy(x):
    indices_1 = x._indices().numpy()
    values_1 = x._values().numpy()
    return sparse.coo_matrix((values_1, (indices_1[0], indices_1[1])),
                             shape=x.shape)

def scipy_coo_matmul(mat1, mat2):
    if isspmatrix(mat1) and isspmatrix(mat2):
        result = mat1.dot(mat2).tocoo()
        return torch.sparse_coo_tensor([result.row, result.col], result.data,
                                       result.shape)
    else:
        return mat1.dot(mat2)


def read_matrix_params(path):
    sys.stdin = open(path)
    nrows, ncols, nnz = map(lambda el: int(el), input().split(', '))
    return (nrows, ncols), nnz

def csr_to_coo(indices, indptr, shape):
    n_rows, n_cols = shape
    cols = indices
    rows = [0] * len(cols)
    for i in range(n_rows):
        for j in range(indptr[i], indptr[i + 1]):
            rows[j] = i
    return torch.tensor([rows, cols], dtype=torch.long)

def load_sparse_matrix(path, device):
    sys.stdin = open(path)
    nrows, ncols, nnz = map(lambda el: int(el), input().split(', '))
    index_pointers = map(lambda el: int(el), input().split())
    indices = map(lambda el: int(el), input().split())

    index_pointers = list(index_pointers)
    indices = list(indices)
    data = torch.randn(nnz, dtype=torch.double)
    shape = (nrows, ncols)
    return torch.sparse_coo_tensor(csr_to_coo(indices, index_pointers, shape), data, shape).to(device=device)


def gen_vector(path, device):
    sys.stdin = open(path)
    nrows, ncols, nnz = map(lambda el: int(el), input().split(', '))
    index_pointers = map(lambda el: int(el), input().split())
    indices = map(lambda el: int(el), input().split())
    return torch.randn(nrows, dtype=torch.double, device=device)


def gen_matrix(path, device):
    sys.stdin = open(path)
    nrows, ncols, nnz = map(lambda el: int(el), input().split(', '))
    index_pointers = map(lambda el: int(el), input().split())
    indices = map(lambda el: int(el), input().split())
    return torch.randn(nrows, ncols, dtype=torch.double, device=device)


def load_spmv_dataset(dataset_path, hidden_size, sparsity, device, n_limit=math.inf):
    current_folder_path = f"{dataset_path}/{sparsity}"
    path = Path(current_folder_path)
    files = path.glob('**/*.smtx')
    print(dataset_path, hidden_size, sparsity)
    index = 0
    x_files, y_files = [], []
    for f in files:
        if index >= n_limit:
            break
        print('.', end='')
        size, nnz = read_matrix_params(f.as_posix())
        if size[1] == hidden_size:
            x_files.append(f.as_posix())
        if size[0] == hidden_size:
            y_files.append(f.as_posix())
        index += 1
    print()

    for fx, fy in zip(x_files, y_files):
        x = load_sparse_matrix(fx, device)
        y = gen_vector(fy, device)
        yield (x, y)


def load_spmm_dataset(dataset_path, hidden_size, sparsity, spmm_type, device, n_limit=math.inf):
    current_folder_path = f"{dataset_path}/{sparsity}"
    path = Path(current_folder_path)
    files = path.glob('**/*.smtx')
    print(dataset_path, hidden_size, sparsity)
    index = 0
    x_files, y_files = [], []
    for f in files:
        if index >= n_limit:
            break
        print('.', end='')
        size, nnz = read_matrix_params(f.as_posix())
        if size[1] == hidden_size:
            x_files.append(f.as_posix())
        if size[0] == hidden_size:
            y_files.append(f.as_posix())
        index += 1
    print()

    for fx, fy in zip(x_files, y_files):
        x = load_sparse_matrix(fx, device)
        y = gen_matrix(fy, device) if spmm_type == 'sparse-dense' else load_sparse_matrix(fy, device)
        yield (x, y)
