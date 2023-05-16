import torch
from pathlib import Path
from scipy import sparse
import math


def to_coo_scipy(x):
    indices_1 = x._indices().numpy()
    values_1 = x._values().numpy()
    return sparse.coo_matrix((values_1, (indices_1[0], indices_1[1])),
                             shape=x.shape)


def sparse_grad_output(a, b):
    c = torch.sparse.mm(a, b)
    if c.is_sparse:
        c2 = torch.rand_like(c.to_dense())
        return c2.sparse_mask(c.coalesce())
    else:
        return torch.rand_like(c)


def read_matrix_params(path):
    with open(path) as file:
        line = file.readline()
        nrows, ncols, nnz = (int(el) for el in line.split(', '))
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
    with open(path) as file:
        nrows, ncols, nnz = (int(el) for el in file.readline().split(', '))
        index_pointers = (int(el) for el in file.readline().split())
        indices = (int(el) for el in file.readline().split())

    index_pointers = list(index_pointers)
    indices = list(indices)
    data = torch.randn(nnz, dtype=torch.double)
    shape = (nrows, ncols)
    return torch.sparse_coo_tensor(csr_to_coo(indices, index_pointers, shape), data, shape, device=device)


def gen_vector(path, device):
    with open(path) as file:
        nrows, ncols, nnz = (int(el) for el in file.readline().split(', '))
        index_pointers = (int(el) for el in file.readline().split())
        indices = (int(el) for el in file.readline().split())
        return torch.randn(nrows, dtype=torch.double, device=device)


def gen_matrix(path, device):
    with open(path) as file:
        nrows, ncols, nnz = (int(el) for el in file.readline().split(', '))
        index_pointers = (int(el) for el in file.readline().split())
        indices = (int(el) for el in file.readline().split())
        return torch.randn(nrows, ncols, dtype=torch.double, device=device)


def load_spmv_dataset(dataset_path, hidden_size, sparsity, device, n_limit=math.inf):
    """load_spmv_dataset loads a DLMC dataset for a sparse matrix-vector multiplication (SPMV) performance test.
    Args:
        dataset_path:
            path of the dataset from DLMC collection.
        hidden_size
            This value allows tensors of varying sizes.
        sparsity:
            This value allows tensors of varying sparsities.
        device:
            Whether to place the Tensor on a GPU or CPU.
        n_limit:
            This value allows a dataset with some limit size.
    """
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
    """load_spmm_dataset loads a DLMC dataset for a sparse matrix-matrix multiplication (SPMM) performance test.
    Args:
        dataset_path:
            path of the dataset from DLMC collection.
        hidden_size
            This value allows tensors of varying sizes.
        sparsity:
            This value allows tensors of varying sparsities.
        spmm_type:
            This value allows tensors for `sparse@sparse` or `sparse@dense` operations.
        device:
            Whether to place the Tensor on a GPU or CPU.
        n_limit:
            This value allows a dataset with some limit size.
    """
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
        y = gen_matrix(fy, device) if spmm_type == 'sparse@dense' else load_sparse_matrix(fy, device)
        yield (x, y)


def load_dlmc_dataset(dataset_path, operation, hidden_size, sparsity, device, requires_grad, n_limit=math.inf):
    """load_dlmc_dataset loads a DLMC dataset for a matmul performance test.
    Args:
        dataset_path:
            path of the dataset from DLMC collection.
        operation:
            This value allows tensors for `sparse@sparse`|`sparse@dense`|`sparse@vector` operations.
        hidden_size
            This value allows tensors of varying sizes.
        sparsity:
            This value allows tensors of varying sparsities.
        device:
            Whether to place the Tensor on a GPU or CPU.
        requires_grad:
            Loads the dataset for backward test.
        n_limit:
            This value allows a dataset with some limit size.
    """
    if operation == 'sparse@sparse' or operation == "sparse@dense":
        collection = load_spmm_dataset(dataset_path, hidden_size, sparsity, operation, device, n_limit)
    elif operation == 'sparse@vector':
        collection = load_spmv_dataset(dataset_path, hidden_size, sparsity, device, n_limit)
    scipy_vars = {}
    backward_vars = {}
    for x, y in collection:
        if device == 'cpu':
            scipy_vars = {
                "sx": to_coo_scipy(x) if x.is_sparse else x.numpy(),
                "sy": to_coo_scipy(y) if y.is_sparse else y.numpy(),
            }
        if not requires_grad:
            dx = x.to_dense() if x.is_sparse else x
            dy = y.to_dense() if y.is_sparse else y
        else:
            c = sparse_grad_output(x, y)
            backward_vars = {
                "sparse_grad_output": c,
                "grad_output": c.to_dense() if c.is_sparse else c,
            }
            x.requires_grad_(True)
            y.requires_grad_(True)
            dx = x.to_dense().detach() if x.is_sparse else x.clone().detach()
            dy = y.to_dense().detach() if y.is_sparse else y.clone().detach()
            dx.requires_grad_(True)
            dy.requires_grad_(True)
        yield {
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            **scipy_vars,
            **backward_vars
        }
