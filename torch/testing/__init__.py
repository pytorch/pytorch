"""
The testing package contains testing-specific utilities.
"""

import torch
import random

__all__ = [
    'make_non_contiguous', 'rand_like', 'randn_like'
]


def make_non_contiguous(tensor):
    if tensor.numel() <= 1:  # can't make non-contiguous
        return tensor.clone()
    osize = list(tensor.size())

    # randomly inflate a few dimensions in osize
    for _ in range(2):
        dim = random.randint(0, len(osize) - 1)
        add = random.randint(4, 15)
        osize[dim] = osize[dim] + add

    # narrow doesn't make a non-contiguous tensor if we only narrow the 0-th dimension,
    # (which will always happen with a 1-dimensional tensor), so let's make a new
    # right-most dimension and cut it off

    input = tensor.new(torch.Size(osize + [random.randint(2, 3)]))
    input = input.select(len(input.size()) - 1, random.randint(0, 1))
    # now extract the input of correct size from 'input'
    for i in range(len(osize)):
        if input.size(i) != tensor.size(i):
            bounds = random.randint(1, input.size(i) - tensor.size(i))
            input = input.narrow(i, bounds, tensor.size(i))

    input.copy_(tensor)
    return input


def get_all_dtypes():
    cpu_dtypes = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                  torch.float16, torch.float32, torch.float64]
    cuda_dtypes = [torch.cuda.uint8, torch.cuda.int8, torch.cuda.int16, torch.cuda.int32, torch.cuda.int64,
                   torch.cuda.float16, torch.cuda.float32, torch.cuda.float64]
    cpum = torch.sparse
    cpu_sparse_dtypes = [cpum.uint8, cpum.int8, cpum.int16, cpum.int32, cpum.int64,
                         cpum.float32, cpum.float64]
    cudam = torch.cuda.sparse
    cuda_sparse_dtypes = [cudam.uint8, cudam.int8, cudam.int16, cudam.int32, cudam.int64,
                          cudam.float32, cudam.float64]
    return cpu_dtypes + cuda_dtypes + cpu_sparse_dtypes + cuda_sparse_dtypes

rand_like = torch._C._VariableFunctions.rand_like
randn_like = torch._C._VariableFunctions.randn_like
