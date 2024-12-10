

import torch

from . import _dtypes


def finfo(dtyp):
    torch_dtype = _dtypes.dtype(dtyp).torch_dtype
    return torch.finfo(torch_dtype)


def iinfo(dtyp):
    torch_dtype = _dtypes.dtype(dtyp).torch_dtype
    return torch.iinfo(torch_dtype)
