import os
import ctypes
import warnings
import torch.cuda
from torch.backends.cudnn import int_array

__all__ = ['all_reduce', 'reduce', 'broadcast', 'all_gather', 'reduce_scatter']

SUM = 0  # ncclRedOp_t


def is_available(tensors):
    devices = set()
    for tensor in tensors:
        if tensor.is_sparse:
            return False
        if not tensor.is_contiguous():
            return False
        if not tensor.is_cuda:
            return False
        device = tensor.get_device()
        if device in devices:
            return False
        devices.add(device)

    if not hasattr(torch._C, '_nccl_all_reduce'):
        warnings.warn('PyTorch is not compiled with NCCL support')
        return False

    return True


def all_reduce(inputs, outputs=None, op=SUM):
    if outputs is None:
        outputs = inputs
    torch._C._nccl_all_reduce(inputs, outputs, op)


def reduce(inputs, outputs=None, root=0, op=SUM, streams=None):
    assert(root >= 0 and root < len(inputs))
    if outputs is None:
        outputs = inputs
    if streams is None:
        streams = [None] * len(inputs)
    torch._C._nccl_reduce(inputs, outputs, streams, root, op)


def broadcast(inputs, root=0):
    assert(root >= 0 and root < len(inputs))
    torch._C._nccl_broadcast(inputs, root)


def all_gather(inputs, outputs):
    torch._C._nccl_all_gather(inputs, outputs)


def reduce_scatter(inputs, outputs, op=SUM):
    torch._C._nccl_reduce_scatter(inputs, outputs, op)
