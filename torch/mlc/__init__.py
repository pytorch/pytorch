# The Tensor classes are added to this module by python_tensor.cpp

r"""
This package adds support for MLC tensor types, that implement the same
function as CPU tensors, but they utilize MLC resources for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports MLC.

"""

import contextlib
import platform
import ctypes
import os
import sys
import torch
import traceback
import warnings
import threading
# from torch._six import raise_from
from subprocess import Popen, PIPE
import torch._C

from ._utils import ReversibleDict
from ..storage import _StorageBase

def is_available():
    r"""Returns a bool indicating if MLC is currently available."""
    return torch._C.has_mlc


################################################################################
# Define device
################################################################################

_mlc_device_name = ReversibleDict({
    'cpu': 0,
    'gpu': 1,
    'any': 2
})

def get_device():
    r"""
    Returns the current MLC device which we are running the model.
    """
    device_val = torch._C._get_mlc_device()
    return _mlc_device_name.reversed()[device_val]


def set_device(device):
    r"""Sets the current device.

    Select MLC device to run model.

    Arguments:
        device: Choice of 'cpu', 'gpu', 'any'
    """
    device_lower = device.lower()
    if device_lower not in _mlc_device_name:
        raise ValueError("Device type {} not supported by MLC (choices: 'cpu', 'gpu', 'any')".format(device))
    device_map = _mlc_device_name[device_lower]
    torch._C._set_mlc_device(device_map)

def get_cache_parameters():
    return {"node_limit": torch._C._get_mlc_cache_node_limit(),
            "memory_limit_mb": torch._C._get_mlc_cache_memory_limit_mb()}

def set_cache_node_limit(i):
    assert(i > 0)
    torch._C._set_mlc_cache_node_limit(i)

def set_cache_memory_limit(mb):
    assert(mb > 0)
    torch._C._set_mlc_cache_memory_limit_mb(mb)
################################################################################
# Define Storage and Tensor classes
################################################################################

def _dummy_type(name):
    def init_err(self):
        class_name = self.__class__.__name__
        raise RuntimeError(
            "Tried to instantiate dummy base class {}".format(class_name))
    return type(storage_name, (object,), {"__init__": init_err})


if not hasattr(torch._C, 'MLCDoubleStorageBase'):
    # Define dummy base classes
    for t in ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte', 'Half', 'Bool', 'BFloat16']:
        storage_name = 'MLC{0}StorageBase'.format(t)
        tensor_name = 'MLC{0}TensorBase'.format(t)

        torch._C.__dict__[storage_name] = _dummy_type(storage_name)
        torch._C.__dict__[tensor_name] = _dummy_type(tensor_name)


class _MLCBase(object):
    is_mlc = True

    def type(self, *args, **kwargs):
        return super(_MLCBase, self).type(*args, **kwargs)


# Currently, MLC only supports Float32 for CPU (BNNS), and Float16 for GPU (MPS)
# However, the storage outlined here does not necessarily map to BNNS/MPS tensor
# types. This is simply an abstraction of ATen CPU Tensor which allows us to place
# the OPS in MLC device when we perform the dynamic dispatch of the PyTorch ops.
# As discussed earlier, we should allow the casting of the output of user's MLCompute graph
# to the ATen input to the graph.

class DoubleStorage(_MLCBase, torch._C.MLCDoubleStorageBase, _StorageBase):
    pass


class FloatStorage(_MLCBase, torch._C.MLCFloatStorageBase, _StorageBase):
    pass


class LongStorage(_MLCBase, torch._C.MLCLongStorageBase, _StorageBase):
    pass


class IntStorage(_MLCBase, torch._C.MLCIntStorageBase, _StorageBase):
    pass


class ShortStorage(_MLCBase, torch._C.MLCShortStorageBase, _StorageBase):
    pass


class CharStorage(_MLCBase, torch._C.MLCCharStorageBase, _StorageBase):
    pass


class ByteStorage(_MLCBase, torch._C.MLCByteStorageBase, _StorageBase):
    pass


class HalfStorage(_MLCBase, torch._C.MLCHalfStorageBase, _StorageBase):
    pass


class BoolStorage(_MLCBase, torch._C.MLCBoolStorageBase, _StorageBase):
    pass


class BFloat16Storage(_MLCBase, torch._C.MLCBFloat16StorageBase, _StorageBase):
    pass

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)

# Note we cannot directly import torch.mlc.optim here since that will
# result in a circular dependency as torch.mlc is imported early by
# _C._initExtension(manager_path()) in torch/__init__.py.
#
# Instead torch.mlc.optim is only imported in
# to torch/backends/mlc/__init__.py which is imported much later.
# torch/mlc/optim/__init__.py then injects its classes into the right places
