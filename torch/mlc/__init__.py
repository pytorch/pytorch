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

# Instead torch.mlc.optim is only imported in
# to torch/backends/mlc/__init__.py which is imported much later.
# torch/mlc/optim/__init__.py then injects its classes into the right places
