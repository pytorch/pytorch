"""Default-dtype context management, called from C++ torch.dtype.__enter__/__exit__."""

import threading

import torch


_dtype_stack = threading.local()


def _enter_dtype(dtype_obj):
    if not hasattr(_dtype_stack, "stack"):
        _dtype_stack.stack = []
    _dtype_stack.stack.append(torch.get_default_dtype())
    torch.set_default_dtype(dtype_obj)


def _exit_dtype():
    old = _dtype_stack.stack.pop()
    torch.set_default_dtype(old)
