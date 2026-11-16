# mypy: allow-untyped-defs
from contextlib import contextmanager

import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


__all__ = ["is_available", "flags", "set_flags"]


def is_available():
    r"""Return whether PyTorch is built with NNPACK support."""
    return torch._nnpack_available()


def set_flags(_enabled):
    r"""Set if nnpack is enabled globally"""
    orig_flags = (torch._C._get_nnpack_enabled(),)
    torch._C._set_nnpack_enabled(_enabled)
    return orig_flags


@contextmanager
def flags(enabled=False):
    r"""Context manager for setting if nnpack is enabled globally"""
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(orig_flags[0])
