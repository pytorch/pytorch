"""
cuDNN attention implementation, via the cuDNN frontend Python API.

The kernels live in the out-of-tree ``nvidia-cudnn-frontend`` package, which
registers itself into this registry when imported. This module exists only so
the name is selectable *before* that import has happened: without it,
``activate_flash_attention_impl("CUDNN")`` fails on a fresh interpreter and
callers have to know to ``import cudnn.torch`` first.
"""
# mypy: allow-untyped-defs

from __future__ import annotations

import importlib

from . import _registry


__all__ = [
    "register_flash_attention_cudnn",
]


_CUDNN_MODULE_PATH = "cudnn.torch"


def register_flash_attention_cudnn(module_path: str = _CUDNN_MODULE_PATH):
    """
    Register cuDNN attention kernels with the PyTorch dispatcher.

    Args:
        module_path: Python module path to the cuDNN frontend torch provider.

    Unlike FA3/FA4, the implementation is not shimmed here: importing
    ``module_path`` re-registers "CUDNN" with the provider's own callable, and
    this function hands off to it. That keeps the kernel set (dense SDPA plus
    the varlen ops) owned entirely by the package that ships the kernels, so a
    provider update does not need a matching change here.
    """
    importlib.import_module(module_path)

    register_fn = _registry._FLASH_ATTENTION_IMPLS.get("CUDNN")
    if register_fn is None or register_fn is register_flash_attention_cudnn:
        # The package imported but did not take over the registration, so it
        # predates the flash-impl registry. Say so, rather than recursing.
        raise RuntimeError(
            f"'{module_path}' did not register a 'CUDNN' flash attention "
            f"implementation; a newer nvidia-cudnn-frontend is required"
        )
    return register_fn()


_registry.register_flash_attention_impl(
    "CUDNN", register_fn=register_flash_attention_cudnn
)
