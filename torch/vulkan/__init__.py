# mypy: allow-untyped-defs
r"""
This package enables an interface for accessing the Vulkan backend in Python.
"""
from typing import Union

import torch
from torch import Tensor

def device_count() -> int:
    r"""Returns the number of available Vulkan devices."""
    # TODO: actually get the number!
    return int(torch.is_vulkan_available())


def compile_shader(name: str, source: str):
    r"""Compiles compute shader from source and allows one to invoke kernels
    defined there from the comfort of Python runtime
    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_MPS)
        >>> lib = torch.mps.compile_shader(
        ... "kernel void full(device float* out, constant float& val, uint idx [[thread_position_in_grid]]) { out[idx] = val; }"
        ...  )
        >>> x = torch.zeros(16, device="mps")
        >>> lib.full(x, 3.14)
    """
    from pathlib import Path

    from torch.utils._cpp_embed_headers import _embed_headers

    if not hasattr(torch._C, "_vulkan_compileShader"):
        raise RuntimeError("Vulkan is not available")
    source = _embed_headers(
        [l + "\n" for l in source.split("\n")],
        [Path(__file__).parent.parent / "include"],
        set(),
    )
    return torch._C._vulkan_compileShader(name, source)


def is_available() -> bool:
    return device_count() > 0


__all__ = [
    "compile_shader",
    "device_count",
    "is_available",
    "synchronize",
]
