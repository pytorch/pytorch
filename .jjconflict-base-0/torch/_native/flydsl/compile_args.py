"""Shared FlyDSL compile-time and launch-time tensor argument helpers."""

from __future__ import annotations

from typing import Any

import torch


def read_only_tensor(tensor: torch.Tensor) -> Any:
    """Return a read-only view that does not materialize copy-on-write tensors."""
    if not torch._C._is_cow_tensor(tensor):  # pyrefly: ignore[missing-attribute]
        return tensor
    from torch._native.const_tensor_wrapper import ConstTensorWrapper

    return ConstTensorWrapper(tensor)


def make_compile_arg(
    tensor: torch.Tensor, *, read_only: bool = False, dynamic_dim: int = 0
) -> Any:
    """Build a FlyDSL compile argument with one dynamic tensor dimension."""
    # Imported here so this module stays importable without flydsl installed.
    import flydsl.compiler as flyc

    tensor_arg = read_only_tensor(tensor) if read_only else tensor
    return flyc.from_torch_tensor(tensor_arg).mark_shape_dynamic(dynamic_dim)
