# mypy: allow-untyped-defs
from functools import lru_cache as _lru_cache
from typing import Optional, TYPE_CHECKING

import torch
from torch.library import Library as _Library


__all__ = ["is_built", "is_available", "is_macos13_or_newer", "is_macos_or_newer"]


def is_built() -> bool:
    r"""Return whether PyTorch is built with MPS support.

    Note that this doesn't necessarily mean MPS is available; just that
    if this PyTorch binary were run a machine with working MPS drivers
    and devices, we would be able to use it.
    """
    return torch._C._has_mps


@_lru_cache
def is_available() -> bool:
    r"""Return a bool indicating if MPS is currently available."""
    return torch._C._mps_is_available()


@_lru_cache
def is_macos_or_newer(major: int, minor: int) -> bool:
    r"""Return a bool indicating whether MPS is running on given MacOS or newer."""
    return torch._C._mps_is_on_macos_or_newer(major, minor)


@_lru_cache
def is_macos13_or_newer(minor: int = 0) -> bool:
    r"""Return a bool indicating whether MPS is running on MacOS 13 or newer."""
    return torch._C._mps_is_on_macos_or_newer(13, minor)


_lib: Optional[_Library] = None


def _init():
    r"""Register prims as implementation of var_mean and group_norm."""
    global _lib

    if _lib is not None or not is_built():
        return

    from torch._decomp.decompositions import native_group_norm_backward
    from torch._refs import native_group_norm

    _lib = _Library("aten", "IMPL")  # noqa: TOR901
    _lib.impl("native_group_norm", native_group_norm, "MPS")
    _lib.impl("native_group_norm_backward", native_group_norm_backward, "MPS")
