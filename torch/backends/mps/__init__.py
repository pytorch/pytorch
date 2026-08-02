from functools import lru_cache as _lru_cache
from typing import Optional, TYPE_CHECKING

import torch
from torch.library import Library as _Library


__all__ = [
    "get_core_count",
    "get_name",
    "is_built",
    "is_available",
    "is_macos13_or_newer",
    "is_macos_or_newer",
]


def is_built() -> bool:
    r"""Return whether PyTorch is built with MPS support.

    Note that this doesn't necessarily mean MPS is available; just that
    if this PyTorch binary were run on a machine with working MPS drivers
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


@_lru_cache
def get_name() -> str:
    r"""Return Metal device name"""
    return torch._C._mps_get_name()


@_lru_cache
def get_core_count() -> int:
    r"""Return GPU core count.

    According to the documentation, one core is comprised of 16 Execution Units.
    One execution Unit has 8 ALUs.
    And one ALU can run 24 threads, i.e. one core is capable of executing 3072 threads concurrently.
    """
    return torch._C._mps_get_core_count()
