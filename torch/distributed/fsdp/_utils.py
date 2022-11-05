import dataclasses
import traceback
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel.scatter_gather import (  # type: ignore[attr-defined]
    _is_namedtuple,
)
from torch.nn.utils.rnn import PackedSequence


def _contains_batchnorm(module):
    return any(isinstance(mod, _BatchNorm) for mod in module.modules())


def _override_batchnorm_mixed_precision(module):
    for mod in module.modules():
        if isinstance(mod, _BatchNorm):
            mod._wrap_overrides = {"mixed_precision": None}  # type: ignore[assignment]

@torch.no_grad()
def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> bool:
    """
    Allocate storage for ``tensor`` with the given size.

    Returns:
        bool: ``True`` if this method allocated storage and ``False`` if the
        storage was already allocated.
    """
    already_allocated = tensor.storage().size() == size.numel()
    if not already_allocated:
        tensor_storage_size = tensor.storage().size()
        p_assert(
            tensor_storage_size == 0,
            f"Tensor storage should have been resized to be 0 but got {tensor_storage_size}",
        )
        tensor.storage().resize_(size.numel())
    return not already_allocated


@torch.no_grad()
def _free_storage(tensor: torch.Tensor) -> bool:
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
    already_freed = tensor.storage().size() == 0
    if not already_freed:
        p_assert(
            tensor.storage_offset() == 0,
            "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
            f"storage offset: {tensor.storage_offset()}\n"
            f"storage size: {tensor.storage().size()}\n"
            f"tensor shape: {tensor.shape}",
        )
        tensor.storage().resize_(0)
    return not already_freed


def _same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Returns if ``x`` and ``y`` share the same storage."""
    # NOTE: CPU and GPU tensors are ensured to have different data pointers.
    return x.storage().data_ptr() == y.storage().data_ptr()


def p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """This is used as an alternate to ``assert`` when in the backward context
    to print the error message ``s`` since otherwise, it is swallowed."""
    if not cond:
        print(s)
        traceback.print_stack()
        if raise_assertion_error:
            raise AssertionError
