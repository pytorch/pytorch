from __future__ import annotations

import dataclasses
import sys

import torch


@dataclasses.dataclass(frozen=True)
class TensorRefInfo:
    """Reference count and alias information for a tensor.

    Weak counts include a +1 when the strong count is > 0 (intrusive_ptr
    convention).  Subtract 1 to get the number of external weak references.
    """

    # TensorImpl refcounts
    tensor_use_count: int
    tensor_weak_use_count: int

    # StorageImpl refcounts
    storage_use_count: int
    storage_weak_use_count: int

    # Python object refcount (via sys.getrefcount, includes the call's own ref)
    python_refcount: int

    # Alias / view information
    is_view: bool
    is_cow: bool
    data_ptr: int
    storage_offset: int


def tensor_ref_info(tensor: torch.Tensor) -> TensorRefInfo:
    """Collect reference count and alias information for a tensor."""
    return TensorRefInfo(
        tensor_use_count=tensor._use_count(),
        tensor_weak_use_count=tensor._weak_use_count(),
        storage_use_count=tensor._storage_use_count(),
        storage_weak_use_count=tensor._storage_weak_use_count(),
        python_refcount=sys.getrefcount(tensor),
        is_view=tensor._is_view(),
        is_cow=torch._C._is_cow_tensor(tensor),
        data_ptr=tensor.data_ptr(),
        storage_offset=tensor.storage_offset(),
    )


def is_safe_to_inplace(tensor: torch.Tensor) -> bool:
    """Check whether an in-place mutation of ``tensor`` is safe.

    "Safe" means no other tensor will observe the side effects.  This requires:
    - The storage is not shared (storage_use_count == 1).
    - The data is not copy-on-write shared.

    Weak references do not prevent mutation.  Autograd safety (leaf tensors,
    requires_grad) is a separate concern handled by autograd itself and is
    not checked here.
    """
    if torch._C._is_cow_tensor(tensor):
        return False
    return tensor._storage_use_count() == 1
