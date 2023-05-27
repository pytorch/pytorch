from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

import torch
from torch import Tensor
from torch.autograd.grad_mode import no_grad

__all__ = ["get_foreach_kernels_supported_devices", "get_fused_kernels_supported_devices"]

def get_foreach_kernels_supported_devices() -> List[str]:
    r"""
    Return the device type list that supports foreach kernels.
    """
    return ["cuda", torch._C._get_privateuse1_backend_name()]

def get_fused_kernels_supported_devices() -> List[str]:
    r"""
    Return the device type list that supports fused kernels in optimizer.
    """
    return ["cuda", torch._C._get_privateuse1_backend_name()]

# This util function splits tensors into groups by device and dtype, which is useful before sending
# tensors off to a foreach implementation, which requires tensors to be on one device and dtype.
# If tensorlistlist contains more than one tensorlist, the following assumptions are made BUT NOT verified:
#   - tensorlists CAN be None
#   - all tensors in the first specified list cannot be None
#   - given an index i, all specified tensorlist[i]s match in dtype and device
# with_indices (bool, optional): whether to track previous indices as the last list per dictionary entry.
#   It comes in handy if there are Nones or literals in the tensorlists that are getting scattered out.
#   Whereas mutating a tensor in the resulting split-up tensorlists WILL propagate changes back to the
#   original input tensorlists, changing up Nones/literals WILL NOT propagate, and manual propagation
#   may be necessary. Check out torch/optim/sgd.py for an example.
@no_grad()
def _group_tensors_by_device_and_dtype(tensorlistlist: List[List[Tensor]],
                                       with_indices: Optional[bool] = False) -> \
        Dict[Tuple[torch.device, torch.dtype], List[List[Union[Tensor, int]]]]:
    assert all(not x or len(x) == len(tensorlistlist[0]) for x in tensorlistlist), (
           "all specified tensorlists must match in length")
    per_device_and_dtype_tensors: Dict[Tuple[torch.device, torch.dtype], List[List[Union[Tensor, int]]]] = defaultdict(
        lambda: [[] for _ in range(len(tensorlistlist) + (1 if with_indices else 0))])
    for i, t in enumerate(tensorlistlist[0]):
        key = (t.device, t.dtype)
        for j in range(len(tensorlistlist)):
            # a tensorlist may be empty/None
            if tensorlistlist[j]:
                per_device_and_dtype_tensors[key][j].append(tensorlistlist[j][i])
        if with_indices:
            # tack on previous index
            per_device_and_dtype_tensors[key][j + 1].append(i)
    return per_device_and_dtype_tensors

def _has_foreach_support(tensors: List[Tensor], device: torch.device) -> bool:
    if device.type not in ["cpu", "cuda", torch._C._get_privateuse1_backend_name()] or torch.jit.is_scripting():
        return False
    return all(t is None or type(t) == torch.Tensor for t in tensors)

def _check_same_device(params: List[Tensor]) -> bool:
    r"""
    Checks that all Tensors in params have the same device.
    """
    device_type = ""
    for p in params:
        if p is None:
            continue
        if device_type == "":
            device_type = p.device.type
        if device_type != p.device.type:
            return False
    return True
