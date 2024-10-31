from typing import List, Dict, Tuple, Optional

import torch
from torch import Tensor
from torch.autograd.grad_mode import no_grad
from typing_extensions import TypeAlias

def _get_foreach_kernels_supported_devices() -> List[str]:
    r"""Return the device type list that supports foreach kernels."""
    return ["cuda", "xpu", torch._C._get_privateuse1_backend_name()]

def _get_fused_kernels_supported_devices() -> List[str]:
    r"""Return the device type list that supports fused kernels in optimizer."""
    return ["mps", "cuda", "xpu", "cpu", torch._C._get_privateuse1_backend_name()]

TensorListList: TypeAlias = List[List[Optional[Tensor]]]
Indices: TypeAlias = List[int]
_foreach_supported_types = [torch.Tensor]


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
def _group_tensors_by_device_and_dtype(
    tensorlistlist: TensorListList,
    with_indices: bool = False,
) -> Dict[Tuple[torch.device, torch.dtype], Tuple[TensorListList, Indices]]:
    return torch._C._group_tensors_by_device_and_dtype(tensorlistlist, with_indices)

def _device_has_foreach_support(device: torch.device) -> bool:
    return device.type in (_get_foreach_kernels_supported_devices() + ["cpu"]) and not torch.jit.is_scripting()


def _has_foreach_support(tensors: List[Tensor], device: torch.device) -> bool:
    return _device_has_foreach_support(device) and all(t is None or type(t) in _foreach_supported_types for t in tensors)
