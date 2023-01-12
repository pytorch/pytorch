from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from torch import Tensor


# This util function splits tensors into groups by device and dtype, which is useful before sending
# tensors off to a foreach implementation, which requires tensors to be on one device and dtype.
# Currently, this function is only used in torch.optim.
# If tensorlistlist contains more than one tensorlist, the following assumptions are made BUT NOT verified:
#   - tensorlists CAN be None
#   - all tensors in the first specified list cannot be None
#   - given an index i, all specified tensorlist[i]s match in dtype and device
@torch.no_grad()
def _group_tensors_by_device_and_dtype(tensorlistlist: List[List[Tensor]]) -> Dict[Tuple[str, torch.dtype], List[List[Tensor]]]:
    assert all([not x or len(x) == len(tensorlistlist[0]) for x in tensorlistlist]), (
           "all specified tensorlists must match in length")
    per_device_and_dtype_tensors: Dict[Tuple[str, torch.dtype], List[List[Tensor]]] = defaultdict(
        lambda: [[] for _ in range(len(tensorlistlist))])
    for i, t in enumerate(tensorlistlist[0]):
        key = (str(t.device), t.dtype)
        for j in range(len(tensorlistlist)):
            # a tensorlist may be empty/None
            if tensorlistlist[j]:
                per_device_and_dtype_tensors[key][j].append(tensorlistlist[j][i])
    return per_device_and_dtype_tensors
