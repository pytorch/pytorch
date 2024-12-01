from typing import *  # noqa: F403

import torch._dynamo
from torch.nested._internal.offload_tensor import _make_offload_tensor, OffloadTensor


# Wrap constructors in allow_in_graph
# Put in a separate file so that this doesn't get imported
@torch._dynamo.allow_in_graph
def make_offload_tensor(
    device_tensor: Optional[torch.Tensor] = None,
    host_tensor: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> OffloadTensor:
    return _make_offload_tensor(device_tensor, host_tensor, device) 
