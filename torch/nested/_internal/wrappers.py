from typing import *  # noqa: F403

import torch._dynamo
from torch.nested._internal.offload_tensor import _make_offload_tensor, OffloadTensor
from torch.nested._internal.cached_tensor import _make_cached_tensor, CachedTensor


# Wrap constructors in allow_in_graph
# Put in a separate file so that this doesn't get imported
@torch._dynamo.allow_in_graph
def make_offload_tensor(
    device_tensor: Optional[torch.Tensor] = None,
    host_tensor: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> OffloadTensor:
    return _make_offload_tensor(device_tensor, host_tensor, device) 

@torch._dynamo.allow_in_graph
def make_cached_tensor(metadata, source_fields, extra_fields, target_field=None) -> CachedTensor:
    return _make_cached_tensor(metadata, source_fields, extra_fields, target_field)


# Nested-specific unwrapping
@torch._dynamo.allow_in_graph
def unpack(x):
    from torch.nested._internal.nested_tensor import UnpackResult

    # Need to hide the attribute access from dynamo. GetAttrVariable does not
    # support any operations.
    return UnpackResult(**{k: x.metadata.get(k) for k in x.all_fields})
