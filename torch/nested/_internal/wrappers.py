from typing import *  # noqa: F403

import torch._dynamo
from torch.nested._internal.cached_tensor import CachedTensor


# Wrap constructors in allow_in_graph
# Put in a separate file to avoid circular imports


# NJT-specific (rename?)
# Rename the source/target/extra
# is this allow_in_graph necessary?
@torch._dynamo.allow_in_graph  # type: ignore[misc]
def make_cached_tensor_for_nested(metadata: Dict[str, torch.Tensor]) -> CachedTensor:
    from torch.nested._internal.nested_tensor import make_cached_tensor_for_nested_impl

    return make_cached_tensor_for_nested_impl(metadata)
