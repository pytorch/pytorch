# This file has moved to under torch/_functorch. It is not public API.
# If you are not a PyTorch developer and you are relying on the following
# imports, please file an issue.
from torch._functorch.vmap import (
    _add_batch_dim,
    _broadcast_to_and_flatten,
    _create_batched_inputs,
    _get_name,
    _process_batched_inputs,
    _remove_batch_dim,
    _unwrap_batched,
    _validate_and_get_batch_size,
    Tensor,
    tree_flatten,
    tree_unflatten,
)
