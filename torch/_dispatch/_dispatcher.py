import torch
import torch._C as _C
from torch.utils._pytree import tree_flatten

"""
This is a dispatcher (in Python)
- You can define new operations (in Python) without schemas
- It interfaces with the PyTorch dispatcher
"""

class PyDispatcher:
    # operator is a PyOperator
    @staticmethod
    def call(operator, *args, **kwargs):
        dispatch_key_set = compute_keyset(args, kwargs)
        kernel = operator.lookup(dispatch_key_set)
        return kernel(*args, **kwargs)

    # operator is a PyOperator
    @staticmethod
    def redispatch(operator, dispatch_key_set, *args, **kwargs):
        kernel = operator.lookup(dispatch_key_set)
        return kernel(*args, **kwargs)


def compute_keyset(args, kwargs):
    tensors = get_tensors(args, kwargs)
    return key_extractor(tensors)


# Note - this should maintain identical impl to the C++ dispatcher key extraction logic
# at ATen/core/dispatch/DispatchKeyExtractor.h
def key_extractor(tensors):
    key_set = _C._dispatch_tls_local_include_set()  # type: ignore[attr-defined]
    for tensor in tensors:
        key_set = key_set | _C._dispatch_keys(tensor)  # type: ignore[attr-defined]
    key_set = key_set - _C._dispatch_tls_local_exclude_set()  # type: ignore[attr-defined]
    return key_set


def to_flat_tuple(args, kwargs):
    flat_args, _ = tree_flatten(args)
    flat_kwargs, _ = tree_flatten(kwargs)
    flat_all = flat_args + flat_kwargs
    return flat_all

def get_tensors(args, kwargs):
    flat_all = to_flat_tuple(args, kwargs)
    tensor_args = [t for t in flat_all if isinstance(t, torch.Tensor)]
    return tuple(tensor_args)
