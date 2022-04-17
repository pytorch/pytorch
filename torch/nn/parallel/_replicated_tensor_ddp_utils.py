from contextlib import contextmanager

_DDP_WITH_REPLICATED_TENSOR = False

@contextmanager
def _ddp_replicated_tensor(val):
    """
    A context manager to tag tensors in the forward pass of DDP to be
    ``ReplicatedTensor``. This can be used by ReplicatedTensor inter-op
    during the forward pass to perform appropriate optimizations.

    This context manager needs to wrap DDP creation and modifying the underlying
    module passed into DDP after leaving this context manager would cause
    inconsitencies and the changes will not be picked up during the forward
    pass.
    """
    global _DDP_WITH_REPLICATED_TENSOR
    old_val = _DDP_WITH_REPLICATED_TENSOR
    _DDP_WITH_REPLICATED_TENSOR = val
    try:
        yield
    finally:
        _DDP_WITH_REPLICATED_TENSOR = old_val

def _ddp_with_replicated_tensor_enabled():
    global _DDP_WITH_REPLICATED_TENSOR
    return _DDP_WITH_REPLICATED_TENSOR

def _set_ddp_with_replicated_tensor(value):
    global _DDP_WITH_REPLICATED_TENSOR
    _DDP_WITH_REPLICATED_TENSOR = value
