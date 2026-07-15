import torch
from torch import Tensor


def my_from_blob_with_deleter(data_ptr, sizes, strides, device, dtype) -> Tensor:
    """
    Creates a Tensor from existing memory with a deleter callback.

    The deleter will be called when the tensor's storage is deallocated. For
    this test, the deleter just updates a global call count, which allows us to
    assert that is was called from get_deleter_call_count().

    Args:
        data_ptr: int - pointer to the data buffer
        sizes: tuple[int] - size of the tensor
        strides: tuple[int] - strides of the tensor
        device: Device - device on which the tensor resides
        dtype: ScalarType - data type of the tensor

    Returns: Tensor - tensor wrapping the existing memory
    """
    return torch.ops.libtorch_agn_2_11.my_from_blob_with_deleter.default(
        data_ptr, sizes, strides, device, dtype
    )


def get_deleter_call_count() -> int:
    """
    Returns the number of times the test deleter has been called.
    """
    return torch.ops.libtorch_agn_2_11.get_deleter_call_count.default()


def reset_deleter_call_count() -> None:
    """
    Resets the deleter call counter to zero.
    """
    torch.ops.libtorch_agn_2_11.reset_deleter_call_count.default()


def my_from_blob_with_cuda_deleter(numel: int, device) -> Tensor:
    """
    Creates a CUDA tensor that owns its memory via cudaMalloc.

    The tensor's memory is allocated with cudaMalloc and will be freed
    with cudaFree when the tensor is destroyed (via from_blob's deleter).
    This is useful for testing that the deleter properly frees memory.

    Args:
        numel: int - number of elements in the tensor
        device: Device - CUDA device

    Returns: Tensor - a 1D float32 tensor of zeros
    """
    return torch.ops.libtorch_agn_2_11.my_from_blob_with_cuda_deleter.default(
        numel, device
    )


# =============================================================================
# Proxy for inherited ops (from libtorch_agn_2_9 and libtorch_agn_2_10 csrc/)
#
# Ops compiled from previous versions' csrc directories are accessible via
# the module-level __getattr__. For example:
#     libtorch_agn_2_11.ops.sgd_out_of_place(...)  # from 2.9
#     libtorch_agn_2_11.ops.my_sum(...)            # from 2.10
# =============================================================================

_NAMESPACE = "libtorch_agn_2_11"


def __getattr__(name):
    """Proxy for inherited ops from previous versions."""
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    ops_namespace = getattr(torch.ops, _NAMESPACE)
    op = getattr(ops_namespace, name, None)
    if op is None:
        raise AttributeError(f"No op named '{name}' in {_NAMESPACE}")
    return op.default


def __dir__():
    """List all available ops (native + inherited)."""
    native = [
        name
        for name in globals()
        if not name.startswith("_") and callable(globals().get(name))
    ]
    ops_namespace = getattr(torch.ops, _NAMESPACE)
    inherited = [n for n in dir(ops_namespace) if not n.startswith("_")]
    return sorted(set(native + inherited))
