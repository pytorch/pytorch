import torch
from torch import Tensor


def my_from_blob_with_lambda_deleter(data_ptr, sizes, strides, device, dtype) -> Tensor:
    """
    Creates a Tensor from existing memory with a capturing-lambda deleter.

    The lambda deleter captures a pointer to a global counter and increments it,
    exercising the torch_from_blob_v2 code path (deleter + context).

    Args:
        data_ptr: int - pointer to the data buffer
        sizes: tuple[int] - size of the tensor
        strides: tuple[int] - strides of the tensor
        device: Device - device on which the tensor resides
        dtype: ScalarType - data type of the tensor

    Returns: Tensor - tensor wrapping the existing memory
    """
    return torch.ops.libtorch_agn_2_12.my_from_blob_with_lambda_deleter.default(
        data_ptr, sizes, strides, device, dtype
    )


def get_lambda_deleter_call_count() -> int:
    """
    Returns the number of times the lambda deleter has been called.
    """
    return torch.ops.libtorch_agn_2_12.get_lambda_deleter_call_count.default()


def reset_lambda_deleter_call_count() -> None:
    """
    Resets the lambda deleter call counter to zero.
    """
    torch.ops.libtorch_agn_2_12.reset_lambda_deleter_call_count.default()


def my_from_blob_with_cuda_lambda_deleter(numel: int, device) -> Tensor:
    """
    Creates a CUDA tensor that owns its memory via cudaMalloc with a lambda deleter.

    The tensor's memory is allocated with cudaMalloc and will be freed
    with cudaFree when the tensor is destroyed (via from_blob's lambda deleter).

    Args:
        numel: int - number of elements in the tensor
        device: Device - CUDA device

    Returns: Tensor - a 1D float32 tensor of zeros
    """
    return torch.ops.libtorch_agn_2_12.my_from_blob_with_cuda_lambda_deleter.default(
        numel, device
    )


# =============================================================================
# Proxy for inherited ops (from libtorch_agn_2_9, 2_10, and 2_11 csrc/)
#
# Ops compiled from previous versions' csrc directories are accessible via
# the module-level __getattr__. For example:
#     libtorch_agn_2_12.ops.sgd_out_of_place(...)  # from 2.9
#     libtorch_agn_2_12.ops.my_sum(...)            # from 2.10
# =============================================================================

_NAMESPACE = "libtorch_agn_2_12"


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
