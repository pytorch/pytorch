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
    return torch.ops.libtorch_agn_2_11_cuda.my_from_blob_with_cuda_deleter.default(
        numel, device
    )
