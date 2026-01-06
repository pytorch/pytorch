import numpy as np

import torch


def unpack_float4x2_as_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a float4x2 tensor to unpacked uint8 np array."""
    assert tensor.dtype == torch.float4_e2m1fn_x2
    data = tensor.view(torch.uint8).numpy(force=True).flatten()
    result_size = tensor.numel() * 2
    result = np.empty([result_size], dtype=np.uint8)
    array_low = data & np.uint8(0x0F)
    array_high = data & np.uint8(0xF0)
    array_high >>= np.uint8(4)
    result[0::2] = array_low
    result[1::2] = array_high
    result.resize(get_float4_shape(tensor), refcheck=False)
    return result


def get_float4_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    """Get the shape of an unpacked float4 tensor.

    The float4_e2m1fn_x2 type is a shell type described in
    https://github.com/pytorch/pytorch/issues/146414.

    the shell dtype is takes up 1 byte per element and semantically represents
    two fp4 values packed into 1 byte. Semantically it represents (*tensor.shape[:-1], tensor.shape[-1]*2)
    fp4 elements.
    """
    assert tensor.dtype == torch.float4_e2m1fn_x2
    return (*tensor.shape[:-1], tensor.shape[-1] * 2)
