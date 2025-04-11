import numpy as np

import torch


def unpack_float4x2_as_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a float4x2 tensor to unpacked uint8 np array."""
    # FIXME: Figure out what the shape really means
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
    """Get the shape of an unpacked float4 tensor."""
    # TODO(justinchuby): Ensure this is correct
    return (*tensor.shape, 2)
