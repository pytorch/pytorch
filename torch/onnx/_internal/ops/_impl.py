import typing
from typing import Any, Callable, Optional

import torch
import torch.fx

_T = typing.TypeVar("_T", bound=Callable)

_ONNX_DECOMP_TABLE = {}


def onnx_aten_decomp_table() -> dict[Any, Callable]:
    """Return the ONNX to ATen decomp table."""
    return _ONNX_DECOMP_TABLE


def _register_op(func: _T) -> _T:
    func_name = func.__name__
    torch_op = torch.library.custom_op(f"onnx::{func_name}", mutates_args=())(func)
    _ONNX_DECOMP_TABLE[getattr(torch.ops.onnx, func_name).default] = func
    # Use the same implementation for the fake implementation
    # This is possible because we use pure aten ops to implement ONNX ops
    torch_op.register_fake(func)
    return torch_op


@_register_op
def rotary_embedding(
    input: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    interleaved: bool = False,
    num_heads: int = 0,
    rotary_embedding_dim: int = 0,
) -> torch.Tensor:
    # First ensure input has shape [batch_size, num_heads, seq_len, head_size]
    batch_size = input.shape[0]
    sequence_length = input.shape[1]
    if len(input.shape) == 3:
        hidden_size = input.shape[2]
        assert num_heads != 0
        head_size = int(hidden_size / num_heads)
        new_shape = [batch_size, sequence_length, num_heads, head_size]
        input = torch.reshape(input, new_shape)
    assert len(input.shape) == 4
    head_size = input.shape[3]

    # Fully or partially perform rotation on input based on rotary_embedding_dim attribute
    if rotary_embedding_dim == 0:
        # If rotary_embedding_dim not provided, perform full rotation by using head_size
        rotary_embedding_dim = head_size
    x_rotate = input[:, :, :, :rotary_embedding_dim]
    x_not_rotate = input[:, :, :, rotary_embedding_dim:]
    rotary_embedding_dim_half = int(rotary_embedding_dim / 2)

    # Retrieve sin and cos caches using position ids
    if position_ids is not None:
        cos = cos_cache[
            position_ids
        ]  # Shape: [batch_size, sequence_length, head_size/2]
        sin = sin_cache[
            position_ids
        ]  # Shape: [batch_size, sequence_length, head_size/2]
    else:
        cos = cos_cache
        sin = sin_cache
    cos = cos[
        :, :, :rotary_embedding_dim_half
    ]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
    sin = sin[
        :, :, :rotary_embedding_dim_half
    ]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
    cos = torch.unsqueeze(
        cos, 2
    )  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]
    sin = torch.unsqueeze(
        sin, 2
    )  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]

    # Either divide the input in halves or interleave (based on interleaved attribute)
    if interleaved:
        x1 = x_rotate[:, :, :, 0::2]
        x2 = x_rotate[:, :, :, 1::2]
    else:
        x1, x2 = torch.chunk(x_rotate, 2, dim=-1)

    # Calculate real and imaginary values
    real = cos * x1 - sin * x2
    imag = sin * x1 + cos * x2

    # Inserted rotated embeddings back to the original input
    if interleaved:
        # x_rotate[:, :, :, 0::2] = real
        # x_rotate[:, :, :, 1::2] = imag
        real = torch.unsqueeze(real, -1)
        imag = torch.unsqueeze(imag, -1)
        x_rotate_concat = torch.cat((real, imag), dim=-1)
        x_rotate = torch.reshape(x_rotate_concat, x_rotate.shape)
    else:
        x_rotate = torch.cat((real, imag), dim=-1)
    output = torch.cat((x_rotate, x_not_rotate), dim=-1)
    if len(input.shape) == 3:
        output = torch.reshape(output, input.shape)
    return output


"""
input_data = torch.rand(2, 3, 4, 8)
position_ids_data = torch.randint(0, 50, (2, 3)).long()
sin_cache_data = torch.rand(50, 4)
cos_cache_data = torch.rand(50, 4)

result = torch.ops.onnx.rotary_embedding(input_data, cos_cache_data, sin_cache_data, position_ids_data)

"""
