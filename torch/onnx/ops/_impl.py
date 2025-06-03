import typing
from typing import Callable, Optional

import torch


_T = typing.TypeVar("_T", bound=Callable)

# ONNX to ATen decomp table
ONNX_ATEN_DECOMP_TABLE: dict[torch._ops.OpOverload, Callable] = {}


def _onnx_op(op_type: str, opset_version: int) -> Callable[[_T], _T]:
    """Decorator to register an ONNX operator with a custom implementation."""

    def decorator(func: _T) -> _T:
        overload = f"opset{opset_version}"
        torch_op = torch.library.custom_op(
            f"onnx::{op_type}.{overload}", mutates_args=()
        )(func)
        ONNX_ATEN_DECOMP_TABLE[getattr(getattr(torch.ops.onnx, op_type), overload)] = (
            func  # type: ignore[assignment]
        )
        # Use the same implementation for the fake implementation
        # This is possible because we use pure aten ops to implement ONNX ops
        torch_op.register_fake(func)
        return torch_op  # type: ignore[return-value]

    return decorator


@_onnx_op("RotaryEmbedding", 23)
def rotary_embedding(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    *,
    interleaved: bool = False,
    num_heads: int = 0,
    rotary_embedding_dim: int = 0,
) -> torch.Tensor:
    """RotaryEmbedding-23 https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html#rotaryembedding-23"""
    # First ensure x has shape [batch_size, num_heads, seq_len, head_size]
    batch_size = x.shape[0]
    sequence_length = x.shape[1]
    if len(x.shape) == 3:
        hidden_size = x.shape[2]
        assert num_heads != 0
        head_size = hidden_size // num_heads
        new_shape = [batch_size, sequence_length, num_heads, head_size]
        x = torch.reshape(x, new_shape)
    assert len(x.shape) == 4
    head_size = x.shape[3]

    # Fully or partially perform rotation on x based on rotary_embedding_dim attribute
    if rotary_embedding_dim == 0:
        # If rotary_embedding_dim not provided, perform full rotation by using head_size
        rotary_embedding_dim = head_size
    x_rotate = x[:, :, :, :rotary_embedding_dim]
    x_not_rotate = x[:, :, :, rotary_embedding_dim:]
    rotary_embedding_dim_half = rotary_embedding_dim // 2

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

    # Either divide the x in halves or interleave (based on interleaved attribute)
    if interleaved:
        x1 = x_rotate[:, :, :, 0::2]
        x2 = x_rotate[:, :, :, 1::2]
    else:
        x1, x2 = torch.chunk(x_rotate, 2, dim=-1)

    # Calculate real and imaginary values
    real = cos * x1 - sin * x2
    imag = sin * x1 + cos * x2

    # Inserted rotated embeddings back to the original x
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
    if len(x.shape) == 3:
        output = torch.reshape(output, x.shape)
    return output
