# flake8: noqa: B950
import math
import typing
from typing import Callable, Optional

import torch
from torch.onnx.ops import _dtype_mappings


_T = typing.TypeVar("_T", bound=Callable)

# ONNX to ATen decomp table
ONNX_ATEN_DECOMP_TABLE: dict[torch._ops.OpOverload, Callable] = {}
_ATTENTION_23_ALLOWED_INTERMEDIATE_PRECISIONS = frozenset(
    {
        1,  # FLOAT
        10,  # FLOAT16
        11,  # DOUBLE
        16,  # BFLOAT16
    }
)

# Saturation ranges for different quantization types
_QUANTIZATION_SATURATION_RANGES = {
    torch.uint16: (0, 65535),
    torch.int16: (-32768, 32767),
    torch.uint8: (0, 255),
    torch.int8: (-128, 127),
    # Note: uint4 and int4 are not directly supported in PyTorch
    # They would need special handling if encountered
}


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
def rotary_embedding_23(
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
        torch._check(
            num_heads != 0,
            lambda: f"num_heads must be provided for 3D inputs. Received input tensor with shape {x.shape}",
        )
        head_size = hidden_size // num_heads
        new_shape = [batch_size, sequence_length, num_heads, head_size]
        x = torch.reshape(x, new_shape)
    torch._check(len(x.shape) == 4, lambda: "x should be a 4D tensor by now")
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


def _get_scale_factor(scale: Optional[float], head_size: int) -> float:
    """Get the scale factor for attention computation."""
    return scale if scale is not None else (1.0 / math.sqrt(head_size))


def _reshape_3d_to_4d(
    tensor: torch.Tensor, batch_size: int, num_heads: int
) -> torch.Tensor:
    """Reshape 3D tensor to 4D for multi-head attention."""
    sequence_length, hidden_size = tensor.shape[1], tensor.shape[2]
    head_size = hidden_size // num_heads
    return (
        tensor.view(batch_size, sequence_length, num_heads, head_size)
        .transpose(1, 2)
        .contiguous()
    )


def _get_qk_output_for_aten_spda(
    Q: torch.Tensor,
    K: torch.Tensor,
    current_q_num_heads: int,
    current_kv_num_heads: int,
    scale: Optional[float],
    qk_matmul_output_mode: int,
) -> torch.Tensor:
    """Get QK output tensor based on the specified mode."""
    if qk_matmul_output_mode == 0:
        return _compute_qk_output_for_mode_0(
            Q, K, current_q_num_heads, current_kv_num_heads, scale
        )
    else:
        # For other modes, return a zero tensor with correct shape
        return torch.zeros_like(torch.matmul(Q, K.transpose(-2, -1)))


def _validate_gqa_configuration(
    current_q_num_heads: int, current_kv_num_heads: int
) -> None:
    """Validate Group Query Attention configuration."""
    torch._check(
        current_q_num_heads % current_kv_num_heads == 0,
        lambda: f"q_num_heads ({current_q_num_heads}) must be divisible by kv_num_heads ({current_kv_num_heads}) for GQA",
    )


def _compute_qk_output_for_mode_0(
    Q: torch.Tensor,
    K: torch.Tensor,
    current_q_num_heads: int,
    current_kv_num_heads: int,
    scale: Optional[float],
) -> torch.Tensor:
    """Helper function to compute QK output for qk_matmul_output_mode == 0."""
    # Handle GQA manually for QK output
    K_for_qk = K
    if current_q_num_heads != current_kv_num_heads:
        repeat_factor = current_q_num_heads // current_kv_num_heads
        K_for_qk = K.repeat_interleave(repeat_factor, dim=1)

    scale_factor = _get_scale_factor(scale, Q.shape[3])
    # Scale both Q and K by sqrt(scale_factor) for numerical stability
    sqrt_scale = math.sqrt(scale_factor)
    Q_scaled = Q * sqrt_scale
    K_scaled = K_for_qk * sqrt_scale
    return torch.matmul(Q_scaled, K_scaled.transpose(-2, -1))


@_onnx_op("Attention", 23)
def attention_23(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    past_key: Optional[torch.Tensor] = None,
    past_value: Optional[torch.Tensor] = None,
    *,
    is_causal: bool = False,
    kv_num_heads: int = 0,
    q_num_heads: int = 0,
    qk_matmul_output_mode: int = 0,
    scale: Optional[float] = None,
    softcap: float = 0.0,
    softmax_precision: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention-23 https://onnx.ai/onnx/operators/onnx__Attention.html#attention-23"""

    num_head_dim, sequence_dim, head_dim = 1, 2, 3

    # Store original input shape to determine output shape
    input_shape_len = len(Q.shape)
    batch_size = Q.shape[0]

    # Reshape 3D inputs to 4D format
    if len(Q.shape) == 3:
        torch._check(
            q_num_heads != 0 and kv_num_heads != 0,
            lambda: "q_num_heads and kv_num_heads must be provided for 3D inputs",
        )
        q_sequence_length = Q.shape[1]
        Q = _reshape_3d_to_4d(Q, batch_size, q_num_heads)
        K = _reshape_3d_to_4d(K, batch_size, kv_num_heads)
        V = _reshape_3d_to_4d(V, batch_size, kv_num_heads)

    torch._check(
        len(Q.shape) == 4 and len(K.shape) == 4 and len(V.shape) == 4,
        lambda: "Q, K, and V should be 4D tensors by now",
    )

    # Calculate scale factor if not provided
    q_head_size = Q.shape[head_dim]
    scale = _get_scale_factor(scale, q_head_size)

    # Handle past key/value caches
    present_key = (
        torch.cat([past_key, K], dim=sequence_dim)
        if past_key is not None
        else K.clone()
    )
    present_value = (
        torch.cat([past_value, V], dim=sequence_dim)
        if past_value is not None
        else V.clone()
    )

    # Update K and V to include past states
    K, V = present_key, present_value

    # Get current dimensions
    current_q_num_heads = Q.shape[num_head_dim]
    current_kv_num_heads = K.shape[num_head_dim]
    q_sequence_length = Q.shape[sequence_dim]
    kv_sequence_length = K.shape[sequence_dim]

    # Check if we can use the optimized scaled_dot_product_attention (most optimized)
    can_use_sdpa = (
        softcap == 0.0  # No softcap
        and qk_matmul_output_mode == 0  # Default QK output mode
        and softmax_precision is None  # No custom softmax precision
        and (attn_mask is None or attn_mask.dtype == torch.bool)
    )

    _validate_gqa_configuration(current_q_num_heads, current_kv_num_heads)

    if can_use_sdpa:
        # Use PyTorch's optimized scaled_dot_product_attention

        # Prepare attention mask for SDPA
        sdpa_attn_mask = None
        if attn_mask is not None:
            # Convert boolean mask: True means participate, SDPA expects True to mask out
            sdpa_attn_mask = ~attn_mask if attn_mask.dtype == torch.bool else attn_mask

        output = torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=sdpa_attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=bool(
                current_q_num_heads != current_kv_num_heads
            ),  # Ensure enable_gqa is not SymBool
        )

        qk_output = _get_qk_output_for_aten_spda(
            Q,
            K,
            current_q_num_heads,
            current_kv_num_heads,
            scale,
            qk_matmul_output_mode,
        )
    else:
        # Fallback to manual implementation for complex cases

        # Handle Group Query Attention (GQA) and Multi-Query Attention (MQA)
        if current_q_num_heads != current_kv_num_heads:
            repeat_factor = current_q_num_heads // current_kv_num_heads
            K = K.repeat_interleave(repeat_factor, dim=num_head_dim)
            V = V.repeat_interleave(repeat_factor, dim=num_head_dim)

        # Create attention bias
        attn_bias = torch.zeros(
            q_sequence_length, kv_sequence_length, dtype=Q.dtype, device=Q.device
        )

        # Apply causal masking
        if is_causal:
            torch._check(
                attn_mask is None, lambda: "Cannot use both is_causal and attn_mask"
            )
            causal_mask = torch.tril(
                torch.ones(
                    q_sequence_length,
                    kv_sequence_length,
                    dtype=torch.bool,
                    device=Q.device,
                )
            )
            attn_bias = attn_bias.masked_fill(~causal_mask, float("-inf"))

        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # Boolean mask: True means participate in attention
                attn_bias = attn_bias.masked_fill(~attn_mask, float("-inf"))
            else:
                # Float mask: added to attention scores
                attn_bias = attn_bias + attn_mask

        # Apply scaling factor
        scale_factor = _get_scale_factor(scale, Q.shape[3])

        # Scale both Q and K by sqrt(scale_factor) for numerical stability
        sqrt_scale = math.sqrt(scale_factor)
        Q_scaled = Q * sqrt_scale
        K_scaled = K * sqrt_scale

        # Compute Q @ K^T
        qk_matmul_output = torch.matmul(Q_scaled, K_scaled.transpose(-2, -1))

        # Initialize QK output based on mode
        qk_output = qk_matmul_output  # Default case for mode 0

        # Add attention bias
        qk_with_bias = qk_matmul_output + attn_bias

        if qk_matmul_output_mode == 1:
            qk_output = qk_with_bias

        # Apply softcap if provided
        if softcap > 0.0:
            qk_with_bias = softcap * torch.tanh(qk_with_bias / softcap)

        if qk_matmul_output_mode == 2:
            qk_output = qk_with_bias

        # Apply softmax with optional precision casting
        if softmax_precision is not None:
            # Map ONNX data type to torch dtype
            if softmax_precision in _ATTENTION_23_ALLOWED_INTERMEDIATE_PRECISIONS:
                original_dtype = qk_with_bias.dtype
                qk_with_bias = qk_with_bias.to(
                    _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[softmax_precision]
                )
                qk_softmax = torch.softmax(qk_with_bias, dim=-1)
                qk_softmax = qk_softmax.to(original_dtype)
            else:
                qk_softmax = torch.softmax(qk_with_bias, dim=-1)
        else:
            qk_softmax = torch.softmax(qk_with_bias, dim=-1)

        if qk_matmul_output_mode == 3:
            qk_output = qk_softmax

        # Compute attention output
        output = torch.matmul(qk_softmax, V)

    # Reshape output back to 3D if input was 3D
    if input_shape_len == 3:
        # output: (batch_size, q_num_heads, q_sequence_length, v_head_size) -> (batch_size, q_sequence_length, hidden_size)
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, q_sequence_length, -1)
        )

    return output, present_key, present_value, qk_output


def _get_quantization_dtype_from_zero_point(
    y_zero_point: Optional[torch.Tensor], output_dtype: int
) -> torch.dtype:
    """Get the quantization output data type from zero point or output_dtype attribute."""
    if output_dtype != 0:
        # Use the specified output_dtype
        return _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[output_dtype]
    elif y_zero_point is not None:
        # Infer from y_zero_point data type
        return y_zero_point.dtype
    else:
        # Default to uint8
        return torch.uint8


def _apply_saturation(values: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Apply saturation to values based on target dtype ranges."""
    if target_dtype in _QUANTIZATION_SATURATION_RANGES:
        min_val, max_val = _QUANTIZATION_SATURATION_RANGES[target_dtype]
        return torch.clamp(values, min_val, max_val)
    # For float8 and other types, clamp based on dtype info
    elif hasattr(torch, "finfo") and target_dtype.is_floating_point:
        try:
            info = torch.finfo(target_dtype)
            return torch.clamp(values, info.min, info.max)
        except (TypeError, ValueError):
            # If finfo is not available for the dtype, return as-is
            return values
    return values


def _round_to_nearest_even(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest even (banker's rounding)."""
    return torch.round(x)


def _compute_blocked_quantization_scale(
    x: torch.Tensor, y_scale: torch.Tensor, axis: int, block_size: int
) -> torch.Tensor:
    """Expand scale tensor for blocked quantization."""
    # For blocked quantization, we need to repeat each scale value block_size times
    # along the specified axis
    x_shape = list(x.shape)

    # Calculate how many times to repeat each scale value
    expanded_scale = y_scale.repeat_interleave(block_size, dim=axis)

    # Trim to match input size if necessary
    if expanded_scale.shape[axis] > x_shape[axis]:
        slices = [slice(None)] * len(x_shape)
        slices[axis] = slice(0, x_shape[axis])
        expanded_scale = expanded_scale[tuple(slices)]

    return expanded_scale


@_onnx_op("QuantizeLinear", 23)
def quantize_linear_23(
    x: torch.Tensor,
    y_scale: torch.Tensor,
    y_zero_point: Optional[torch.Tensor] = None,
    *,
    axis: int = 1,
    block_size: int = 0,
    output_dtype: int = 0,
    precision: int = 0,
    saturate: int = 1,
) -> torch.Tensor:
    """QuantizeLinear-23 https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-23"""

    # Determine output dtype
    target_dtype = _get_quantization_dtype_from_zero_point(y_zero_point, output_dtype)

    # Validate that y_zero_point and target dtype match if both are specified
    if y_zero_point is not None and output_dtype != 0:
        expected_dtype = _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[output_dtype]
        torch._check(
            y_zero_point.dtype == expected_dtype,
            lambda: f"y_zero_point dtype ({y_zero_point.dtype}) must match output_dtype ({expected_dtype})",
        )

    # Handle axis normalization
    if axis < 0:
        axis = len(x.shape) + axis

    torch._check(
        0 <= axis < len(x.shape),
        lambda: f"axis ({axis}) out of range for tensor with {len(x.shape)} dimensions",
    )

    # Determine quantization granularity and prepare scale
    if y_scale.numel() == 1:
        # Per-tensor quantization
        effective_scale = y_scale
    elif len(y_scale.shape) == 1:
        # Per-axis quantization
        torch._check(
            y_scale.shape[0] == x.shape[axis],
            lambda: f"y_scale length ({y_scale.shape[0]}) must match input dimension at axis {axis} ({x.shape[axis]})",
        )
        # Reshape scale to broadcast correctly
        new_shape = [1] * len(x.shape)
        new_shape[axis] = y_scale.shape[0]
        effective_scale = y_scale.view(new_shape)
    else:
        # Blocked quantization
        torch._check(
            block_size > 0,
            lambda: f"block_size must be positive for blocked quantization. block_size: {block_size}",
        )
        effective_scale = _compute_blocked_quantization_scale(
            x, y_scale, axis, block_size
        )

    # Handle zero point
    if y_zero_point is None:
        # Default zero point
        effective_zero_point = torch.tensor(0, dtype=target_dtype, device=x.device)
    else:
        torch._check(
            y_zero_point.shape == y_scale.shape,
            lambda: f"y_zero_point shape ({y_zero_point.shape}) must match y_scale shape ({y_scale.shape})",
        )

        if y_scale.numel() == 1:
            effective_zero_point = y_zero_point
        elif len(y_scale.shape) == 1:
            # Reshape zero point to broadcast correctly
            new_shape = [1] * len(x.shape)
            new_shape[axis] = y_zero_point.shape[0]
            effective_zero_point = y_zero_point.view(new_shape)
        else:
            # Blocked quantization
            effective_zero_point = _compute_blocked_quantization_scale(
                x, y_zero_point, axis, block_size
            )

    # Handle precision casting for division
    if precision != 0:
        division_dtype = _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[precision]
        x_for_div = x.to(division_dtype)
        scale_for_div = effective_scale.to(division_dtype)
    else:
        # Use y_scale precision
        x_for_div = x.to(effective_scale.dtype)
        scale_for_div = effective_scale

    # Perform quantization: y = saturate((x / y_scale) + y_zero_point)
    # Step 1: Divide x by scale
    divided = x_for_div / scale_for_div

    # Step 2: Round to nearest even
    rounded = _round_to_nearest_even(divided)

    # Step 3: Add zero point
    with_zero_point = rounded + effective_zero_point.to(rounded.dtype)

    # Step 4: Apply saturation if enabled
    if saturate == 1:
        saturated = _apply_saturation(with_zero_point, target_dtype)
    else:
        saturated = with_zero_point

    # Step 5: Convert to target dtype
    result = saturated.to(target_dtype)

    return result
