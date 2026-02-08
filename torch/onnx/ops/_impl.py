"""Implementations of ONNX operators as native Torch ops.

NOTE: Fake implementations:
    Refer to https://docs.pytorch.org/docs/stable/library.html#torch.library.register_fake
    for more details on how to create fake kernels.
"""

# flake8: noqa: B950
import math
from collections.abc import Callable
from typing import Optional, TypeVar
from typing_extensions import ParamSpec

import torch
from torch.onnx.ops import _dtype_mappings


# Use ParamSpec for better type preservation instead of bound Callable TypeVar
_P = ParamSpec("_P")
_R = TypeVar("_R")

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


def _onnx_op(
    op_type: str, opset_version: int, fake_impl: Callable[_P, _R]
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator to register an ONNX operator with a custom implementation."""

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        overload = f"opset{opset_version}"
        torch_op = torch.library.custom_op(
            f"onnx::{op_type}.{overload}", mutates_args=()
        )(func)
        ONNX_ATEN_DECOMP_TABLE[getattr(getattr(torch.ops.onnx, op_type), overload)] = (
            func  # type: ignore[assignment]
        )
        torch_op.register_fake(fake_impl)
        return torch_op  # type: ignore[return-value]

    return decorator


# Basic arithmetic operations
def _add_13_fake_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Fake implementation for Add-13 for torch.compile purposes."""
    return torch.add(A, B)


@_onnx_op("Add", 13, _add_13_fake_impl)
def add_13(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Add-13 https://onnx.ai/onnx/operators/onnx__Add.html"""
    return torch.add(A, B)


# Helper functions for Attention
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


def _attention_23_fake_impl(
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
    """Fake implementation for Attention-23 for torch.compile purposes."""
    batch_size = Q.shape[0]

    # Handle 3D vs 4D input shapes
    if len(Q.shape) == 3:
        # 3D input: (batch_size, sequence_length, hidden_size)
        q_sequence_length = Q.shape[1]
        output_shape = Q.shape  # Same shape as Q for 3D output

        # For present_key and present_value, we need 4D shapes
        if past_key is not None:
            present_key_shape = (
                batch_size,
                kv_num_heads,
                past_key.shape[2] + K.shape[1],  # Combined sequence length
                K.shape[2] // kv_num_heads,  # head_size
            )
        else:
            present_key_shape = (
                batch_size,
                kv_num_heads,
                K.shape[1],  # sequence_length
                K.shape[2] // kv_num_heads,  # head_size
            )
        present_value_shape = present_key_shape  # Same shape as present_key

        # QK output shape for 3D input (reshaped to 4D internally)
        qk_output_shape = (
            batch_size,
            q_num_heads,
            q_sequence_length,
            present_key_shape[2],  # kv_sequence_length
        )
    else:
        # 4D input: (batch_size, num_heads, sequence_length, head_size)
        q_sequence_length = Q.shape[2]
        # Same shape as Q for 4D output
        output_shape = Q.shape  # type: ignore[assignment]

        # Handle past key/value concatenation
        if past_key is not None:
            present_key_shape = (
                K.shape[0],  # batch_size
                K.shape[1],  # num_heads
                past_key.shape[2] + K.shape[2],  # Combined sequence length
                K.shape[3],  # head_size
            )
        else:
            present_key_shape = K.shape  # type: ignore[assignment]
        present_value_shape = present_key_shape  # Same shape as present_key

        # QK output shape
        qk_output_shape = (
            Q.shape[0],  # batch_size
            Q.shape[1],  # q_num_heads
            Q.shape[2],  # q_sequence_length
            present_key_shape[2],  # kv_sequence_length
        )

    # Create fake tensors with correct shapes and dtypes
    output = torch.empty(output_shape, dtype=Q.dtype, device=Q.device)
    present_key = torch.empty(present_key_shape, dtype=K.dtype, device=K.device)
    present_value = torch.empty(present_value_shape, dtype=V.dtype, device=V.device)
    qk_output = torch.empty(qk_output_shape, dtype=Q.dtype, device=Q.device)

    return output, present_key, present_value, qk_output


@_onnx_op("Attention", 23, _attention_23_fake_impl)
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
        output = torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
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


# Pooling operations
def _average_pool_11_fake_impl(
    X: torch.Tensor,
    *,
    kernel_shape: list[int],
    strides: Optional[list[int]] = None,
    auto_pad: str = "NOTSET",
    pads: Optional[list[int]] = None,
    count_include_pad: bool = False,
) -> torch.Tensor:
    """Fake implementation for AveragePool-11 for torch.compile purposes."""
    if strides is None:
        strides = [1] * len(kernel_shape)

    # Handle padding
    if auto_pad == "NOTSET":
        padding = [0] * len(kernel_shape) if pads is None else pads[: len(kernel_shape)]
    else:
        # Auto padding not fully implemented, use default
        padding = [0] * len(kernel_shape)

    if len(kernel_shape) == 2:
        return torch.nn.functional.avg_pool2d(
            X,
            kernel_size=kernel_shape,
            stride=strides,
            padding=padding,
            count_include_pad=count_include_pad,
        )
    elif len(kernel_shape) == 1:
        return torch.nn.functional.avg_pool1d(
            X,
            kernel_size=kernel_shape[0],
            stride=strides[0],
            padding=padding[0],
            count_include_pad=count_include_pad,
        )
    elif len(kernel_shape) == 3:
        return torch.nn.functional.avg_pool3d(
            X,
            kernel_size=kernel_shape,
            stride=strides,
            padding=padding,
            count_include_pad=count_include_pad,
        )
    else:
        raise ValueError(f"Unsupported kernel_shape length: {len(kernel_shape)}")


@_onnx_op("AveragePool", 11, _average_pool_11_fake_impl)
def average_pool_11(
    X: torch.Tensor,
    *,
    kernel_shape: list[int],
    strides: Optional[list[int]] = None,
    auto_pad: str = "NOTSET",
    pads: Optional[list[int]] = None,
    count_include_pad: bool = False,
) -> torch.Tensor:
    """AveragePool-11 https://onnx.ai/onnx/operators/onnx__AveragePool.html"""
    if strides is None:
        strides = [1] * len(kernel_shape)

    # Handle padding
    if auto_pad == "NOTSET":
        padding = [0] * len(kernel_shape) if pads is None else pads[: len(kernel_shape)]
    else:
        # Auto padding not fully implemented, use default
        padding = [0] * len(kernel_shape)

    if len(kernel_shape) == 2:
        return torch.nn.functional.avg_pool2d(
            X,
            kernel_size=kernel_shape,
            stride=strides,
            padding=padding,
            count_include_pad=count_include_pad,
        )
    elif len(kernel_shape) == 1:
        return torch.nn.functional.avg_pool1d(
            X,
            kernel_size=kernel_shape[0],
            stride=strides[0],
            padding=padding[0],
            count_include_pad=count_include_pad,
        )
    elif len(kernel_shape) == 3:
        return torch.nn.functional.avg_pool3d(
            X,
            kernel_size=kernel_shape,
            stride=strides,
            padding=padding,
            count_include_pad=count_include_pad,
        )
    else:
        raise ValueError(f"Unsupported kernel_shape length: {len(kernel_shape)}")


def _cast_13_fake_impl(input: torch.Tensor, *, to: int) -> torch.Tensor:
    """Fake implementation for Cast-13 for torch.compile purposes."""
    target_dtype = _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE.get(to, input.dtype)
    return torch.empty_like(input, dtype=target_dtype)


@_onnx_op("Cast", 13, _cast_13_fake_impl)
def cast_13(input: torch.Tensor, *, to: int) -> torch.Tensor:
    """Cast-13 https://onnx.ai/onnx/operators/onnx__Cast.html"""
    target_dtype = _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE.get(to)
    torch._check(
        target_dtype is not None,
        lambda: f"Unsupported ONNX data type: {to}",
    )
    return input.to(target_dtype)


def _concat_13_fake_impl(*tensors: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Fake implementation for Concat-13 for torch.compile purposes."""
    return tensors[0].clone()


@_onnx_op("Concat", 13, _concat_13_fake_impl)
def concat_13(*tensors: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Concat-13 https://onnx.ai/onnx/operators/onnx__Concat.html"""
    return torch.cat(tensors, dim=axis)


# Convolution operation
def _conv_11_fake_impl(
    X: torch.Tensor,
    W: torch.Tensor,
    B: Optional[torch.Tensor] = None,
    *,
    auto_pad: str = "NOTSET",
    dilations: Optional[list[int]] = None,
    group: int = 1,
    kernel_shape: Optional[list[int]] = None,
    pads: Optional[list[int]] = None,
    strides: Optional[list[int]] = None,
) -> torch.Tensor:
    """Fake implementation for Conv-11 for torch.compile purposes."""
    spatial_dims = W.ndim - 2

    if strides is None:
        strides = [1] * spatial_dims
    if dilations is None:
        dilations = [1] * spatial_dims

    # Handle padding
    if auto_pad == "NOTSET":
        padding = [0] * spatial_dims if pads is None else pads[:spatial_dims]
    else:
        # Auto padding not fully implemented, use default
        padding = [0] * spatial_dims

    if spatial_dims == 2:
        return torch.nn.functional.conv2d(
            X, W, B, stride=strides, padding=padding, dilation=dilations, groups=group
        )
    elif spatial_dims == 1:
        return torch.nn.functional.conv1d(
            X,
            W,
            B,
            stride=strides[0],
            padding=padding[0],
            dilation=dilations[0],
            groups=group,
        )
    elif spatial_dims == 3:
        return torch.nn.functional.conv3d(
            X, W, B, stride=strides, padding=padding, dilation=dilations, groups=group
        )
    else:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")


@_onnx_op("Conv", 11, _conv_11_fake_impl)
def conv_11(
    X: torch.Tensor,
    W: torch.Tensor,
    B: Optional[torch.Tensor] = None,
    *,
    auto_pad: str = "NOTSET",
    dilations: Optional[list[int]] = None,
    group: int = 1,
    kernel_shape: Optional[list[int]] = None,
    pads: Optional[list[int]] = None,
    strides: Optional[list[int]] = None,
) -> torch.Tensor:
    """Conv-11 https://onnx.ai/onnx/operators/onnx__Conv.html"""
    spatial_dims = W.ndim - 2

    if strides is None:
        strides = [1] * spatial_dims
    if dilations is None:
        dilations = [1] * spatial_dims

    # Handle padding
    if auto_pad == "NOTSET":
        padding = [0] * spatial_dims if pads is None else pads[:spatial_dims]
    else:
        # Auto padding not fully implemented, use default
        padding = [0] * spatial_dims

    if spatial_dims == 2:
        return torch.nn.functional.conv2d(
            X, W, B, stride=strides, padding=padding, dilation=dilations, groups=group
        )
    elif spatial_dims == 1:
        return torch.nn.functional.conv1d(
            X,
            W,
            B,
            stride=strides[0],
            padding=padding[0],
            dilation=dilations[0],
            groups=group,
        )
    elif spatial_dims == 3:
        return torch.nn.functional.conv3d(
            X, W, B, stride=strides, padding=padding, dilation=dilations, groups=group
        )
    else:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")


def _dft_20_fake_impl(
    input: torch.Tensor,
    dft_length: Optional[torch.Tensor] = None,
    *,
    axis: int = -1,
    inverse: bool = False,
    onesided: bool = False,
) -> torch.Tensor:
    """Fake implementation for DFT-20 for torch.compile purposes."""
    # Normalize axis to positive index
    normalized_axis = (
        axis if axis >= 0 else input.ndim + axis - 1
    )  # -1 for the complex dimension

    # Determine output shape along the FFT axis
    if inverse:
        if onesided:
            # irfft output: real-valued, then converted to [..., 2] format
            if dft_length is not None:
                n = dft_length.item()
            else:
                # Default irfft size
                n = 2 * (input.shape[normalized_axis] - 1)
            output_shape = list(input.shape)
            output_shape[normalized_axis] = n
        else:
            # ifft output: same size or dft_length
            output_shape = list(input.shape)
            if dft_length is not None:
                output_shape[normalized_axis] = dft_length.item()
    else:
        if onesided:
            # rfft output: (n // 2) + 1 for real input
            if dft_length is not None:
                n = dft_length.item()
            else:
                n = input.shape[normalized_axis]
            output_size = (n // 2) + 1
            output_shape = list(input.shape)
            output_shape[normalized_axis] = output_size
        else:
            # fft output: same size or dft_length
            output_shape = list(input.shape)
            if dft_length is not None:
                output_shape[normalized_axis] = dft_length.item()

    return torch.empty(output_shape, dtype=input.dtype, device=input.device)


@_onnx_op("DFT", 20, _dft_20_fake_impl)
def dft_20(
    input: torch.Tensor,
    dft_length: Optional[torch.Tensor] = None,
    *,
    axis: int = -1,
    inverse: bool = False,
    onesided: bool = False,
) -> torch.Tensor:
    """DFT-20 https://onnx.ai/onnx/operators/onnx__DFT.html"""
    # ONNX DFT expects input shape [..., 2] where last dimension is [real, imag]
    # Convert to complex tensor
    torch._check(
        input.shape[-1] == 2,
        lambda: f"DFT input must have last dimension of size 2 (real, imag). Got shape {input.shape}",
    )

    complex_input = torch.view_as_complex(input.contiguous())

    # Determine FFT length
    n = None
    if dft_length is not None:
        n = dft_length.item()

    # Perform FFT or IFFT
    if inverse:
        if onesided:
            result = torch.fft.irfft(complex_input, n=n, dim=axis)
            # Convert real result back to [..., 2] format with zero imaginary part
            result_complex = torch.view_as_real(
                torch.complex(result, torch.zeros_like(result))
            )
            return result_complex
        else:
            result = torch.fft.ifft(complex_input, n=n, dim=axis)
    else:
        if onesided:
            # For onesided forward FFT, input should be real-valued
            # But ONNX format is [..., 2], so extract real part
            real_input = input[..., 0]
            result = torch.fft.rfft(real_input, n=n, dim=axis)
        else:
            result = torch.fft.fft(complex_input, n=n, dim=axis)

    # Convert back to [..., 2] format
    return torch.view_as_real(result)


def _div_13_fake_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Fake implementation for Div-13 for torch.compile purposes."""
    return torch.div(A, B)


@_onnx_op("Div", 13, _div_13_fake_impl)
def div_13(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Div-13 https://onnx.ai/onnx/operators/onnx__Div.html"""
    return torch.div(A, B)


def _expand_13_fake_impl(input: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Fake implementation for Expand-13 for torch.compile purposes."""
    target_shape = shape.tolist()
    return torch.empty(target_shape, dtype=input.dtype, device=input.device)


@_onnx_op("Expand", 13, _expand_13_fake_impl)
def expand_13(input: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Expand-13 https://onnx.ai/onnx/operators/onnx__Expand.html"""
    target_shape = shape.tolist()
    return input.expand(target_shape)


def _gather_13_fake_impl(
    data: torch.Tensor, indices: torch.Tensor, *, axis: int = 0
) -> torch.Tensor:
    """Fake implementation for Gather-13 for torch.compile purposes."""
    return torch.gather(data, axis, indices)


@_onnx_op("Gather", 13, _gather_13_fake_impl)
def gather_13(
    data: torch.Tensor, indices: torch.Tensor, *, axis: int = 0
) -> torch.Tensor:
    """Gather-13 https://onnx.ai/onnx/operators/onnx__Gather.html"""
    return torch.gather(data, axis, indices)


def _matmul_13_fake_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Fake implementation for MatMul-13 for torch.compile purposes."""
    return torch.matmul(A, B)


@_onnx_op("MatMul", 13, _matmul_13_fake_impl)
def matmul_13(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """MatMul-13 https://onnx.ai/onnx/operators/onnx__MatMul.html"""
    return torch.matmul(A, B)


# Element-wise Min/Max operations
def _max_13_fake_impl(*data_tensors: torch.Tensor) -> torch.Tensor:
    """Fake implementation for Max-13 for torch.compile purposes."""
    if not data_tensors:
        raise ValueError("Max requires at least one input")
    result_shape = data_tensors[0].shape
    for tensor in data_tensors[1:]:
        result_shape = torch.broadcast_shapes(result_shape, tensor.shape)
    return torch.empty(
        result_shape, dtype=data_tensors[0].dtype, device=data_tensors[0].device
    )


@_onnx_op("Max", 13, _max_13_fake_impl)
def max_13(*data_tensors: torch.Tensor) -> torch.Tensor:
    """Max-13 https://onnx.ai/onnx/operators/onnx__Max.html"""
    result = data_tensors[0]
    for tensor in data_tensors[1:]:
        result = torch.maximum(result, tensor)
    return result


def _max_pool_12_fake_impl(
    X: torch.Tensor,
    *,
    kernel_shape: list[int],
    strides: Optional[list[int]] = None,
    auto_pad: str = "NOTSET",
    pads: Optional[list[int]] = None,
    dilations: Optional[list[int]] = None,
    storage_order: int = 0,
) -> torch.Tensor:
    """Fake implementation for MaxPool-12 for torch.compile purposes."""
    if strides is None:
        strides = [1] * len(kernel_shape)
    if dilations is None:
        dilations = [1] * len(kernel_shape)

    # Handle padding
    if auto_pad == "NOTSET":
        padding = [0] * len(kernel_shape) if pads is None else pads[: len(kernel_shape)]
    else:
        # Auto padding not fully implemented, use default
        padding = [0] * len(kernel_shape)

    if len(kernel_shape) == 2:
        return torch.nn.functional.max_pool2d(
            X,
            kernel_size=kernel_shape,
            stride=strides,
            padding=padding,
            dilation=dilations,
        )
    elif len(kernel_shape) == 1:
        return torch.nn.functional.max_pool1d(
            X,
            kernel_size=kernel_shape[0],
            stride=strides[0],
            padding=padding[0],
            dilation=dilations[0],
        )
    elif len(kernel_shape) == 3:
        return torch.nn.functional.max_pool3d(
            X,
            kernel_size=kernel_shape,
            stride=strides,
            padding=padding,
            dilation=dilations,
        )
    else:
        raise ValueError(f"Unsupported kernel_shape length: {len(kernel_shape)}")


@_onnx_op("MaxPool", 12, _max_pool_12_fake_impl)
def max_pool_12(
    X: torch.Tensor,
    *,
    kernel_shape: list[int],
    strides: Optional[list[int]] = None,
    auto_pad: str = "NOTSET",
    pads: Optional[list[int]] = None,
    dilations: Optional[list[int]] = None,
    storage_order: int = 0,
) -> torch.Tensor:
    """MaxPool-12 https://onnx.ai/onnx/operators/onnx__MaxPool.html"""
    if strides is None:
        strides = [1] * len(kernel_shape)
    if dilations is None:
        dilations = [1] * len(kernel_shape)

    # Handle padding
    if auto_pad == "NOTSET":
        padding = [0] * len(kernel_shape) if pads is None else pads[: len(kernel_shape)]
    else:
        # Auto padding not fully implemented, use default
        padding = [0] * len(kernel_shape)

    if len(kernel_shape) == 2:
        return torch.nn.functional.max_pool2d(
            X,
            kernel_size=kernel_shape,
            stride=strides,
            padding=padding,
            dilation=dilations,
        )
    elif len(kernel_shape) == 1:
        return torch.nn.functional.max_pool1d(
            X,
            kernel_size=kernel_shape[0],
            stride=strides[0],
            padding=padding[0],
            dilation=dilations[0],
        )
    elif len(kernel_shape) == 3:
        return torch.nn.functional.max_pool3d(
            X,
            kernel_size=kernel_shape,
            stride=strides,
            padding=padding,
            dilation=dilations,
        )
    else:
        raise ValueError(f"Unsupported kernel_shape length: {len(kernel_shape)}")


def _min_13_fake_impl(*data_tensors: torch.Tensor) -> torch.Tensor:
    """Fake implementation for Min-13 for torch.compile purposes."""
    if not data_tensors:
        raise ValueError("Min requires at least one input")
    result_shape = data_tensors[0].shape
    for tensor in data_tensors[1:]:
        result_shape = torch.broadcast_shapes(result_shape, tensor.shape)
    return torch.empty(
        result_shape, dtype=data_tensors[0].dtype, device=data_tensors[0].device
    )


@_onnx_op("Min", 13, _min_13_fake_impl)
def min_13(*data_tensors: torch.Tensor) -> torch.Tensor:
    """Min-13 https://onnx.ai/onnx/operators/onnx__Min.html"""
    result = data_tensors[0]
    for tensor in data_tensors[1:]:
        result = torch.minimum(result, tensor)
    return result


def _mul_13_fake_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Fake implementation for Mul-13 for torch.compile purposes."""
    return torch.mul(A, B)


@_onnx_op("Mul", 13, _mul_13_fake_impl)
def mul_13(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Mul-13 https://onnx.ai/onnx/operators/onnx__Mul.html"""
    return torch.mul(A, B)


def _pow_13_fake_impl(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Fake implementation for Pow-13 for torch.compile purposes."""
    return torch.empty(
        torch.broadcast_shapes(X.shape, Y.shape), dtype=X.dtype, device=X.device
    )


@_onnx_op("Pow", 13, _pow_13_fake_impl)
def pow_13(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Pow-13 https://onnx.ai/onnx/operators/onnx__Pow.html"""
    return torch.pow(X, Y)


def _range_11_fake_impl(
    start: torch.Tensor, limit: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    """Fake implementation for Range-11 for torch.compile purposes."""
    # Calculate number of elements: ceil((limit - start) / delta)
    start_val = start.item()
    limit_val = limit.item()
    delta_val = delta.item()
    num_elements = max(0, int((limit_val - start_val) / delta_val))
    return torch.empty(num_elements, dtype=start.dtype, device=start.device)


@_onnx_op("Range", 11, _range_11_fake_impl)
def range_11(
    start: torch.Tensor, limit: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    """Range-11 https://onnx.ai/onnx/operators/onnx__Range.html"""
    start_val = start.item()
    limit_val = limit.item()
    delta_val = delta.item()
    return torch.arange(
        start_val, limit_val, delta_val, dtype=start.dtype, device=start.device
    )


# Reduction operations
def _reduce_max_13_fake_impl(
    data: torch.Tensor, *, axes: Optional[list[int]] = None, keepdims: bool = True
) -> torch.Tensor:
    """Fake implementation for ReduceMax-13 for torch.compile purposes."""
    if axes is None:
        if keepdims:
            return torch.empty([1] * data.ndim, dtype=data.dtype, device=data.device)
        else:
            return torch.empty((), dtype=data.dtype, device=data.device)
    output_shape = list(data.shape)
    for axis in sorted(axes, reverse=True):
        if keepdims:
            output_shape[axis] = 1
        else:
            output_shape.pop(axis)
    return torch.empty(output_shape, dtype=data.dtype, device=data.device)


@_onnx_op("ReduceMax", 13, _reduce_max_13_fake_impl)
def reduce_max_13(
    data: torch.Tensor, *, axes: Optional[list[int]] = None, keepdims: bool = True
) -> torch.Tensor:
    """ReduceMax-13 https://onnx.ai/onnx/operators/onnx__ReduceMax.html"""
    if axes is None:
        return torch.max(data).view(1) if keepdims else torch.max(data)
    result = data
    for axis in sorted(axes, reverse=True):
        result = torch.max(result, dim=axis, keepdim=keepdims).values
    return result


def _reduce_mean_13_fake_impl(
    data: torch.Tensor, *, axes: Optional[list[int]] = None, keepdims: bool = True
) -> torch.Tensor:
    """Fake implementation for ReduceMean-13 for torch.compile purposes."""
    if axes is None:
        if keepdims:
            return torch.empty([1] * data.ndim, dtype=data.dtype, device=data.device)
        else:
            return torch.empty((), dtype=data.dtype, device=data.device)
    output_shape = list(data.shape)
    for axis in sorted(axes, reverse=True):
        if keepdims:
            output_shape[axis] = 1
        else:
            output_shape.pop(axis)
    return torch.empty(output_shape, dtype=data.dtype, device=data.device)


@_onnx_op("ReduceMean", 13, _reduce_mean_13_fake_impl)
def reduce_mean_13(
    data: torch.Tensor, *, axes: Optional[list[int]] = None, keepdims: bool = True
) -> torch.Tensor:
    """ReduceMean-13 https://onnx.ai/onnx/operators/onnx__ReduceMean.html"""
    if axes is None:
        axes = list(range(data.ndim))
    return torch.mean(data, dim=axes, keepdim=keepdims)


def _reduce_min_13_fake_impl(
    data: torch.Tensor, *, axes: Optional[list[int]] = None, keepdims: bool = True
) -> torch.Tensor:
    """Fake implementation for ReduceMin-13 for torch.compile purposes."""
    if axes is None:
        if keepdims:
            return torch.empty([1] * data.ndim, dtype=data.dtype, device=data.device)
        else:
            return torch.empty((), dtype=data.dtype, device=data.device)
    output_shape = list(data.shape)
    for axis in sorted(axes, reverse=True):
        if keepdims:
            output_shape[axis] = 1
        else:
            output_shape.pop(axis)
    return torch.empty(output_shape, dtype=data.dtype, device=data.device)


@_onnx_op("ReduceMin", 13, _reduce_min_13_fake_impl)
def reduce_min_13(
    data: torch.Tensor, *, axes: Optional[list[int]] = None, keepdims: bool = True
) -> torch.Tensor:
    """ReduceMin-13 https://onnx.ai/onnx/operators/onnx__ReduceMin.html"""
    if axes is None:
        return torch.min(data).view(1) if keepdims else torch.min(data)
    result = data
    for axis in sorted(axes, reverse=True):
        result = torch.min(result, dim=axis, keepdim=keepdims).values
    return result


# Activation functions
def _relu_13_fake_impl(X: torch.Tensor) -> torch.Tensor:
    """Fake implementation for Relu-13 for torch.compile purposes."""
    return X.clone()


@_onnx_op("Relu", 13, _relu_13_fake_impl)
def relu_13(X: torch.Tensor) -> torch.Tensor:
    """Relu-13 https://onnx.ai/onnx/operators/onnx__Relu.html"""
    return torch.relu(X)


# Tensor manipulation operations
def _reshape_13_fake_impl(
    data: torch.Tensor, shape: torch.Tensor, *, allowzero: bool = False
) -> torch.Tensor:
    """Fake implementation for Reshape-13 for torch.compile purposes."""
    target_shape = shape.tolist()
    if not allowzero:
        # Replace 0 with corresponding input dimension
        target_shape = [
            target_shape[i] if target_shape[i] != 0 else data.shape[i]
            for i in range(len(target_shape))
        ]
    return torch.reshape(data, target_shape)


@_onnx_op("Reshape", 13, _reshape_13_fake_impl)
def reshape_13(
    data: torch.Tensor, shape: torch.Tensor, *, allowzero: bool = False
) -> torch.Tensor:
    """Reshape-13 https://onnx.ai/onnx/operators/onnx__Reshape.html"""
    target_shape = shape.tolist()
    if not allowzero:
        # Replace 0 with corresponding input dimension
        target_shape = [
            target_shape[i] if target_shape[i] != 0 else data.shape[i]
            for i in range(len(target_shape))
        ]
    return torch.reshape(data, target_shape)


def _rotary_embedding_23_fake_impl(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    *,
    interleaved: bool = False,
    num_heads: int = 0,
    rotary_embedding_dim: int = 0,
) -> torch.Tensor:
    """Fake implementation for RotaryEmbedding-23 for torch.compile purposes."""
    return x.clone()


@_onnx_op("RotaryEmbedding", 23, _rotary_embedding_23_fake_impl)
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
    # x has shape (batch_size, num_heads, sequence_length, head_size)
    # or (batch_size, sequence_length, hidden_size)
    input_shape = x.shape
    input_rank = len(input_shape)
    batch_size = input_shape[0]
    sequence_length = input_shape[-2]

    # Validate position_ids and caches match x
    if position_ids is not None:
        torch._check(
            position_ids.dim() == 2,
            lambda: f"position_ids must be 2D when provided. Received shape {position_ids.shape}",
        )
        torch._check(
            position_ids.shape[0] == batch_size,
            lambda: f"position_ids first dim (batch) must match x.shape[0] ({batch_size}). Received {position_ids.shape[0]}",
        )
        torch._check(
            position_ids.shape[1] == sequence_length,
            lambda: f"position_ids second dim (sequence) must match x.shape[-2] ({sequence_length}). Received {position_ids.shape[1]}",
        )
        torch._check(
            cos_cache.dim() == 2 and sin_cache.dim() == 2,
            lambda: "cos_cache/sin_cache must be 2D when position_ids is provided. "
            f"Received cos_cache shape {cos_cache.shape}, sin_cache shape {sin_cache.shape}",
        )
    else:
        torch._check(
            cos_cache.dim() == 3 and sin_cache.dim() == 3,
            lambda: "cos_cache/sin_cache must be 3D when position_ids is not provided. "
            f"Received cos_cache shape {cos_cache.shape}, sin_cache shape {sin_cache.shape}",
        )

    # First ensure x has shape [batch_size, num_heads, seq_len, head_size]
    # So that the rotation logic can be shared with reshaped 3D inputs
    if input_rank == 4:
        # Reshape from (batch_size, num_heads, seq_len, head_size)
        # to [batch_size, seq_len, num_heads, head_size]
        x = torch.permute(x, (0, 2, 1, 3))
    elif input_rank == 3:
        torch._check(
            num_heads != 0,
            lambda: f"num_heads must be provided for 3D inputs. Received input tensor with shape {input_shape}",
        )
        hidden_size = input_shape[2]
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
        cos = cos_cache  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
        sin = sin_cache  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]

    torch._check(
        cos.shape[0] == batch_size and cos.shape[1] == sequence_length,
        lambda: f"cos has shape {cos.shape} but expected (batch={batch_size}, seq={sequence_length}, ...)",
    )
    torch._check(
        sin.shape[0] == batch_size and sin.shape[1] == sequence_length,
        lambda: f"sin has shape {sin.shape} but expected (batch={batch_size}, seq={sequence_length}, ...)",
    )
    torch._check(
        cos.shape[-1] == rotary_embedding_dim_half,
        lambda: f"Last dimension of cos cache ({cos.shape[-1]}) should match rotary_embedding_dim/2 ({rotary_embedding_dim_half}).",
    )
    torch._check(
        sin.shape[-1] == rotary_embedding_dim_half,
        lambda: f"Last dimension of sin cache ({sin.shape[-1]}) should match rotary_embedding_dim/2 ({rotary_embedding_dim_half}).",
    )
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
    if input_rank == 3:
        return torch.reshape(output, input_shape)

    # Return the dimensions to the original order
    return torch.permute(output, (0, 2, 1, 3))


# Shape operations
def _shape_13_fake_impl(data: torch.Tensor) -> torch.Tensor:
    """Fake implementation for Shape-13 for torch.compile purposes."""
    return torch.empty(len(data.shape), dtype=torch.int64, device=data.device)


@_onnx_op("Shape", 13, _shape_13_fake_impl)
def shape_13(data: torch.Tensor) -> torch.Tensor:
    """Shape-13 https://onnx.ai/onnx/operators/onnx__Shape.html"""
    return torch.tensor(data.shape, device=data.device)


# Sigmoid operations
def _sigmoid_13_fake_impl(X: torch.Tensor) -> torch.Tensor:
    """Fake implementation for Sigmoid-13 for torch.compile purposes."""
    return X.clone()


@_onnx_op("Sigmoid", 13, _sigmoid_13_fake_impl)
def sigmoid_13(X: torch.Tensor) -> torch.Tensor:
    """Sigmoid-13 https://onnx.ai/onnx/operators/onnx__Sigmoid.html"""
    return torch.sigmoid(X)


# Slice operations
def _slice_13_fake_impl(
    data: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    axes: Optional[torch.Tensor] = None,
    steps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fake implementation for Slice-13 for torch.compile purposes."""
    starts_list = starts.tolist()
    ends_list = ends.tolist()
    axes_list = axes.tolist() if axes is not None else list(range(len(starts_list)))
    steps_list = steps.tolist() if steps is not None else [1] * len(starts_list)

    output_shape = list(data.shape)
    for i, axis in enumerate(axes_list):
        dim_size = data.shape[axis]
        start = max(0, min(starts_list[i], dim_size))
        end = max(0, min(ends_list[i], dim_size))
        step = steps_list[i]
        output_shape[axis] = max(0, (end - start + step - 1) // step)

    return torch.empty(output_shape, dtype=data.dtype, device=data.device)


@_onnx_op("Slice", 13, _slice_13_fake_impl)
def slice_13(
    data: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    axes: Optional[torch.Tensor] = None,
    steps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Slice-13 https://onnx.ai/onnx/operators/onnx__Slice.html"""
    starts_list = starts.tolist()
    ends_list = ends.tolist()
    axes_list = axes.tolist() if axes is not None else list(range(len(starts_list)))
    steps_list = steps.tolist() if steps is not None else [1] * len(starts_list)

    # Build slice objects for each dimension
    slices = [slice(None)] * data.ndim
    for i, axis in enumerate(axes_list):
        slices[axis] = slice(starts_list[i], ends_list[i], steps_list[i])

    return data[tuple(slices)]


# Split operations
def _split_13_fake_impl(
    input: torch.Tensor,
    split: Optional[torch.Tensor] = None,
    *,
    axis: int = 0,
    num_outputs: Optional[int] = None,
) -> tuple[torch.Tensor, ...]:
    """Fake implementation for Split-13 for torch.compile purposes."""
    if split is not None:
        split_sizes = split.tolist()
        outputs = []
        for size in split_sizes:
            output_shape = list(input.shape)
            output_shape[axis] = size
            outputs.append(
                torch.empty(output_shape, dtype=input.dtype, device=input.device)
            )
        return tuple(outputs)
    else:
        if num_outputs is None:
            num_outputs = input.shape[axis]
        size = input.shape[axis] // num_outputs
        output_shape = list(input.shape)
        output_shape[axis] = size
        return tuple(
            torch.empty(output_shape, dtype=input.dtype, device=input.device)
            for _ in range(num_outputs)
        )


@_onnx_op("Split", 13, _split_13_fake_impl)
def split_13(
    input: torch.Tensor,
    split: Optional[torch.Tensor] = None,
    *,
    axis: int = 0,
    num_outputs: Optional[int] = None,
) -> tuple[torch.Tensor, ...]:
    """Split-13 https://onnx.ai/onnx/operators/onnx__Split.html"""
    if split is not None:
        split_sizes = split.tolist()
        return tuple(torch.split(input, split_sizes, dim=axis))
    else:
        # Equal split
        if num_outputs is None:
            num_outputs = input.shape[axis]
        size = input.shape[axis] // num_outputs
        return tuple(torch.split(input, size, dim=axis))


# Squeeze operations
def _squeeze_13_fake_impl(
    data: torch.Tensor, axes: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Fake implementation for Squeeze-13 for torch.compile purposes."""
    if axes is None:
        # Squeeze all dimensions of size 1
        output_shape = [s for s in data.shape if s != 1]
    else:
        axes_list = sorted(axes.tolist(), reverse=True)
        output_shape = list(data.shape)
        for axis in axes_list:
            if output_shape[axis] == 1:
                output_shape.pop(axis)
    return torch.empty(output_shape, dtype=data.dtype, device=data.device)


@_onnx_op("Squeeze", 13, _squeeze_13_fake_impl)
def squeeze_13(data: torch.Tensor, axes: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Squeeze-13 https://onnx.ai/onnx/operators/onnx__Squeeze.html"""
    if axes is None:
        # Squeeze all dimensions of size 1
        return torch.squeeze(data)
    else:
        # Squeeze specific axes
        axes_list = sorted(axes.tolist(), reverse=True)
        result = data
        for axis in axes_list:
            result = torch.squeeze(result, dim=axis)
        return result


# Sub operations
def _sub_13_fake_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Fake implementation for Sub-13 for torch.compile purposes."""
    return torch.sub(A, B)


@_onnx_op("Sub", 13, _sub_13_fake_impl)
def sub_13(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Sub-13 https://onnx.ai/onnx/operators/onnx__Sub.html"""
    return torch.sub(A, B)


# Transpose operations
def _transpose_13_fake_impl(
    data: torch.Tensor, *, perm: Optional[list[int]] = None
) -> torch.Tensor:
    """Fake implementation for Transpose-13 for torch.compile purposes."""
    if perm is None:
        perm = list(range(data.ndim - 1, -1, -1))
    output_shape = [data.shape[i] for i in perm]
    return torch.empty(output_shape, dtype=data.dtype, device=data.device)


@_onnx_op("Transpose", 13, _transpose_13_fake_impl)
def transpose_13(
    data: torch.Tensor, *, perm: Optional[list[int]] = None
) -> torch.Tensor:
    """Transpose-13 https://onnx.ai/onnx/operators/onnx__Transpose.html"""
    if perm is None:
        # Default behavior: reverse the dimensions
        perm = list(range(data.ndim - 1, -1, -1))
    return torch.permute(data, perm)


# Unsqueeze operations
def _unsqueeze_13_fake_impl(data: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    """Fake implementation for Unsqueeze-13 for torch.compile purposes."""
    axes_list = sorted(axes.tolist())
    result = data
    for axis in axes_list:
        result = torch.unsqueeze(result, axis)
    return result


@_onnx_op("Unsqueeze", 13, _unsqueeze_13_fake_impl)
def unsqueeze_13(data: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    """Unsqueeze-13 https://onnx.ai/onnx/operators/onnx__Unsqueeze.html"""
    axes_list = sorted(axes.tolist())
    result = data
    for axis in axes_list:
        result = torch.unsqueeze(result, axis)
    return result


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


def _attention_23_fake_impl(
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
    """Fake implementation for Attention-23 for torch.compile purposes."""
    batch_size = Q.shape[0]

    # Handle 3D vs 4D input shapes
    if len(Q.shape) == 3:
        # 3D input: (batch_size, sequence_length, hidden_size)
        q_sequence_length = Q.shape[1]
        output_shape = Q.shape  # Same shape as Q for 3D output

        # For present_key and present_value, we need 4D shapes
        if past_key is not None:
            present_key_shape = (
                batch_size,
                kv_num_heads,
                past_key.shape[2] + K.shape[1],  # Combined sequence length
                K.shape[2] // kv_num_heads,  # head_size
            )
        else:
            present_key_shape = (
                batch_size,
                kv_num_heads,
                K.shape[1],  # sequence_length
                K.shape[2] // kv_num_heads,  # head_size
            )
        present_value_shape = present_key_shape  # Same shape as present_key

        # QK output shape for 3D input (reshaped to 4D internally)
        qk_output_shape = (
            batch_size,
            q_num_heads,
            q_sequence_length,
            present_key_shape[2],  # kv_sequence_length
        )
    else:
        # 4D input: (batch_size, num_heads, sequence_length, head_size)
        q_sequence_length = Q.shape[2]
        # Same shape as Q for 4D output
        output_shape = Q.shape  # type: ignore[assignment]

        # Handle past key/value concatenation
        if past_key is not None:
            present_key_shape = (
                K.shape[0],  # batch_size
                K.shape[1],  # num_heads
                past_key.shape[2] + K.shape[2],  # Combined sequence length
                K.shape[3],  # head_size
            )
        else:
            present_key_shape = K.shape  # type: ignore[assignment]
        present_value_shape = present_key_shape  # Same shape as present_key

        # QK output shape
        qk_output_shape = (
            Q.shape[0],  # batch_size
            Q.shape[1],  # q_num_heads
            Q.shape[2],  # q_sequence_length
            present_key_shape[2],  # kv_sequence_length
        )

    # Create fake tensors with correct shapes and dtypes
    output = torch.empty(output_shape, dtype=Q.dtype, device=Q.device)
    present_key = torch.empty(present_key_shape, dtype=K.dtype, device=K.device)
    present_value = torch.empty(present_value_shape, dtype=V.dtype, device=V.device)
    qk_output = torch.empty(qk_output_shape, dtype=Q.dtype, device=Q.device)

    return output, present_key, present_value, qk_output


@_onnx_op("Attention", 23, _attention_23_fake_impl)
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
        output = torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
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
