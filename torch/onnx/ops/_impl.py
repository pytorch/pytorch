import typing
from typing import Callable, Optional

import torch
from torch.nn.attention import flex_attention
from torch.onnx.ops import _dtype_mappings


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
    enable_gqa = current_q_num_heads != current_kv_num_heads
    if enable_gqa:
        assert current_q_num_heads % current_kv_num_heads == 0, (
            f"q_num_heads ({current_q_num_heads}) must be divisible by kv_num_heads ({current_kv_num_heads}) for GQA"
        )
        repeat_factor = current_q_num_heads // current_kv_num_heads
        K_for_qk = K.repeat_interleave(repeat_factor, dim=1)

    scale_factor = scale if scale is not None else (1.0 / (Q.shape[3] ** 0.5))
    return torch.matmul(Q, K_for_qk.transpose(-2, -1)) * scale_factor


@_onnx_op("Attention", 23)
def attention(
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

    # Store original input shape to determine output shape
    input_shape_len = len(Q.shape)
    batch_size = Q.shape[0]

    # Reshape 3D inputs to 4D format
    if len(Q.shape) == 3:
        assert q_num_heads != 0 and kv_num_heads != 0, (
            "q_num_heads and kv_num_heads must be provided for 3D inputs"
        )

        # Q: (batch_size, q_sequence_length, q_hidden_size) -> (batch_size, q_num_heads, q_sequence_length, head_size)
        q_sequence_length = Q.shape[1]
        q_hidden_size = Q.shape[2]
        head_size_q = q_hidden_size // q_num_heads
        Q = (
            Q.view(batch_size, q_sequence_length, q_num_heads, head_size_q)
            .transpose(1, 2)
            .contiguous()
        )

        # K: (batch_size, kv_sequence_length, k_hidden_size) -> (batch_size, kv_num_heads, kv_sequence_length, head_size)
        kv_sequence_length = K.shape[1]
        k_hidden_size = K.shape[2]
        head_size_k = k_hidden_size // kv_num_heads
        K = (
            K.view(batch_size, kv_sequence_length, kv_num_heads, head_size_k)
            .transpose(1, 2)
            .contiguous()
        )

        # V: (batch_size, kv_sequence_length, v_hidden_size) -> (batch_size, kv_num_heads, kv_sequence_length, v_head_size)
        v_hidden_size = V.shape[2]
        v_head_size = v_hidden_size // kv_num_heads
        V = (
            V.view(batch_size, kv_sequence_length, kv_num_heads, v_head_size)
            .transpose(1, 2)
            .contiguous()
        )

    assert len(Q.shape) == 4 and len(K.shape) == 4 and len(V.shape) == 4

    # Calculate scale factor if not provided
    if scale is None:
        q_head_size = Q.shape[3]
        scale = 1.0 / (q_head_size**0.5)

    # Handle past key/value caches
    if past_key is not None:
        present_key = torch.cat([past_key, K], dim=2)
    else:
        present_key = K

    if past_value is not None:
        present_value = torch.cat([past_value, V], dim=2)
    else:
        present_value = V

    # Update K and V to include past states
    K = present_key
    V = present_value

    # Get current dimensions
    current_q_num_heads = Q.shape[1]
    current_kv_num_heads = K.shape[1]
    q_sequence_length = Q.shape[2]
    kv_sequence_length = K.shape[2]

    # Check if we can use the optimized scaled_dot_product_attention (most optimized)
    can_use_sdpa = (
        softcap == 0.0  # No softcap
        and qk_matmul_output_mode == 0  # Default QK output mode
        and softmax_precision is None  # No custom softmax precision
        and (
            attn_mask is None
            or not (attn_mask.dtype != torch.bool and torch.any(attn_mask != 0))
        )  # No float mask or zero float mask
    )

    # Check if we can use flex_attention (flexible and optimized)
    can_use_flex_attention = (
        not can_use_sdpa  # Use flex_attention only if SDPA is not available
        and qk_matmul_output_mode == 0  # Default QK output mode
        and softmax_precision is None  # No custom softmax precision
    )

    if can_use_sdpa:
        # Use PyTorch's optimized scaled_dot_product_attention
        enable_gqa = current_q_num_heads != current_kv_num_heads

        # Validate GQA configuration
        if enable_gqa:
            assert current_q_num_heads % current_kv_num_heads == 0, (
                f"q_num_heads ({current_q_num_heads}) must be divisible by kv_num_heads ({current_kv_num_heads}) for GQA"
            )

        # Prepare attention mask for SDPA
        sdpa_attn_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # Convert boolean mask: True means participate, SDPA expects True to mask out
                sdpa_attn_mask = ~attn_mask
            else:
                sdpa_attn_mask = attn_mask

        output = torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=sdpa_attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

        # For QK output mode 0, we need to compute QK manually since SDPA doesn't return it
        if qk_matmul_output_mode == 0:
            qk_output = _compute_qk_output_for_mode_0(
                Q, K, current_q_num_heads, current_kv_num_heads, scale
            )
        else:
            # For other modes, we need fallback implementation
            qk_output = torch.zeros_like(torch.matmul(Q, K.transpose(-2, -1)))
    elif can_use_flex_attention:
        # Use PyTorch's flexible flex_attention for complex cases
        enable_gqa = current_q_num_heads != current_kv_num_heads

        # Validate GQA configuration
        if enable_gqa:
            assert current_q_num_heads % current_kv_num_heads == 0, (
                f"q_num_heads ({current_q_num_heads}) must be divisible by kv_num_heads ({current_kv_num_heads}) for GQA"
            )

        # Create score modification function for flex_attention
        def score_mod(score, b, h, q_idx, kv_idx):
            # Apply causal masking
            if is_causal:
                score = torch.where(q_idx >= kv_idx, score, float("-inf"))

            # Apply attention mask
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    # Boolean mask: True means participate in attention
                    if attn_mask.ndim == 4:  # (batch, heads, q_seq, kv_seq)
                        mask_value = attn_mask[b, h, q_idx, kv_idx]
                    elif attn_mask.ndim == 3:  # (batch, q_seq, kv_seq)
                        mask_value = attn_mask[b, q_idx, kv_idx]
                    else:  # (q_seq, kv_seq)
                        mask_value = attn_mask[q_idx, kv_idx]
                    score = torch.where(mask_value, score, float("-inf"))
                else:
                    # Float mask: added to attention scores
                    if attn_mask.ndim == 4:  # (batch, heads, q_seq, kv_seq)
                        mask_value = attn_mask[b, h, q_idx, kv_idx]
                    elif attn_mask.ndim == 3:  # (batch, q_seq, kv_seq)
                        mask_value = attn_mask[b, q_idx, kv_idx]
                    else:  # (q_seq, kv_seq)
                        mask_value = attn_mask[q_idx, kv_idx]
                    score = score + mask_value

            # Apply softcap if provided
            if softcap > 0.0:
                score = softcap * torch.tanh(score / softcap)

            return score

        # FlexAttention returns a single tensor of shape (B, Hq, L, Ev)
        output = flex_attention.flex_attention(
            Q,
            K,
            V,
            score_mod=score_mod
            if (is_causal or attn_mask is not None or softcap > 0.0)
            else None,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        assert not isinstance(output, tuple), (
            "flex_attention should return a single tensor"
        )

        # For QK output mode 0, we need to compute QK manually since flex_attention doesn't return it
        if qk_matmul_output_mode == 0:
            qk_output = _compute_qk_output_for_mode_0(
                Q, K, current_q_num_heads, current_kv_num_heads, scale
            )
        else:
            # For other modes, we need fallback implementation
            qk_output = torch.zeros_like(torch.matmul(Q, K.transpose(-2, -1)))
    else:
        # Fallback to manual implementation for complex cases
        # Handle Group Query Attention (GQA) and Multi-Query Attention (MQA)
        if current_q_num_heads != current_kv_num_heads:
            assert current_q_num_heads % current_kv_num_heads == 0, (
                "q_num_heads must be divisible by kv_num_heads"
            )
            repeat_factor = current_q_num_heads // current_kv_num_heads
            K = K.repeat_interleave(repeat_factor, dim=1)
            V = V.repeat_interleave(repeat_factor, dim=1)

        # Create attention bias
        attn_bias = torch.zeros(
            q_sequence_length, kv_sequence_length, dtype=Q.dtype, device=Q.device
        )

        # Apply causal masking
        if is_causal:
            assert attn_mask is None, "Cannot use both is_causal and attn_mask"
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
        scale_factor = scale if scale is not None else (1.0 / (Q.shape[3] ** 0.5))

        # Compute Q @ K^T
        qk_matmul_output = torch.matmul(Q, K.transpose(-2, -1)) * scale_factor

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
            allowed_intermediate_precisions = {
                1,  # FLOAT
                10,  # FLOAT16
                11,  # DOUBLE
                16,  # BFLOAT16
            }
            if softmax_precision in allowed_intermediate_precisions:
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
