"""ONNX operators as native torch.fx operators.

This module provides a set of functions to create ONNX operators in the FX graph
which are exportable to ONNX.
"""

# flake8: noqa: B950
from __future__ import annotations


__all__ = [
    "aten_decompositions",
    "symbolic",
    "symbolic_multi_out",
    "rotary_embedding",
    "attention",
]


from typing import Callable, TYPE_CHECKING

import torch
from torch.onnx.ops import _impl, _symbolic_impl


if TYPE_CHECKING:
    from collections.abc import Sequence


# https://github.com/onnx/onnx/blob/f542e1f06699ea7e1db5f62af53355b64338c723/onnx/onnx.proto#L597
_TORCH_DTYPE_TO_ONNX_DTYPE = {
    torch.float32: 1,  # FLOAT
    torch.uint8: 2,  # UINT8
    torch.int8: 3,  # INT8
    torch.uint16: 4,  # UINT16
    torch.int16: 5,  # INT16
    torch.int32: 6,  # INT32
    torch.int64: 7,  # INT64
    str: 8,  # STRING
    torch.bool: 9,  # BOOL
    torch.float16: 10,  # FLOAT16
    torch.double: 11,  # DOUBLE
    torch.uint32: 12,  # UINT32
    torch.uint64: 13,  # UINT64
    torch.complex64: 14,  # COMPLEX64
    torch.complex128: 15,  # COMPLEX128
    torch.bfloat16: 16,  # BFLOAT16
    torch.float8_e4m3fn: 17,  # FLOAT8E4M3FN
    torch.float8_e4m3fnuz: 18,  # FLOAT8E4M3FNUZ
    torch.float8_e5m2: 19,  # FLOAT8E5M2
    torch.float8_e5m2fnuz: 20,  # FLOAT8E5M2FNUZ
    # 21 = UINT4
    # 22 = INT4
    torch.float4_e2m1fn_x2: 23,  # FLOAT4E2M1
}


def aten_decompositions() -> dict[torch._ops.OpOverload, Callable]:
    """Return the ONNX to ATen decomp table."""
    return _impl.ONNX_ATEN_DECOMP_TABLE


def _parse_domain_op_type(domain_op: str) -> tuple[str, str]:
    splitted = domain_op.split("::", 1)
    if len(splitted) == 1:
        domain = ""
        op_type = splitted[0]
    else:
        domain = splitted[0]
        op_type = splitted[1]
    return domain, op_type


def symbolic(
    domain_op: str,
    /,
    inputs: Sequence[torch.Tensor | None],
    attrs: dict[
        str,
        int
        | float
        | str
        | bool
        | Sequence[int]
        | Sequence[float]
        | Sequence[str]
        | Sequence[bool],
    ]
    | None = None,
    *,
    dtype: torch.dtype | int,
    shape: Sequence[int | torch.SymInt],
    version: int | None = None,
    metadata_props: dict[str, str] | None = None,
) -> torch.Tensor:
    """Create a symbolic FX operator to represent an arbitrary ONNX operator.

    This function is used to create a symbolic operator with a single output.
    To create an operator with multiple outputs, use :func:`symbolic_multi_out`.

    You may use ``if torch.onnx.is_in_onnx_export()`` to conditionally enable the
    symbolic logic only during ``torch.onnx.export()``.

    Example::

        class CustomOp(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Normal torch operators can interleave with the symbolic ops during ONNX export
                x = x + 1

                # Create a symbolic ONNX operator with the name "CustomOp" in the "custom_domain" domain.
                # The output tensor will have the specified dtype and shape
                val = torch.onnx.ops.symbolic(
                    "custom_domain::CustomOp",
                    (x,),
                    dict(attr_key="attr_value"),
                    dtype=x.dtype,
                    shape=x.shape,
                    version=1,
                )

                # The result of the symbolic op can be used in normal torch operations during ONNX export
                return torch.nn.functional.relu(val)


        # You may then export this model to ONNX using torch.onnx.export(..., dynamo=True).

    Args:
        domain_op: The domain and operator name, separated by "::". For example,
            "custom_domain::CustomOp".
        inputs: The input tensors to the operator.
        attrs: The attributes of the operator. The keys are attribute names and
            the values are attribute values. Valid attribute types are int, float,
            str, bool, and lists of int, float, str, and bool. Tensor attributes
            are unsupported.
        dtype: The data type of the output tensor.This can be either a torch.dtype
            or an integer representing the ONNX data type.
        shape: The shape of the output tensor. This can be a list of integers or
            SymInt values.
        version: The version of the opset used for the operator.
        metadata_props: Metadata properties for the ONNX node.
            This is a dictionary of str-str pairs.

    Returns:
        The output tensor of the operator.
    """
    if not isinstance(dtype, int):
        torch._check(
            dtype in _TORCH_DTYPE_TO_ONNX_DTYPE, lambda: f"Unsupported dtype: {dtype}"
        )
        dtype = _TORCH_DTYPE_TO_ONNX_DTYPE[dtype]
    domain, op_type = _parse_domain_op_type(domain_op)
    if attrs is None:
        attrs = {}
    encoded_attrs = _symbolic_impl.EncodedAttrs.from_dict(attrs)
    # TODO: Parse domain
    return _symbolic_impl._symbolic(
        inputs,
        op_type,
        dtype,
        shape=shape,
        attr_keys=encoded_attrs.attr_keys,
        attr_types=encoded_attrs.attr_types,
        attr_pos=encoded_attrs.attr_pos,
        attr_ints=encoded_attrs.attr_ints,
        attr_floats=encoded_attrs.attr_floats,
        attr_strs=encoded_attrs.attr_strs,
        metadata_props_keys=metadata_props.keys() if metadata_props else [],
        metadata_props_values=metadata_props.values() if metadata_props else [],
        domain=domain,
        version=version,
    )


def symbolic_multi_out(
    domain_op: str,
    /,
    inputs: Sequence[torch.Tensor | None],
    attrs: dict[
        str,
        int
        | float
        | str
        | bool
        | Sequence[int]
        | Sequence[float]
        | Sequence[str]
        | Sequence[bool],
    ]
    | None = None,
    *,
    dtypes: Sequence[torch.dtype | int],
    shapes: Sequence[Sequence[int | torch.SymInt]],
    version: int | None = None,
    metadata_props: dict[str, str] | None = None,
) -> Sequence[torch.Tensor]:
    """Create a symbolic FX operator to represent an arbitrary ONNX operator with multiple outputs.

    You may use ``if torch.onnx.is_in_onnx_export()`` to conditionally enable the
    symbolic logic only during ``torch.onnx.export()``.

    Example::

        class CustomOp(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Normal torch operators can interleave with the symbolic ops during ONNX export
                x = x + 1

                # Create a symbolic ONNX operator with the name "CustomOp" in the "custom_domain" domain.
                # The output tensors will have the specified dtypes and shapes
                (out1, out2) = torch.onnx.ops.symbolic(
                    "custom_domain::CustomOp",
                    (x,),
                    dict(attr_key="attr_value"),
                    dtypes=(x.dtype, torch.float32),
                    shapes=(x.shape, [1, 2, 3]),
                    version=1,
                )

                # The result of the symbolic op can be used in normal torch operations during ONNX export
                return torch.nn.functional.relu(out1 + out2)


        # You may then export this model to ONNX using torch.onnx.export(..., dynamo=True).

    Args:
        domain_op: The domain and operator name, separated by "::". For example,
            "custom_domain::CustomOp".
        inputs: The input tensors to the operator.
        attrs: The attributes of the operator. The keys are attribute names and
            the values are attribute values. Valid attribute types are int, float,
            str, bool, and lists of int, float, str, and bool. Tensor attributes
            are unsupported.
        dtypes: The data types of the output tensors. This can be a list of
            torch.dtype or integers representing the ONNX data types. The length
            of this list must be the number of outputs.
        shapes: The shapes of the output tensors. This can be a list of lists of
            integers or SymInt values. The length of this list must be the number of outputs.
        version: The version of the opset used for the operator.
        metadata_props: Metadata properties for the ONNX node.
            This is a dictionary of str-str pairs.

    Returns:
        A list of output tensors of the operator.
    """
    torch._check(
        len(shapes) == len(dtypes),
        lambda: f"Number of shapes ({len(shapes)}) must match number of dtypes ({len(dtypes)})",
    )
    onnx_dtypes = []
    for dtype in dtypes:
        if not isinstance(dtype, int):
            torch._check(
                dtype in _TORCH_DTYPE_TO_ONNX_DTYPE,
                lambda: f"Unsupported dtype: {dtype}",
            )
            onnx_dtypes.append(_TORCH_DTYPE_TO_ONNX_DTYPE[dtype])
        else:
            onnx_dtypes.append(dtype)
    domain, op_type = _parse_domain_op_type(domain_op)
    if attrs is None:
        attrs = {}
    encoded_attrs = _symbolic_impl.EncodedAttrs.from_dict(attrs)
    # Use the size of dtypes to determine the number of outputs
    return _symbolic_impl._symbolic_multi_out(
        inputs,
        op_type,
        onnx_dtypes,
        shapes=shapes,
        attr_keys=encoded_attrs.attr_keys,
        attr_types=encoded_attrs.attr_types,
        attr_pos=encoded_attrs.attr_pos,
        attr_ints=encoded_attrs.attr_ints,
        attr_floats=encoded_attrs.attr_floats,
        attr_strs=encoded_attrs.attr_strs,
        metadata_props_keys=metadata_props.keys() if metadata_props else [],
        metadata_props_values=metadata_props.values() if metadata_props else [],
        domain=domain,
        version=version,
    )


def rotary_embedding(
    X: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    *,
    interleaved: bool = False,
    num_heads: int = 0,
    rotary_embedding_dim: int = 0,
) -> torch.Tensor:
    """RotaryEmbedding op in ONNX.

    https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html

    RotaryEmbedding is the implementation of rotary positional embeddings (RoPE) based on the paper https://arxiv.org/pdf/2104.09864.
    The key advantage of RoPE is that it allows the model to understand both the absolute position of a token and the relative distances
    between tokens. This is achieved through a rotational mechanism where the extent of rotation is computed based on the token's absolute position (position_ids).

    The rotational mechanism is defined by sine and cosine functions that are used to represent the rotation angles.
    For each token in the sequence, its positional embedding is computed by rotating its embedding vector. This is done by splitting the
    embedding vector either into two halves or interleaving every alternate token and applying the rotation matrix to each half of the embedding vector.
    The rotation matrix is parameterized by the token's position in the sequence. The rotated halves of the embedding vector are concatenated
    to form the final positional embedding for each token. The rotated positional embeddings are used in the self-attention mechanism.
    The rotation ensures that the model captures both absolute and relative positional information.

    Args:
        X: The input tensor representing the token embeddings. 4D tensor with
            shape `(batch_size, num_heads, sequence_length, head_size)` or 3D tensor
            with shape `(batch_size, sequence_length, hidden_size)`. For cases with
            a 4D input tensor, `head_size` has to be even. For cases with a 3D input
            tensor, `num_heads` attribute must be provided and `hidden_size` must
            be an even multiple of `num_heads` where `hidden_size = num_heads * head_size`
        cos_cache: The cosine values for the rotation. 2D tensor with shape `(max_position_id_plus_1, head_size / 2)`
            for full rotation or `(max_position_id_plus_1, rotary_embedding_dim / 2)`
            for partial rotation when `position_ids` are provided. 3D tensor with shape
            `(batch_size, sequence_length, head_size / 2)` for full rotation or
            `(batch_size, sequence_length, rotary_embedding_dim / 2)` for partial
            rotation when `position_ids` are not provided. `max_position_id_plus_1`
            is a parameter to the model.
        sin_cache: The sine values for the rotation. 2D tensor with shape `(max_position_id_plus_1, head_size / 2)`
            for full rotation or `(max_position_id_plus_1, rotary_embedding_dim / 2)`
            for partial rotation when `position_ids` are provided. 3D tensor with shape
            `(batch_size, sequence_length, head_size / 2)` for full rotation or
            `(batch_size, sequence_length, rotary_embedding_dim / 2)` for partial rotation
            when `position_ids` are not provided. `max_position_id_plus_1` is a parameter
            to the model.
        position_ids: The position indices for the tokens. 2D tensor with shape
            `(batch_size, sequence_length)`.
        interleaved: Rotate using interleaved pattern. Default value is 0 (False).
        num_heads: Number of attention heads. Must be provided when input is a 3D tensor.
        rotary_embedding_dim: Rotary embedding dimension used to apply partial rotary embeddings.

    Returns:
        Tensor with same shape as input.
    """
    return _impl.rotary_embedding_23(
        X,
        cos_cache,
        sin_cache,
        position_ids=position_ids,
        interleaved=interleaved,
        num_heads=num_heads,
        rotary_embedding_dim=rotary_embedding_dim,
    )


def attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    past_key: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None,
    *,
    is_causal: bool = False,
    kv_num_heads: int = 0,
    q_num_heads: int = 0,
    qk_matmul_output_mode: int = 0,
    scale: float | None = None,
    softcap: float = 0.0,
    softmax_precision: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention op in ONNX.

    https://onnx.ai/onnx/operators/onnx__Attention.html

    Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed.

    This operator covers self and cross variants of the attention operation based on sequence lengths of K, Q and V.

    For self attention, ``kv_sequence_length`` equals to ``q_sequence_length``.

    For cross attention, query and key might have different lengths.

    This operator also covers the 3 following variants based on the number of heads:

    1. Multi-headed Attention (MHA): Described in the paper https://arxiv.org/pdf/1706.03762, `q_num_heads = kv_num_heads`.
    2. Group-query Attention (GQA): Described in the paper https://arxiv.org/pdf/2305.13245, `q_num_heads > kv_num_heads`, `q_num_heads % kv_num_heads == 0`.
    3. Multi-query Attention (MQA): Described in the paper https://arxiv.org/pdf/1911.02150, `q_num_heads > kv_num_heads`, `kv_num_heads=1`.

    Attention bias to be added is calculated based on ``attn_mask`` input and ``is_causal` `attribute``, only one of which can be provided.

    1. If ``is_causal`` is set to `1`, the attention masking is a lower triangular matrix when the mask is a square matrix. The attention masking has the form of the upper left causal bias due to the alignment.
    2. `attn_mask`: A boolean mask where a value of `True` indicates that the element should take part in attention or a float mask of the same type as query, key, value that is added to the attention score.

    Both past and present state key/values are optional. They shall be used together, and not allowed to use only one of them.
    The following pattern is applied to the Q, K and V inputs after appropriate reshaping of K and V inputs based on sequence lengths and num heads provided::

        The following pattern is applied by this operator:
                Q          K          V
                |          |          |
        Q*sqrt(scale) K*sqrt(scale) |
                |          |          |
                |       Transpose     |
                |          |          |
                ---MatMul---          |
                    |               |
        at_mask---Add              |
                    |               |
            softcap (if provided)     |
                    |               |
                Softmax            |
                    |               |
                    -----MatMul------
                            |
                            Y

    Args:
        Q: Query tensor. 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, head_size)` or 3D tensor
            with shape `(batch_size, q_sequence_length, q_hidden_size)`. For cases with a 3D input tensor,
            `q_hidden_size = q_num_heads * head_size`
        K: Key tensor. 4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, head_size)` or 3D tensor
            with shape `(batch_size, kv_sequence_length, k_hidden_size)`. For cases with a 3D input tensor,
            `k_hidden_size = kv_num_heads * head_size`
        V: Value tensor. 4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, v_head_size)` or 3D tensor
            with shape `(batch_size, kv_sequence_length, v_hidden_size)`. For cases with a 3D input tensor,
            `v_hidden_size = kv_num_heads * v_head_size`
        attn_mask: Attention mask. Shape must be broadcastable to 4D tensor with shape
            `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)` where
            `total_sequence_length = past_sequence_length + kv_sequence_length`. Two types of masks are supported.
            A boolean mask where a value of True indicates that the element should take part in attention.
            Also supports a float mask of the same type as query, key, value that is added to the attention score.
        past_key: Past state cache for key with shape `(batch_size, kv_num_heads, past_sequence_length, head_size)`
        past_value: Past state cache for value with shape `(batch_size, kv_num_heads, past_sequence_length, v_head_size)`
        is_causal: If set to True, the attention masking is a lower triangular matrix when the mask is a square matrix.
            The attention masking has the form of the upper left causal bias due to the alignment.
        kv_num_heads: Number of heads of key and value. Must be used with 3D inputs of Q, K and V.
        q_num_heads: Number of heads of query. Must be used with 3D inputs of Q, K and V.
        qk_matmul_output_mode: If set to 0, qk_matmul_output is the output of qk matmul. If set to 1,
            qk_matmul_output includes the addition of the attention mask to the output of qk matmul.
            If set to 2, qk_matmul_output is the output after the softcap operation. If set to 3,
            qk_matmul_output is the output after the softmax operation. Default value is 0.
        scale: Scaling factor applied to Q*K^T. Default value is 1/sqrt(head_size). To prevent numerical overflow,
            scale Q, K by sqrt(scale) before matmul.
        softcap: Softcap value for attention weights. Default value is 0.
        softmax_precision: The floating-point precision used in softmax computation. If softmax precision is not provided,
            the same precision as the input of softmax (Q and K) is used.

    Returns:
        A tuple containing:
        - The output tensor. 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, v_head_size)` or 3D tensor
          with shape `(batch_size, q_sequence_length, hidden_size)`. For cases with a 3D input tensor,
          `hidden_size = q_num_heads * v_head_size`
        - Updated key cache with shape `(batch_size, kv_num_heads, total_sequence_length, head_size)` where
          `total_sequence_length = past_sequence_length + kv_sequence_length`.
        - Updated value cache with shape `(batch_size, kv_num_heads, total_sequence_length, v_head_size)` where
          `total_sequence_length = past_sequence_length + kv_sequence_length`.
        - The output of QK matmul. 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)`
          where `total_sequence_length = past_sequence_length + kv_sequence_length`.
    """
    return _impl.attention_23(
        Q,
        K,
        V,
        attn_mask=attn_mask,
        past_key=past_key,
        past_value=past_value,
        is_causal=is_causal,
        kv_num_heads=kv_num_heads,
        q_num_heads=q_num_heads,
        qk_matmul_output_mode=qk_matmul_output_mode,
        scale=scale,
        softcap=softcap,
        softmax_precision=softmax_precision,
    )
