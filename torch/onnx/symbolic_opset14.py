# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""This file exports ONNX ops for opset 14.

Note [ONNX operators that are added/updated in opset 14]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
    HardSwish, Trilu

Updated operators:
    Reshape
    Add, Sub, Mul, Div
    GRU, LSTM, RNN
    BatchNorm, Cumsum, Relu
"""

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md
from __future__ import annotations

import functools

import torch
from torch.onnx import _constants, _type_utils, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import jit_utils, registration


__all__ = [
    "hardswish",
    "tril",
    "triu",
    "reshape",
    "batch_norm",
    "quantized_hardswish",
    "scaled_dot_product_attention",
]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=14)


@_onnx_symbolic("aten::hardswish")
@symbolic_helper.parse_args("v")
def hardswish(g: jit_utils.GraphContext, self):
    return g.op("HardSwish", self)


@_onnx_symbolic("aten::tril")
def tril(g: jit_utils.GraphContext, self, diagonal, out=None):
    return g.op("Trilu", self, diagonal, upper_i=0)


@_onnx_symbolic("aten::triu")
def triu(g: jit_utils.GraphContext, self, diagonal, out=None):
    return g.op("Trilu", self, diagonal, upper_i=1)


@_onnx_symbolic("aten::reshape")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v")
def reshape(g: jit_utils.GraphContext, self, shape):
    # NOTE: Due to bug in ORT https://github.com/microsoft/onnxruntime/issues/10664
    #       Reshape export cannot utilize the new allowzero attribute introduced in opset 14.
    return symbolic_helper._reshape_helper(g, self, shape, allowzero=0)


@_onnx_symbolic("aten::batch_norm")
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
def batch_norm(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    cudnn_enabled,
):
    if (
        torch.is_autocast_enabled()
        and not symbolic_helper.args_have_same_dtype(
            [input, weight, bias, running_mean, running_var]
        )
        and GLOBALS.export_onnx_opset_version < 15
    ):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "BatchNormalization",
            14,
            15,
            "All input tensors must have the same `dtype`."
            " Turn off Autocast or export using opset version 15.",
            input,
        )

    symbolic_helper.check_training_mode(training, "batch_norm")
    weight, bias, running_mean, running_var = symbolic_helper._batchnorm_helper(
        g, input, weight, bias, running_mean, running_var
    )
    out = g.op(
        "BatchNormalization",
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon_f=eps,
        momentum_f=1 - momentum,
        training_mode_i=0 if not training else 1,
        outputs=1 if not training else 3,
    )
    if not training:
        return out
    else:
        res, new_running_mean, new_running_var = out
        new_running_mean.setType(running_mean.type())
        new_running_var.setType(running_var.type())
        return res


@_onnx_symbolic("quantized::hardswish")
def quantized_hardswish(g: jit_utils.GraphContext, x, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    output = hardswish(g, x)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# Ported from
# https://github.com/microsoft/onnxscript/blob/6b1b81700b4523f31d8c6d3321e5d8ef5d42b764/onnxscript/function_libs/torch_aten/ops/nn.py#L1504
# aten_scaled_dot_product_attention
# NOTE: Need op.Trilu
@_onnx_symbolic("aten::scaled_dot_product_attention")
@symbolic_helper.parse_args("v", "v", "v", "v", "f", "b", "v", "b")
def scaled_dot_product_attention(
    g: jit_utils.GraphContext,
    query: torch._C.Value,
    key: torch._C.Value,
    value: torch._C.Value,
    attn_mask: torch._C.Value | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: torch._C.Value | None = None,
    enable_gqa: bool = False,
):
    assert (not is_causal) or (
        is_causal and symbolic_helper._is_none(attn_mask)
    ), "is_causal and attn_mask cannot be set at the same time"
    assert (
        not enable_gqa
    ), "conversion of scaled_dot_product_attention not implemented if enable_gqa is True"

    scale = symbolic_helper._maybe_get_const(scale, "f")
    if symbolic_helper._is_none(scale):
        scale = _attention_scale(g, query)

    if is_causal:
        attn_mask = _causal_attention_mask(g, query, key)

    # Swap the last two axes of key
    # NOTE: onnx-script has different logic here, because the attribute perms in
    # transpose needs list of ints
    key_shape_builtin = symbolic_helper._get_tensor_rank(key)
    key_transposed_axes = list(range(key_shape_builtin))
    key_transposed_axes[-1], key_transposed_axes[-2] = (
        key_transposed_axes[-2],
        key_transposed_axes[-1],
    )
    key_transposed = g.op("Transpose", key, perm_i=key_transposed_axes)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = g.op("Mul", query, g.op("Sqrt", scale))
    key_transposed_scaled = g.op("Mul", key_transposed, g.op("Sqrt", scale))
    mul_qk = g.op("MatMul", query_scaled, key_transposed_scaled)

    if symbolic_helper._is_none(attn_mask):
        mul_qk_add = mul_qk
    elif (
        _type_utils.JitScalarType.from_value(attn_mask)
        == _type_utils.JitScalarType.BOOL
    ):
        # Turn the Boolean mask to float: attn_mask.masked_fill(not attn_mask, -float('inf'))
        const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
        const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
        attn_mask = g.op("Where", attn_mask, const_zero, const_neg_inf)
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    elif _type_utils.JitScalarType.from_value(attn_mask) in (
        _type_utils.JitScalarType.FLOAT,
        _type_utils.JitScalarType.HALF,
        _type_utils.JitScalarType.BFLOAT16,
    ):
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    else:
        raise ValueError(
            f"Unsupported type for attn_mask: {_type_utils.JitScalarType.from_value(attn_mask)}"
        )

    attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)

    if dropout_p != 0:
        attn_weight = g.op(
            "Dropout",
            attn_weight,
            g.op("Constant", value_t=torch.tensor(dropout_p, dtype=torch.float)),
        )

    return g.op("MatMul", attn_weight, value)


def _attention_scale(
    g: jit_utils.GraphContext, query: torch._C.Value
) -> torch._C.Value:
    """Calculate the scale factor for the attention result.

    Args:
        query: Tensor of shape [..., L, E]

    Returns:
        Scalar scale factor := 1 / math.sqrt(query.size(-1))
    """
    query_shape = g.op("Shape", query)
    query_shape_last = g.op(
        "Slice",
        query_shape,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)),
        g.op(
            "Constant", value_t=torch.tensor([_constants.INT64_MAX], dtype=torch.int64)
        ),
    )
    embedding_size = g.op(
        "Cast",
        query_shape_last,
        to_i=_type_utils.JitScalarType.from_value(query).onnx_type(),
    )
    const_one = g.op("Constant", value_t=torch.tensor([1.0], dtype=torch.float))
    scale = g.op("Div", const_one, g.op("Sqrt", embedding_size))
    # Add a Cast to convert the scale back to original type
    scale = g.op(
        "Cast",
        scale,
        to_i=_type_utils.JitScalarType.from_value(query).onnx_type(),
    )
    return scale


def _causal_attention_mask(
    g: jit_utils.GraphContext, query: torch._C.Value, key: torch._C.Value
) -> torch._C.Value:
    """Create a causal mask for the given query and key tensors.

    Equivalent to::
        mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_mask = torch.zeros(L, S, dtype=torch.float)
        attn_mask = attn_mask.masked_fill(not mask, -float('inf'))

    Args:
        query: Tensor of shape [..., L, E]
        key: Tensor of shape [..., S, E]

    Returns:
        Tensor of shape [L, S]
    """

    query_shape = g.op("Shape", query)
    key_shape = g.op("Shape", key)

    last_idx = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
    second_last_idx = g.op("Constant", value_t=torch.tensor([-2], dtype=torch.int64))
    target_length = g.op("Slice", query_shape, second_last_idx, last_idx)
    source_length = g.op("Slice", key_shape, second_last_idx, last_idx)
    # attn_mask = torch.ones(L, S) := {
    size = g.op("Concat", target_length, source_length, axis_i=0)
    const_one = g.op("Constant", value_t=torch.tensor([1.0]))
    attn_mask = g.op("Expand", const_one, size)
    # }
    attn_mask = g.op("Trilu", attn_mask, upper_i=0)
    # The causal mask has 0s in the lower triangle and -inf in the upper triangle.
    const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
    const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
    attn_mask = g.op(
        "Where", g.op("Equal", attn_mask, const_zero), const_neg_inf, const_zero
    )
    return attn_mask
