"""Dispatcher for AtenLib functions from onnx-script."""

from __future__ import annotations

import operator

from typing import Callable, Dict, Mapping, Union

import onnxscript  # type: ignore[import]
from onnxscript import opset18  # type: ignore[import]
from onnxscript.function_libs.torch_aten import (  # type: ignore[import,attr-defined]
    ops as onnxscript_ops,
)

import torch._decomp
import torch._ops as torch_ops
from torch.onnx._internal import _beartype


TORCH_ONNX_OPSET = onnxscript.values.Opset(domain="torch.onnx", version=1)


@onnxscript.script(opset=TORCH_ONNX_OPSET)  # type: ignore[arg-type]
def prims_convert_element_type(tensor, dtype: int):
    return opset18.Cast(tensor, to=dtype)


@onnxscript.script(opset=TORCH_ONNX_OPSET)  # type: ignore[arg-type]
def aten_getitem(self, i):
    # TODO(justinchuby): Support
    # i = opset18.Unsqueeze(i, opset18.Constant(value_ints=[0]))
    # return opset18.Gather(self, i, axis=0)
    return opset18.SequenceAt(self, i)


# A simple lookup table for atenlib functions
_ATENLIB_FUNCTIONS = {
    "aten::abs": onnxscript_ops.core.aten_abs,
    "aten::acos": onnxscript_ops.core.aten_acos,
    "aten::acosh": onnxscript_ops.core.aten_acosh,
    "aten::adaptive_avg_pool1d": onnxscript_ops.nn.aten_adaptive_avg_pool1d,
    "aten::adaptive_avg_pool2d": onnxscript_ops.nn.aten_adaptive_avg_pool2d,
    "aten::adaptive_avg_pool3d": onnxscript_ops.nn.aten_adaptive_avg_pool3d,
    "aten::add": onnxscript_ops.core.aten_add,
    "aten::addmm": onnxscript_ops.core.aten_addmm,
    "aten::alias": onnxscript_ops.core.aten_alias,
    "aten::amax": onnxscript_ops.core.aten_amax,
    "aten::amin": onnxscript_ops.core.aten_amin,
    "aten::arange": onnxscript_ops.core.aten_arange_start,
    "aten::argmax": onnxscript_ops.core.aten_argmax,
    "aten::argmin": onnxscript_ops.core.aten_argmin,
    "aten::asin": onnxscript_ops.core.aten_asin,
    "aten::asinh": onnxscript_ops.core.aten_asinh,
    "aten::atan": onnxscript_ops.core.aten_atan,
    "aten::atanh": onnxscript_ops.core.aten_atanh,
    "aten::baddbmm": onnxscript_ops.core.aten_baddbmm,
    "aten::bitwise_not": onnxscript_ops.core.aten_bitwise_not_bool,
    "aten::bmm": onnxscript_ops.core.aten_bmm,
    "aten::ceil": onnxscript_ops.core.aten_ceil,
    "aten::celu": onnxscript_ops.nn.aten_celu,
    "aten::clamp_max": onnxscript_ops.core.aten_clamp_max,
    "aten::clamp_min": onnxscript_ops.core.aten_clamp_min,
    "aten::clamp": onnxscript_ops.core.aten_clamp,
    "aten::clone": onnxscript_ops.core.aten_clone,
    "aten::convolution": onnxscript_ops.core.aten_convolution,
    "aten::cos": onnxscript_ops.core.aten_cos,
    "aten::cosh": onnxscript_ops.core.aten_cosh,
    "aten::cumsum": onnxscript_ops.core.aten_cumsum,
    "aten::detach": onnxscript_ops.core.aten_detach,
    "aten::div": onnxscript_ops.core.aten_div,
    "aten::dot": onnxscript_ops.core.aten_dot,
    "aten::elu": onnxscript_ops.nn.aten_elu,
    "aten::embedding": onnxscript_ops.core.aten_embedding,
    "aten::empty_like": onnxscript_ops.core.aten_empty_like,
    "aten::empty": onnxscript_ops.core.aten_empty,
    "aten::eq": onnxscript_ops.core.aten_eq,
    "aten::equal": onnxscript_ops.core.aten_equal,
    "aten::erf": onnxscript_ops.core.aten_erf,
    "aten::exp": onnxscript_ops.core.aten_exp,
    "aten::exp2": onnxscript_ops.core.aten_exp2,
    "aten::expand": onnxscript_ops.core.aten_expand,
    "aten::fmod": onnxscript_ops.core.aten_fmod,
    "aten::full_like": onnxscript_ops.core.aten_full_like,
    "aten::full": onnxscript_ops.core.aten_full,
    "aten::ge": onnxscript_ops.core.aten_ge,
    "aten::gelu": onnxscript_ops.nn.aten_gelu,
    "aten::gt": onnxscript_ops.core.aten_gt,
    "aten::isinf": onnxscript_ops.core.aten_isinf,
    "aten::le": onnxscript_ops.core.aten_le,
    "aten::leaky_relu": onnxscript_ops.nn.aten_leaky_relu,
    "aten::linear": onnxscript_ops.nn.aten_linear,
    "aten::log_softmax": onnxscript_ops.special.aten_special_log_softmax,
    "aten::log": onnxscript_ops.core.aten_log,
    "aten::log10": onnxscript_ops.core.aten_log10,
    "aten::log1p": onnxscript_ops.core.aten_log1p,
    "aten::log2": onnxscript_ops.core.aten_log2,
    "aten::logaddexp": onnxscript_ops.core.aten_logaddexp,
    "aten::logaddexp2": onnxscript_ops.core.aten_logaddexp2,
    "aten::logcumsumexp": onnxscript_ops.core.aten_logcumsumexp,
    "aten::logdet": onnxscript_ops.core.aten_logdet,
    "aten::logsigmoid": onnxscript_ops.nn.aten_log_sigmoid,
    "aten::logsumexp": onnxscript_ops.core.aten_logsumexp,
    "aten::lt": onnxscript_ops.core.aten_lt,
    "aten::masked_fill": onnxscript_ops.core.aten_masked_fill,
    "aten::matmul": onnxscript_ops.core.aten_matmul,
    "aten::maximum": onnxscript_ops.core.aten_maximum,
    "aten::minimum": onnxscript_ops.core.aten_minimum,
    "aten::mm": onnxscript_ops.core.aten_mm,
    "aten::mul": onnxscript_ops.core.aten_mul,
    "aten::native_layer_norm": onnxscript_ops.core.aten_native_layer_norm,
    "aten::ne": onnxscript_ops.core.aten_ne,
    "aten::neg": onnxscript_ops.core.aten_neg,
    "aten::new_full": onnxscript_ops.core.aten_new_full,
    "aten::nonzero": onnxscript_ops.core.aten_nonzero,
    "aten::ones_like": onnxscript_ops.core.aten_ones_like,
    "aten::ones": onnxscript_ops.core.aten_ones,
    "aten::permute": onnxscript_ops.core.aten_permute,
    "aten::pow": onnxscript_ops.core.aten_pow,
    "aten::reciprocal": onnxscript_ops.core.aten_reciprocal,
    "aten::relu": onnxscript_ops.nn.aten_relu,
    "aten::relu6": onnxscript_ops.nn.aten_relu6,
    "aten::remainder": onnxscript_ops.core.aten_remainder,
    "aten::repeat": onnxscript_ops.core.aten_repeat,
    "aten::reshape": onnxscript_ops.core.aten_reshape,
    "aten::round": onnxscript_ops.core.aten_round,
    "aten::rsqrt": onnxscript_ops.core.aten_rsqrt,
    "aten::rsub": onnxscript_ops.core.aten_rsub,
    "aten::select": onnxscript_ops.core.aten_select,
    "aten::selu": onnxscript_ops.core.aten_selu,
    "aten::sigmoid": onnxscript_ops.core.aten_sigmoid,
    "aten::sign": onnxscript_ops.core.aten_sign,
    "aten::sin": onnxscript_ops.core.aten_sin,
    "aten::sinh": onnxscript_ops.core.aten_sinh,
    "aten::slice": onnxscript_ops.core.aten_slice,
    "aten::softmax": onnxscript_ops.special.aten_special_softmax,
    "aten::split": onnxscript_ops.core.aten_split,
    "aten::sqrt": onnxscript_ops.core.aten_sqrt,
    "aten::stack": onnxscript_ops.core.aten_stack,
    "aten::sub": onnxscript_ops.core.aten_sub,
    "aten::sum": onnxscript_ops.core.aten_sum_dim_IntList,
    "aten::sym_size": onnxscript_ops.core.aten_sym_size,
    "aten::t": onnxscript_ops.core.aten_t,
    "aten::tan": onnxscript_ops.core.aten_tan,
    "aten::tanh": onnxscript_ops.core.aten_tanh,
    "aten::topk": onnxscript_ops.core.aten_topk,
    "aten::transpose": onnxscript_ops.core.aten_transpose,
    "aten::unsqueeze": onnxscript_ops.core.aten_unsqueeze,
    "aten::upsample_nearest2d": onnxscript_ops.nn.aten_upsample_nearest2d,
    "aten::view": onnxscript_ops.core.aten_view,
    "aten::where": onnxscript_ops.core.aten_where,
    "aten::xlogy": onnxscript_ops.special.aten_special_xlogy,
    "aten::zeros_like": onnxscript_ops.core.aten_zeros_like,
    "aten::zeros": onnxscript_ops.core.aten_zeros,
    "getitem": aten_getitem,
    "prims::convert_element_type": prims_convert_element_type,
}


def _create_op_overload_to_exporter_key_table() -> (
    Mapping[Union[torch_ops.OpOverload, Callable], str]
):
    # TODO(justinchuby): Improve how the table is constructed.
    table: Dict[Union[torch_ops.OpOverload, Callable], str] = {}

    for op_namespace in (torch_ops.ops.aten, torch_ops.ops.prims):
        for attr_name in dir(op_namespace):
            op_overload_packet = getattr(op_namespace, attr_name)
            if not isinstance(op_overload_packet, torch_ops.OpOverloadPacket):
                continue

            exporter_look_up_key = op_overload_packet._qualified_op_name
            if _ATENLIB_FUNCTIONS.get(exporter_look_up_key) is None:
                # This aten op doesn't have ONNX exporter.
                continue

            for overload_name in op_overload_packet.overloads():
                op_overload = getattr(op_overload_packet, overload_name)
                # This line maps torch_ops.ops.aten.add.Tensor, torch_ops.ops.aten.add.Scalar, torch_ops.ops.aten.add.out, etc
                # to "aten::add". This means the exporter for "aten::add" is used for all overloads of "aten::add".
                # This is applied to all ops under torch_ops.ops.aten.
                #
                # TODO(wechi): in the future, we might want to write individual exporter for each overload, if,
                # for example, they have different type promotion rules. If so, just map different overloads to
                # different exporter keys.

                table[op_overload] = op_overload_packet._qualified_op_name
    # NOTE: Below are not in torch_ops.ops.aten/torch_ops.ops.prim
    table[torch_ops.ops.aten.sym_size.int] = "aten::sym_size"
    return table


# Dictionary that maps torch_ops.ops.aten.* to exporter look up key; e.g.,
# _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[torch.add.Tensor] is "aten::add".
_OP_OVERLOAD_TO_EXPORTER_KEY_TABLE = _create_op_overload_to_exporter_key_table()
_SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE = {
    operator.mul: "aten::mul",
    operator.add: "aten::add",
    operator.pow: "aten::pow",
    operator.sub: "aten::sub",
}


@_beartype.beartype
def _create_onnx_friendly_decomposition_table() -> (
    Mapping[torch_ops.OpOverload, Callable]
):
    decomposition_table: Dict[torch_ops.OpOverload, Callable] = {}
    for op_overload, decomp_fn in torch._decomp.decomposition_table.items():
        # Skip decomposition into "prim::*" ops, because they are not generally supported by ONNX.
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX exporter.
        if (
            "torch._refs" in decomp_fn.__module__
            or op_overload in _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
        ):
            continue
        decomposition_table[op_overload] = decomp_fn
    return decomposition_table


# Subset of PyTorch's built-in aten-to-aten decomposition.
# If an aten op (e.g., torch.ops.aten.add.Tensor) has n ONNX counterpart,
# we exclude the op's decomposition from _DEFAULT_ONNX_EXPORTER_DECOMPOSITION_TABLE.
_DEFAULT_ONNX_EXPORTER_DECOMPOSITION_TABLE: Mapping[
    torch_ops.OpOverload, Callable
] = _create_onnx_friendly_decomposition_table()
