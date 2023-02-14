"""Dispatcher for AtenLib functions from onnx-script."""

from __future__ import annotations

from typing import Callable, Dict, Union

import onnxscript  # type: ignore[import]
from onnxscript import opset18  # type: ignore[import]
from onnxscript.function_libs.torch_aten import ops  # type: ignore[import]

import torch
from torch.onnx._internal import _beartype


TORCH_ONNX_OPSET = onnxscript.values.Opset(domain="torch.onnx", version=1)


@onnxscript.script(opset=TORCH_ONNX_OPSET)
def prims_convert_element_type(tensor, dtype: int):
    return opset18.Cast(tensor, to=dtype)


@onnxscript.script(opset=TORCH_ONNX_OPSET)
def aten_getitem(self, i):
    # TODO(justinchuby): Support
    # i = opset18.Unsqueeze(i, opset18.Constant(value_ints=[0]))
    # return opset18.Gather(self, i, axis=0)
    return opset18.SequenceAt(self, i)


# A simple lookup table for atenlib functions
_ATENLIB_FUNCTIONS = {
    "aten::abs": ops.core.aten_abs,
    "aten::acos": ops.core.aten_acos,
    "aten::acosh": ops.core.aten_acosh,
    "aten::adaptive_avg_pool1d": ops.nn.aten_adaptive_avg_pool1d,
    "aten::adaptive_avg_pool2d": ops.nn.aten_adaptive_avg_pool2d,
    "aten::adaptive_avg_pool3d": ops.nn.aten_adaptive_avg_pool3d,
    "aten::add": ops.core.aten_add,
    "aten::addmm": ops.core.aten_addmm,
    "aten::amax": ops.core.aten_amax,
    "aten::amin": ops.core.aten_amin,
    "aten::arange": ops.core.aten_arange_start,
    "aten::argmax": ops.core.aten_argmax,
    "aten::argmin": ops.core.aten_argmin,
    "aten::asin": ops.core.aten_asin,
    "aten::asinh": ops.core.aten_asinh,
    "aten::atan": ops.core.aten_atan,
    "aten::atanh": ops.core.aten_atanh,
    "aten::bmm": ops.core.aten_bmm,
    "aten::ceil": ops.core.aten_ceil,
    "aten::celu": ops.nn.aten_celu,
    "aten::clamp_max": ops.core.aten_clamp_max,
    "aten::clamp_min": ops.core.aten_clamp_min,
    "aten::clamp": ops.core.aten_clamp,
    "aten::clone": ops.core.aten_clone,
    "aten::convolution": ops.core.aten_convolution,
    "aten::cos": ops.core.aten_cos,
    "aten::cosh": ops.core.aten_cosh,
    "aten::detach": ops.core.aten_detach,
    "aten::div": ops.core.aten_div,
    "aten::dot": ops.core.aten_dot,
    "aten::elu": ops.nn.aten_elu,
    "aten::embedding": ops.core.aten_embedding,
    "aten::empty_like": ops.core.aten_empty_like,
    "aten::empty": ops.core.aten_empty,
    "aten::eq": ops.core.aten_eq,
    "aten::equal": ops.core.aten_equal,
    "aten::erf": ops.core.aten_erf,
    "aten::exp": ops.core.aten_exp,
    "aten::exp2": ops.core.aten_exp2,
    "aten::expand": ops.core.aten_expand,
    "aten::fmod": ops.core.aten_fmod,
    "aten::full_like": ops.core.aten_full_like,
    "aten::full": ops.core.aten_full,
    "aten::ge": ops.core.aten_ge,
    "aten::gelu": ops.nn.aten_gelu,
    "aten::gt": ops.core.aten_gt,
    "aten::isinf": ops.core.aten_isinf,
    "aten::le": ops.core.aten_le,
    "aten::leaky_relu": ops.nn.aten_leaky_relu,
    "aten::linear": ops.nn.aten_linear,
    "aten::log_softmax": ops.special.aten_special_log_softmax,
    "aten::log": ops.core.aten_log,
    "aten::log10": ops.core.aten_log10,
    "aten::log1p": ops.core.aten_log1p,
    "aten::log2": ops.core.aten_log2,
    "aten::logaddexp": ops.core.aten_logaddexp,
    "aten::logaddexp2": ops.core.aten_logaddexp2,
    "aten::logcumsumexp": ops.core.aten_logcumsumexp,
    "aten::logdet": ops.core.aten_logdet,
    "aten::logsigmoid": ops.nn.aten_log_sigmoid,
    "aten::logsumexp": ops.core.aten_logsumexp,
    "aten::lt": ops.core.aten_lt,
    "aten::matmul": ops.core.aten_matmul,
    "aten::maximum": ops.core.aten_maximum,
    "aten::minimum": ops.core.aten_minimum,
    "aten::mm": ops.core.aten_mm,
    "aten::mul": ops.core.aten_mul,
    "aten::native_layer_norm": ops.core.aten_native_layer_norm,
    "aten::ne": ops.core.aten_ne,
    "aten::neg": ops.core.aten_neg,
    "aten::new_full": ops.core.aten_new_full,
    "aten::nonzero": ops.core.aten_nonzero,
    "aten::ones_like": ops.core.aten_ones_like,
    "aten::ones": ops.core.aten_ones,
    "aten::permute": ops.core.aten_permute,
    "aten::pow": ops.core.aten_pow,
    "aten::reciprocal": ops.core.aten_reciprocal,
    "aten::relu": ops.nn.aten_relu,
    "aten::relu6": ops.nn.aten_relu6,
    "aten::remainder": ops.core.aten_remainder,
    "aten::repeat": ops.core.aten_repeat,
    "aten::reshape": ops.core.aten_reshape,
    "aten::round": ops.core.aten_round,
    "aten::rsqrt": ops.core.aten_rsqrt,
    "aten::rsub": ops.core.aten_rsub,
    "aten::selu": ops.core.aten_selu,
    "aten::sigmoid": ops.core.aten_sigmoid,
    "aten::sign": ops.core.aten_sign,
    "aten::sin": ops.core.aten_sin,
    "aten::sinh": ops.core.aten_sinh,
    "aten::slice": ops.core.aten_slice,
    "aten::softmax": ops.special.aten_special_softmax,
    "aten::split": ops.core.aten_split,
    "aten::sqrt": ops.core.aten_sqrt,
    "aten::sub": ops.core.aten_sub,
    "aten::sum": ops.core.aten_sum_dim_IntList,
    "aten::t": ops.core.aten_t,
    "aten::tan": ops.core.aten_tan,
    "aten::tanh": ops.core.aten_tanh,
    "aten::topk": ops.core.aten_topk,
    "aten::transpose": ops.core.aten_transpose,
    "aten::unsqueeze": ops.core.aten_unsqueeze,
    "aten::upsample_nearest2d": ops.nn.aten_upsample_nearest2d,
    "aten::view": ops.core.aten_view,
    "aten::where": ops.core.aten_where,
    "aten::xlogy": ops.special.aten_special_xlogy,
    "aten::zeros_like": ops.core.aten_zeros_like,
    "aten::zeros": ops.core.aten_zeros,
    "getitem": aten_getitem,
    "prims::convert_element_type": prims_convert_element_type,
}


def _create_op_overload_to_exporter_key_table() -> Dict[
    Union[torch._ops.OpOverload, Callable], str
]:
    # TODO(justinchuby): Improve how the table is constructed.
    table: Dict[Union[torch._ops.OpOverload, Callable], str] = {}

    for op_namespace in (torch.ops.aten, torch.ops.prims):
        for attr_name in dir(op_namespace):
            op_overload_packet = getattr(op_namespace, attr_name)

            if not isinstance(op_overload_packet, torch._ops.OpOverloadPacket):
                continue

            exporter_look_up_key = op_overload_packet._qualified_op_name
            if _ATENLIB_FUNCTIONS.get(exporter_look_up_key) is None:
                # This aten op doesn't have ONNX exporter.
                continue

            for overload_name in op_overload_packet.overloads():
                op_overload = getattr(op_overload_packet, overload_name)
                # This line maps torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar, torch.ops.aten.add.out, etc
                # to "aten::add". This means the exporter for "aten::add" is used for all overloads of "aten::add".
                # This is applied to all ops under torch.ops.aten.
                #
                # TODO(wechi): in the future, we might want to write individual exporter for each overload, if,
                # for example, they have different type promotion rules. If so, just map different overloads to
                # different exporter keys.

                table[op_overload] = op_overload_packet._qualified_op_name
    # TODO(justinchuby): is baddbmm different?
    table[torch.ops.aten.baddbmm.default] = "aten::baddbmm"
    return table


# Dictionary that maps torch.ops.aten.* to exporter look up key; e.g.,
# _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[torch.add.Tensor] is "aten::add".
_OP_OVERLOAD_TO_EXPORTER_KEY_TABLE = _create_op_overload_to_exporter_key_table()


@_beartype.beartype
def _create_onnx_friendly_decomposition_table() -> Dict[
    torch._ops.OpOverload, Callable
]:
    decomposition_table: Dict[torch._ops.OpOverload, Callable] = {}
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


# This is a subset of PyTorch's built-in aten-to-aten decomposition. If an aten
# op (e.g., torch.ops.aten.add.Tensor) has exporter, we exclude the op's decomposition
# function in the _ONNX_FRIENDLY_DECOMPOSITION_TABLE.
_ONNX_FRIENDLY_DECOMPOSITION_TABLE = _create_onnx_friendly_decomposition_table()
