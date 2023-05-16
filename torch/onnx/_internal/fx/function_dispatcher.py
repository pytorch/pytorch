"""Dispatcher for AtenLib functions from onnx-script."""

from __future__ import annotations

import operator
import types
from typing import Any, Callable, Dict, Mapping, Union

import onnxscript  # type: ignore[import]
from onnxscript import opset18  # type: ignore[import]
from onnxscript.function_libs.torch_lib import ops  # type: ignore[import]

import torch._ops

import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import diagnostics


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
    "aten::alias": ops.core.aten_alias,
    "aten::all": ops.core.aten_all,
    "aten::allclose": ops.core.aten_allclose,
    "aten::amax": ops.core.aten_amax,
    "aten::amin": ops.core.aten_amin,
    "aten::any": ops.core.aten_any,
    "aten::arange": ops.core.aten_arange_start,
    "aten::argmax": ops.core.aten_argmax,
    "aten::argmin": ops.core.aten_argmin,
    "aten::as_strided": ops.core.aten_as_strided,
    "aten::asin": ops.core.aten_asin,
    "aten::asinh": ops.core.aten_asinh,
    "aten::atan": ops.core.aten_atan,
    "aten::atanh": ops.core.aten_atanh,
    "aten::avg_pool2d": ops.nn.aten_avg_pool2d,
    "aten::baddbmm": ops.core.aten_baddbmm,
    "aten::bitwise_not": ops.core.aten_bitwise_not_bool,
    "aten::bmm": ops.core.aten_bmm,
    "aten::broadcast_to": ops.core.aten_broadcast_to,
    "aten::cat": ops.core.aten_cat,
    "aten::ceil": ops.core.aten_ceil,
    "aten::celu": ops.nn.aten_celu,
    "aten::chunk": ops.core.aten_chunk,
    "aten::clamp_max": ops.core.aten_clamp_max,
    "aten::clamp_min": ops.core.aten_clamp_min,
    "aten::clamp": ops.core.aten_clamp,
    "aten::clone": ops.core.aten_clone,
    "aten::col2im": ops.nn.aten_col2im,
    "aten::constant_pad_nd": ops.core.aten_constant_pad_nd,
    "aten::contiguous": ops.core.aten_contiguous,
    "aten::conv1d": ops.core.aten_conv1d,
    "aten::conv2d": ops.core.aten_conv2d,
    "aten::conv3d": ops.core.aten_conv3d,
    "aten::convolution": ops.core.aten_convolution,
    "aten::copy": ops.core.aten_copy,
    "aten::cos": ops.core.aten_cos,
    "aten::cosh": ops.core.aten_cosh,
    "aten::cross_entropy_loss": ops.nn.aten_cross_entropy_loss,
    "aten::cross": ops.core.aten_cross,
    "aten::cumsum": ops.core.aten_cumsum,
    "aten::detach": ops.core.aten_detach,
    "aten::div": ops.core.aten_div,
    "aten::dot": ops.core.aten_dot,
    "aten::dropout": ops.core.aten_dropout,
    "aten::elu": ops.nn.aten_elu,
    "aten::embedding": ops.core.aten_embedding,
    "aten::empty_like": ops.core.aten_empty_like,
    "aten::empty_strided": ops.core.aten_empty_strided,
    "aten::empty": ops.core.aten_empty,
    "aten::eq": ops.core.aten_eq,
    "aten::equal": ops.core.aten_equal,
    "aten::erf": ops.core.aten_erf,
    "aten::exp": ops.core.aten_exp,
    "aten::exp2": ops.core.aten_exp2,
    "aten::expand_as": ops.core.aten_expand_as,
    "aten::expand": ops.core.aten_expand,
    "aten::fill": ops.core.aten_fill,
    "aten::flip": ops.core.aten_flip,
    "aten::floor": ops.core.aten_floor,
    "aten::fmod": ops.core.aten_fmod,
    "aten::full_like": ops.core.aten_full_like,
    "aten::full": ops.core.aten_full,
    "aten::gather": ops.core.aten_gather,
    "aten::ge": ops.core.aten_ge,
    "aten::gelu": ops.nn.aten_gelu,
    "aten::greater_equal": ops.core.aten_greater_equal,
    "aten::greater": ops.core.aten_greater,
    "aten::grid_sampler_2d": ops.core.aten_grid_sampler_2d,
    "aten::grid_sampler": ops.core.aten_grid_sampler,
    "aten::gt": ops.core.aten_gt,
    "aten::hardtanh": ops.nn.aten_hardtanh,
    "aten::index_put": ops.core.aten_index_put,
    "aten::index_select": ops.core.aten_index_select,
    "aten::is_nonzero": ops.core.aten_is_nonzero,
    "aten::is_same_size": ops.core.aten_is_same_size,
    "aten::isclose": ops.core.aten_isclose,
    "aten::isfinite": ops.core.aten_isfinite,
    "aten::isinf": ops.core.aten_isinf,
    "aten::isnan": ops.core.aten_isnan,
    "aten::isneginf": ops.core.aten_isneginf,
    "aten::isposinf": ops.core.aten_isposinf,
    "aten::layer_norm": ops.core.aten_layer_norm,
    "aten::le": ops.core.aten_le,
    "aten::leaky_relu": ops.nn.aten_leaky_relu,
    "aten::linear": ops.nn.aten_linear,
    "aten::log_sigmoid": ops.nn.aten_log_sigmoid,
    "aten::log_sigmoid_forward": ops.nn.aten_log_sigmoid,
    "aten::log_softmax": ops.special.aten_special_log_softmax,
    "aten::log": ops.core.aten_log,
    "aten::log10": ops.core.aten_log10,
    "aten::log1p": ops.core.aten_log1p,
    "aten::log2": ops.core.aten_log2,
    "aten::logaddexp": ops.core.aten_logaddexp,
    "aten::logaddexp2": ops.core.aten_logaddexp2,
    "aten::logcumsumexp": ops.core.aten_logcumsumexp,
    "aten::logdet": ops.core.aten_logdet,
    "aten::logsumexp": ops.core.aten_logsumexp,
    "aten::lt": ops.core.aten_lt,
    "aten::masked_fill": ops.core.aten_masked_fill,
    "aten::matmul": ops.core.aten_matmul,
    "aten::max_pool2d_with_indices": ops.nn.aten_max_pool2d_with_indices,
    "aten::max_pool2d": ops.nn.aten_max_pool2d,
    "aten::max_pool3d_with_indices": ops.nn.aten_max_pool3d_with_indices,
    "aten::max_pool3d": ops.nn.aten_max_pool3d,
    "aten::max": ops.core.aten_max,
    "aten::maximum": ops.core.aten_maximum,
    "aten::min": ops.core.aten_min,
    "aten::minimum": ops.core.aten_minimum,
    "aten::mm": ops.core.aten_mm,
    "aten::mse_loss": ops.nn.aten_mse_loss,
    "aten::mul": ops.core.aten_mul,
    "aten::narrow": ops.core.aten_narrow,
    "aten::native_dropout": ops.core.aten_native_dropout,
    "aten::native_batch_norm": ops.core.aten_native_batch_norm,
    "aten::native_layer_norm": ops.core.aten_native_layer_norm,
    "aten::ne": ops.core.aten_ne,
    "aten::neg": ops.core.aten_neg,
    "aten::new_empty": ops.core.aten_new_empty,
    "aten::new_full": ops.core.aten_new_full,
    "aten::new_ones": ops.core.aten_new_ones,
    "aten::new_zeros": ops.core.aten_new_zeros,
    "aten::nll_loss": ops.nn.aten_nll_loss,
    "aten::nonzero": ops.core.aten_nonzero,
    "aten::normal": ops.core.aten_normal,
    "aten::ones_like": ops.core.aten_ones_like,
    "aten::ones": ops.core.aten_ones,
    "aten::permute": ops.core.aten_permute,
    "aten::pow": ops.core.aten_pow,
    "aten::rand": ops.core.aten_rand,
    "aten::randn": ops.core.aten_randn,
    "aten::reciprocal": ops.core.aten_reciprocal,
    "aten::reflection_pad2d": ops.nn.aten_reflection_pad2d,
    "aten::relu": ops.nn.aten_relu,
    "aten::relu6": ops.nn.aten_relu6,
    "aten::remainder": ops.core.aten_remainder,
    "aten::repeat": ops.core.aten_repeat,
    "aten::replication_pad2d": ops.nn.aten_replication_pad2d,
    "aten::replication_pad3d": ops.nn.aten_replication_pad3d,
    "aten::reshape": ops.core.aten_reshape,
    "aten::resolve_conj": ops.core.aten_resolve_conj,
    "aten::resolve_neg": ops.core.aten_resolve_neg,
    "aten::round": ops.core.aten_round,
    "aten::rsqrt": ops.core.aten_rsqrt,
    "aten::rsub": ops.core.aten_rsub,
    "aten::scalar_tensor": ops.core.aten_scalar_tensor,
    "aten::scaled_dot_product_attention": ops.nn.aten_scaled_dot_product_attention,
    "aten::scatter_add": ops.core.aten_scatter_add,
    "aten::scatter_reduce": ops.core.aten_scatter_reduce,
    "aten::select": ops.core.aten_select,
    "aten::selu": ops.core.aten_selu,
    "aten::sigmoid": ops.core.aten_sigmoid,
    "aten::sign": ops.core.aten_sign,
    "aten::sin": ops.core.aten_sin,
    "aten::sinh": ops.core.aten_sinh,
    "aten::slice_scatter": ops.core.aten_slice_scatter,
    "aten::slice": ops.core.aten_slice,
    "aten::softmax": ops.special.aten_special_softmax,
    "aten::split_with_sizes": ops.core.aten_split_with_sizes,
    "aten::split": ops.core.aten_split,
    "aten::sqrt": ops.core.aten_sqrt,
    "aten::squeeze": ops.core.aten_squeeze,
    "aten::stack": ops.core.aten_stack,
    "aten::sub": ops.core.aten_sub,
    "aten::sum": ops.core.aten_sum_dim_IntList,
    "aten::sym_size": ops.core.aten_sym_size,
    "aten::t": ops.core.aten_t,
    "aten::tan": ops.core.aten_tan,
    "aten::tanh": ops.core.aten_tanh,
    "aten::topk": ops.core.aten_topk,
    "aten::transpose": ops.core.aten_transpose,
    "aten::tril": ops.core.aten_tril,
    "aten::triu": ops.core.aten_triu,
    "aten::trunc": ops.core.aten_trunc,
    "aten::unflatten": ops.core.aten_unflatten,
    "aten::unsqueeze": ops.core.aten_unsqueeze,
    "aten::upsample_bilinear2d": ops.nn.aten_upsample_bilinear2d,
    "aten::upsample_nearest2d": ops.nn.aten_upsample_nearest2d,
    "aten::var_mean": ops.core.aten_var_mean,
    "aten::view": ops.core.aten_view,
    "aten::where": ops.core.aten_where,
    "aten::xlogy": ops.special.aten_special_xlogy,
    "aten::zeros_like": ops.core.aten_zeros_like,
    "aten::zeros": ops.core.aten_zeros,
    "getitem": aten_getitem,
    "prims::convert_element_type": prims_convert_element_type,
}


def _create_op_overload_to_exporter_key_table() -> (
    Mapping[Union[torch._ops.OpOverload, Callable], str]
):
    # TODO(justinchuby): Improve how the table is constructed.
    table: Dict[Union[torch._ops.OpOverload, Callable], str] = {}

    # Some ops in `torch.ops.aten` are not discoverable through `dir(torch.ops.aten)`,
    # but retrievable via explicit lookup.
    # https://github.com/pytorch/pytorch/issues/99681
    # This is a workaround to make sure we register ONNX symbolic functions for these.
    onnx_supported_aten_lookup_table = [
        k.split("::")[1] for k in _ATENLIB_FUNCTIONS.keys() if k.startswith("aten::")
    ]

    for op_namespace in (torch.ops.aten, torch.ops.prims):
        attr_names = dir(op_namespace)
        if op_namespace is torch.ops.aten:
            attr_names += onnx_supported_aten_lookup_table
        for attr_name in attr_names:
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
    return table


# Dictionary that maps torch.ops.aten.* to exporter look up key; e.g.,
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
    Dict[torch._ops.OpOverload, Callable]
):
    decomposition_table: Dict[torch._ops.OpOverload, Callable] = {}
    for op_overload, decomp_fn in torch._decomp.decomposition_table.items():
        # Skip decomposition into "prim::*" ops (defined in 'torch._refs'), because they
        # are not generally supported by ONNX.
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX
        # symbolic function.
        if (
            "torch._refs" in decomp_fn.__module__
            or op_overload in _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
        ):
            continue
        decomposition_table[op_overload] = decomp_fn
    return decomposition_table


# This is a subset of PyTorch's built-in aten-to-aten decomposition. If an aten
# op (e.g., torch.ops.aten.add.Tensor) has exporter, we exclude the op's decomposition
# function in the DEFAULT_ONNX_EXPORTER_DECOMPOSITION_TABLE.
DEFAULT_ONNX_EXPORTER_DECOMPOSITION_TABLE: Dict[
    torch._ops.OpOverload, Callable
] = _create_onnx_friendly_decomposition_table()


def get_symbolic_function(
    diagnostic_context: diagnostics.DiagnosticContext,
    node: torch.fx.Node,
) -> Union[onnxscript.OnnxFunction, Callable[..., Any]]:
    if node.target == operator.getitem:
        # __getitem__ on Tensor or Sequence of tensors. Not tuple.
        exporter_key = "getitem"
    elif (
        isinstance(node.target, types.BuiltinFunctionType)
        and node.target in _SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE
    ):
        for node_arg in node.args:
            if (not isinstance(node_arg, (torch.fx.Node, int, float))) or (
                isinstance(node_arg, torch.fx.Node)
                and not isinstance(node_arg.meta["val"], (torch.SymInt, torch.SymFloat))
            ):
                # TODO: reduce number of explicit initializations.
                # TODO: Log location, stack.
                diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
                    diagnostics.rules.no_symbolic_function_for_call_function,
                    diagnostics.levels.ERROR,
                    f"Unsupported node arg: {node_arg} with builtin function: {node.target},"
                    " only int/float/SymInt/SymFloat is supported with built-in ops!",
                    unsupported_fx_node=node,
                )
                diagnostic_context.log(diagnostic)
                raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)

        # symbolic fx.graph contains built-in functions to calculate python values.
        exporter_key = _SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE[
            node.target  # type: ignore[index]
        ]
    elif (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target in _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
    ):
        exporter_key = _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[node.target]
    elif isinstance(node.target, torch._ops.OpOverloadPacket):
        # aten::sym_size is the only OverloadPacket that we support.
        # schema: aten::sym_size(Tensor self, int dim) -> Tensor
        if node.target != torch.ops.aten.sym_size:
            diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
                diagnostics.rules.no_symbolic_function_for_call_function,
                diagnostics.levels.ERROR,
                f"Unsupported OverloadPacket: {node.target}, aten.sym_size is the only allowed OverloadPacket!",
                unsupported_fx_node=node,
            )
            diagnostic_context.log(diagnostic)
            raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
        # TODO(titaiwang): aten::sym_size has overload, but fx graph is using
        # overloadpacket for some reasons.
        # https://github.com/pytorch/pytorch/issues/97201
        # We manually assigned overload for aten::sym_size.
        exporter_key = _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[torch.ops.aten.sym_size.int]
    else:
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
            diagnostics.rules.no_symbolic_function_for_call_function,
            diagnostics.levels.ERROR,
            f"Unknown call_function target: {node.target}",
            unsupported_fx_node=node,
        )
        diagnostic_context.log(diagnostic)
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
    # Only the latest opset version is only supported in atenlib for now
    symbolic_fn = _ATENLIB_FUNCTIONS.get(exporter_key)
    if symbolic_fn is None:
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
            diagnostics.rules.no_symbolic_function_for_call_function,
            diagnostics.levels.ERROR,
            f"Cannot find symbolic function for {exporter_key}, "
            f"which should be registered under {node.target}.",
            unsupported_fx_node=node,
        )
        diagnostic_context.log(diagnostic)
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)

    return symbolic_fn
