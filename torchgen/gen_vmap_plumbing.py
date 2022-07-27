import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    OptionalType,
    Return,
    SchemaKind,
    Type,
)
from torchgen.utils import mapMaybe


def is_tensor(typ: Type) -> bool:
    return isinstance(typ, BaseType) and typ.name == BaseTy.Tensor


def is_optional_tensor(typ: Type) -> bool:
    return isinstance(typ, OptionalType) and is_tensor(typ.elem)


def is_tensor_list(typ: Type) -> bool:
    return isinstance(typ, ListType) and is_tensor(typ.elem)


def unwrap_tensor(name: str, cur_level_var: str) -> List[str]:
    result = f"""\
    Tensor {name}_value;
    optional<int64_t> {name}_bdim;
    std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}, {cur_level_var});"""
    return textwrap.dedent(result).split("\n")


def unwrap_optional_tensor(name: str, cur_level_var: str) -> List[str]:
    result = f"""\
    optional<Tensor> {name}_value;
    optional<int64_t> {name}_bdim;
    if ({name}) {{
        std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}.value(), {cur_level_var});
    }}"""
    return textwrap.dedent(result).split("\n")


def gen_unwraps(
    flat_arguments: Sequence[Argument], cur_level_var: str
) -> Tuple[str, List[str]]:
    arg_names = [a.name for a in flat_arguments]
    arg_types = [a.type for a in flat_arguments]

    tensors = [name for typ, name in zip(arg_types, arg_names) if is_tensor(typ)]
    optional_tensors = [
        name for typ, name in zip(arg_types, arg_names) if is_optional_tensor(typ)
    ]

    unwraps = []
    for tensor in tensors:
        unwraps += unwrap_tensor(tensor, cur_level_var)

    for opt_tensor in optional_tensors:
        unwraps += unwrap_optional_tensor(opt_tensor, cur_level_var)
    unwrap_code = "\n".join(unwraps)

    unwrapped_arg_list = []
    for arg in arg_names:
        if arg in tensors or arg in optional_tensors:
            unwrapped_arg_list += [f"{arg}_value", f"{arg}_bdim"]
        else:
            unwrapped_arg_list.append(arg)
    return unwrap_code, unwrapped_arg_list


def gen_case_where_all_bdims_are_none(
    schema: FunctionSchema, cur_level_var: str
) -> str:
    conditions = []
    flat_args = schema.arguments.flat_all
    for arg in flat_args:
        if not arg.type.is_tensor_like():
            continue
        conditions.append(f"!isBatchedAtLevel({arg.name}, {cur_level_var})")

    sig = DispatcherSignature.from_schema(schema)
    translated_args = ", ".join(
        e.expr for e in translate(sig.arguments(), sig.arguments())
    )
    return f"""\
if ({' && '.join(conditions)}) {{
  return at::_ops::{sig.func.name.unambiguous_name()}::call({translated_args});
}}"""


def gen_returns(
    returns: Tuple[Return, ...], cur_level_var: str, results_var: str
) -> str:
    idx = 0
    wrapped_returns = []
    for ret in returns:
        if is_tensor(ret.type):
            wrapped_returns.append(
                f"makeBatched(std::get<{idx}>({results_var}), std::get<{idx + 1}>({results_var}), {cur_level_var})"
            )
            idx += 2
        elif is_tensor_list(ret.type):
            wrapped_returns.append(
                f"makeBatchedVector(std::get<{idx}>({results_var}), std::get<{idx+1}>({results_var}), {cur_level_var})"
            )
            idx += 2
        else:
            wrapped_returns.append(f"std::get<{idx}>({results_var})")
            idx += 1
    if len(wrapped_returns) == 1:
        result = f"return {wrapped_returns[0]};"
    else:
        result = f'return std::make_tuple({", ".join(wrapped_returns)});'
    return result


def accepts_at_least_one_tensor_input(schema: FunctionSchema) -> bool:
    return any(a.type.is_tensor_like() for a in schema.arguments.flat_all)


def is_mutated_arg(argument: Argument) -> bool:
    return argument.annotation is not None and argument.annotation.is_write


def gen_vmap_inplace_plumbing(native_function: NativeFunction) -> Optional[str]:
    # Assumptions:
    # - only one argument is being modified in-place
    # - the argument that is being modified in-place is the first argument
    # - all returns are either Tensor, tuple of Tensor, or TensorList
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    returns = schema.returns

    # Check assumptions. If these are invalid we return None
    # and punt the work to handle them to the future.
    assert schema.kind() == SchemaKind.inplace
    if not is_mutated_arg(schema.arguments.flat_all[0]):
        return None
    if not len([arg for arg in schema.arguments.flat_all if is_mutated_arg(arg)]) == 1:
        return None

    # Only support cases where all returns are Tensors or vector<Tensor>
    if len(returns) == 0:
        return None
    if not all(is_tensor(ret.type) or is_tensor_list(ret.type) for ret in returns):
        return None
    if not accepts_at_least_one_tensor_input(schema):
        return None

    cur_level_var = "cur_level"

    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(schema, cur_level_var)

    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t {cur_level_var} = maybe_layer->layerId();
{textwrap.indent(bdims_all_none_case, "  ")}
{textwrap.indent(unwraps, "  ")}
  batch_rule({', '.join(unwrapped_arg_list)});
  return {schema.arguments.flat_all[0].name};
}}"""


def gen_vmap_plumbing_no_returns(native_function: NativeFunction) -> str:
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    cur_level_var = "cur_level"

    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(schema, cur_level_var)

    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t {cur_level_var} = maybe_layer->layerId();
{textwrap.indent(bdims_all_none_case, "  ")}
{textwrap.indent(unwraps, "  ")}
  batch_rule({', '.join(unwrapped_arg_list)});
}}"""


def gen_vmap_plumbing(native_function: NativeFunction) -> Optional[str]:
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    returns = schema.returns

    # Only support cases where all returns are Tensors or vector<Tensor>
    if len(returns) == 0:
        return gen_vmap_plumbing_no_returns(native_function)
    if not all(ret.type.is_tensor_like() for ret in returns):
        return None
    if not accepts_at_least_one_tensor_input(schema):
        return None
    # in-place views need special handling
    if "inplace_view" in native_function.tags:
        return None

    if schema.kind() == SchemaKind.inplace:
        return gen_vmap_inplace_plumbing(native_function)

    # Don't support these
    if schema.kind() == SchemaKind.out:
        return None

    # From now on, assume we're dealing with a functional (out-of-place) operation
    assert schema.kind() == SchemaKind.functional

    results_var = "results"
    cur_level_var = "cur_level"

    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(schema, cur_level_var)

    wrapped_returns = gen_returns(returns, cur_level_var, results_var)
    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t {cur_level_var} = maybe_layer->layerId();
{textwrap.indent(bdims_all_none_case, "  ")}
{textwrap.indent(unwraps, "  ")}
  auto {results_var} = batch_rule({', '.join(unwrapped_arg_list)});
  {wrapped_returns}
}}"""


allowlist = {
    "glu_backward",
    "glu",
    "prelu",
    "prelu_backward",
    "__lshift__.Tensor",
    "__lshift__.Scalar",
    "__rshift__.Tensor",
    "__rshift__.Scalar",
    "add.Tensor",
    "add.Scalar",
    "atan2",
    "bitwise_and.Tensor",
    "bitwise_and.Scalar",
    "bitwise_or.Tensor",
    "bitwise_xor.Tensor",
    "bitwise_left_shift.Tensor",
    "bitwise_left_shift.Tensor_Scalar",
    "bitwise_right_shift.Tensor",
    "bitwise_right_shift.Tensor_Scalar",
    "clamp",
    "clamp_min.Tensor",
    "clamp_min",
    "clamp_max.Tensor",
    "clamp_max",
    "_cdist_forward",
    "_cdist_backward",
    "div.Tensor",
    "div.Scalar",
    "div.Tensor_mode",
    "div.Scalar_mode",
    "floor_divide",
    "floor_divide.Scalar",
    "fmax",
    "fmin",
    "fmod.Tensor",
    "fmod.Scalar",
    "heaviside",
    "hypot",
    "gcd",
    "igamma",
    "igammac",
    "linalg_householder_product",
    "logaddexp",
    "logaddexp2",
    "lcm",
    "_linalg_check_errors",
    "maximum",
    "minimum",
    "mul.Tensor",
    "mul.Scalar",
    "nextafter",
    "pow.Tensor_Tensor",
    "pow.Tensor_Scalar",
    "polar",
    "sub.Tensor",
    "sub.Scalar",
    "remainder.Tensor",
    "remainder.Scalar",
    "rrelu_with_noise",
    "rsub.Tensor",
    "rsub.Scalar",
    "special_xlog1py",
    "special_xlog1py.other_scalar",
    "special_xlogy",
    "special_xlogy.other_scalar",
    "special_zeta",
    "special_zeta.other_scalar",
    "where.self",
    "xlogy.Tensor",
    "xlogy.Scalar_Other",
    "hardsigmoid_backward",
    "hardtanh_backward",
    "hardshrink_backward",
    "hardswish_backward",
    "leaky_relu_backward",
    "logit_backward",
    "gelu_backward",
    "sigmoid_backward",
    "softshrink_backward",
    "tanh_backward",
    "threshold_backward",
    "silu_backward",
    "add_.Scalar",
    "sub_.Tensor",
    "sub_.Scalar",
    "mul_.Tensor",
    "mul_.Scalar",
    "div_.Tensor",
    "div_.Scalar",
    "clamp_min_.Tensor",
    "clamp_max_.Tensor",
    "masked_fill_.Scalar",
    "copy_",
    "eq.Tensor",
    "eq.Scalar",
    "gt.Tensor",
    "gt.Scalar",
    "ge.Tensor",
    "ge.Scalar",
    "le.Tensor",
    "le.Scalar",
    "lt.Tensor",
    "lt.Scalar",
    "ne.Tensor",
    "ne.Scalar",
    "logical_and",
    "logical_and_",
    "logical_or",
    "logical_or_",
    "logical_xor",
    "logical_xor_",
    "masked_select",
    "masked_select_backward",
    "convolution",
    "ones_like",
    "zeros_like",
    "empty_like",
    "randn_like",
    "rand_like",
    "full_like",
    "new_empty",
    "new_zeros",
    "new_ones",
    "new_full",
    "_new_zeros_with_same_feature_meta",
    "bmm",
    "dot",
    "mv",
    "mm",
    "linalg_pinv",
    "cholesky",
    "cholesky_inverse",
    "logdet",
    "matrix_exp",
    "pinverse",
    "inverse",
    "mse_loss",
    "mse_loss_backward",
    "im2col",
    "im2col_backward",
    "embedding",
    "embedding_dense_backward",
    "grid_sampler_2d",
    "grid_sampler_2d_backward",
    "grid_sampler_3d",
    "grid_sampler_3d_backward",
    "cudnn_grid_sampler_backward",
    "cudnn_grid_sampler",
    "cross",
    "pixel_shuffle",
    "pixel_unshuffle",
    "constant_pad_nd",
    "reflection_pad1d",
    "reflection_pad2d",
    "reflection_pad3d",
    "replication_pad1d",
    "replication_pad2d",
    "replication_pad3d",
    "upsample_bicubic2d.vec",
    "upsample_bicubic2d",
    "upsample_bilinear2d.vec",
    "upsample_bilinear2d",
    "upsample_linear1d.vec",
    "upsample_linear1d",
    "upsample_nearest1d.vec",
    "upsample_nearest1d",
    "upsample_nearest2d.vec",
    "upsample_nearest2d",
    "upsample_nearest3d.vec",
    "upsample_nearest3d",
    "upsample_trilinear3d.vec",
    "upsample_trilinear3d",
    "upsample_bicubic2d_backward.vec",
    "upsample_bilinear2d_backward.vec",
    "upsample_linear1d_backward.vec",
    "upsample_nearest1d_backward.vec",
    "upsample_nearest2d_backward.vec",
    "upsample_nearest3d_backward.vec",
    "upsample_trilinear3d_backward.vec",
    "native_batch_norm",
    "cudnn_batch_norm",
    "miopen_batch_norm",
    "native_layer_norm",
    "_adaptive_avg_pool2d",
    "_adaptive_avg_pool3d",
    "avg_pool2d",
    "avg_pool3d",
    "max_pool2d_with_indices",
    "aminmax",
    "_log_softmax_backward_data",
    "_softmax_backward_data",
    "index_add",
    "diagonal_scatter",
    "gather",
    "gather_backward",
    "scatter.value",
    "scatter.src",
    "scatter_add",
    "scatter.reduce",
    "scatter.value_reduce",
    "imag",
    "real",
    "view_as_real",
    "view_as_complex",
    "clone",
    "contiguous",
    "to.device",
    "to.dtype",
    "to.dtype_layout",
    "to.other",
    "_to_copy",
    "alias",
    "abs",
    "acos",
    "acosh",
    "angle",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "bitwise_not",
    "ceil",
    "cos",
    "cosh",
    "_conj",
    "deg2rad",
    "detach",
    "digamma",
    "erf",
    "exp",
    "expm1",
    "floor",
    "frac",
    "isfinite",
    "isnan",
    "isinf",
    "isposinf",
    "isneginf",
    "isreal",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_not",
    "logit",
    "mish",
    "mvlgamma",
    "nan_to_num",
    "neg",
    "positive",
    "rad2deg",
    "reciprocal",
    "masked_fill.Scalar",
    "round",
    "round.decimals",
    "rsqrt",
    "sgn",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "tan",
    "threshold",
    "trunc",
    "i0",
    "erfc",
    "erfinv",
    "exp2",
    "special_entr",
    "special_erf",
    "special_erfc",
    "special_erfcx",
    "special_erfinv",
    "special_expit",
    "special_expm1",
    "special_digamma",
    "special_psi",
    "special_exp2",
    "special_gammaln",
    "special_i0",
    "special_i0e",
    "special_i1",
    "special_i1e",
    "special_log1p",
    "special_ndtr",
    "special_ndtri",
    "special_round",
    "special_sinc",
    "elu",
    "hardshrink",
    "hardsigmoid",
    "hardtanh",
    "hardswish",
    "leaky_relu",
    "log_sigmoid",
    "relu",
    "relu6",
    "selu",
    "celu",
    "gelu",
    "sigmoid",
    "silu",
    "softplus",
    "softshrink",
    "tanh",
    "diag",
    "chunk",
    "flip",
    "tril",
    "triu",
    "repeat",
    "_unsafe_view",
    "unsqueeze",
    "select.int",
    "squeeze",
    "squeeze.dim",
    "_reshape_alias",
    "roll",
    "permute",
    "diagonal",
    "diagonal_backward",
    "select_backward",
    "slice_backward",
    "view",
    "expand",
    "expand_copy",
    "unfold",
    "movedim.intlist",
    "slice.Tensor",
    "transpose.int",
    "diag_embed",
    "searchsorted.Tensor",
}


@dataclass(frozen=True)
class ComputeBatchRulePlumbing:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        opname = str(f.func.name)
        if opname not in allowlist:
            return None
        result = gen_vmap_plumbing(f)
        return result


def gen_all_vmap_plumbing(native_functions: Sequence[NativeFunction]) -> str:
    body = "\n".join(list(mapMaybe(ComputeBatchRulePlumbing(), native_functions)))
    return f"""
#pragma once
#include <ATen/Operators.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/Constants.h>

namespace at {{ namespace functorch {{

{body}

}}}} // namespace at::functorch
"""
