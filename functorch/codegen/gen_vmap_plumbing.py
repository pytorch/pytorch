from tools.codegen.api.types import (
    DispatcherSignature,
)
from tools.codegen.model import (
    BaseTy, OptionalType, BaseType, ListType, NativeFunction, Type,
    Argument, Return, SchemaKind, Tag
)
from tools.codegen.api.translate import translate
from tools.codegen.context import method_with_native_function
from tools.codegen.utils import mapMaybe
from dataclasses import dataclass
from typing import List, Optional, Tuple
import textwrap


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
    return textwrap.dedent(result).split('\n')


def unwrap_optional_tensor(name: str, cur_level_var: str) -> List[str]:
    result = f"""\
    optional<Tensor> {name}_value;
    optional<int64_t> {name}_bdim;
    if ({name}) {{
        std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}.value(), {cur_level_var});
    }}"""
    return textwrap.dedent(result).split('\n')


def gen_unwraps(flat_arguments: List[Argument], cur_level_var: str) -> Tuple[List[str], List[str]]:
    arg_names = [a.name for a in flat_arguments]
    arg_types = [a.type for a in flat_arguments]

    tensors = [name for typ, name in zip(arg_types, arg_names) if is_tensor(typ)]
    optional_tensors = [name for typ, name in zip(arg_types, arg_names) if is_optional_tensor(typ)]

    unwraps = []
    for tensor in tensors:
        unwraps += unwrap_tensor(tensor, cur_level_var)

    for opt_tensor in optional_tensors:
        unwraps += unwrap_optional_tensor(opt_tensor, cur_level_var)
    unwraps = '\n'.join(unwraps)

    unwrapped_arg_list = []
    for arg in arg_names:
        if arg in tensors or arg in optional_tensors:
            unwrapped_arg_list += [f'{arg}_value', f'{arg}_bdim']
        else:
            unwrapped_arg_list.append(arg)
    return unwraps, unwrapped_arg_list


def get_aten_op_call(schema) -> str:
    if schema.name.overload_name:
        return f'ATEN_FN2({schema.name.name}, {schema.name.overload_name})'
    return f'ATEN_FN({schema.name.name})'


def gen_case_where_all_bdims_are_none(schema, cur_level_var) -> str:
    conditions = []
    flat_args = schema.arguments.flat_all
    for arg in flat_args:
        if not arg.type.is_tensor_like():
            continue
        conditions.append(f'!isBatchedAtLevel({arg.name}, {cur_level_var})')

    sig = DispatcherSignature.from_schema(schema)
    translated_args = ', '.join(e.expr for e in translate(sig.arguments(), sig.arguments()))
    return f"""\
if ({' && '.join(conditions)}) {{
  return at::_ops::{sig.func.name.unambiguous_name()}::call({translated_args});
}}"""


def gen_returns(returns: List[Return], cur_level_var: str, results_var: str) -> str:
    idx = 0
    wrapped_returns = []
    for ret in returns:
        if is_tensor(ret.type):
            wrapped_returns.append(
                f'makeBatched(std::get<{idx}>({results_var}), std::get<{idx + 1}>({results_var}), {cur_level_var})')
            idx += 2
        elif is_tensor_list(ret.type):
            wrapped_returns.append(
                f'makeBatchedVector(std::get<{idx}>({results_var}), std::get<{idx+1}>({results_var}), {cur_level_var})'
            )
            idx += 2
        else:
            wrapped_returns.append(f'std::get<{idx}>({results_var})')
            idx += 1
    if len(wrapped_returns) == 1:
        wrapped_returns = f'return {wrapped_returns[0]};'
    else:
        wrapped_returns = f'return std::make_tuple({", ".join(wrapped_returns)});'
    return wrapped_returns


def accepts_at_least_one_tensor_input(schema):
    return any(a.type.is_tensor_like() for a in schema.arguments.flat_all)


def is_mutated_arg(argument):
    return argument.annotation is not None and argument.annotation.is_write


def gen_vmap_inplace_plumbing(native_function):
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

    cur_level_var = 'cur_level'

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


def gen_vmap_plumbing(native_function: NativeFunction) -> str:
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    returns = schema.returns

    # Only support cases where all returns are Tensors or vector<Tensor>
    if len(returns) == 0:
        return None
    if not all(ret.type.is_tensor_like() for ret in returns):
        return None
    if not accepts_at_least_one_tensor_input(schema):
        return None
    # in-place views need special handling
    if native_function.tag == Tag.inplace_view:
        return None

    if schema.kind() == SchemaKind.inplace:
        return gen_vmap_inplace_plumbing(native_function)

    # Don't support these
    if schema.kind() == SchemaKind.out:
        return None

    # From now on, assume we're dealing with a functional (out-of-place) operation
    assert schema.kind() == SchemaKind.functional

    results_var = 'results'
    cur_level_var = 'cur_level'

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


@dataclass(frozen=True)
class ComputeBatchRulePlumbing:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        result = gen_vmap_plumbing(f)
        return result


def gen_all_vmap_plumbing(native_functions):
    body = '\n'.join(list(mapMaybe(ComputeBatchRulePlumbing(), native_functions)))
    return f"""
#pragma once
#include <ATen/Operators.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/Constants.h>

namespace at {{ namespace functorch {{

{body}

}}}} // namespace at::functorch
"""
