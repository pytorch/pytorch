from tools.codegen.api.types import (
    DispatcherSignature,
)
from tools.codegen.model import (
    BaseTy, Variant, OptionalType, BaseType, ListType, NativeFunction, Type,
    Argument, Return, SchemaKind, Tag
)
from tools.codegen.context import method_with_native_function
from tools.codegen.utils import mapMaybe
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set, Any, Union, Sequence, TypeVar
import textwrap


def is_tensor(typ: Type) -> bool:
    return isinstance(typ, BaseType) and typ.name == BaseTy.Tensor


def is_optional_tensor(typ: Type) -> bool:
    return isinstance(typ, OptionalType) and is_tensor(typ.elem)


def is_vector_tensor(typ: Type) -> bool:
    return isinstance(typ, ListType) and is_tensor(typ.elem)


def unwrap_tensor(name: str) -> List[str]:
    result = f"""\
    Tensor {name}_value;
    optional<int64_t> {name}_bdim;
    std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}, cur_level);"""
    return textwrap.dedent(result).split('\n')


def unwrap_optional_tensor(name: str) -> List[str]:
    result = f"""\
    optional<Tensor> {name}_value;
    optional<int64_t> {name}_bdim;
    if ({name}) {{
        std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}.value(), cur_level);
    }}"""
    return textwrap.dedent(result).split('\n')


def gen_unwraps(flat_arguments: List[Argument]) -> Tuple[List[str], List[str]]:
    arg_names = [a.name for a in flat_arguments]
    arg_types = [a.type for a in flat_arguments]

    tensors = [name for typ, name in zip(arg_types, arg_names) if is_tensor(typ)]
    optional_tensors = [name for typ, name in zip(arg_types, arg_names) if is_optional_tensor(typ)]

    unwraps = []
    for tensor in tensors:
        unwraps += unwrap_tensor(tensor)

    for opt_tensor in optional_tensors:
        unwraps += unwrap_optional_tensor(opt_tensor)
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


def gen_case_where_all_bdims_are_none(flat_args: List[Argument], schema) -> str:
    conditions = []
    for arg in flat_args:
        if not arg.type.is_tensor_like():
            continue
        conditions.append(f'!isBatchedAtLevel({arg.name}, cur_level)')
    aten_op = get_aten_op_call(schema)
    arg_names = [a.name for a in flat_args]

    return f"""\
if ({' && '.join(conditions)}) {{
  return {aten_op}({', '.join(arg_names)});
}}"""


def gen_returns(returns: List[Return]) -> str:
    idx = 0
    wrapped_returns = []
    for ret in returns:
        if is_tensor(ret.type):
            wrapped_returns.append(f'makeBatched(std::get<{idx}>(results), std::get<{idx + 1}>(results), cur_level)')
            idx += 2
        elif is_vector_tensor(ret.type):
            wrapped_returns.append(
                f'makeBatchedVector(std::get<{idx}>(results), std::get<{idx + 1}>(results), cur_level)'
            )
            idx += 2
        else:
            wrapped_returns.append(f'std::get<{idx}>(results)')
            idx += 1
    if len(wrapped_returns) == 1:
        wrapped_returns = f'return {wrapped_returns[0]};'
    else:
        wrapped_returns = f'return std::make_tuple({", ".join(wrapped_returns)});'
    return wrapped_returns


def accepts_at_least_one_tensor_input(schema):
    for arg in schema.arguments.flat_all:
        if arg.type.is_tensor_like():
            return True
    return False


def gen_vmap_inplace_plumbing(native_function):
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    returns = schema.returns

    assert schema.kind() == SchemaKind.inplace

    # Only support cases where all returns are Tensors or vector<Tensor>
    if len(returns) == 0:
        return None
    if not all(is_tensor(ret.type) or is_vector_tensor(ret.type) for ret in returns):
        return None
    if not accepts_at_least_one_tensor_input(schema):
        return None

    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(schema.arguments.flat_all, schema)

    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
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
    if not all(is_tensor(ret.type) or is_vector_tensor(ret.type) for ret in returns):
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

    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(schema.arguments.flat_all, schema)

    wrapped_returns = gen_returns(returns)
    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
{textwrap.indent(bdims_all_none_case, "  ")}
{textwrap.indent(unwraps, "  ")}
  auto results = batch_rule({', '.join(unwrapped_arg_list)});
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
