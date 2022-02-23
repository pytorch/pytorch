from tools.codegen.api.types import (
    DispatcherSignature,
)
from tools.codegen.model import (
    BaseTy, Variant, OptionalType, BaseType, ListType, NativeFunction, Type,
    Argument, Return, SchemaKind,
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


def gen_vmap_plumbing(native_function: NativeFunction) -> str:
    schema = native_function.func

    # Don't support these yet
    if schema.kind() == SchemaKind.inplace or schema.kind() == SchemaKind.out:
        return None

    sig = DispatcherSignature.from_schema(schema)
    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all)
    returns = schema.returns

    # Only support cases where all returns are Tensors or vector<Tensor>
    if len(returns) == 0:
        return None
    if not all(is_tensor(ret.type) or is_vector_tensor(ret.type) for ret in returns):
        return None

    wrapped_returns = gen_returns(returns)
    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
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
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/Constants.h>

namespace at {{ namespace functorch {{

{body}

}}}} // namespace at::functorch
"""
