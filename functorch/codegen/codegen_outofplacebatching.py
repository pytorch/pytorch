# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Tuple, List
from textwrap import dedent
import re


def is_tensor(typ: str) -> bool:
    if typ == 'Tensor':
        return True
    if typ == 'const Tensor &':
        return True
    return False


def is_optional_tensor(typ: str) -> bool:
    if typ == 'c10::optional<Tensor>':
        return True
    if typ == 'const c10::optional<Tensor> &':
        return True
    return False


def is_vector_tensor(typ: str) -> bool:
    # (chilli): I don't really understand why there's 2 dots in front?
    return (typ == '::std::vector<Tensor>')


def unwrap_tensor(name: str) -> List[str]:
    result = f"""\
    Tensor {name}_value;
    optional<int64_t> {name}_bdim;
    std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}, cur_level);"""
    return dedent(result).split('\n')


def unwrap_optional_tensor(name: str) -> List[str]:
    result = f"""\
    optional<Tensor> {name}_value;
    optional<int64_t> {name}_bdim;
    if ({name}) {{
        std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}.value(), cur_level);
    }}"""
    return dedent(result).split('\n')


def gen_unwraps(arg_types, arg_names):
    tensors = [name for typ, name in zip(arg_types, arg_names) if is_tensor(typ)]
    optional_tensors = [name for typ, name in zip(arg_types, arg_names) if is_optional_tensor(typ)]

    unwraps = []
    for tensor in tensors:
        unwraps += unwrap_tensor(tensor)

    for opt_tensor in optional_tensors:
        unwraps += unwrap_optional_tensor(opt_tensor)
    unwraps = ('\n' + ' ' * 6).join(unwraps)

    unwrapped_arg_list = []
    for arg in arg_names:
        if arg in tensors or arg in optional_tensors:
            unwrapped_arg_list += [f'{arg}_value', f'{arg}_bdim']
        else:
            unwrapped_arg_list.append(arg)
    return unwraps, unwrapped_arg_list


def gen_plumbing(op: str, returns: Tuple[str], args: List[Tuple[str, str]]) -> str:
    arg_types, arg_names = zip(*args)
    if '.' in op:
        name, overload = op.split('.')
        definition_name = f'{name}_{overload}'
    else:
        definition_name = op

    return_t = returns[0] if len(returns) == 1 else f'std::tuple<{",".join(returns)}>'
    arg_list = ', '.join([f'{arg[0]} {arg[1]}' for arg in args])

    unwraps, unwrapped_arg_list = gen_unwraps(arg_types, arg_names)

    idx = 0
    wrapped_returns = []
    for ret in returns:
        if is_tensor(ret):
            wrapped_returns.append(f'makeBatched(std::get<{idx}>(results), std::get<{idx + 1}>(results), cur_level)')
            idx += 2
        elif is_vector_tensor(ret):
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

    result = f"""\
    template <typename batch_rule_t, batch_rule_t batch_rule>
    {return_t} {definition_name}_generated_plumbing({arg_list}) {{
      c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
      auto maybe_layer = maybeCurrentDynamicLayer();
      TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
      int64_t cur_level = maybe_layer->layerId();
      {unwraps}
      auto results = batch_rule({', '.join(unwrapped_arg_list)});
      {wrapped_returns}
    }}"""
    return dedent(result)


def parse_return(return_t):
    if 'std::tuple' not in return_t:
        return (return_t,)
    m = re.match(r'std::tuple<(.*)>', return_t)
    if m is None:
        m = re.match(r'::std::tuple<(.*)>', return_t)
    return tuple([x.strip() for x in m.group(1).split(',')])


def parse_args(args_t):
    # There is an assumption made that args are separated with comma-space
    # and types like std::array<bool,2> do not contain spaces after the comma
    args = args_t.split(', ')
    result = []
    for arg in args:
        split_idx = arg.rfind(' ')
        result.append((arg[:split_idx].strip(), arg[split_idx:].strip()))
    return tuple(result)


def get_signatures(path='build/aten/src/ATen/RegistrationDeclarations.h', include_op=False):
    with open(path, 'r') as f:
        txt = f.read()
    lines = txt.split('\n')
    schemas = []
    for line in lines:
        if 'void' in line:
            continue
        m = re.match(r'(.*) \w+\((.*)\); // {"schema": "aten::(\w+\.?\w*)\(.*', line)
        if m is None:
            continue
        return_t = m.group(1)
        args_t = m.group(2)
        op = m.group(3)
        # TODO: some namedtuple return. Also, testing for codegen
        if include_op:
            result = (op, parse_return(return_t), parse_args(args_t))
        else:
            result = (parse_return(return_t), parse_args(args_t))
        schemas.append(result)
    return tuple(schemas)


def is_schema_outplace(schema):
    _, returns, args = schema
    for arg in args:
        typ, _ = arg
        if typ == 'Tensor &' or typ == "TensorList":
            return False

    types, _ = zip(*args)
    if all(not is_tensor(typ) for typ in types):
        return False
    for ret in returns:
        if ret == "std::vector<Tensor>":
            return False
        if ret == "const Tensor &":
            return False
        if ret == "Tensor &":
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        default='build/aten/src/ATen/RegistrationDeclarations.h',
                        help='link to RegistrationDeclarations.h')
    args = parser.parse_args()

    schemas = get_signatures(args.path, include_op=True)
    schemas = [schema for schema in schemas if is_schema_outplace(schema)]
    codes = [gen_plumbing(*schema) for schema in schemas]
    print("#pragma once")
    print("#include <functorch/csrc/PlumbingHelper.h>")
    print("#include <functorch/csrc/Constants.h>")

    print("")
    print("namespace at { namespace functorch {")
    for code in codes:
        print(code)
        print('')
    print("}}")
