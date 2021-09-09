# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Tuple, List
import re

def num_leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip())

def min_leading_spaces(lines):
    num_spaces = [num_leading_spaces(line) for line in lines if len(line) > 0]
    if len(num_spaces) == 0:
        return None
    return min(num_spaces)

def deindent(code: str) -> str:
    lines = code.split('\n')
    mls = min_leading_spaces(lines)
    lines = [line[mls:] for line in lines]
    return '\n'.join(lines)

def indent(code: str, num) -> str:
    lines = code.split('\n')
    indented_lines = [' ' * num + line for line in lines]
    indented_lines[0] = lines[0]
    return '\n'.join(indented_lines)

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

def add_bdim_after_tensor(types: Tuple[str]) -> Tuple[str]:
    result = []
    for typ in types:
        result.append(typ)
        if is_tensor(typ) or is_optional_tensor(typ) or is_vector_tensor(typ):
            result.append('c10::optional<int64_t>')
    return tuple(result)

def batch_rule_type(
        op_returns: Tuple[str],
        op_args: Tuple[str],
        unique_count: int) -> Tuple[str, str]:
    returns = add_bdim_after_tensor(op_returns)
    args = add_bdim_after_tensor(op_args)
    br_t = f'batch_rule_{unique_count}_t'

    result = f"typedef std::tuple<{','.join(returns)}> (*{br_t})({', '.join(args)});"
    return result, br_t

def unwrap_tensor(name: str) -> List[str]:
    result = f"""\
    Tensor {name}_value;
    optional<int64_t> {name}_bdim;
    std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}, cur_level);"""
    return deindent(result).split('\n')

def unwrap_optional_tensor(name: str) -> List[str]:
    result = f"""\
    optional<Tensor> {name}_value;
    optional<int64_t> {name}_bdim;
    if ({name}) {{
        std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}.value(), cur_level);
    }}"""
    return deindent(result).split('\n')

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

def lower(returns: Tuple[str], args: List[Tuple[str, str]], unique_count: int) -> str:
    arg_types, arg_names = zip(*args)
    batch_rule_typedef, batch_rule_t = batch_rule_type(returns, arg_types, unique_count)

    return_t = returns[0] if len(returns) == 1 else f'std::tuple<{",".join(returns)}>'
    args_t = ', '.join(arg_types)
    arg_list = ', '.join([f'{arg[0]} {arg[1]}' for arg in args])

    unwraps, unwrapped_arg_list = gen_unwraps(arg_types, arg_names)

    idx = 0
    wrapped_returns = []
    for ret in returns:
        if is_tensor(ret):
            wrapped_returns.append(f'makeBatched(std::get<{idx}>(results), std::get<{idx + 1}>(results), cur_level)')
            idx += 2
        elif is_vector_tensor(ret):
            wrapped_returns.append(f'makeBatchedVector(std::get<{idx}>(results), std::get<{idx + 1}>(results), cur_level)')
            idx += 2
        else:
            wrapped_returns.append(f'std::get<{idx}>(results)')
            idx += 1
    if len(wrapped_returns) == 1:
        wrapped_returns = f'return {wrapped_returns[0]};'
    else:
        wrapped_returns = f'return std::make_tuple({", ".join(wrapped_returns)});'

    result = f"""\
    {batch_rule_typedef}
    template <>
    {return_t} lowerToNextLayer<{batch_rule_t},{return_t},{args_t}>(
      {batch_rule_t} batch_rule,
      {arg_list}
    ) {{
      c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
      auto maybe_layer = maybeCurrentDynamicLayer();
      TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
      int64_t cur_level = maybe_layer->layerId();
      {unwraps}
      auto results = batch_rule({', '.join(unwrapped_arg_list)});
      {wrapped_returns}
    }}"""
    return deindent(result)

def parse_return(return_t):
    if 'std::tuple' not in return_t:
        return (return_t,)
    m = re.match(r'std::tuple<(.*)>', return_t)
    if m is None:
        m = re.match(r'::std::tuple<(.*)>', return_t)
    return tuple([x.strip() for x in m.group(1).split(',')])

def parse_args(args_t):
    args = args_t.split(',')
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
        if 'std::array' in line:
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
    returns, args = schema
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
    return True

def get_hash(schema):
    ret_t, args = schema
    args_t, _ = tuple(zip(*args))
    return (ret_t, args_t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        default='build/aten/src/ATen/RegistrationDeclarations.h',
                        help='link to RegistrationDeclarations.h')
    args = parser.parse_args()

    schemas = get_signatures(args.path)
    schemas = [schema for schema in schemas if is_schema_outplace(schema)]
    unique_schemas = {}
    for schema in schemas:
        unique_schemas[get_hash(schema)] = schema
    schemas = list(unique_schemas.values())
    codes = [lower(*schema, i) for i, schema in enumerate(schemas)]
    print("#include <functorch/csrc/OutOfPlacePlumbing.h>")
    print("#include <functorch/csrc/PlumbingHelper.h>")
    print("#include <functorch/csrc/Constants.h>")

    print("")
    print("namespace at { namespace functorch {")
    for code in codes:
        print(code)
        print('')
    print("}}")
