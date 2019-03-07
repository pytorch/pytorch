#!/bin/env python

# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

import sys
import yaml
import argparse
import os
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--template_dir", default=".", help="where template.h is")
parser.add_argument("--yaml_dir", default="aten/src/ATen/ATen",
                    help="where ATen yaml files are")
parser.add_argument("--output_prefix", default="", help="")
parser.add_argument(
    "--install_dir", default=".", help="where to put generated file")
parser.add_argument("--aten_root", default="", help="root directory of aten")
args, _ = parser.parse_known_args()

if args.aten_root:
    if not os.path.exists(args.aten_root):
        raise ValueError('aten_root ({}) does not exist'.format(
            args.aten_root))
    sys.path.append(os.path.join(args.aten_root, 'src', 'ATen'))
    from code_template import CodeTemplate as CT
else:
    from src.ATen.code_template import CodeTemplate as CT

OP_TEMPLATE = CT.from_file(
    os.path.join(args.template_dir, 'aten_op_template.h'))


try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def write(filename, s):
    with open(filename, "w") as f:
        f.write(s)


def read(filename):
    with open(filename, "r") as f:
        return f.read()


def value_has_tensors(v):
    # Sparse shouldn't appear in public API, seems to be temporary bug
    return "Tensor" in v['dynamic_type'] and "Sparse" not in v['dynamic_type']


def value_is_tensor_type(v):
    return value_has_tensors(v) and v['dynamic_type'] != 'TensorList'


# for each aten type, how do we handle a return value of that type?
RETURN_MAP = {
    'Tensor': 'assignTo(Output(${offset}),${output});',
    'Scalar': 'assignTo(Output(${offset}),*inferred_type, ${output});',
    'bool': 'assignToValue<int64_t>(Output(${offset}),${output});',
    'int64_t': 'assignToValue<int64_t>(Output(${offset}),${output});',
    'std::vector<Tensor>': 'assignListStartingAt(${offset}, ${output});',
}

# for each non-Tensor aten argument, how to we read it from caffe2's
# attribute list. Most of these call runtime functions defined in the
# template class.
ARGUMENT_MAP = {
    'Scalar': 'at::Scalar ${arg} = readScalarAttribute("${arg}");',
    'bool': 'bool ${arg} = readAttribute<int64_t>("${arg}");',
    'int': 'int ${arg} = readAttribute<int64_t>("${arg}");',
    'double': 'double ${arg} = readAttribute<float>("${arg}");',
    'int64_t': 'int64_t ${arg} = readAttribute<int64_t>("${arg}");',
    'IntArrayRef': 'auto ${arg} = readIntArrayRef("${arg}");',
    'std::array<bool, 2>': 'auto ${arg} = readBoolMask<2>("${arg}");',
    'std::array<bool, 3>': 'auto ${arg} = readBoolMask<3>("${arg}");',
}


def expand(o):
    num_defaults = sum(1 if 'default' in arg else 0 for arg in o['arguments'])
    results = [o]
    for i in range(0, num_defaults):
        # last num_default values should be default
        assert('default' in o['arguments'][-(i + 1)])
        v = deepcopy(o)
        v['arguments'] = v['arguments'][:-(i + 1)]
        results.append(v)
    return results


# filter the list of declarations removing things we cannot support
def supports(o, factory_methods):
    # Ignore all families (!) of functions that have TensorOptions (i.e. tensor factory methods).
    if o['name'] in factory_methods:
        if factory_methods[o['name']] == 0:
            print("Skipping {} because it is a factory method".format(o['name']))
        factory_methods[o['name']] += 1
        return False

    # skip all in-place operators for now since aten cannot Resize
    # caffe2 memory inside an operator
    if o['inplace']:
        return False

    # _out variants also work in-place on arguments taken as destinations
    # we also cannot handle these because aten cannot resize caffe2 Tensors
    if "_out" in o['name']:
        return False

    # skip return types we cannot handle
    for ret in o['returns']:
        if not value_has_tensors(ret) and ret['type'] not in RETURN_MAP:
            print("Skipping {} Because of Ret: {} ({})".format(
                  o['name'], ret['type'], ret['dynamic_type']))
            return False

    # skip arguments we cannot handle
    for arg in o['arguments']:
        if not value_has_tensors(arg) and arg['type'] not in ARGUMENT_MAP:
            print("Skipping {} Because of Arg: {} ({}) ".format(
                  o['name'], arg['type'], arg['dynamic_type']))
            return False
    return True


# template for each potential operator.
# each operator has an integer 'key' associated with it, and
# a lambda that defines the operator
# non-tensor attributes are created in ${initialization}
# and then saved as arguments to the lambda
# Inputs/Outputs are read inside the lambda
OPTION_TEMPLATE = CT("""\
case ${key}: { // ${name}
    ${initialization}
    run_op = [=] {
        ${statements}
        auto the_result = ${invocation};
        ${assignments}
        return true;
    };
} break;
""")


def get_output(o, i):
    if len(o['returns']) == 1:
        return 'the_result'
    else:
        return 'std::get<{}>(the_result)'.format(i)


def attribute_names(o):
    return sorted([a['name'] for a in o['arguments'] if not value_has_tensors(a)])


def required_attribute_names(o):
    return sorted([a['name'] for a in o['arguments'] if not value_has_tensors(a) and 'default' not in a])


def self_as_first_argument(arguments):
    return ([a for a in arguments if a['name'] == 'self'] +
            [a for a in arguments if a['name'] != 'self'])


def get_num_inputs(o):
    args = 0
    for a in o['arguments']:
        if a['type'] == 'TensorList':
            return '*'
        elif value_has_tensors(a):
            args += 1
    return str(args)


def find_factory_methods(decls):
    factory_methods = {}
    for o in decls:
        if any(arg['dynamic_type'] == 'TensorOptions' for arg in o['arguments']):
            factory_methods[o['name']] = 0
    return factory_methods


if __name__ == '__main__':
    decls = yaml.load(read(os.path.join(args.yaml_dir, 'Declarations.yaml')), Loader=Loader)
    factory_methods = find_factory_methods(decls)
    filtered = [expanded for o in decls for expanded in expand(o) if supports(expanded, factory_methods)]
    top_env = {
        'mappings': [],
        'implementations': [],
    }
    seen = set()
    key = 0
    for o in filtered:
        # [DESCRIPTORS]
        # each option is associated with a descriptor string that is used
        # to figure out which version of an op is being used:
        # The format is:
        #     opname-num_inputs-attribute_1-attribute2
        # Example:
        #  lerp-2-weight
        #  the operator lerp takes 2 arguments and has the attribute weight
        attr_names = attribute_names(o)
        num_inputs = get_num_inputs(o)
        descriptor = '-'.join([o['name']] + attr_names + [num_inputs])
        if descriptor in seen:
            continue
        seen.add(descriptor)

        # map from descriptor string to the integer key in the switch statements
        # that initializes the operators
        top_env['mappings'].append('{{ "{}", {} }},'.format(descriptor, key))
        env = {
            'name': o['name'],
            'statements': [],
            'arguments': [],
            'assignments': [],
            'initialization': [],
            'key': str(key),
        }
        defined_inferred_type = False

        if 'namespace' not in o['method_of'] and 'Tensor' not in o['method_of']:
            # methods on type like 'ones' or 'zeros' always take a
            # string attribute that is translated into the at::Type object
            # e.g. "Float" is at::kFloat
            assert('Type' in o['method_of'])
            defined_inferred_type = True
            env['initialization'].append(
                'auto inferred_type = readTypeAttribute("type");')

        static_tensor_inputs = sum(arg['type'] != 'TensorList' and value_is_tensor_type(arg) for arg in o['arguments'])
        has_tensorlist = any(arg['type'] == 'TensorList' for arg in o['arguments'])
        if has_tensorlist:
            tensorlist_idx = [i for i, arg in enumerate(o['arguments']) if arg['type'] == 'TensorList'][0]

        real_inputs = 0
        for i, arg in enumerate(o['arguments']):
            env['arguments'].append(arg['name'])
            # Emulate logic in gen_jit_dispatch.py. Pretend the flat argument
            # list is a stack where the end is the top.
            view_length = 'InputSize()' if has_tensorlist and i < tensorlist_idx else static_tensor_inputs
            if arg['type'] == 'TensorList':
                # NOTE: do not advance real_inputs here. After this we will
                # switch to indexing the "stack" from the end as if we only had
                env['statements'].append(
                    'auto {} = peekSlice({}, InputSize() - {}, InputSize());'
                        .format(arg['name'], real_inputs, static_tensor_inputs))
            elif value_is_tensor_type(arg):
                # load tensor inputs from Caffe2

                env['statements'].append(
                    'auto {} = peek({}, {});'.format(arg['name'], real_inputs, view_length))
                real_inputs += 1
                if arg['dynamic_type'] == 'Tensor' and not defined_inferred_type:
                    # first tensor input is used to define the output type.
                    defined_inferred_type = True
                    env['statements'].append(
                        'auto inferred_type = &at::getType({});'.format(
                            arg['name']))
            else:
                init = CT(ARGUMENT_MAP[arg['type']]).substitute(env, arg=arg['name'])
                env['initialization'].append(init)

        for i, r in enumerate(o['returns']):
            t = RETURN_MAP[r['type'] if not value_is_tensor_type(r) else 'Tensor']
            assignment = CT(t).substitute(env, offset=i, output=get_output(o, i))
            env['assignments'].append(assignment)

        if 'namespace' in o['method_of']:
            env['invocation'] = CT("at::${name}(${arguments})").substitute(env)
        elif 'Tensor' in o['method_of']:
            env['invocation'] = "self.{}({})".format(
                o['name'], ', '.join(env['arguments'][1:]))
        else:
            assert('Type' in o['method_of'])
            env['invocation'] = CT(
                'inferred_type->${name}(${arguments})').substitute(env)

        top_env['implementations'].append(OPTION_TEMPLATE.substitute(env))
        key += 1
    write(os.path.join(args.install_dir, args.output_prefix + "aten_op.h"), OP_TEMPLATE.substitute(top_env))
