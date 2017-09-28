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
project_root = sys.argv[1]
sys.path.append(project_root + "/third_party/aten/src/ATen")
from code_template import CodeTemplate as CT

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

OP_TEMPLATE = CT.from_file(project_root+'/caffe2/contrib/aten/aten_op_template.h')


def write(filename, s):
    with open(filename, "w") as f:
        f.write(s)


def read(filename):
    with open(filename, "r") as f:
        return f.read()


decls = yaml.load(read('aten/src/ATen/ATen/Declarations.yaml'), Loader=Loader)

top_env = {
    'mappings': [],
    'implementations': [],
}


def is_tensor_type(t):
    return "Tensor" in t


def value_is_tensor_type(v):
    return is_tensor_type(v['dynamic_type'])

# for each aten type, how do we handle a return value of that type?
RETURN_MAP = {
    'Tensor': 'assignTo(Output(${offset}),${output});',
    'Scalar': 'assignTo(Output(${offset}),*inferred_type, ${output});',
    'bool': 'assignToValue<int64_t>(Output(${offset}),${output});',
    'int64_t': 'assignToValue<int64_t>(Output(${offset}),${output});',
}

# for each non-Tensor aten argument, how to we read it from caffe2's
# attribute list. Most of these call runtime functions defined in the
# template class.
ARGUMENT_MAP = {
    'Scalar': 'at::Scalar ${arg} = readScalarAttribute("${arg}");',
    'bool': 'bool ${arg} = readAttribute<int64_t>("${arg}");',
    'int': 'int ${arg} = readAttribute<int64_t>("${arg}");',
    'int64_t': 'int64_t ${arg} = readAttribute<int64_t>("${arg}");',
    'IntList': 'auto ${arg} = readIntList("${arg}");',
}


# filter the list of declarations removing things we cannot support
def supports(o):

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
        if not value_is_tensor_type(ret) and ret['type'] not in RETURN_MAP:
            print("Skipping {} Because of Ret: {} ({})".format(o['name'], ret['type'], ret['dynamic_type']))
            return False

    # skip arguments we cannot handle
    for arg in o['arguments']:
        if not value_is_tensor_type(arg) and arg['type'] not in ARGUMENT_MAP:
            print("Skipping {} Because of Arg: {} ({}) ".format(o['name'], arg['type'], arg['dynamic_type']))
            return False
    return True


filtered = [o for o in decls if supports(o)]

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
    return sorted([a['name'] for a in o['arguments'] if not value_is_tensor_type(a)])


def self_as_first_argument(arguments):
    return ([a for a in arguments if a['name'] == 'self'] +
            [a for a in arguments if a['name'] != 'self'])

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
    num_inputs = len(o['arguments']) - len(attr_names)
    descriptor = '-'.join([o['name'], str(num_inputs)] + attr_names)

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

    if 'Tensor' in o['method_of']:
        # make sure 'self' is the first argument. currently Declarations.yaml
        # does not always do this. Instead it keeps the argument list the same order
        # as the Type method.
        o['arguments'] = self_as_first_argument(o['arguments'])
    elif 'namespace' not in o['method_of']:
        # methods on type like 'ones' or 'zeros' always take a
        # string attribute that is translated into the at::Type object
        # e.g. "Float" is at::kFloat
        assert('Type' in o['method_of'])
        defined_inferred_type = True
        env['initialization'].append('auto inferred_type = readTypeAttribute("type");')

    i = 0
    for arg in o['arguments']:
        env['arguments'].append(arg['name'])
        if value_is_tensor_type(arg):
            # load tensor inputs from Caffe2
            env['statements'].append("auto {}_ = Input({});".format(arg['name'], i))
            i += 1
            env['statements'].append(CT(
                "auto ${name} = tensorWrapping(${name}_);").substitute(arg))
            if arg['dynamic_type'] == 'Tensor' and not defined_inferred_type:
                # first tensor input is used to define the output type.
                defined_inferred_type = True
                env['statements'].append('auto inferred_type = &({}.type());'.format(arg['name']))
        else:
            init = CT(ARGUMENT_MAP[arg['type']]).substitute(env, arg=arg['name'])
            env['initialization'].append(init)

    for i, r in enumerate(o['returns']):
        t = RETURN_MAP[r['type'] if not value_is_tensor_type(r) else 'Tensor']
        assignment = CT(t).substitute(env, offset=i, output=get_output(o, i))
        env['assignments'].append(assignment)

    if 'Tensor' in o['method_of']:
        env['invocation'] = "self.{}({})".format(o['name'], ', '.join(env['arguments'][1:]))
    elif 'namespace' in o['method_of']:
        env['invocation'] = CT("at::${name}(${arguments})").substitute(env)
    else:
        assert('Type' in o['method_of'])
        env['invocation'] = CT('inferred_type->${name}(${arguments})').substitute(env)

    top_env['implementations'].append(OPTION_TEMPLATE.substitute(env))
    key += 1
write("aten_op.h", OP_TEMPLATE.substitute(top_env))
