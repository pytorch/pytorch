"""
To run this file by hand from the root of the PyTorch
repository, run:

python -m tools.jit.gen_jit_dispatch \
       build/aten/src/ATen/Declarations.yaml \
       $OUTPUT_DIR \
       tools/jit/templates

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/csrc/jit/generated/
"""

import argparse
import copy
import re
import yaml
from itertools import groupby
from ..autograd.utils import CodeTemplate, YamlLoader, write
from ..autograd.gen_autograd import load_aten_declarations
from ..autograd.gen_autograd import RETURNS_VIEWS_OF_INPUT

# JIT has a type system of
# Scalar = int | float | bool # int is the largest int (int64_t),
# float is the largest float (double) we don't have the others because they are never held in tensors
# Type = Scalar # primitive numbers
#      | Tensor # any tensor, as defined by at::Tensor
#      | Type[] # a dynamically sized list[ of a type
#      | Scalar[N] # a homogenous fixed size scalar list, single scalars can expand to this list
#      | (Type1, Type2, ...) # a heterogeneous tuple
#      | Layout | ScalarType | Device | Generator # special singleton types for built-in concepts in tensor lib

# clean up the variety of C++ types in the ATen declarations
# to be in the restricted set of types that the IR represents
# note: no default values for this map, to make it clear what types
# can be passedthrough

TYPE_MAP = {
    'std::array<bool,2>': 'bool[2]',
    'std::array<bool,3>': 'bool[3]',
    'std::array<bool,4>': 'bool[4]',
    'std::string': 'str',
    'Scalar': 'Scalar',
    'MemoryFormat': 'MemoryFormat',
    'MemoryFormat?': 'MemoryFormat?',
    'QScheme': 'QScheme',
    'Scalar?': 'Scalar?',
    'Tensor': 'Tensor',
    'Tensor?': 'Tensor?',
    'TensorList': 'Tensor[]',
    # this appears in return values instead of TensorList
    # since TensorList is a ArrayRef in arguments but a vector
    # in returns
    'std::vector<Tensor>': 'Tensor[]',
    'IntArrayRef': 'int[]',
    'Layout': 'Layout',
    'Layout?': 'Layout?',
    'Device': 'Device',
    'Device?': 'Device?',
    'ScalarType': 'ScalarType',
    'ScalarType?': 'ScalarType?',
    'int64_t': 'int',
    'int64_t?': 'int?',
    'double': 'float',
    'double?': 'float?',
    'bool': 'bool',
    'bool?': 'bool?',
    'Generator': 'Generator?',
}


def optional_type_of(arg, typ):
    # optional type special handling for Tensor?[] and Tensor
    # types that is missing a optional annotation
    if arg.get('is_nullable') and '?' not in typ:
        if typ == 'TensorList' or typ == 'Tensor[]':
            typ = 'Tensor?[]'
        else:
            typ = '{}?'.format(typ)
    return typ


def annotated_type_of(arg, typ):
    anno = arg.get('annotation')
    if anno:
        typ = '{}({})'.format(typ, anno)
    return typ


def jit_type_of(arg):
    jit_type = arg.get('jit_type')
    if not jit_type:
        jit_type = TYPE_MAP[arg['simple_type']]
        if is_sized_intlist_arg(arg):
            jit_type = 'int[{}]'.format(arg['size'])
        jit_type = optional_type_of(arg, jit_type)
        jit_type = annotated_type_of(arg, jit_type)
        arg['jit_type'] = jit_type
    return jit_type


# map from aten 'simple_type' to the function that will turn a tensor into
# that type
FROM_IVALUE = {
    'Device': '{}.toDevice()',
    'Device?': '{}.toOptional<c10::Device>()',
    'IntArrayRef': '{}.toIntVector()',
    'Layout': '{}.toLayout()',
    'Layout?': '{}.toOptional<c10::Layout>()',
    'MemoryFormat': '{}.toMemoryFormat()',
    'MemoryFormat?': '{}.toOptional<c10::MemoryFormat>()',
    'QScheme': '{}.toQScheme()',
    'Scalar': '{}.toScalar()',
    'Scalar?': '{}.toOptional<Scalar>()',
    'ScalarType': '{}.toScalarType()',
    'ScalarType?': '{}.toOptional<ScalarType>()',
    'Tensor': '{}.toTensor()',
    'Tensor?': 'toOptionalTensor({})',
    'Tensor?[]': 'toListOfOptionalTensor({})',
    'TensorList': '{}.toTensorVector()',
    'bool': '{}.toBool()',
    'bool?': '{}.toOptional<bool>()',
    'double': '{}.toDouble()',
    'double?': '{}.toOptional<double>()',
    'int64_t': '{}.toInt()',
    'int64_t?': '{}.toOptional<int64_t>()',
    'std::string': '{}.toStringRef()',
    'Generator': 'nullptr',
    'std::array<bool,2>': 'as_bool_array<2>({}.toBoolList())',
    'std::array<bool,3>': 'as_bool_array<3>({}.toBoolList())',
    'std::array<bool,4>': 'as_bool_array<4>({}.toBoolList())',
}


def from_ivalue(arg, value):
    typ = optional_type_of(arg, arg['simple_type'])
    return FROM_IVALUE[typ].format(value)


CALL_UNBOXED_KERNEL = CodeTemplate("""\
auto result_ = callUnboxedKernel<${return_type}${formals_types_with_leading_comma}>(unboxedKernel${args_with_leading_comma});
""")
CALL_NAMESPACE = CodeTemplate("""\
auto result_ = at::${name}(
    ${args}
);
""")
CALL_METHOD = CodeTemplate("""\
auto result_ = (${first}).${name}(
    ${args}
);
""")
CALL_NAMESPACE_WITH_TENSOR_OPTIONS = CodeTemplate("""\
const auto options = TensorOptions()
        .dtype(${dtype})
        .layout(${layout})
        .device(${device})
        .pinned_memory(${pin_memory});
#ifdef USE_STATIC_DISPATCH
    auto result_ = at::${name}(${args_with_tensor_options});
#else
    auto result_ = torch::${name}(${args_with_tensor_options});
#endif
""")
CALL_METHOD_WITH_TENSOR_OPTIONS = CodeTemplate("""\
const auto options = TensorOptions()
        .dtype(${dtype})
        .layout(${layout})
        .device(${device})
        .pinned_memory(${pin_memory});
auto result_ = (${first}).${name}(${args_with_tensor_options});
""")

CONSTRUCTOR = CodeTemplate("""\
[](OperatorKernel* unboxedKernel, const OperatorHandle&, Stack* stack) {
    using namespace at;
    ${lvalues}
    ${call}
    drop(*stack, ${num_inputs});
    pack(*stack, std::move(result_));
}
""")

CONSTRUCTOR_JITONLY = CodeTemplate("""\
[](Stack* stack) {
    ${lvalues}
    ${call}
    drop(*stack, ${num_inputs});
    pack(*stack, std::move(result_));
    return 0;
}
""")

OPERATOR = CodeTemplate("""\
  .op("${signature}",
    ${op})
""")

OPERATOR_JITONLY = CodeTemplate("""\
  .jitOnlyOp("${signature}",
    ${op})
""")


blacklisted_types = {
    'Storage',
    'DimnameList?',
    'ConstQuantizerPtr',
    'Dimname',
    'DimnameList',
}

default_only_types = {'Generator'}


def is_jit_arg(i, arg):
    simple_type = arg['simple_type']
    if simple_type in blacklisted_types:
        return False
    if simple_type in default_only_types and 'default' not in arg:
        return False
    if simple_type == 'Type':
        return False
    return True


def is_jit_op(decl):
    # We currently don't support functions that return nothing
    assert all(r['type'] != 'void' for r in decl['returns'])
    if len(decl['returns']) == 0:
        return False

    arguments = decl['arguments']

    # there must be a single out variant
    if is_out_variant(decl) and sum([not not arg.get('output') for arg in arguments]) > 1:
        return False

    return (('namespace' in decl['method_of'] or 'Tensor' in decl['method_of']) and
            all(is_jit_arg(i, arg) for i, arg in enumerate(decl['arguments'])) and
            all(is_jit_arg(i, arg) for i, arg in enumerate(decl['returns'])))


def is_tensor_arg(arg):
    return arg['simple_type'] in {'Tensor', 'TensorList'}


def is_sized_intlist_arg(arg):
    """Returns True for arguments declared as IntArrayRef[k], but False for IntArrayRef."""
    return (arg['simple_type'] == 'IntArrayRef') and ('size' in arg)


def base_name(decl):
    name = decl['name']
    return name[:-1] if decl.get('inplace', False) else name[:-4] if name.endswith('_out') else name


def is_view(decl):
    return base_name(decl) in RETURNS_VIEWS_OF_INPUT


def is_out_variant(decl):
    return decl['name'].endswith('_out')


# Copied from ..autograd.gen_python_functions.SKIP_PYTHON_BINDINGS
BACKWARD_OP_PATTERNS = [
    '.*_backward',
    '.*_backward_(out|input|weight|bias)',
]

def is_backward_op(decl):
    for pattern in BACKWARD_OP_PATTERNS:
        if re.match('^' + pattern + '$', decl['name']):
            return True
    return False


# for each argument in decl, the location it should appear in the
# jit schema declaration. e.g.
# arguments = [x, y, z] # the order in aten
# jit_argument_order = [2, 0, 1]
# aten::my_arg(Tensor y, Tensor z, Tensor x) # the order in schema
# used to move 'out' arguments to the end of the list
def argument_order(decl):
    return decl.get('jit_argument_order') or list(range(len(decl['arguments'])))


def load_op_list(path):
    with open(path, 'r') as f:
        op_list = yaml.load(f, Loader=YamlLoader)
    return op_list


def gen_jit_dispatch(declarations, out, template_path, disable_autograd=False, selected_op_list_path=None):
    REGISTER_ATEN_OPS_CPP = CodeTemplate.from_file(template_path + '/register_aten_ops.cpp')

    ops = []

    def get_invocation(decl, args, num_inputs):

        # because the arg list can get lengthy we put them on a separate line
        def pack_arguments(args):
            return ',\n'.join(args)
        is_namespace_function = 'namespace' in decl['method_of']
        tensor_options_arg_index = decl.get('tensor_options_arg_index', None)
        if tensor_options_arg_index is not None:
            dtype = args[tensor_options_arg_index]
            layout = args[tensor_options_arg_index + 1]
            device = args[tensor_options_arg_index + 2]
            pin_memory = args[tensor_options_arg_index + 3]
            args_with_tensor_options = args[:tensor_options_arg_index] + \
                ['options'] + args[(tensor_options_arg_index + 4):]
            if is_namespace_function:
                return CALL_NAMESPACE_WITH_TENSOR_OPTIONS.substitute(
                    name=decl['name'], dtype=dtype, layout=layout,
                    device=device, pin_memory=pin_memory,
                    args_with_tensor_options=pack_arguments(args_with_tensor_options))
            else:
                return CALL_METHOD_WITH_TENSOR_OPTIONS.substitute(
                    name=decl['name'], dtype=dtype, layout=layout,
                    device=device, pin_memory=pin_memory,
                    args_with_tensor_options=pack_arguments(args_with_tensor_options[1:]),
                    first=args_with_tensor_options[0], num_inputs=num_inputs)
        # The use_c10_dispatcher setting in native_functions.yaml now has a new option
        # 'with_codegenerated_unboxing_wrapper' which means we take the codegened unboxing wrapper from
        # register_aten_ops.cpp and stuff it into c10. This new argument is the default, 'unboxed_only' is not the
        # default anymore. For the (very few) ops that don't support boxed dispatch yet (i.e. ops taking TensorOptions
        # arguments), we set them to 'unboxed_only' and they follow the old behavior of having register_aten_ops.cpp
        # register the jit op.
        elif decl['use_c10_dispatcher'] == 'with_codegenerated_unboxing_wrapper':
            if len(decl['returns']) == 0:
                return_type = "void"
            elif len(decl['returns']) == 1:
                return_type = decl['returns'][0]['type']
            else:
                return_type = "std::tuple<{}>".format(", ".join([r['type'] for r in decl['returns']]))
            for a in decl['arguments']:
                if 'type' not in a:
                    raise Exception(decl)
            argument_types_with_leading_comma = ", ".join([a['type'] for a in decl['arguments']])
            if argument_types_with_leading_comma != "":
                argument_types_with_leading_comma = ", " + argument_types_with_leading_comma
            args_with_leading_comma = pack_arguments(args)
            if args_with_leading_comma != "":
                args_with_leading_comma = ", " + args_with_leading_comma
            return CALL_UNBOXED_KERNEL.substitute(name=decl['name'],
                                                  args_with_leading_comma=args_with_leading_comma,
                                                  num_inputs=num_inputs,
                                                  return_type=return_type,
                                                  formals_types_with_leading_comma=argument_types_with_leading_comma)
        else:
            assert decl['use_c10_dispatcher'] in ['unboxed_only', 'full']
            if is_namespace_function:
                return CALL_NAMESPACE.substitute(name=decl['name'],
                                                 args=pack_arguments(args),
                                                 num_inputs=num_inputs)
            else:
                return CALL_METHOD.substitute(
                    name=decl['name'], first=args[0],
                    args=pack_arguments(args[1:]), num_inputs=num_inputs)

    def requires_lvalue(arg):
        jit_type = jit_type_of(arg)
        return jit_type.startswith('Tensor') and '!' in jit_type

    def emit_decl_variant(decl):
        if ('emit_dummy_placeholder' in decl):
            if decl['use_c10_dispatcher'] == 'unboxed_only':
                return "DUMMY_OPERATION_JITONLY"
            else:
                return "DUMMY_OPERATION"
        kw_assignments = []

        # mutable arguments in aten are passed as non const references
        # these must be lvalues, so we have to put them in variables
        # before calling the function
        lvalues = []

        arguments = []
        num_inputs = len(decl['arguments'])
        op_capture = ''
        order = argument_order(decl)
        for i, arg in enumerate(decl['arguments']):
            value = from_ivalue(arg, '(std::move(peek(*stack, {}, {})))'.format(order[i], num_inputs))
            if requires_lvalue(arg):
                lvalues.append('auto {} = {};\n'.format(arg['name'], value))
                value = arg['name']
            arguments.append(value)

        call = get_invocation(decl, arguments, num_inputs)

        returns = decl['returns']

        if decl['use_c10_dispatcher'] == 'unboxed_only':
            # Ops taking TensorOptions aren't supported in this mechanism yet because boxed dispatch doesn't
            # work for them. They use the old mechanism of registering a jitonly op for now.
            # TODO We should get rid of this once TensorOptions are supported.
            constructor = CONSTRUCTOR_JITONLY.substitute(name=decl['name'],
                                                         call=call,
                                                         kw_assignments=kw_assignments,
                                                         num_inputs=num_inputs,
                                                         op_capture=op_capture,
                                                         lvalues=lvalues)
        elif decl['use_c10_dispatcher'] == 'with_codegenerated_unboxing_wrapper':
            constructor = CONSTRUCTOR.substitute(name=decl['name'],
                                                 call=call,
                                                 kw_assignments=kw_assignments,
                                                 num_inputs=num_inputs,
                                                 op_capture=op_capture,
                                                 lvalues=lvalues)
        else:
            assert decl['use_c10_dispatcher'] == 'full'

        return constructor

    def filter_decls(jit_decls, disable_autograd, selected_op_list):
        result = []
        for decl in jit_decls:
            if disable_autograd and is_backward_op(decl):
                continue
            if selected_op_list and signature_without_args(decl) not in selected_op_list:
                decl['emit_dummy_placeholder'] = True
            result.append(decl)
        return result

    # This function declares an order on declarations. This is necessary because
    # there is some ambiguity in the choice of overload: if an argument is overloaded
    # to accept both Scalar and Tensor, the schema with the Tensor should come first
    # TODO: this can (probably) be removed when we remove the implicit conversion
    # from Tensor -> Number.
    def sort_decls(jit_decls):
        def declkey(decl):
            # key = sum_{i < len(args)} {1 if arg is tensor else 2} * (3 ** i)
            # This is a ternary encoding where
            # 0: No argument at this position
            # 1: Tensor argument at this position
            # 2: Some other argument at this position.
            args = decl['arguments']
            result = 0
            for i in range(len(args)):
                result += (3 ** i) * (1 if args[i]['simple_type'] == 'Tensor' else 2)
            return result

        # NB: itertools.groupby requires the list be sorted.
        sorted_decls = sorted(jit_decls, key=lambda decl: decl['name'])
        grouped_decls = [list(g) for _, g in
                         groupby(sorted_decls, key=lambda decl: decl['name'])]
        return [sorted(g, key=declkey) for g in grouped_decls]

    # We need to add methods implemented manually in TensorImpl
    # TODO: This seems to claim sizes() returns an int64_t.  Really?
    tensor_impl_methods = [{
        'name': name,
        'api_name': name,
        'schema_string': schema_string,
        'overload_name': '',
        'method_of': ['Tensor'],
        'arguments': [{'name': 'self', 'simple_type': 'Tensor'}],
        'returns': [{'name': 'result', 'type': 'int64_t', 'dynamic_type': 'int64_t', 'simple_type': 'int64_t'}],
        'use_c10_dispatcher': 'unboxed_only',
    } for name, schema_string in [
        ('sizes', 'aten::sizes(Tensor self) -> int'),
        ('strides', 'aten::strides(Tensor self) -> int'),
        ('dim', 'aten::dim(Tensor self) -> int'),
        ('numel', 'aten::numel(Tensor self) -> int'),
        ('element_size', 'aten::element_size(Tensor self) -> int'),
    ]]

    aten_decls = load_aten_declarations(declarations) + tensor_impl_methods
    jit_decls = [d for d in aten_decls if is_jit_op(d)]

    # add arguments dtype and device for functions like zeros
    def expand_options(decl, i, arg):
        if arg['simple_type'] != 'TensorOptions':
            return [arg]
        assert decl.get('tensor_options_arg_index') != i
        decl['tensor_options_arg_index'] = i
        tensor_options_expansion = [
            # XXX - until we actually have first-class interpreter types for these
            # concepts, the default values to be encoded in Tensors
            # If you change this, you also need to update [TensorOptions in script]
            # in the tracer code.
            # dtype is specified as an int64_t of at::ScalarType
            {'name': 'dtype', 'simple_type': 'ScalarType'},
            # layout is specified as an int64_t of at::Layout
            {'name': 'layout', 'simple_type': 'Layout'},
            # device is specified as an IntArrayRef of { at::Device::Type, device_id }
            {'name': 'device', 'simple_type': 'Device'},
            # pin_memory is specified as a boolean
            {'name': 'pin_memory', 'simple_type': 'bool', 'default': False},
        ]
        # TODO: Don't repack this into TensorOptions. Needs various changes in downstream code.
        if 'default' in arg:
            for el in tensor_options_expansion:
                el['simple_type'] += '?'
                el['default'] = 'None'
        if 'default' in arg and arg['default'] == 'at::kLong':
            tensor_options_expansion[0]['default'] = 'long'
        if 'kwarg_only' in arg and arg['kwarg_only']:
            for el in tensor_options_expansion:
                el['kwarg_only'] = True
        return tensor_options_expansion

    additional_jit_decls = []

    for decl in jit_decls:
        decl['arguments'] = [a for i, arg in enumerate(decl['arguments']) for a in expand_options(decl, i, arg)]
        if is_out_variant(decl):
            reorder_out_args(decl)
        if needs_hacked_twin(decl):
            additional_jit_decls.append(hacked_twin(decl))

    jit_decls.extend(additional_jit_decls)
    selected_op_list = load_op_list(selected_op_list_path) if selected_op_list_path else None
    jit_decls = filter_decls(jit_decls, disable_autograd, selected_op_list)

    # generation is deterministic
    jit_decl_groups = sort_decls(jit_decls)

    # NOTE: see Note [Sharded File] at the top of the register_aten_ops.cpp
    # template regarding sharding of the generated files.
    #
    # If you edit the number of shards here, you will also have to
    # modify generate_code.py, torch/CMakeLists.txt, and the TARGETS
    # files.
    num_shards = 3
    shards = [[] for _ in range(num_shards)]

    # ops are assigned arbitrarily but stably to a file based on hash
    for group in jit_decl_groups:
        x = sum(ord(c) for c in group[0]['name']) % num_shards
        for decl in group:
            if decl['use_c10_dispatcher'] == 'unboxed_only':
                shards[x].append(OPERATOR_JITONLY.substitute(signature=decl['schema_string'],
                                                             op=emit_decl_variant(decl)))
            elif decl['use_c10_dispatcher'] == 'with_codegenerated_unboxing_wrapper':
                shards[x].append(OPERATOR.substitute(signature=decl['schema_string'],
                                                     op=emit_decl_variant(decl)))
            else:
                assert decl['use_c10_dispatcher'] == 'full'

    for i, shard in enumerate(shards):
        env = {
            'constructors': shard,
        }
        write(out, 'register_aten_ops_%d.cpp' % i, REGISTER_ATEN_OPS_CPP, env)


default_map = {'{}': 'None', 'nullptr': 'None', 'c10::nullopt': 'None'}


def reorder_out_args(decl):
    first_arg = decl['arguments'][0]
    assert(first_arg['output'])
    # the output variant must go at the end
    # note: this is an annoying side effect of using a single '*'
    # to denote kwarg_only
    nargs = len(decl['arguments'])
    decl['jit_argument_order'] = [nargs - 1] + list(range(nargs - 1))


def is_kwarg_only(a):
    return a.get('kwarg_only') or a.get('output')


#
# create a clone of these declarations
# with nullability scrubbed from TensorList arg types
# TOOD find out why this exists and how to do it without the hack
#

NEEDS_HACKED_TWIN_NAMES = [
    "aten::_index_put_impl_",
    "aten::index.Tensor",
    "aten::index_put",
    "aten::index_put_",
]

def needs_hacked_twin(decl):
    schema_string = decl['schema_string']
    return any([schema_string.startswith(name) for name in NEEDS_HACKED_TWIN_NAMES])


def hacked_twin(decl):
    decl_copy = copy.deepcopy(decl)
    old_overload_name = decl['overload_name']
    schema_string = decl['schema_string']
    name = decl['name']
    schema_string = schema_string.replace('Tensor?[]', 'Tensor[]')
    if old_overload_name:
        new_overload_name = old_overload_name + "_hacked_twin"
        decl_copy['overload_name'] = new_overload_name
        decl_copy['schema_string'] = schema_string.replace(name + "." + old_overload_name,
                                                           name + "." + new_overload_name)
    else:
        new_overload_name = "hacked_twin"
        decl_copy['overload_name'] = new_overload_name
        decl_copy['schema_string'] = schema_string.replace(name, name + "." + new_overload_name)
    for arg in decl_copy['arguments']:
        if arg['simple_type'] == 'TensorList' and arg.get('is_nullable'):
            arg['is_nullable'] = False
    return decl_copy


def signature_without_args(decl):
    name = decl['name'] if not is_out_variant(decl) else decl['name'][:-4]
    overload_name = '.' + decl['overload_name'] if not decl['overload_name'] == '' else ''
    return 'aten::{}{}'.format(name, overload_name)


def main():
    parser = argparse.ArgumentParser(
        description='Generate JIT op dispatch')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    parser.add_argument('template_path', metavar='TEMPLATE_PATH',
                        help='path to templates directory')
    args = parser.parse_args()
    gen_jit_dispatch(args.declarations, args.out, args.template_path)


if __name__ == '__main__':
    main()
