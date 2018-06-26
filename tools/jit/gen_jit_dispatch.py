import os
import argparse
from itertools import count, combinations, groupby
from ..autograd.utils import CodeTemplate, write, uninplace_api_name
from ..autograd.gen_autograd import load_aten_declarations
from collections import OrderedDict

ATTR_METHOD_MAP = {
    'int64_t': 'i',
    'IntList': 'is',
    'Scalar': 't',
    'bool': 'i',
    'double': 'f',
    'std::array<bool,2>': 'is',
    'std::array<bool,3>': 'is',
    'std::array<bool,4>': 'is',
    'Layout': 'i',
    'Device': 'is',
    'ScalarType': 'i',
}

TYPE_CASTS = {
    'std::array<bool,2>': 'as_bool_array<2>',
    'std::array<bool,3>': 'as_bool_array<3>',
    'std::array<bool,4>': 'as_bool_array<4>',
    'Scalar': 'Scalar',
    'IntList': 'std::vector<int64_t>',
    'Layout': 'int64_t',
    'Device': 'std::vector<int64_t>',
    'ScalarType': 'int64_t',
}

FROM_TENSOR = {
    'Device': 'tensor_as<IntList>',
    'ScalarType': 'tensor_as<int64_t>',
    'Layout': 'tensor_as<int64_t>',
}

KW_ASSIGNMENT = CodeTemplate("""\
auto ${name} = ${type_cast}(node->${method}(Symbol::attr("${name}")));\
""")

POS_ASSIGNMENT = CodeTemplate("""\
auto ${name} = ${from_tensor}(std::move(peek(stack, ${i}, ${N})));\
""")

CALL_NAMESPACE = CodeTemplate("""\
auto result = at::${name}(${args});
""")
CALL_METHOD = CodeTemplate("""\
DeviceGuard device_guard(deviceForInputs(stack, ${num_dynamic_inputs}));
auto result = (${first}).${name}(${args});
""")
CALL_TENSOR_OPTIONS = CodeTemplate("""\
const auto device_index = static_cast<int32_t>(device[1]);
const auto options = TensorOptions()
        .dtype(static_cast<at::ScalarType>(dtype))
        .layout(static_cast<at::Layout>(layout))
        .device({static_cast<at::Device::Type>(device[0]), device_index});
auto result = torch::${name}(${args}, options);
""")

CONSTRUCTOR = CodeTemplate("""\
{"${descriptor}", [](Node *node) {
  ${kw_assignments}
  return TensorOp([=](Stack & stack) {
    autograd::profiler::RecordFunction record("${name}");
    ${pos_assignments}
    ${call}
    drop(stack, ${num_dynamic_inputs});
    pack(stack, std::move(result));
    return 0;
  }, "${name}", ${num_dynamic_inputs}, ${num_outputs});
}},
""")


def is_magic_method(api_name):
    return api_name.startswith('__') and api_name.endswith('__')


blacklisted_types = {'SparseTensorRef', 'Storage', 'ScalarType', 'optional<ScalarType>', 'std::string', 'void*'}
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
    if all(r['type'] == 'void' for r in decl['returns']):
        return False

    # we currently only support vararg tensor lists when they are the _first_ argument
    # and the only tensor argument
    arguments = decl['arguments']
    # Only support a single TensorList arg
    if sum(arg['simple_type'] == 'TensorList' for arg in arguments) > 1:
        return False

    return ((not decl['api_name'].endswith('_') or is_magic_method(decl['api_name'])) and
            not decl['name'].endswith('_out') and
            ('namespace' in decl['method_of'] or 'Tensor' in decl['method_of']) and
            all(is_jit_arg(i, arg) for i, arg in enumerate(decl['arguments'])) and
            all(is_jit_arg(i, arg) for i, arg in enumerate(decl['returns'])))

# Scalar overloads like add(Tensor self, Scalar other) are not supported atm.
# TODO: Why are they not supported?
skip_scalar_overload = {
    'lt-2': [1], 'gt-2': [1], 'le-2': [1], 'ge-2': [1], 'eq-2': [1], 'ne-2': [1],
    'pow-2': [0, 1], 'add-3': [1], 'sub-3': [1],
    'mul-2': [1], 'th_mul-2': [1], 'native_mul-2': [1],
    'div-2': [1], 'th_div-2': [1], 'native_div-2': [1],
    'fmod-2': [1], 'remainder-2': [1], '__and__-2': [1], '__or__-2': [1],
    '__iand__-2': [1], '__ior__-2': [1], '__xor__-2': [1], '__ixor__-2': [1],
    '__lshift__-2': [1], '__ilshift__-2': [1], '__rshift__-2': [1], '__irshift__-2': [1],
    'normal-2': [0, 1], 'bernoulli-2': [0, 1],
}


def is_tensor_arg(arg):
    return arg['simple_type'] in {'Tensor', 'TensorList'}


def is_sized_intlist_arg(arg):
    """Returns True for arguments declared as IntList[k], but False for IntList."""
    return (arg['simple_type'] == 'IntList') and ('size' in arg)


def gen_jit_dispatch(declarations, out, template_path):
    ATEN_DISPATCH_CPP = CodeTemplate.from_file(template_path + '/aten_dispatch.cpp')
    ATEN_INTERNED_STRINGS_H = CodeTemplate.from_file(template_path + '/aten_interned_strings.h')

    ops = {}

    def get_invocation(decl, args, num_dynamic_inputs):
        if decl.get('has_tensor_options'):
            return CALL_TENSOR_OPTIONS.substitute(name=decl['name'], args=args[:-3])
        elif 'namespace' in decl['method_of']:
            return CALL_NAMESPACE.substitute(name=decl['name'], args=args, num_dynamic_inputs=num_dynamic_inputs)
        else:
            return CALL_METHOD.substitute(
                name=decl['name'], first=args[0], args=args[1:],
                num_dynamic_inputs=num_dynamic_inputs)

    def emit_decl_variant(decl, is_positional_arg, has_tensorlist):
        # is_positional_arg is a boolean list the same length as decl['arguments']
        # that indicates if the argument should come from the postional list
        # of inputs. If false, the argument comes from the constant attributes
        kw_assignments = []
        attr_names = []
        pos_assignments = []
        arguments = []

        if has_tensorlist:
            kw_assignments.append('size_t varargs_length = node->inputs().size();')
            # arguments look like: [tensor list], arg1, arg2, arg3
            # we use peek(<i>, static_inputs) to read the non-vararg inputs
            # from the end of the stack
            static_inputs = sum(is_positional_arg) - 1
            num_dynamic_inputs = 'varargs_length'
            tensorlist_idx = [i for i, arg in enumerate(decl['arguments']) if arg['simple_type'] == 'TensorList'][0]
        else:
            static_inputs = sum(is_positional_arg)
            num_dynamic_inputs = static_inputs

        real_inputs = 0
        for i, arg in enumerate(decl['arguments']):
            # This conditional allows us to process argument lists with a flattened argument list
            # with a single TensorList. Given the sequence of arguments:
            # a b c [d e f g] h i # [] is the list
            #
            # 1. For the section where we are processing positional inputs before the
            #    TensorList:
            #    a b c [d e f g] h i # [] is the list
            #    ~~~~~~~~~~~~ <- N
            #   we set this view_length to the total number of varargs inputs (i.e. the length)
            #   of the whole argument list. This means that indexing into the list using peek()
            #   we will retrieve arguments ar their true indices (i.e. peek at 0 points to a,
            #   1 points to b, etc...). Similarly, we can use peekSlice() to index into the
            #   list itself this way.
            # 2. After the list:
            #    a b c [d e f g] h i # [] is the list
            #                 ~~~~~~ <- N
            #   Here we set the view length to static_inputs. In our example,
            #   we effectively ignore the fact that we have a list here. What is
            #   significant is that our index i is equivalent when the view length
            #   is right-justified, whether we have the list or not. Concretely,
            #   indexing h or i from `a b c [d e f g] h i` is equvalent to indexing
            #   h or i from `a b c h i`.
            view_length = 'varargs_length' if has_tensorlist and i < tensorlist_idx else static_inputs

            if arg['simple_type'] == 'TensorList':
                # NOTE: don't advance real_inputs here. After this we are going
                # to switch over to indexing from the end as if we only had
                # the static arguments.
                arguments.append('peekSlice(stack, {}, varargs_length - {}, varargs_length)'
                                 .format(real_inputs, static_inputs))
            elif arg['simple_type'] in default_only_types:
                arguments.append(arg['default'])
            elif is_tensor_arg(arg):
                arguments.append('std::move(peek(stack, {}, {}))'.format(real_inputs, view_length))
                real_inputs += 1
            elif is_positional_arg[i]:
                from_tensor = FROM_TENSOR.get(arg['simple_type'], 'tensor_as<{}>'.format(arg['simple_type']))
                template_kwargs = dict(from_tensor=from_tensor,
                                       name=arg['name'],
                                       i=real_inputs,
                                       N=view_length)
                real_inputs += 1

                assign = POS_ASSIGNMENT.substitute(**template_kwargs)

                pos_assignments.append(assign)
                arguments.append(arg['name'])
            else:
                simple_type = arg['simple_type']

                assert simple_type in ATTR_METHOD_MAP, (decl['name'], simple_type)
                attr_method = ATTR_METHOD_MAP[simple_type]
                assign = KW_ASSIGNMENT.substitute(type_cast=TYPE_CASTS.get(simple_type, simple_type),
                                                  name=arg['name'],
                                                  method=attr_method)
                kw_assignments.append(assign)
                attr_names.append('{}_{}'.format(arg['name'], attr_method))
                arguments.append(arg['name'])

        call = get_invocation(decl, arguments, num_dynamic_inputs)

        # Descriptor is a unique identifier for a particular overload of an op.
        attr_names = sorted(attr_names)
        num_inputs = '*' if has_tensorlist else static_inputs
        descriptor = '-'.join([decl['name'], str(num_inputs)] + attr_names)

        # If there are two overloads with the same descriptor, that differ only by a type of a
        # single argument, where one of them takes a tensor, while another one takes an
        # at::Scalar as a positional scalar arg, then prefer the tensor overload.
        # It should get broadcasted correctly.
        if descriptor in skip_scalar_overload:
            if any(decl['arguments'][idx]['simple_type'] in {'Scalar', 'double'}
                   for idx in skip_scalar_overload[descriptor]):
                return

        returns = decl['returns']
        all_scalars = all(r['dynamic_type'] != 'TensorList' for r in returns)
        num_outputs = str(len(returns)) if all_scalars else 'UNKNOWN_OUTPUTS'

        constructor = CONSTRUCTOR.substitute(descriptor=descriptor, name=decl['name'],
                                             call=call,
                                             kw_assignments=kw_assignments,
                                             pos_assignments=pos_assignments,
                                             num_dynamic_inputs=num_dynamic_inputs,
                                             num_outputs=num_outputs)

        assert descriptor not in ops, descriptor
        ops[descriptor] = constructor

    def emit_decl(decl):
        arguments = decl['arguments']
        has_tensorlist = any(arg['simple_type'] == 'TensorList' for arg in arguments)
        num_tensor_args = sum(map(is_tensor_arg, arguments))

        # Right now, we generate dispatch methods that either take all non-tensor arguments
        # as attributes, or don't use any attributes at all. In the future we might want to
        # have something in the middle too (might be useful for e.g. constant propagation
        # into attributes, as that would allow us to avoid reparsing tensors into scalar
        # args at every invocation).

        all_real_arguments_are_inputs = tuple(arg['simple_type'] not in default_only_types for arg in arguments)
        only_tensors_are_inputs = tuple(is_tensor_arg(arg) for arg in arguments)

        # NB: if there are no scalar args then both options on LHS are equivalent, so deduplicate them.
        for variant in {all_real_arguments_are_inputs, only_tensors_are_inputs}:
            emit_decl_variant(decl, variant, has_tensorlist)

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
        result = []
        for group in grouped_decls:
            sorted_decls = sorted(group, key=declkey)
            result.extend(sorted_decls)
        return result

    # We need to add methods implemented manually in TensorImpl
    tensor_impl_methods = [{
        'name': name,
        'api_name': name,
        'method_of': ['Tensor'],
        'arguments': [{'name': 'self', 'simple_type': 'Tensor'}],
        'returns': [{'name': 'result', 'type': 'int64_t', 'dynamic_type': 'int64_t', 'simple_type': 'int64_t'}],
    } for name in ['sizes', 'strides', 'dim']]
    aten_decls = load_aten_declarations(declarations) + tensor_impl_methods

    jit_decls = [d for d in aten_decls if is_jit_op(d)]

    # add arguments dtype and device for functions like zeros
    for decl in jit_decls:
        arguments = decl['arguments']
        for n, arg in enumerate(arguments):
            if arg['simple_type'] == 'TensorOptions':
                del arguments[n]
                arguments.extend([
                    # XXX - until we actually have first-class interpreter types for these
                    # concepts, the default values to be encoded in Tensors

                    # dtype is specified as an int64_t of at::ScalarType
                    {'name': 'dtype', 'simple_type': 'ScalarType', 'default': 'float', 'kwarg_only': True},
                    # layout is specified as an int64_t of at::Layout
                    {'name': 'layout', 'simple_type': 'Layout', 'default': 'strided', 'kwarg_only': True},
                    # device is specified as an IntList of { at::Device::Type, device_id }
                    {'name': 'device', 'simple_type': 'Device', 'kwarg_only': True,
                        'default': '[cpu, -1]'},
                ])
                decl['has_tensor_options'] = True

    jit_decls = sort_decls(jit_decls)
    for decl in jit_decls:
        emit_decl(decl)

    # Sort the generated snippets to ensure that the generation is deterministic
    env = {
        'constructors': sorted(ops.values()),
    }
    write(out, 'aten_dispatch.cpp', ATEN_DISPATCH_CPP, env)

    emit_schema(jit_decls, out, template_path)

    # NB: Operate on aten_decls, not jit_decls, because VariableType is
    # a client for these symbols as well
    # NB: This means we DON'T generate interned strings for inplace ops.
    # Change this when you do!
    # NB: Keep this code synchronized with the code in
    # tool/autograd/gen_variable_type.py
    # NB: Some operations have inplace versions, but NOT non-inplace
    # versions! Thus uninplace_api_name() is mandatory (if you remove
    # it, you will get missing symbols.)
    names = set(uninplace_api_name(decl['api_name']) for decl in aten_decls)
    # NB: This grabs non keyword arguments too, but it's harmless
    attrs = set(arg['name'] for decl in aten_decls for arg in decl['arguments'])
    strings_env = {
        'aten_symbols': ["_(aten, {}) \\".format(n) for n in sorted(names)],
        'attr_symbols': ["_(attr, {}) \\".format(n) for n in sorted(attrs)]
    }
    write(out, 'aten_interned_strings.h', ATEN_INTERNED_STRINGS_H, strings_env)


# JIT has a type system of
# Scalar = int | float | bool # int is the largest int (int64_t),
# float is the largest float (double) we don't have the others because they are never held in tensors
# Type = Scalar # primitive numbers
#      | Tensor # any tensor, as defined by at::Tensor
#      | Type[] # a dynamically sizes list of a type
#      | Scalar[N] # a homogenous fixed size scalar list, single scalars can expand to this list
#      | (Type1, Type2, ...) # a heterogenous tuple
#      | Layout | ScalarType | Device | Generator # special singleton types for built-in concepts in tensor lib

# clean up the variety of C++ types in the ATen declarations
# to be in the restricted set of types that the IR represents
# note: no default values for this map, to make it clear what types
# can be passedthrough

TYPE_MAP = {
    'std::array<bool,2>': 'bool[2]',
    'std::array<bool,3>': 'bool[3]',
    'std::array<bool,4>': 'bool[4]',
    'Scalar': 'Scalar',
    'Tensor': 'Tensor',
    'TensorList': 'Tensor[]',
    # this appears in return values instead of TensorList
    # since TensorList is a ArrayRef in arguments but a vector
    # in returns
    'std::vector<Tensor>': 'Tensor[]',
    'IntList': 'int[]',
    'Layout': 'Layout',
    'Device': 'Device',
    'ScalarType': 'ScalarType',
    'int64_t': 'int',
    'double': 'float',
    'bool': 'bool',
    'Generator': 'Generator',
}

default_map = {'{}': 'None', 'nullptr': 'None'}


def signature(decl):
    def format_arg(arg):
        name = arg['name']
        typ = format_type(arg)
        decl = '{} {}'.format(typ, name)
        if 'default' in arg:
            # clean up initializer lists {{true, true}} -> [true, true]
            default = str(arg['default']) \
                .replace('{{', '[') \
                .replace('}}', ']') \
                .replace('true', 'True') \
                .replace('false', 'False') \
                .replace('nullptr', 'None') \
                .replace('{}', 'None' if is_tensor_arg(arg) else '[]')

            default = default_map.get(default, default)
            decl = '{}={}'.format(decl, default)
        return decl

    def format_type(arg):
        typ = TYPE_MAP[arg['simple_type']]
        if is_sized_intlist_arg(arg):
            typ = 'int[{}]'.format(arg['size'])

        if arg.get('is_nullable'):
            typ = '{}?'.format(typ)

        return typ

    args = []
    kwarg_only = False
    for a in decl['arguments']:
        if not kwarg_only and a.get('kwarg_only'):
            args.append('*')
            kwarg_only = True
        args.append(format_arg(a))

    arg_list = ', '.join(args)
    if len(decl['returns']) == 1:
        ret_list = format_type(decl['returns'][0])
    else:
        ret_list = '({})'.format(', '.join(format_type(r) for r in decl['returns']))
    return '{}({}) -> {}'.format(decl['name'], arg_list, ret_list)


def emit_schema(jit_decls, out, template_path):
    ATEN_SCHEMA_CPP = CodeTemplate.from_file(template_path + '/aten_schema_declarations.cpp')

    # see [aten_schema encoding] for how this gets translated to C++ object
    env = {
        'declarations': [],
    }

    for decl in jit_decls:
        env['declarations'].append(signature(decl))

    write(out, 'aten_schema_declarations.cpp', ATEN_SCHEMA_CPP, env)


def main():
    parser = argparse.ArgumentParser(
        description='Generate JIT op dispatch')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    parser.add_argument('template-path', metavar='TEMPLATE_PATH',
                        help='path to templates directory')
    args = parser.parse_args()
    gen_jit_dispatch(args.declarations, args.out, args.template_path)


if __name__ == '__main__':
    main()
