import os
import argparse
from collections import defaultdict
from tools.shared.module_loader import import_module
from itertools import count
from ..autograd.gen_variable_type import load_aten_declarations, CodeTemplate, write, \
    FALLTHROUGH_RETURN_TYPES, FALLTHROUGH_FUNCTIONS, GENERATED_COMMENT

template_path = os.path.join(os.path.dirname(__file__), 'templates')

ATEN_DISPATCH_H = CodeTemplate.from_file(template_path + '/aten_dispatch.h')
ATEN_DISPATCH_CPP = CodeTemplate.from_file(template_path + '/aten_dispatch.cpp')

ATTR_METHOD_MAP = {
    'int64_t': 'i',
    'IntList': 'is',
    'Scalar': 't',
    'bool': 'i',
    'double': 'f',
    'std::array<bool,2>': 'is',
    'std::array<bool,3>': 'is',
}

TYPE_CASTS = {
    'std::array<bool,2>': 'as_bool_array<2>',
    'std::array<bool,3>': 'as_bool_array<3>',
    'Scalar': 'Scalar',
    'IntList': 'std::vector<int64_t>',
}

ATTR_ASSIGNMENT = CodeTemplate("""\
auto ${name} = ${type_cast}(node->${method}(stringToSymbol("${name}")));\
""")

CALL_NAMESPACE = CodeTemplate("at::${name}(${args})")
CALL_METHOD = CodeTemplate("vars[0].${name}(${args})")

CONSTRUCTOR = CodeTemplate("""\
{"${descriptor}", [](Node *node) {
  ${assignments}
  return TensorOp([=](const variable_list& vars) -> variable_list {
    return pack_list(${call});
  }, "${name}", ${num_inputs});
}},
""")


def is_jit_op(decl):
    return (not decl['api_name'].endswith('_') and
            not decl['name'].endswith('_out') and
            not decl['name'].endswith('_forward') and
            not any(arg['simple_type'] == 'Generator' for arg in decl['arguments']) and
            not any(arg['simple_type'] == 'SparseTensor' for arg in decl['arguments']) and
            not decl['return_type'] in FALLTHROUGH_RETURN_TYPES and
            not decl['name'] in FALLTHROUGH_FUNCTIONS)


def gen_jit_dispatch(declarations, out):
    aten_decls = load_aten_declarations(declarations)
    jit_decls = [d for d in aten_decls if is_jit_op(d)]

    def is_tensor_arg(arg):
        return arg['simple_type'] in {'Tensor', 'TensorList'}

    ops = {}
    for decl in jit_decls:
        arguments = decl['arguments']
        name = decl['name']
        scalar_args = [arg for arg in arguments if not is_tensor_arg(arg)]

        # Descriptor is a unique identified for a particular overload of an op
        attr_names = sorted([arg['name'] for arg in scalar_args])
        num_inputs = len(arguments) - len(scalar_args)
        descriptor = '-'.join([decl['name'], str(num_inputs)] + attr_names)

        # All scalar args need to be assigned, so they can be captured by a lambda
        assignments = [ATTR_ASSIGNMENT.substitute(type=arg['simple_type'],
                                                  type_cast=TYPE_CASTS.get(arg['simple_type'], arg['simple_type']),
                                                  name=arg['name'],
                                                  method=ATTR_METHOD_MAP[arg['simple_type']])
                       for arg in scalar_args]

        # Generate the actuall ATen call. This gets a bit tricky because of
        # TensorList arguments, and functions that are only available as methods.
        if 'namespace' in decl['method_of']:
            if any(arg['simple_type'] == 'TensorList' for arg in arguments):
                assert sum(map(is_tensor_arg, arguments)) == 1
                args = ['as_tensor_list(vars)' if is_tensor_arg(arg) else arg['name']
                        for arg in arguments]
            else:
                tensor_id = iter(count(start=0))
                args = ['vars[{}]'.format(next(tensor_id)) if is_tensor_arg(arg) else arg['name']
                        for arg in arguments]
            call = CALL_NAMESPACE.substitute(name=name, args=args)
        else:
            tensor_id = iter(count(start=1))
            args = ['vars[{}]'.format(next(tensor_id)) if is_tensor_arg(arg) else arg['name']
                    for arg in arguments[1:]]
            call = CALL_METHOD.substitute(name=name, args=args)

        constructor = CONSTRUCTOR.substitute(descriptor=descriptor, name=name, call=call,
                                             assignments=assignments, num_inputs=num_inputs)
        assert descriptor not in ops, descriptor
        ops[descriptor] = constructor

    # Sort the generated snippets to ensure that the generation is deterministic
    env = {'constructors': sorted(list(ops.values()))}
    write(out, 'aten_dispatch.h', ATEN_DISPATCH_H, env)
    write(out, 'aten_dispatch.cpp', ATEN_DISPATCH_CPP, env)


def main():
    parser = argparse.ArgumentParser(
        description='Generate JIT op dispatch')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    args = parser.parse_args()
    gen_jit_dispatch(args.declarations, args.out)


if __name__ == '__main__':
    main()
