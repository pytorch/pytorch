import os
import argparse
from itertools import count
from ..autograd.utils import CodeTemplate, write
from ..autograd.gen_autograd import load_aten_declarations

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
CALL_METHOD = CodeTemplate("TensorTemporary(inputs[0]).value().${name}(${args})")

CONSTRUCTOR = CodeTemplate("""\
{"${descriptor}", [](Node *node) {
  ${assignments}
  return TensorOp([=](const list_of_retainable & inputs,
                      list_of_retainable & outputs) {
    autograd::profiler::RecordFunction record("${name}");
    AutoGPU device_guard(deviceForInputs(inputs));
    pack_list(outputs, ${call});
  }, "${name}", ${num_inputs});
}},
""")


def is_jit_op(decl):
    return (not decl['api_name'].endswith('_') and
            not decl['name'].endswith('_out') and
            not decl['name'].endswith('_forward') and
            not any(arg['simple_type'] == 'Generator' for arg in decl['arguments']) and
            not any(arg['simple_type'] == 'SparseTensor' for arg in decl['arguments']) and
            not any(arg['simple_type'] == 'Storage' for arg in decl['arguments']) and
            any(arg['simple_type'] in {'Tensor', 'TensorList'} for arg in decl['arguments']) and
            'Tensor' in decl['return_type'])


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
        has_tensorlist = any(arg['simple_type'] == 'TensorList' for arg in arguments)

        # Descriptor is a unique identified for a particular overload of an op
        attr_names = sorted([arg['name'] for arg in scalar_args])
        num_inputs = len(arguments) - len(scalar_args) if not has_tensorlist else "*"
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
            if has_tensorlist:
                if sum(map(is_tensor_arg, arguments)) != 1:
                    # TODO: support this
                    continue
                args = ['TensorTemporaryList(inputs)' if is_tensor_arg(arg) else arg['name']
                        for arg in arguments]
            else:
                tensor_id = iter(count(start=0))
                args = ['TensorTemporary(inputs[{}]).value()'.format(
                    next(tensor_id)) if is_tensor_arg(arg) else arg['name']
                    for arg in arguments]
            call = CALL_NAMESPACE.substitute(name=name, args=args)
        else:
            tensor_id = iter(count(start=1))
            args = ['TensorTemporary(inputs[{}]).value()'.format(next(tensor_id)) if is_tensor_arg(arg) else arg['name']
                    for arg in arguments[1:]]
            call = CALL_METHOD.substitute(name=name, args=args)

        constructor = CONSTRUCTOR.substitute(descriptor=descriptor, name=name, call=call,
                                             assignments=assignments,
                                             # num_inputs is only used in AutogradClosure, which
                                             # is going to be removed soon anyway. There's no good value
                                             # we can provide for cat.
                                             num_inputs=num_inputs if num_inputs != "*" else 0)
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
