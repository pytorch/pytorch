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
    'std::array<bool,4>': 'is',
}

TYPE_CASTS = {
    'std::array<bool,2>': 'as_bool_array<2>',
    'std::array<bool,3>': 'as_bool_array<3>',
    'std::array<bool,4>': 'as_bool_array<4>',
    'Scalar': 'Scalar',
    'IntList': 'std::vector<int64_t>',
}

KW_ASSIGNMENT = CodeTemplate("""\
auto ${name} = ${type_cast}(node->${method}(Symbol("${name}")));\
""")

POS_ASSIGNMENT = CodeTemplate("""\
auto ${name} = tensor_as<${type}>(std::move(fromLast(stack, ${arg_idx})));\
""")

CALL_NAMESPACE = CodeTemplate("at::${name}(${args})")
CALL_METHOD = CodeTemplate("(${first}).${name}(${args})")

CONSTRUCTOR = CodeTemplate("""\
{"${descriptor}", [](Node *node) {
  ${kw_assignments}
  return TensorOp([=](Stack & stack) {
    autograd::profiler::RecordFunction record("${name}");
    AutoGPU device_guard(deviceForInputs(stack, ${num_inputs} + ${num_dropped_args}));
    ${pos_assignments}
    ${pos_arg_drop}
    auto result = ${call};
    drop(stack, ${num_inputs});
    pack(stack, std::move(result));
    return 0;
  }, "${name}", ${num_inputs});
}},
""")


def is_jit_op(decl):
    uses_tensors = any(arg['simple_type'] in {'Tensor', 'TensorList'} for arg in decl['arguments']) or \
        'Tensor' in decl['method_of']
    return (not decl['api_name'].endswith('_') and
            not decl['name'].endswith('_out') and
            not any(arg['simple_type'] == 'Generator' for arg in decl['arguments']) and
            not any(arg['simple_type'] == 'SparseTensor' for arg in decl['arguments']) and
            not any(arg['simple_type'] == 'Storage' for arg in decl['arguments']) and
            not any(arg['simple_type'] == 'Type' for arg in decl['arguments']) and
            uses_tensors)


skip_scalar_overload = {
    'lt-2': [1], 'gt-2': [1], 'le-2': [1], 'ge-2': [1], 'eq-2': [1], 'ne-2': [1],
    'pow-2': [0, 1], 'add-3': [1], 'sub-3': [1], 'mul-2': [1], 'div-2': [1],
    'fmod-2': [1], 'remainder-2': [1]
}


def gen_jit_dispatch(declarations, out):
    # We need to add methods implemented manually in TensorImpl
    tensor_impl_methods = [{
        'name': name,
        'api_name': name,
        'method_of': ['Tensor'],
        'arguments': [{'name': 'self', 'simple_type': 'Tensor'}],
    } for name in ['sizes', 'strides', 'dim']]
    aten_decls = load_aten_declarations(declarations) + tensor_impl_methods
    jit_decls = [d for d in aten_decls if is_jit_op(d)]

    def is_tensor_arg(arg):
        return arg['simple_type'] in {'Tensor', 'TensorList'}

    ops = {}
    for decl in jit_decls:
        arguments = decl['arguments']
        name = decl['name']
        has_tensorlist = any(arg['simple_type'] == 'TensorList' for arg in arguments)
        scalar_arg_idx = [i for i, arg in enumerate(arguments) if not is_tensor_arg(arg)]
        num_tensor_args = sum(map(is_tensor_arg, arguments))
        # TODO: support this
        if has_tensorlist and (num_tensor_args != 1 or not is_tensor_arg(arguments[0])):
            continue

        # Right now, we generate dispatch methods that either take all non-tensor arguments
        # as attributes, or don't use any attributes at all. In the future we might want to
        # have something in the middle too (might be useful for e.g. constant propagation
        # into attributes, as that would allow us to avoid reparsing tensors into scalar
        # args at every invocation).
        # NB: if there are no scalar args then both options on LHS are equivalent, so deduplicate them.
        scalar_arg_idx_iter = ([], scalar_arg_idx) if scalar_arg_idx else ([],)
        for pos_scalar_arg_idx in scalar_arg_idx_iter:
            num_args = len(arguments)
            num_inputs = num_tensor_args + len(pos_scalar_arg_idx) if not has_tensorlist else '*'

            # Scatter arguments into positional and keyword, and compute stack offsets
            # of posiitional args.
            pos_scalar_args, kw_scalar_args = [], []
            scalar_stack_off, tensor_stack_off = [], []
            for i, arg in enumerate(arguments):
                # XXX: we currently support only TensorList ops that have a TensorList as
                # the first argument, that is then followed by a number of positional args.
                stack_off = (num_args if num_inputs == '*' else num_inputs) - i - 1
                if is_tensor_arg(arg):
                    tensor_stack_off.append(stack_off)
                else:
                    if i in pos_scalar_arg_idx:
                        pos_scalar_args.append(arg)
                        scalar_stack_off.append(stack_off)
                    else:
                        kw_scalar_args.append(arg)

            # Descriptor is a unique identifier for a particular overload of an op.
            attr_names = sorted([arg['name'] for arg in kw_scalar_args])
            descriptor = '-'.join([decl['name'], str(num_inputs)] + attr_names)

            # If there are two overloads with the same descriptor, that differ only by a type of a
            # single argument, where one of them takes a tensor, while another one takes an
            # at::Scalar as a positional scalar arg, then prefer the tensor overload.
            # It should get broadcasted correctly.
            if descriptor in skip_scalar_overload:
                if any(arguments[idx]['simple_type'] == 'Scalar'
                       for idx in skip_scalar_overload[descriptor]):
                    continue

            kw_assignments = [KW_ASSIGNMENT.substitute(type_cast=TYPE_CASTS.get(arg['simple_type'], arg['simple_type']),
                                                       name=arg['name'],
                                                       method=ATTR_METHOD_MAP[arg['simple_type']])
                              for arg in kw_scalar_args]
            if num_inputs == "*":
                kw_assignments.append('size_t varargs_length = node->inputs().size();')
                num_inputs = 'varargs_length'
            pos_assignments = [POS_ASSIGNMENT.substitute(type=arg['simple_type'],
                                                         name=arg['name'],
                                                         arg_idx=arg_idx)
                               for arg_idx, arg in zip(scalar_stack_off, pos_scalar_args)]

            # Generate the actuall ATen call. This gets a bit tricky because of
            # TensorList arguments, and functions that are only available as methods.
            pos_arg_drop = ''
            num_dropped_args = 0
            if 'namespace' in decl['method_of']:
                if has_tensorlist:
                    # We need to drop the scalar args following varargs before we use last
                    if pos_scalar_args:
                        num_dropped_args = len(pos_scalar_args)
                        pos_arg_drop = 'drop(stack, {});'.format(num_dropped_args)
                    args = ['last(stack, varargs_length)' if is_tensor_arg(arg) else arg['name']
                            for arg in arguments]
                else:
                    tensor_id = iter(tensor_stack_off)
                    args = ['std::move(fromLast(stack, {}))'.format(1 + next(tensor_id))
                            if is_tensor_arg(arg) else arg['name']
                            for arg in arguments]
                call = CALL_NAMESPACE.substitute(name=name, args=args)
            else:
                tensor_id = iter(tensor_stack_off)
                args = ['std::move(fromLast(stack, {}))'.format(1 + next(tensor_id))
                        if is_tensor_arg(arg) else arg['name']
                        for arg in arguments]
                call = CALL_METHOD.substitute(name=name, first=args[0], args=args[1:])

            constructor = CONSTRUCTOR.substitute(descriptor=descriptor, name=name,
                                                 num_dropped_args=num_dropped_args,
                                                 pos_arg_drop=pos_arg_drop,
                                                 call=call,
                                                 kw_assignments=kw_assignments,
                                                 pos_assignments=pos_assignments,
                                                 num_inputs=num_inputs)

            assert descriptor not in ops, descriptor
            ops[descriptor] = constructor

    # Sort the generated snippets to ensure that the generation is deterministic
    env = {'constructors': sorted(ops.values())}
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
