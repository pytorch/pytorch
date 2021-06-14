import argparse
import re

from codegen_outofplacebatching import deindent, get_signatures, gen_unwraps


def get_signature(op, path):
    signatures = get_signatures(path, include_op=True)
    result = [sig for sig in signatures if sig[0] == op]
    if len(result) != 1:
        raise ValueError("")
    return result[0]


def gen_return_sig(return_t):
    if len(return_t) == 1:
        return return_t[0]
    return f'std::tuple<{".".join(return_t)}>'


def gen_args_sig(args_t):
    args = [f'{typ} {argname}' for typ, argname in args_t]
    return ', '.join(args)


def gen_args_list(args_t):
    args = [f'{argname}' for _, argname in args_t]
    return ', '.join(args)


def gen_plumbing(signature):
    # "add.Tensor"
    op, return_t, args_t = signature

    maybe_op_and_variant = op.split('.')
    if len(maybe_op_and_variant) == 1:
        op = maybe_op_and_variant[0]
        variant = ''
        opname = op
    else:
        op, variant = maybe_op_and_variant
        opname = f'{op}_{variant}'

    if op.endswith('_'):
        raise ValueError('Codegen doesn\'t handle in-place ops')

    arg_types, arg_names = zip(*args_t)
    unwraps, _ = gen_unwraps(arg_types, arg_names)

    result = deindent(f"""\
    {gen_return_sig(return_t)} {opname}_plumbing({gen_args_sig(args_t)}) {{
      auto maybe_layer = maybeCurrentDynamicLayer();
      TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
      int64_t cur_level = maybe_layer->layerId();

      {unwraps}

      // Your logic here

      static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{op}", "{variant}");
      return slow_fallback<{','.join(return_t)}>(op, {{ {gen_args_list(args_t)} }});
    }}
    """)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate the batch rule plumbing for an op')
    parser.add_argument('op',
                        help='the operator name (with overload name)')
    parser.add_argument('path',
                        help='link to RegistrationDeclarations.h')

    # Sample usage:
    # gen_plumbing.py add.Tensor ~/pytorch/build/aten/src/ATen/RegistrationDeclarations.h
    args = parser.parse_args()
    signature = get_signature(args.op, args.path)
    result = gen_plumbing(signature)
    print(result)
