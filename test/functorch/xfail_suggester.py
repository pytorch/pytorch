import re
import torch

"""
Instructions:

1. pytest -n 8 test/test_vmap.py test/test_ops.py test/test_aotdispatch.py > result.txt
2. python test/xfail_suggester.py
"""

with open('result.txt') as f:
    lines = f.readlines()

failed = [line for line in lines if line.startswith('FAILED')]
p = re.compile('FAILED test/test_\w+.py::\w+::(\S+)')  # noqa: W605


def get_failed_test(line):
    m = p.match(line)
    if m is None:
        return None
    return m.group(1)


base_names = {
    'test_grad_',
    'test_vjp_',
    'test_vmapvjp_',
    'test_vmapvjp_has_batch_rule_',
    'test_vjpvmap_',
    'test_jvp_',
    'test_vmapjvp_',
    'test_vmapjvpall_has_batch_rule_',
    'test_vmapjvpall_',
    'test_jvpvjp_',
    'test_vjpvjp_',
    'test_decomposition_',
    'test_make_fx_exhaustive_',
    'test_vmap_exhaustive_',
    'test_op_has_batch_rule_',
    'test_vmap_autograd_grad_',
}

failed_tests = [get_failed_test(line) for line in lines]
failed_tests = [match for match in failed_tests if match is not None]
failed_tests = sorted(failed_tests)

suggested_xfails = {}


def remove_device_dtype(test):
    return '_'.join(test.split('_')[:-2])


def belongs_to_base(test, base):
    if not test.startswith(base):
        return False
    candidates = [try_base for try_base in base_names if len(try_base) > len(base)]
    for candidate in candidates:
        if test.startswith(candidate):
            return False
    return True


def parse_namespace(base):
    mappings = {
        'nn_functional_': 'nn.functional',
        'fft_': 'fft',
        'linalg_': 'linalg',
        '_masked_': '_masked',
        'sparse_': 'sparse',
        'speical_': 'special',
    }
    for heading in mappings.keys():
        if base.startswith(heading):
            return mappings[heading], base[len(heading):]
    return None, base


def get_torch_module(namespace):
    if namespace is None:
        return torch
    if namespace == 'nn.functional':
        return torch.nn.functional
    return getattr(torch, namespace)


def parse_base(base):
    namespace, rest = parse_namespace(base)

    apis = dir(get_torch_module(namespace))
    apis = sorted(apis, key=lambda x: -len(x))

    api = rest
    variant = ''
    for candidate in apis:
        if rest.startswith(candidate):
            api = candidate
            variant = rest[len(candidate) + 1:]
            break
    print(base, namespace, api, variant)
    return namespace, api, variant


def any_starts_with(strs, thing):
    for s in strs:
        if s.startswith(thing):
            return True
    return False


def get_suggested_xfails(base, tests):
    result = []
    tests = [test[len(base):] for test in tests if
             belongs_to_base(test, base)]

    base_tests = set([remove_device_dtype(test) for test in tests])
    tests = set(tests)
    for base in base_tests:
        cpu_variant = base + '_cpu_float32'
        cuda_variant = base + '_cuda_float32'
        namespace, api, variant = parse_base(base)
        if namespace is None:
            api = api
        else:
            api = f'{namespace}.{api}'
        if cpu_variant in tests and cuda_variant in tests:
            result.append(f"xfail('{api}', '{variant}'),")
            continue
        if cpu_variant in tests:
            result.append(f"xfail('{api}', '{variant}', device_type='cpu'),")
            continue
        if cuda_variant in tests:
            result.append(f"xfail('{api}', '{variant}', device_type='cuda'),")
            continue
        result.append(f"skip('{api}', '{variant}',")
    return result


result = {base: get_suggested_xfails(base, failed_tests) for base in base_names}
for k, v in result.items():
    print('=' * 50)
    print(k)
    print('=' * 50)
    print('\n'.join(v))
