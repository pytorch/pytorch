import re
import pprint

"""
Instructions:

1. pytest -n 8 test/test_vmap.py test/test_ops.py test/test_pythonkey.py > result.txt
2. test/xfail_suggester.py
"""

with open('result.txt') as f:
    lines = f.readlines()

failed = [line for line in lines if line.startswith('FAILED')]
p = re.compile('FAILED test/test_\w+.py::\w+::(\S+)')

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
    'test_vmap_exhaustive_',
    'test_op_has_batch_rule_',
    'test_jvp_',
    'test_vmapjvp_',
    'test_decomposition_',
    'test_make_fx_',
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

def sanitize_base(base):
    if base.startswith('nn_functional_'):
        base = f'nn.functional.{base[len("nn_functional_"):]}'
    if base.startswith('fft_'):
        base = f'fft.{base[len("fft_"):]}'
    if base.startswith('linalg_'):
        base = f'linalg.{base[len("linalg."):]}'
    if base.startswith('_masked_'):
        base = f'_masked.{base[len("_masked_"):]}'
    return base

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
        sanitized_base = sanitize_base(base)
        if cpu_variant in tests and cuda_variant in tests:
            result.append(f"xfail('{sanitized_base}'),")
            continue
        if cpu_variant in tests:
            result.append(f"xfail('{sanitized_base}', device_type='cpu'),")
            continue
        if cuda_variant in tests:
            result.append(f"xfail('{sanitized_base}', device_type='cuda'),")
            continue
        result.append(f"skip('{sanitized_base}'),")
    return result

result = {base: get_suggested_xfails(base, failed_tests) for base in base_names}
for k, v in result.items():
    print('=' * 50)
    print(k)
    print('=' * 50)
    print('\n'.join(v))
