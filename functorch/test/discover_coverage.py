import torch
import copy
from torch.testing._internal.common_methods_invocations import op_db
from enum import Enum
from functorch_lagging_op_db import functorch_lagging_op_db

# Importing these files make modifications to the op_db that we need
import test_ops
import test_vmap

all_overridable = list(torch.overrides.get_testing_overrides().keys())

public_docs = [
    (torch.nn.functional, 'torch.nn.functional', 'docs/source/nn.functional.rst'),
    (torch.fft, 'torch.fft', 'docs/source/fft.rst'),
    (torch.special, 'torch.special', 'docs/source/special.rst'),
    (torch.linalg, 'torch.linalg', 'docs/source/linalg.rst'),
    (torch, 'torch', 'docs/source/torch.rst'),
    (torch.Tensor, 'torch.Tensor', 'docs/source/tensors.rst'),
]

# torch.abs, Tensor.abs, Tensor.abs_ are all considered to be different
def get_public_overridable_apis(pytorch_root='/raid/rzou/pt/whiteboard'):
    results = {}
    all_overridable_apis = set(torch.overrides.get_testing_overrides().keys())
    for module, module_name, src in public_docs:
        with open(f'{pytorch_root}/{src}') as f:
            lines = f.readlines()
        # APIs eitehr begin with 4 spaces or ".. autofunction::"
        api_lines1 = [line.strip() for line in lines if line.startswith(' ' * 4)]
        api_lines2 = [line.strip()[len('.. autofunction:: '):]
                      for line in lines if line.startswith('.. autofunction::')]
        lines = api_lines1 + api_lines2
        lines = [line[7:] if line.startswith('Tensor.') else line for line in lines]
        lines = [line for line in lines if hasattr(module, line)]
        for line in lines:
            api = getattr(module, line)
            if api in all_overridable_apis:
                results[f'{module_name}.{line}'] = api
    return results

# Deduplicates torch.abs and Tensor.abs
def get_public_overridable_ops():
    results = get_public_overridable_apis()
    cpy = copy.deepcopy(results)
    for key, _ in cpy.items():
        if not key.startswith('torch.Tensor'):
            continue
        api = key.split('.')[2]
        if f'torch.{api}' in results.keys():
            del results[key]
    return results

def get_public_overridable_outplace_ops():
    results = get_public_overridable_ops()
    cpy = copy.deepcopy(results)
    for key, _ in cpy.items():
        # NB: there are no dunder methods bcs we don't document those
        if key.endswith('_'):
            del results[key]
    return results

def get_public_overridable_outplace_we_care_about():
    results = get_public_overridable_outplace_ops()
    cpy = copy.deepcopy(results)
    for key, _ in cpy.items():
        # quantization
        if 'quant' in key or '.q_' in key:
            del results[key]

        # is_cpu, etc. It doesn't make sense to have OpInfos for these
        if '.is_' in key:
            del results[key]
    return results

# Maps function -> OpInfo
def get_ops_covered_by_opinfos():
    ops = {}
    for opinfo in op_db:
        ops[opinfo.op] = opinfo
        if opinfo.method_variant:
            ops[opinfo.method_variant] = opinfo
        if opinfo.inplace_variant:
            ops[opinfo.inplace_variant] = opinfo
        for alias in opinfo.aliases:
            ops[alias.op] = opinfo
    return ops

def get_covered_ops(ops_list, invert=False):
    ops_covered_by_opinfo = get_ops_covered_by_opinfos()
    overridable_outplace_ops = ops_list
    results = {}
    for key, op in overridable_outplace_ops.items():
        cond = op in ops_covered_by_opinfo
        if invert:
            cond = not cond
        if cond:
            results[key] = op
    return results

class Status(Enum):
    Correct = 0
    Fast = 1

tests = {
    'test_vmap_exhaustive',
    'test_op_has_batch_rule',
    'test_vjp',
    'test_vmapvjp',
    'test_vmapvjp_has_batch_rule',
}

def get_statuses():
    overridable_outplace_we_care_about = get_public_overridable_outplace_we_care_about()
    op_to_opinfo = get_ops_covered_by_opinfos()
    result = {}
    for _, op in get_covered_ops(overridable_outplace_we_care_about).items():
        opinfo = op_to_opinfo[op]
        success = copy.deepcopy(tests)
        for decorator in opinfo.decorators:
            if not hasattr(decorator, 'test_name'):
                continue
            if decorator.test_name in tests and decorator.test_name in success:
                success.remove(decorator.test_name)
        for func in [opinfo.op] + [alias.op for alias in opinfo.aliases]:
            if opinfo.name not in result.keys():
                result[func] = success
            else:
                result[func] = result[opinfo.name].intersection(success)
    return result

def transpose_statuses():
    statuses = get_statuses()
    result = {}
    for test in tests:
        result[test] = set({})
    for op, supported in statuses.items():
        for test in supported:
            result[test].add(op)
    return result

overridable_apis = get_public_overridable_apis()
print(f'Overridable public APIs: {len(overridable_apis)}')

overridable_ops = get_public_overridable_ops()
print(f'Overridable public ops: {len(overridable_ops)}')

overridable_outplace_ops = get_public_overridable_outplace_ops()
print(f'Overridable public outplace ops: {len(overridable_outplace_ops)}')

overridable_outplace_we_care_about = get_public_overridable_outplace_we_care_about()
print(f'Overridable public outplace ops we care about: {len(overridable_outplace_we_care_about)}')

tested_overridable_outplace_ops = get_covered_ops(overridable_outplace_we_care_about)
print(f'OpInfo-tested overridable public outplace ops: {len(tested_overridable_outplace_ops)}')

untested_overridable_outplace_ops = get_covered_ops(overridable_outplace_we_care_about, invert=True)
# print(untested_overridable_outplace_ops.keys())

statuses = transpose_statuses()
for test in tests:
    print(f'{test} coverage {len(statuses[test])}')
