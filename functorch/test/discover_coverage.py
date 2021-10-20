import torch
import copy
from torch.testing._internal.common_methods_invocations import op_db
from enum import Enum
from functorch_lagging_op_db import functorch_lagging_op_db
import functorch._src.top_operators_github_usage as top_ops

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
def get_public_overridable_apis(pytorch_root='/raid/rzou/pt/debug-cpu'):
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

denylist = {
    'torch.Tensor.data_ptr',
    'torch.Tensor.dim',
    'torch.Tensor.element_size',
    'torch.Tensor.backward',
    'torch.Tensor.as_strided',
    'torch.Tensor.register_hook',
    'torch.Tensor.record_stream',
    'torch.Tensor.qscheme',
    'torch.Tensor.ndimension',
    'torch.Tensor.smm',
    'torch.Tensor.sspaddmm',
    'torch.Tensor.retain_grad',
    'torch.Tensor.sparse_mask',
    'torch.Tensor.sparse_dim',
    'torch.Tensor.dense_dim',
    'torch.Tensor.values',
    'torch.Tensor.indices',
    'torch.Tensor.numel',
    'torch.Tensor.size',
    'torch.Tensor.nelement',
    'torch.Tensor.q_scale',
    'torch.Tensor.q_zero_point',
    'torch.Tensor.q_per_channel_scales',
    'torch.Tensor.q_per_channel_zero_points',
    'torch.Tensor.q_per_channel_axis',
    'torch.Tensor.int_repr',
    'torch.Tensor.to_sparse',
    'torch.Tensor.is_inference',
    'torch.Tensor.storage',
    'torch.Tensor.storage_type',
}

def get_method_only_ops_we_care_about():
    apis = get_public_overridable_apis()
    result = []
    for key, _ in apis.items():
        if not key.startswith('torch.Tensor'):
            continue
        if key in denylist:
            continue
        api = key.split('.')[2]
        # filter out in-place
        if api.endswith('_'):
            continue
        if f'torch.{api}' not in apis.keys():
            result.append(api)
    return result

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

        if key in denylist and key in results:
            del results[key]
    return results

# e.g. nn.functional.softmax
def get_op(dotted_name):
    names = dotted_name.split('.')
    mod = torch
    for name in names:
        if not hasattr(mod, name):
            return None
        mod = getattr(mod, name)
    return mod

# Maps function -> OpInfo
def get_ops_covered_by_opinfos():
    ops = {}
    for opinfo in op_db:
        func_op = get_op(opinfo.name)
        if func_op:
            ops[func_op] = opinfo
        if opinfo.method_variant:
            ops[opinfo.method_variant] = opinfo
        if opinfo.inplace_variant:
            ops[opinfo.inplace_variant] = opinfo
        for alias in opinfo.aliases:
            ops[alias.op] = opinfo
    return ops

def get_top_ops(torch_threshold, nn_fn_threshold):
    denylist = set({
        'tensor', 'load', 'zeros', 'no_grad', 'save', 'from_numpy',
        'manual_seed', 'ones', 'randn', 'arange', 'rand',
        'empty', 'randperm', 'linspace', 'set_grad_enabled',
        'isnan', 'set_default_tensor_type', 'set_num_threads',
        'set_printoptions', 'isfinite', 'range', 'numel',
        'set_default_dtype', 'sparse_coo_tensor', 'set_rng_state',
        'get_rng_state', 'get_default_dtype', 'initial_seed',
        'get_num_threads', 'quantize_per_tensor', 'logspace',
        'hann_window', 'is_tensor', 'as_tensor', 'randint', 'full', 'eye',
        'equal', 'enable_grad', 'seed', 'is_storage', 'hamming_window',
        'is_floating_point', 'nn.functional.torch',
    })

    torch_ops = [op[0] for op in top_ops.top_torch[:torch_threshold]]
    nn_fn_ops = [op[0] for op in top_ops.top_nn_functional[:nn_fn_threshold]]
    ops = torch_ops + nn_fn_ops
    ops = [op for op in ops if op not in denylist]
    return ops

def get_top_ops_not_covered_by_opinfo(torch_threshold=0, nn_fn_threshold=0):
    ops = get_top_ops(torch_threshold, nn_fn_threshold)

    ops_with_opinfo = []
    for op in op_db:
        ops_with_opinfo.append(op.name)
        ops_with_opinfo.extend([op.name for op in op.aliases])
    ops_with_opinfo = set(ops_with_opinfo)

    result = [op for op in ops if op not in ops_with_opinfo]
    result = [op for op in result if op not in denylist]
    return result

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

def get_statuses(for_subset=None, invert=False):
    overridable_outplace_we_care_about = get_public_overridable_outplace_we_care_about()
    if for_subset is not None:
        overridable_outplace_we_care_about = {
            k: v
            for k, v in overridable_outplace_we_care_about.items()
            # Removes "torch."
            if k[6:] in for_subset
        }
    op_to_opinfo = get_ops_covered_by_opinfos()
    result = {}
    x = get_covered_ops(overridable_outplace_we_care_about)
    for name, op in get_covered_ops(overridable_outplace_we_care_about).items():
        opinfo = op_to_opinfo[op]
        if invert == False:
            success = copy.deepcopy(tests)
            for decorator in opinfo.decorators:
                if not hasattr(decorator, 'test_name'):
                    continue
                if decorator.test_name in tests and decorator.test_name in success:
                    success.remove(decorator.test_name)
            # NB: disregard aliases, they're too much trouble
            for func in [opinfo.op]:
                if opinfo.name not in result.keys():
                    result[name] = success
                else:
                    result[name] = result[name].intersection(success)
        if invert == True:
            failures = set({})
            for decorator in opinfo.decorators:
                if not hasattr(decorator, 'test_name'):
                    continue
                if decorator.test_name in tests:
                    failures.add(decorator.test_name)

            # NB: disregard aliases, they're too much trouble
            for func in [opinfo.op]:
                if opinfo.name not in result.keys():
                    result[name] = failures
                else:
                    result[name] = result[name].union(failures)
    return result

def transpose_statuses(for_subset=None, invert=False):
    statuses = get_statuses(for_subset, invert=invert)
    result = {}
    for test in tests:
        result[test] = set({})
    for op, supported in statuses.items():
        for test in supported:
            result[test].add(op)
    return result

overridable_apis = get_public_overridable_apis()

overridable_ops = get_public_overridable_ops()

overridable_outplace_ops = get_public_overridable_outplace_ops()

overridable_outplace_we_care_about = get_public_overridable_outplace_we_care_about()

tested_overridable_outplace_ops = get_covered_ops(overridable_outplace_we_care_about)
untested_overridable_outplace_ops = get_covered_ops(overridable_outplace_we_care_about, invert=True)

print("List of OpInfos we need:")
for key in untested_overridable_outplace_ops.keys():
    print(key)
print("-" * 80)
print("")

print(f'Overridable public APIs: {len(overridable_apis)}')
print(f'Overridable public ops: {len(overridable_ops)}')
print(f'Overridable public outplace ops: {len(overridable_outplace_ops)}')
print(f'Overridable public outplace ops we care about: {len(overridable_outplace_we_care_about)}')
print(f'OpInfo-tested overridable public outplace ops: {len(tested_overridable_outplace_ops)}')


statuses = transpose_statuses()
for test in tests:
    print(f'{test} coverage {len(statuses[test])}')

method_only_ops = get_method_only_ops_we_care_about()
# for op in method_only_ops:
#     print(f'    {op},')

# top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(100, 25)
# for op in top_ops_not_covered_by_opinfo:
#     print(op)

# print("top ops not covered by opinfo: ")
# top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(200, 40)
# for op in top_ops_not_covered_by_opinfo:
#     print('- ' + op)

# print("top ops not covered by opinfo: ")
# top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(200, 40)
# for op in top_ops_not_covered_by_opinfo:
#     print('- ' + op)

def print_coverage_info(th=100, nn=25):
    print('=' * 80)
    print(f"top {th}, {nn} coverage")
    statuses = transpose_statuses(get_top_ops(th, nn), invert=True)
    top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(th, nn)
    print(f"total ops in set: {th + nn}")
    print(f"tested by OpInfo: {th + nn - len(top_ops_not_covered_by_opinfo)}")
    for test in tests:
        print(f'{test} failing coverage {len(statuses[test])}')

print_coverage_info(100, 25)
print_coverage_info(200, 50)
