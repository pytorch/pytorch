import copy
import enum
import pprint
import unittest
from enum import Enum

# Importing these files make modifications to the op_db that we need
import test_ops  # noqa: F401

import test_vmap  # noqa: F401
from functorch_additional_op_db import additional_op_db

import torch
import torch._functorch.top_operators_github_usage as top_ops
from torch.testing._internal.common_device_type import toleranceOverride
from torch.testing._internal.common_methods_invocations import op_db


all_overridable = list(torch.overrides.get_testing_overrides().keys())

public_docs = [
    (torch.nn.functional, "torch.nn.functional", "docs/source/nn.functional.rst"),
    (torch.fft, "torch.fft", "docs/source/fft.rst"),
    (torch.special, "torch.special", "docs/source/special.rst"),
    (torch.linalg, "torch.linalg", "docs/source/linalg.rst"),
    (torch, "torch", "docs/source/torch.rst"),
    (torch.Tensor, "torch.Tensor", "docs/source/tensors.rst"),
]

# torch.abs, Tensor.abs, Tensor.abs_ are all considered to be different


def get_public_overridable_apis(pytorch_root="/raid/rzou/pt/debug-cpu"):
    results = {}
    all_overridable_apis = set(torch.overrides.get_testing_overrides().keys())
    for module, module_name, src in public_docs:
        with open(f"{pytorch_root}/{src}") as f:
            lines = f.readlines()
        # APIs either begin with 4 spaces or ".. autofunction::"
        api_lines1 = [line.strip() for line in lines if line.startswith(" " * 4)]
        api_lines2 = [
            line.strip()[len(".. autofunction:: ") :]
            for line in lines
            if line.startswith(".. autofunction::")
        ]
        lines = api_lines1 + api_lines2
        lines = [line.removeprefix("Tensor.") for line in lines]
        lines = [line for line in lines if hasattr(module, line)]
        for line in lines:
            api = getattr(module, line)
            if api in all_overridable_apis:
                results[f"{module_name}.{line}"] = api
    return results


denylist = {
    "torch.Tensor.data_ptr",
    "torch.Tensor.dim",
    "torch.Tensor.element_size",
    "torch.Tensor.backward",
    "torch.Tensor.as_strided",
    "torch.Tensor.register_hook",
    "torch.Tensor.record_stream",
    "torch.Tensor.qscheme",
    "torch.Tensor.ndimension",
    "torch.Tensor.smm",
    "torch.Tensor.sspaddmm",
    "torch.Tensor.retain_grad",
    "torch.Tensor.sparse_mask",
    "torch.Tensor.sparse_dim",
    "torch.Tensor.dense_dim",
    "torch.Tensor.values",
    "torch.Tensor.indices",
    "torch.Tensor.numel",
    "torch.Tensor.size",
    "torch.Tensor.nelement",
    "torch.Tensor.q_scale",
    "torch.Tensor.q_zero_point",
    "torch.Tensor.q_per_channel_scales",
    "torch.Tensor.q_per_channel_zero_points",
    "torch.Tensor.q_per_channel_axis",
    "torch.Tensor.int_repr",
    "torch.Tensor.to_sparse",
    "torch.Tensor.is_inference",
    "torch.Tensor.storage",
    "torch.Tensor.storage_type",
}


def get_method_only_ops_we_care_about():
    apis = get_public_overridable_apis()
    result = []
    for key in apis.keys():
        if not key.startswith("torch.Tensor"):
            continue
        if key in denylist:
            continue
        api = key.split(".")[2]
        # filter out in-place
        if api.endswith("_"):
            continue
        if f"torch.{api}" not in apis.keys():
            result.append(api)
    return result


# Deduplicates torch.abs and Tensor.abs


def get_public_overridable_ops():
    results = get_public_overridable_apis()
    cpy = copy.deepcopy(results)
    for key in cpy.keys():
        if not key.startswith("torch.Tensor"):
            continue
        api = key.split(".")[2]
        if f"torch.{api}" in results.keys():
            del results[key]
    return results


def get_public_overridable_outplace_ops():
    results = get_public_overridable_ops()
    cpy = copy.deepcopy(results)
    for key in cpy.keys():
        # NB: there are no dunder methods bcs we don't document those
        if key.endswith("_"):
            del results[key]
    return results


def get_public_overridable_outplace_we_care_about():
    results = get_public_overridable_outplace_ops()
    cpy = copy.deepcopy(results)
    for key in cpy.keys():
        # quantization
        if "quant" in key or ".q_" in key:
            del results[key]

        # is_cpu, etc. It doesn't make sense to have OpInfos for these
        if ".is_" in key:
            del results[key]

        if key in denylist and key in results:
            del results[key]
    return results


# e.g. nn.functional.softmax


def get_op(dotted_name):
    names = dotted_name.split(".")
    mod = torch
    for name in names:
        if not hasattr(mod, name):
            return None
        mod = getattr(mod, name)
    return mod


# Maps function -> [OpInfo]


def get_ops_covered_by_opinfos():
    ops = {}

    def safe_append(dct, key, val):
        if key in dct:
            dct[key].append(val)
        else:
            dct[key] = [val]

    for opinfo in op_db:
        func_op = get_op(opinfo.name)
        if func_op:
            safe_append(ops, func_op, opinfo)
        if opinfo.method_variant:
            safe_append(ops, opinfo.method_variant, opinfo)
        if opinfo.inplace_variant:
            safe_append(ops, opinfo.inplace_variant, opinfo)
        for alias in opinfo.aliases:
            safe_append(ops, alias.op, opinfo)
    return ops


factory_fns = {
    "tensor",
    "zeros",
    "ones",
    "randn",
    "arange",
    "rand",
    "empty",
    "randperm",
    "linspace",
    "logspace",
    "hann_window",
    "full",
    "eye",
    "blackman_window",
    "bartlett_window",
    "randint",
    "range",
}


def get_top_ops(torch_threshold, nn_fn_threshold, with_counts=False):
    denylist = set(
        {
            # These are either not real "operators", factory functions
            # that trivially work, or not-documented ops.
            "load",
            "no_grad",
            "save",
            "from_numpy",
            "manual_seed",
            "set_grad_enabled",
            "set_default_tensor_type",
            "set_num_threads",
            "set_printoptions",
            "numel",
            "set_default_dtype",
            "sparse_coo_tensor",
            "set_rng_state",
            "get_rng_state",
            "get_default_dtype",
            "initial_seed",
            "get_num_threads",
            "quantize_per_tensor",
            "hann_window",
            "is_tensor",
            "as_tensor",
            "equal",
            "enable_grad",
            "seed",
            "is_storage",
            "is_floating_point",
            "nn.functional.torch",
            "set_flush_denormal",
            "set_num_interop_threads",
            "dequantize",
            "get_num_interop_threads",
            "nn.functional.math",
            "nn.functional.threshold_",
            "nn.functional.selu_",
            "nn.functional.elu_",
            "nn.functional.rrelu_",
            "nn.functional.leaky_relu_",
            "nn.functional.hardtanh_",
            "nn.functional.has_torch_function",
            "nn.functional.has_torch_function_unary",
            "nn.functional.has_torch_function_variadic",
            "nn.functional.handle_torch_function",
            "nn.functional.adaptive_max_pool1d_with_indices",
            "nn.functional.adaptive_max_pool2d_with_indices",
            "nn.functional.adaptive_max_pool3d_with_indices",
            "nn.functional.fractional_max_pool2d_with_indices",
            "nn.functional.fractional_max_pool3d_with_indices",
            "is_complex",
            "grad",
            "quantize_per_channel",
            "nn.functional.max_pool2d_with_indices",
            "nn.functional.max_pool3d_with_indices",
            "nn.functional.max_pool1d_with_indices",
            "nn.functional.celu_",
            "nn.functional.grad",
            "nn.functional.relu_",
            "nn.functional.boolean_dispatch",
            "nn.functional.assert_int_or_pair",
            "fft",  # is namespace
        }
    )

    torch_ops = top_ops.top_torch
    nn_fn_ops = top_ops.get_nn_functional_top_list()
    torch_ops = [op for op in torch_ops if op[0] not in denylist]
    nn_fn_ops = [op for op in nn_fn_ops if op[0] not in denylist]

    ops = torch_ops[:torch_threshold] + nn_fn_ops[:nn_fn_threshold]

    # Now, sort by priority
    ops.sort(reverse=True, key=lambda op: op[1])
    if not with_counts:
        ops = [op[0] for op in ops]
    return ops


def get_ops_percentage(torch_threshold, nn_fn_threshold):
    data = top_ops.top_torch + top_ops.get_nn_functional_top_list()

    def get_num_usages(opname):
        # Ignore this, this is heavily inflated
        if opname == "t":
            return 0
        result = [op[1] for op in data if op[0] == opname]
        assert len(result) == 1
        return result[0]

    # get all operators that are not in the denylist
    all_ops = get_top_ops(999999, 999999)
    total_op_usages = sum(get_num_usages(op) for op in all_ops)

    # get subset of all operators
    subset_ops = get_top_ops(torch_threshold, nn_fn_threshold)
    subset_op_usages = sum(get_num_usages(op) for op in subset_ops)
    return subset_op_usages / total_op_usages


def get_top_ops_not_covered_by_opinfo(torch_threshold=0, nn_fn_threshold=0):
    ops = get_top_ops(torch_threshold, nn_fn_threshold)

    ops_with_opinfo = []
    for op in op_db:
        ops_with_opinfo.append(op.name)
        ops_with_opinfo.extend([op.name for op in op.aliases])
    ops_with_opinfo = set(ops_with_opinfo)

    result = [op for op in ops if op not in ops_with_opinfo]
    result = [op for op in result if op not in denylist]
    result = [op for op in result if op not in factory_fns]
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
    "test_vmap_exhaustive",
    "test_op_has_batch_rule",
    "test_vjp",
    "test_vmapvjp",
    "test_vmapvjp_has_batch_rule",
    "test_jvp",
    "test_vmapjvp",
}


def is_decorateinfo_skip_or_xfail(decorateinfo):
    assert len(decorateinfo.decorators) == 1
    actual_decorator = decorateinfo.decorators[0]
    if isinstance(actual_decorator, toleranceOverride):
        return False
    if actual_decorator is unittest.expectedFailure:
        return True
    # Assume the rest are skips
    return True


def get_all_tested_ops():
    overridable_outplace_we_care_about = get_public_overridable_outplace_we_care_about()
    op_to_opinfo = get_ops_covered_by_opinfos()
    result = set({})
    for op in get_covered_ops(overridable_outplace_we_care_about).values():
        opinfos = op_to_opinfo[op]
        result.update(opinfo.name for opinfo in opinfos)
    return result


def get_skipped_or_xfailed_ops_for(test_name):
    overridable_outplace_we_care_about = get_public_overridable_outplace_we_care_about()
    op_to_opinfo = get_ops_covered_by_opinfos()
    result = set({})
    for op in get_covered_ops(overridable_outplace_we_care_about).values():
        opinfos = op_to_opinfo[op]
        for opinfo in opinfos:
            for decorator in opinfo.decorators:
                if not hasattr(decorator, "test_name"):
                    continue
                if decorator.test_name != test_name:
                    continue
                if is_decorateinfo_skip_or_xfail(decorator):
                    result.add(opinfo.name)
    return result


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
    _ = get_covered_ops(overridable_outplace_we_care_about)

    def get_covered_tests(op):
        opinfos = op_to_opinfo[op]
        result = copy.deepcopy(tests)
        for opinfo in opinfos:
            for decorator in opinfo.decorators:
                if not hasattr(decorator, "test_name"):
                    continue
                if decorator.test_name in tests and decorator.test_name in result:
                    result.remove(decorator.test_name)
        return result

    for name, op in get_covered_ops(overridable_outplace_we_care_about).items():
        successful_tests = get_covered_tests(op)
        failed_tests = tests - successful_tests
        result[name] = failed_tests if invert else successful_tests
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
untested_overridable_outplace_ops = get_covered_ops(
    overridable_outplace_we_care_about, invert=True
)

# print("List of OpInfos we need:")
# for key in untested_overridable_outplace_ops.keys():
#     print(key)
# print("-" * 80)
# print("")

print(f"Overridable public APIs: {len(overridable_apis)}")
print(f"Overridable public ops: {len(overridable_ops)}")
print(f"Overridable public outplace ops: {len(overridable_outplace_ops)}")
print(
    f"Overridable public outplace ops we care about: {len(overridable_outplace_we_care_about)}"
)
print(
    f"OpInfo-tested overridable public outplace ops: {len(tested_overridable_outplace_ops)}"
)


def remove_torch(name):
    assert name[:6] == "torch."
    return name[6:]


def get_list_of_all_tests():
    all_tests = list(tested_overridable_outplace_ops.keys())
    return {remove_torch(test) for test in all_tests}


mytest = {
    "test_vmap_exhaustive",
    "test_op_has_batch_rule",
    "test_vjp",
    "test_vmapvjp",
    "test_vmapvjp_has_batch_rule",
}

print("*" * 80)
all_tests = get_list_of_all_tests()
for test in mytest:
    result = get_skipped_or_xfailed_ops_for(test)
    diff = len(all_tests - result)
    print(f"{test}: {diff}")


def get_jvp_coverage(subset=None):
    # - number that support autograd
    # - number that support forward_ad (in pytorch core)
    # - number that support functorch.jvp
    op_to_opinfo = get_ops_covered_by_opinfos()
    ops_dct = tested_overridable_outplace_ops
    if subset is not None:
        ops_dct = {
            name: op for name, op in ops_dct.items() if remove_torch(name) in subset
        }
    supports_autograd_ops_dct = {
        name: op_to_opinfo[fn]
        for name, fn in ops_dct.items()
        if op_to_opinfo[fn][0].supports_autograd
    }
    supports_forwardad_ops_dct = {
        name: op_to_opinfo[fn]
        for name, fn in ops_dct.items()
        if op_to_opinfo[fn][0].supports_forward_ad
    }

    ops = {remove_torch(test) for test in list(ops_dct.keys())}
    supports_autograd = {
        remove_torch(test) for test in list(supports_autograd_ops_dct.keys())
    }
    supports_forward_ad = {
        remove_torch(test) for test in list(supports_forwardad_ops_dct.keys())
    }
    assert supports_forward_ad.issubset(supports_autograd)
    assert supports_autograd.issubset(ops)

    failed_ops = get_skipped_or_xfailed_ops_for("test_jvp")

    coverage = len(supports_forward_ad - failed_ops)
    no_forward_ad = len(supports_autograd) - len(supports_forward_ad)
    print(f"test_jvp, {coverage}, {no_forward_ad}, {len(ops)}")


get_jvp_coverage()
get_jvp_coverage(get_top_ops(100, 25))
for op in get_top_ops(100, 25):
    print(op)
print("*" * 80)

# result = get_skipped_or_xfailed_ops_for('test_vmap_exhaustive')
# result = get_skipped_or_xfailed_ops_for('test_op_has_batch_rule')
# result = get_skipped_or_xfailed_ops_for('test_vjp')
# result = get_skipped_or_xfailed_ops_for('test_vmapvjp')
# result = get_skipped_or_xfailed_ops_for('test_vmapvjp_has_batch_rule')
# import pdb; pdb.set_trace()

statuses = transpose_statuses()
for test in tests:
    print(f"{test} coverage {len(statuses[test])}")

method_only_ops = get_method_only_ops_we_care_about()
# for op in method_only_ops:
#     print(f'    {op},')

top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(100, 25)
print("=" * 80)
for op in top_ops_not_covered_by_opinfo:
    print(f"{op}, {top_ops.usage_count[op]}")

# print("top ops not covered by opinfo: ")
# top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(200, 50)
# for op in top_ops_not_covered_by_opinfo:
#     print(f'{op}, {top_ops.usage_count[op]}')

# print("top ops not covered by opinfo: ")
# top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(220, 92)
# for op in top_ops_not_covered_by_opinfo:
#    print(f'{op}, {top_ops.usage_count[op]}')

# print("top ops not covered by opinfo: ")
# top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(999, 999)
# for op in top_ops_not_covered_by_opinfo:
#     print(f'{op}, {top_ops.usage_count[op]}')


def remove_from_set(parent, to_remove):
    for to_remove_elt in to_remove:
        if to_remove_elt in parent:
            parent.remove(to_remove_elt)


def print_coverage_info(th=100, nn=25):
    print("=" * 80)
    print(f"top {th}, {nn} coverage")
    statuses = transpose_statuses(get_top_ops(th, nn), invert=True)
    top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(th, nn)

    # testing problems
    exemptions = {
        "torch.nn.functional.dropout",  # randomness
    }

    # Allowed exemptions
    vmap_exemptions = {
        "torch.randn_like",  # randomness
        "torch.rand_like",  # randomness
        "torch.allclose",  # number output
        "torch.unique",  # dynamic
        "torch.nonzero",  # dynamic
        "torch.masked_select",  # dynamic
        "torch.prod",  # dynamic (backward)
        "torch.norm",  # norm with nuc is not commonly used; we support the other cases.
        "torch.svd",  # There isn't a bug, it is just nondeterministic so we can't test it.
        "torch.nn.functional.embedding",  # We support everything except the sparse option.
    }
    remove_from_set(statuses["test_vmap_exhaustive"], vmap_exemptions)
    remove_from_set(statuses["test_vmapvjp"], vmap_exemptions)
    remove_from_set(statuses["test_vmapvjp_has_batch_rule"], vmap_exemptions)
    remove_from_set(statuses["test_op_has_batch_rule"], vmap_exemptions)
    remove_from_set(statuses["test_vmapjvp"], vmap_exemptions)
    for test in tests:
        remove_from_set(statuses[test], exemptions)

    print(f"total ops in set: {th + nn}")
    print(f"tested by OpInfo: {th + nn - len(top_ops_not_covered_by_opinfo)}")
    for test in tests:
        if test in {"test_jvp", "test_vmapjvp"}:
            continue
        print(f"{test} failing coverage {len(statuses[test])}")

    # We don't care about these yet
    del statuses["test_jvp"]
    del statuses["test_vmapjvp"]

    pprint.pprint(statuses)


def get_name_to_opinfo_map():
    dct = {}
    for op in op_db + additional_op_db:

        def add(name, op):
            if name not in dct:
                dct[name] = []
            dct[name].append(op)

        add(op.name, op)
        for alias in op.aliases:
            add(alias.name, op)
    return dct


NAME_TO_OPINFO = get_name_to_opinfo_map()


class Support(enum.Enum):
    NO = 0
    YES = 1
    UNKNOWN = 2


FACTORY_FNS = {
    "tensor",
    "zeros",
    "ones",
    "randn",
    "arange",
    "rand",
    "empty",
    "range",
    "full",
    "randperm",
    "eye",
    "randint",
    "linspace",
    "logspace",
}

VJP_EXEMPTIONS = {
    "nn.functional.dropout",  # not actually problem, randomness testing artifact
    "nn.functional.dropout2d",  # not actually problem, randomness testing artifact
    "nn.functional.rrelu",  # not actually problem, randomness testing artifact
    "bernoulli",  # not actually problem, randomness testing artifact
    "normal",  # not actually problem, randomness testing artifact
}

VMAP_EXEMPTIONS = {
    "randn_like",  # randomness
    "rand_like",  # randomness
    "allclose",  # number output
    "unique",  # dynamic
    "nonzero",  # dynamic
    "masked_select",  # dynamic
    "prod",  # dynamic (backward)
    "norm",  # norm with nuc is not commonly used; we support the other cases.
    "svd",  # There isn't a bug, it is just nondeterministic so we can't test it.
    "nn.functional.embedding",  # We support everything except the sparse option.
    "nn.functional.dropout",  # randomness
    "nn.functional.dropout2d",  # randomness
    "bernoulli",  # randomness
    "multinomial",  # randomness
    "normal",  # randomness
}

JVP_EXEMPTIONS = {
    "nn.functional.dropout",  # not actually problem, randomness testing artifact
    "nn.functional.dropout2d",  # not actually problem, randomness testing artifact
    "nn.functional.rrelu",  # not actually problem, randomness testing artifact
    "normal",  # not actually problem, randomness testing artifact
    "bernoulli",  # not actually problem, randomness testing artifact
}


class Operator:
    def __init__(self, name):
        self.name = name
        self.opinfos = NAME_TO_OPINFO.get(name, None)
        assert self.opinfos is None or len(self.opinfos) > 0

    def has_opinfo(self):
        return self.opinfos is not None

    def __repr__(self):
        return f'Operator("{self.name}")'

    def __hash__(self):
        return hash(self.name)

    def no_opinfos_skip_test(self, test_name):
        """Returns NO if any opinfos have a skip or xfail for the test"""
        if not self.has_opinfo():
            return Support.UNKNOWN
        for opinfo in self.opinfos:
            for decorator in opinfo.decorators:
                if not hasattr(decorator, "test_name"):
                    continue
                if decorator.test_name != test_name:
                    continue
                if is_decorateinfo_skip_or_xfail(decorator):
                    return Support.NO
        return Support.YES

    def any_opinfo_attr(self, attr):
        if not self.has_opinfo():
            raise RuntimeError
        return any(getattr(opinfo, attr) for opinfo in self.opinfos)

    def all_opinfo_attr(self, attr):
        if not self.has_opinfo():
            raise RuntimeError
        return all(getattr(opinfo, attr) for opinfo in self.opinfos)

    def supports_vjp(self):
        if self.name in FACTORY_FNS:
            return Support.YES
        if self.name in VJP_EXEMPTIONS:
            return Support.YES
        return self.no_opinfos_skip_test("test_vjp")

    def supports_vmap(self):
        if self.name in FACTORY_FNS:
            return Support.YES
        if self.name in VMAP_EXEMPTIONS:
            return Support.YES
        return self.no_opinfos_skip_test("test_vmap_exhaustive")

    def supports_fast_vmap(self):
        if self.name in FACTORY_FNS:
            return Support.YES
        if self.name in VMAP_EXEMPTIONS:
            return Support.YES
        return self.no_opinfos_skip_test("test_op_has_batch_rule")

    def supports_vmapvjp(self):
        if self.name in FACTORY_FNS:
            return Support.YES
        if self.name in VMAP_EXEMPTIONS:
            return Support.YES
        return self.no_opinfos_skip_test("test_vmapvjp")

    def supports_fast_vmapvjp(self):
        if self.name in FACTORY_FNS:
            return Support.YES
        if self.name in VMAP_EXEMPTIONS:
            return Support.YES
        return self.no_opinfos_skip_test("test_vmapvjp_has_batch_rule")

    def supports_jvp(self):
        if self.name in FACTORY_FNS:
            return Support.YES
        if self.name in JVP_EXEMPTIONS:
            return Support.YES
        if not self.has_opinfo():
            return Support.UNKNOWN
        if self.any_opinfo_attr("supports_autograd") and not self.all_opinfo_attr(
            "supports_forward_ad"
        ):
            return Support.NO
        return self.no_opinfos_skip_test("test_jvp")

    def supports_jvpvjp(self):
        if self.name in FACTORY_FNS:
            return Support.YES
        exemptions = {
            # we have support (see OpInfo), testing artifact
            "nn.functional.dropout2d",
            "nn.functional.dropout",
            # exception: we dont even support double backward for this
            "nn.functional.hardswish",
            "bernoulli",  # this isn't differentiable
            "normal",  # not differentiable
        }
        if self.name in exemptions:
            return Support.YES
        return self.no_opinfos_skip_test("test_jvpvjp")

    def _supports_vmapjvp_base(self, test):
        if self.name in FACTORY_FNS:
            return Support.YES
        VMAPJVP_EXEMPTIONS = {
            "prod",  # dynamic (backward)
            "nn.functional.batch_norm",  # testing problem
            "normal",  # not actually problem, randomness testing artifact
            "bernoulli",  # not actually problem, randomness testing artifact
            "nn.functional.dropout2d",  # not actually problem, randomness testing artifact
            "nn.functional.dropout",  # not actually problem, randomness testing artifact
            # Not a problem.
            # It's just that the max_norm testing mutates inputs...
            # (we have our own functorch variant of the OpInfo without max_norm)
            "nn.functional.embedding",
        }
        if self.name in VMAPJVP_EXEMPTIONS:
            return Support.YES
        if not self.has_opinfo():
            return Support.UNKNOWN
        if self.any_opinfo_attr("supports_autograd") and not self.all_opinfo_attr(
            "supports_forward_ad"
        ):
            return Support.NO
        return self.no_opinfos_skip_test(test)

    def supports_vmapjvp(self):
        return self._supports_vmapjvp_base("test_vmapjvpall")

    def supports_fast_vmapjvp(self):
        return self._supports_vmapjvp_base("test_vmapjvpall_has_batch_rule")


class OperatorSet:
    def __init__(self, operators):
        self.data = set(operators)

    @classmethod
    def from_names(cls, names):
        return OperatorSet([Operator(name) for name in names])

    @classmethod
    def from_top_ops_threshold(cls, torch_threshold, nn_fn_threshold):
        names = get_top_ops(torch_threshold, nn_fn_threshold)
        return cls.from_names(names)

    @classmethod
    def from_top125(cls):
        return cls.from_top_ops_threshold(100, 25)

    @classmethod
    def from_top160(cls):
        return cls.from_top_ops_threshold(107, 53)

    @classmethod
    def all(cls):
        dct = get_public_overridable_outplace_we_care_about()
        names = dct.keys()
        names_sanitized = []
        for n in names:
            torch_tensor = "torch.Tensor."
            torch_dot = "torch."
            if n.startswith(torch_tensor):
                names_sanitized.append(n[len(torch_tensor) :])
            elif n.startswith(torch_dot):
                names_sanitized.append(n[len(torch_dot) :])
            else:
                raise AssertionError
        return cls.from_names(names_sanitized)

    def query(self, operator_method, filter=(Support.NO, Support.YES, Support.UNKNOWN)):
        result = {}
        for key in filter:
            result[key] = set()
        for op in self.data:
            support_status = operator_method(op)
            if support_status in filter:
                result[support_status].add(op)
        return result

    def summary(self):
        checks = [
            "supports_vjp",
            "supports_vmap",
            "supports_fast_vmap",
            "supports_vmapvjp",
            "supports_fast_vmapvjp",
            "supports_jvp",
            "supports_vmapjvp",
            "supports_fast_vmapjvp",
            "supports_jvpvjp",
        ]
        result = ["test, yes, no, unknown"]
        for check in checks:
            accessor = getattr(Operator, check)
            all_results = self.query(accessor)
            yes_amt = len(all_results[Support.YES])
            no_amt = len(all_results[Support.NO])
            unknown_amt = len(all_results[Support.UNKNOWN])
            result.append(f"{check}, {yes_amt}, {no_amt}, {unknown_amt}")
        return "\n".join(result)


opset = OperatorSet.all()
has_no_opinfo = opset.query(Operator.has_opinfo, (False,))

print("=" * 30 + " Summary " + "=" * 30)
print(f"% of usages on github: {get_ops_percentage(99999, 99999)}")
print(opset.summary())

# sanity checks
result = opset.query(Operator.supports_vjp, (Support.NO, Support.UNKNOWN))
# pprint.pprint(result)

print("=" * 30 + " Top 60 Summary " + "=" * 30)
print(f"% of usages on github: {get_ops_percentage(35, 25)}")
opset = OperatorSet.from_top_ops_threshold(35, 25)
# result = opset.query(Operator.supports_vmapjvp, (Support.NO, Support.UNKNOWN))
# pprint.pprint(result)
# result = opset.query(Operator.supports_jvp, (Support.NO, Support.UNKNOWN))
# pprint.pprint(result)
# kresult = opset.query(Operator.supports_jvpvjp, (Support.NO, Support.UNKNOWN))
# kpprint.pprint(result)
# result = opset.query(Operator.supports_vmapjvp, (Support.NO, Support.UNKNOWN))
# pprint.pprint(result)
# result = opset.query(Operator.supports_fast_vmapjvp, (Support.NO, Support.UNKNOWN))
# pprint.pprint(result)
# pprint.pprint(result)
print(opset.summary())

print("=" * 30 + " Top 125 Summary " + "=" * 30)
print(f"% of usages on github: {get_ops_percentage(100, 25)}")
opset = OperatorSet.from_top125()
# result = opset.query(Operator.supports_vmap, (Support.NO, Support.UNKNOWN))
# pprint.pprint(result)
# result = opset.query(Operator.supports_jvpvjp, (Support.NO, Support.UNKNOWN))
# pprint.pprint(result)
print("supports_vjp")
result = opset.query(Operator.supports_vjp, (Support.NO, Support.UNKNOWN))
pprint.pprint(result)
print("supports_jvp")
result = opset.query(Operator.supports_jvp, (Support.NO, Support.UNKNOWN))
pprint.pprint(result)
print("supports_vmapjvp")
result = opset.query(Operator.supports_vmapjvp, (Support.NO, Support.UNKNOWN))
pprint.pprint(result)
print("supports_jvpvjp")
result = opset.query(Operator.supports_jvpvjp, (Support.NO, Support.UNKNOWN))
pprint.pprint(result)
# result = opset.query(Operator.supports_fast_vmapjvp, (Support.NO, Support.UNKNOWN))
# pprint.pprint(result)
# pprint.pprint(result)
print(opset.summary())

# print("=" * 30 + " Top 160 Summary " + "=" * 30)
# opset = OperatorSet.from_top160()
# result = opset.query(Operator.supports_jvpvjp, (Support.NO, Support.UNKNOWN))
# pprint.pprint(result)
# print(opset.summary())

# Print list of everything in order
# all_ops = get_top_ops(999999, 999999, with_counts=True)
# for op, count in all_ops:
#     print(f'{op}, {count}')
