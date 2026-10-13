import argparse
import collections
import json
import statistics
import sys
import sysconfig
import timeit
import types

import torch
from torch._dynamo.eval_frame import _debug_get_cache_entry_list


# The default shape approximates the guard mix of a decoder-only Transformer
# while keeping the compiled graph to one tensor addition. This makes runtime
# differences attributable to Dynamo rather than model compute.

class _Shared:
    def __init__(self):
        self.flag = True


class _GuardHeavyLeaf(torch.nn.Module):
    def __init__(self, width, has_weight):
        super().__init__()
        self.width = width
        weight = torch.nn.Parameter(torch.randn(width)) if has_weight else None
        self.register_parameter("weight", weight)
        self.register_parameter("bias", None)

    def forward(self):
        weight_width = 0 if self.weight is None else self.weight.shape[0]
        bias_width = 0 if self.bias is None else self.bias.shape[0]
        return self.width + weight_width + bias_width


class _GuardHeavyLayer(torch.nn.Module):
    def __init__(self, idx, n_leaves, weighted_leaves, shared):
        super().__init__()
        self.scale = 0.5 + idx * 1e-3
        self.shared = shared
        self.leaves = torch.nn.ModuleList(
            [
                _GuardHeavyLeaf(8 + leaf, leaf < weighted_leaves)
                for leaf in range(n_leaves)
            ]
        )

    def forward(self):
        value = 0
        if self.shared.flag:
            for leaf in self.leaves:
                value += leaf()
        return value + self.scale


class _GuardHeavyModel(torch.nn.Module):
    def __init__(self, n_layers, n_leaves, weighted_leaves):
        super().__init__()
        shared = _Shared()
        self.layers = torch.nn.ModuleList(
            [
                _GuardHeavyLayer(idx, n_leaves, weighted_leaves, shared)
                for idx in range(n_layers)
            ]
        )

    def forward(self, x):
        value = 0
        for layer in self.layers:
            value += layer()
        return x + value


def _median_us(fn, number, repeat):
    for _ in range(10):
        fn()
    samples = timeit.repeat(fn, number=number, repeat=repeat)
    return statistics.median(samples) * 1e6 / number


def _median_delta_us(fn, baseline, number, repeat):
    for _ in range(10):
        fn()
        baseline()
    samples = []
    for i in range(repeat):
        if i % 2:
            baseline_time = timeit.timeit(baseline, number=number)
            fn_time = timeit.timeit(fn, number=number)
        else:
            fn_time = timeit.timeit(fn, number=number)
            baseline_time = timeit.timeit(baseline, number=number)
        samples.append((fn_time - baseline_time) * 1e6 / number)
    return statistics.median(samples)


def _manager_stats(root):
    if root is None:
        return {
            "managers": 0,
            "accessors": 0,
            "leaf_guards": 0,
            "tag_safe": 0,
            "tag_safe_roots": 0,
            "active_tag_safe_roots": 0,
            "leaf_guard_types": {},
        }

    pending = [root]
    seen = set()
    accessors = 0
    leaf_guards = 0
    tag_safe = 0
    tag_safe_roots = 0
    active_tag_safe_roots = 0
    leaf_guard_types = collections.Counter()
    while pending:
        manager = pending.pop()
        key = (type(manager).__name__, manager.get_source())
        if key in seen:
            continue
        seen.add(key)
        accessors += len(manager.get_accessors())
        manager_leaf_guards = manager.get_leaf_guards()
        leaf_guards += len(manager_leaf_guards)
        leaf_guard_types.update(type(guard).__name__ for guard in manager_leaf_guards)
        tag_safe += manager.is_tag_safe()
        tag_safe_roots += manager.is_tag_safe_root()
        active_tag_safe_roots += (
            manager.is_tag_safe_root()
            and not manager.is_recursive_dict_tag_matching_disabled()
        )
        pending.extend(manager.get_child_managers())
        if hasattr(manager, "get_key_value_managers"):
            for key_manager, value_manager in manager.get_key_value_managers().values():
                if key_manager is not None:
                    pending.append(key_manager)
                if value_manager is not None:
                    pending.append(value_manager)
    return {
        "managers": len(seen),
        "accessors": accessors,
        "leaf_guards": leaf_guards,
        "tag_safe": tag_safe,
        "tag_safe_roots": tag_safe_roots,
        "active_tag_safe_roots": active_tag_safe_roots,
        "leaf_guard_types": dict(leaf_guard_types.most_common()),
    }


def _benchmark_variant(args, name, guard_filter_fn=None, *, fullgraph=True):
    # Guard filters observe requests before individual builders can discard
    # them, so request counts and installed guard-tree counts are kept separate.
    captured_guard_requests = []
    filter_kept_guard_request_count = 0

    def record_guards(guards):
        nonlocal filter_kept_guard_request_count
        captured_guard_requests.extend(guards)
        keep = (
            [True] * len(guards)
            if guard_filter_fn is None
            else guard_filter_fn(guards)
        )
        filter_kept_guard_request_count += sum(keep)
        return keep

    torch._dynamo.reset()
    model = _GuardHeavyModel(args.layers, args.leaves, args.weighted_leaves)
    model.eval().requires_grad_(False)
    x = torch.randn(1)
    compiled = torch.compile(
        model.forward,
        backend="eager",
        fullgraph=fullgraph,
        options={"guard_filter_fn": record_guards},
    )
    expected = compiled(x)

    entry = _debug_get_cache_entry_list(type(model).forward)[0]
    guard_manager = entry.guard_manager
    direct = types.FunctionType(
        entry.code,
        type(model).forward.__globals__,
        type(model).forward.__name__,
    )
    torch.testing.assert_close(direct(model, x), expected)
    f_locals = {"self": model, "x": x}
    if not guard_manager.check(f_locals):
        raise RuntimeError("full guard check failed")
    diff_guard_root = guard_manager.diff_guard_root
    if diff_guard_root is not None and not diff_guard_root.check(f_locals):
        raise RuntimeError("differential guard check failed")

    timings = {
        "full_guard_check_us": _median_us(
            lambda: guard_manager.check(f_locals), args.number, args.repeat
        ),
        "diff_guard_check_us": (
            _median_us(
                lambda: diff_guard_root.check(f_locals), args.number, args.repeat
            )
            if diff_guard_root is not None
            else 0.0
        ),
        "direct_bytecode_us": _median_us(
            lambda: direct(model, x), args.number, args.repeat
        ),
        "cache_hit_us": _median_us(
            lambda: compiled(x), args.number, args.repeat
        ),
        "cache_hit_minus_direct_us": _median_delta_us(
            lambda: compiled(x),
            lambda: direct(model, x),
            args.number,
            args.repeat,
        ),
    }
    with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
        timings["cache_hit_skip_guard_eval_us"] = _median_us(
            lambda: compiled(x), args.number, args.repeat
        )

    return {
        "name": name,
        "generated_guard_request_count": len(captured_guard_requests),
        "filter_kept_guard_request_count": filter_kept_guard_request_count,
        "guard_request_types": dict(
            collections.Counter(
                guard.guard_type for guard in captured_guard_requests
            ).most_common()
        ),
        "root": _manager_stats(guard_manager.root),
        "diff_root": _manager_stats(guard_manager.diff_guard_root),
        "timings": timings,
    }


def _print_results(results):
    print(
        "variant                              cache hit  guard check"
        "  diff check  skip guard   direct  wrapper"
    )
    for result in results:
        timings = result["timings"]
        print(
            f"{result['name']:<36}"
            f"{timings['cache_hit_us']:>8.1f}us"
            f"{timings['full_guard_check_us']:>12.1f}us"
            f"{timings['diff_guard_check_us']:>10.1f}us"
            f"{timings['cache_hit_skip_guard_eval_us']:>12.1f}us"
            f"{timings['direct_bytecode_us']:>9.1f}us"
            f"{timings['cache_hit_minus_direct_us']:>9.1f}us"
        )
    print(
        "\nvariant                              requests      kept"
        "  managers  accessors    leaves"
    )
    for result in results:
        root = result["root"]
        print(
            f"{result['name']:<36}"
            f"{result['generated_guard_request_count']:>8}"
            f"{result['filter_kept_guard_request_count']:>10}"
            f"{root['managers']:>10}"
            f"{root['accessors']:>11}"
            f"{root['leaf_guards']:>10}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--leaves", type=int, default=14)
    parser.add_argument("--weighted-leaves", type=int, default=11)
    parser.add_argument("--number", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    if args.layers <= 0 or args.leaves <= 0:
        raise ValueError("layers and leaves must be positive")
    if not 0 <= args.weighted_leaves <= args.leaves:
        raise ValueError("weighted-leaves must be between zero and leaves")
    if args.number <= 0 or args.repeat <= 0:
        raise ValueError("number and repeat must be positive")

    with torch.inference_mode():
        with torch._dynamo.config.patch(use_recursive_dict_tags_for_guards=False):
            baseline = _benchmark_variant(args, "recursive_dict_tags_off")
        recursive_dict_tags = None
        recursive_dict_tags_available = (
            sys.implementation.name == "cpython"
            and sysconfig.get_config_var("Py_GIL_DISABLED") != 1
        )
        if recursive_dict_tags_available:
            with torch._dynamo.config.patch(use_recursive_dict_tags_for_guards=True):
                recursive_dict_tags = _benchmark_variant(
                    args, "recursive_dict_tags_on"
                )
            recursive_dict_tags_active = (
                recursive_dict_tags["root"]["active_tag_safe_roots"] > 0
            )
            if not recursive_dict_tags_active:
                recursive_dict_tags["name"] = "recursive_dict_tags_inactive"
        else:
            recursive_dict_tags_active = False
        with torch._dynamo.config.patch(use_recursive_dict_tags_for_guards=False):
            skip_modules = _benchmark_variant(
                args,
                "skip_nn_module_guards",
                torch.compiler.skip_guard_on_all_nn_modules_unsafe,
            )
            skip_all = _benchmark_variant(
                args, "skip_all_guards", torch.compiler.skip_all_guards_unsafe
            )
            skip_all_no_fullgraph = _benchmark_variant(
                args,
                "skip_all_guards_no_fullgraph",
                torch.compiler.skip_all_guards_unsafe,
                fullgraph=False,
            )
        results = [
            baseline,
            *([recursive_dict_tags] if recursive_dict_tags is not None else []),
            skip_modules,
            skip_all,
            skip_all_no_fullgraph,
        ]

    compile_wrapper_us = skip_all["timings"]["cache_hit_minus_direct_us"]
    compile_wrapper_no_fullgraph_us = skip_all_no_fullgraph["timings"][
        "cache_hit_minus_direct_us"
    ]
    derived = {
        "full_guard_increment_us": baseline["timings"]["cache_hit_us"]
        - baseline["timings"]["cache_hit_skip_guard_eval_us"],
        "recursive_dict_tag_savings_us": (
            baseline["timings"]["cache_hit_us"]
            - recursive_dict_tags["timings"]["cache_hit_us"]
            if recursive_dict_tags_active
            else None
        ),
        "nn_module_guard_savings_us": baseline["timings"]["cache_hit_us"]
        - skip_modules["timings"]["cache_hit_us"],
        "compile_wrapper_us": compile_wrapper_us,
        "compile_wrapper_no_fullgraph_us": compile_wrapper_no_fullgraph_us,
        "fullgraph_wrapper_increment_us": compile_wrapper_us
        - compile_wrapper_no_fullgraph_us,
    }
    output = {
        "config": {
            "layers": args.layers,
            "leaves": args.leaves,
            "weighted_leaves": args.weighted_leaves,
            "number": args.number,
            "repeat": args.repeat,
            "recursive_dict_tags_available": recursive_dict_tags_available,
            "recursive_dict_tags_active": recursive_dict_tags_active,
            "python_version": ".".join(str(value) for value in sys.version_info[:3]),
            "python_implementation": sys.implementation.name,
            "free_threaded": sysconfig.get_config_var("Py_GIL_DISABLED") == 1,
            "recursive_dict_tag_runtime": (
                "disabled_free_threaded"
                if not recursive_dict_tags_available
                else "dict_watcher"
                if sys.version_info >= (3, 12)
                else "legacy_version_check"
            ),
            "skip_nnmodule_hook_guards": (
                torch._dynamo.config.skip_nnmodule_hook_guards
            ),
        },
        "results": results,
        "derived": derived,
    }
    _print_results(results)
    print("RESULT_JSON=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
