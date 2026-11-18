"""Compare membership guard strategies on the same compiled function.

Run from the repository root:
    python benchmarks/dynamo/microbenchmarks/relational_dict_guard.py

Times cover the complete guard tree, without tensor execution or compilation.
The mutation/replacement cases also include the identical input update in each
strategy. The Python strategy reproduces the previous root lambda; specialized
disables the relational fast path to use the original per-key guards.
"""

import argparse
import contextlib
import functools
import json
import platform
import statistics
import timeit
from unittest import mock

import torch
import torch._dynamo
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
from torch._dynamo.guards import get_verbose_code_parts, GuardBuilder
from torch._dynamo.variables.dicts import ConstDictVariable


@functools.wraps(GuardBuilder.RELATIONAL_DICT_CONTAINS)
def RELATIONAL_DICT_CONTAINS(builder, guard, *, dict_source, expected):
    key_ref = builder.arg_ref(guard)
    dict_ref = builder.source_ref(dict_source)
    code = f"___relational_dict_contains({key_ref}, {dict_ref}) is {expected}"
    if code in builder.already_added_code_parts:
        return
    builder._set_guard_export_info(guard, [code])
    builder.add_python_lambda_leaf_guard_to_root(
        [code], get_verbose_code_parts(code, guard)
    )
    builder.already_added_code_parts.add(code)


def specialize_membership(self, tx, item):
    return None


def strategy_context(strategy):
    if strategy == "python":
        return mock.patch.object(
            GuardBuilder, "RELATIONAL_DICT_CONTAINS", RELATIONAL_DICT_CONTAINS
        )
    if strategy == "specialized":
        return mock.patch.object(
            ConstDictVariable, "_maybe_relational_dict_contains", specialize_membership
        )
    return contextlib.nullcontext()


def make_key(kind, index):
    if kind == "str":
        return str(index)
    if kind == "tuple":
        return (index, str(index))
    return index


def measure(run, repeats):
    # Calibrate each strategy to keep the large Python key scans affordable.
    timer = timeit.Timer(run)
    number = 1
    while timer.timeit(number) < 0.02:
        number *= 2
    samples = [elapsed * 1e9 / number for elapsed in timer.repeat(repeats, number)]
    return {
        "median_ns": statistics.median(samples),
        "min_ns": min(samples),
        "max_ns": max(samples),
    }


def bench(strategy, kind, size, present, workload, repeats):
    torch._dynamo.reset()

    def fn(x, d, key):
        return x + 1 if key in d else x + 2

    mapping = {make_key(kind, i): None for i in range(size)}
    key = make_key(kind, size // 2 if present else -1)
    x = torch.ones(1)
    with strategy_context(strategy):
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        actual = compiled(x, mapping, key)
    torch.testing.assert_close(actual, x + (1 if present else 2))
    entries = _debug_get_cache_entry_list(fn.__code__)
    if len(entries) != 1:
        raise AssertionError(f"Expected one cache entry, got {len(entries)}")
    manager = entries[0].guard_manager
    tree = str(manager)
    if strategy == "cpp" and "RELATIONAL_DICT_CONTAINS" not in tree:
        raise AssertionError(f"C++ membership leaf missing:\n{tree}")
    if strategy == "python" and "___relational_dict_contains" not in tree:
        raise AssertionError(f"Python membership lambda missing:\n{tree}")

    inputs = {"x": x, "d": mapping, "key": key}
    check = manager.root.check
    if workload == "mutate":
        mutation_key = next(iter(mapping))

        def run():
            mapping[mutation_key] = not mapping[mutation_key]
            return check(inputs)

    elif workload == "replace":

        def run():
            inputs["d"] = mapping.copy()
            return check(inputs)

    else:

        def run():
            return check(inputs)

    for _ in range(10):
        if not run():
            raise AssertionError(manager.check_verbose(inputs))
    result = measure(run, repeats)
    if not run():
        raise AssertionError(manager.check_verbose(inputs))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 16, 256, 4096])
    parser.add_argument(
        "--kinds", nargs="+", choices=["int", "str", "tuple"], default=["int", "str"]
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["stable", "mutate", "replace"],
        default=["stable"],
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()
    print(
        f"Python {platform.python_version()}, torch {torch.__version__}, {platform.machine()}",
        flush=True,
    )
    print(
        "Nanoseconds per complete guard check; median of repeated warm batches.",
        flush=True,
    )
    print(
        "kind  size  member  workload   specialized       python          cpp   old/cpp",
        flush=True,
    )
    rows = []
    with torch._dynamo.config.patch(specialize_int=False):
        for kind in args.kinds:
            for size in args.sizes:
                if size < 1:
                    raise ValueError("Sizes must be positive")
                for present in (True, False):
                    for workload in args.workloads:
                        results = {
                            strategy: bench(
                                strategy, kind, size, present, workload, args.repeats
                            )
                            for strategy in ("specialized", "python", "cpp")
                        }
                        old, python, cpp = (
                            results[s]["median_ns"]
                            for s in ("specialized", "python", "cpp")
                        )
                        print(
                            f"{kind:5} {size:5} {str(present):6} {workload:9} {old:12.1f} {python:12.1f} {cpp:12.1f} {old / cpp:9.2f}x",
                            flush=True,
                        )
                        rows.append(
                            dict(
                                kind=kind,
                                size=size,
                                present=present,
                                workload=workload,
                                results=results,
                            )
                        )
    if args.json_path:
        with open(args.json_path, "w") as output:
            json.dump(
                dict(
                    torch_version=torch.__version__,
                    python=platform.python_version(),
                    rows=rows,
                ),
                output,
                indent=2,
            )


if __name__ == "__main__":
    main()
