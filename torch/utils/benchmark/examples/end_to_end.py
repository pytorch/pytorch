# -*- coding: utf-8 -*-
"""End-to-end example to test a PR for regressions:

$ python -m examples.end_to_end --pr 39850
$ python -m examples.end_to_end --pr 39967
$ python -m examples.end_to_end --pr 39744

NOTE:
  This example assumes that you have and environment prefixed with
  `ref_`, and another prefixed with `pr_` for the PR
  in question. (e.g. `ref_39850` and `pr_39850`).

  A helper script (examples/prepare_e2e.sh) is provided to build
  the required environments with the correct configuration.
"""

import argparse
import itertools as it
import multiprocessing
import multiprocessing.dummy
import os
import pickle
import queue
import subprocess
import tempfile
import textwrap

import numpy as np
import torch
from torch.utils.benchmark.op_fuzzers import unary
from torch.utils.benchmark import Timer, Measurement
from typing import Dict, Tuple, List


_MAIN, _SUBPROCESS = "main", "subprocess"

_PR_ENV_TEMPLATE = "pr_{pr}"
_REF_ENV_TEMPLATE = "ref_{pr}"

_PR_LIST = (
    # Optimize topk performance for tensor with a large dimension size
    "39850",

    # Migrate `var` & `std` to ATen
    "39967",

    # Introducing (Const)StridedRandomAccessor + CompositeRandomAccessor + migrate `sort` to ATen (CPU)
    "39744",
)

_CPU, _GPU = "cpu", "gpu"
_MIN_RUN_SEC = 1
_REPLICATES = {
    _CPU: 5,  # CPU has a higher variance.
    _GPU: 1,
}
_RUNS_PER_LOOP = 3
_NUM_LOOPS = {
    _CPU: 32,
    _GPU: 64,
}

_DEVICES_TO_TEST = {
    "39850": {_CPU: False, _GPU: True},
    "39967": {_CPU: True, _GPU: True},
    "39744": {_CPU: True, _GPU: True},
}

_AVAILABLE_GPUS = queue.Queue[int]()
_DTYPES_TO_TEST = {
    "39850": ("int8", "float32", "float64"),
    "39967": ("float32", "float64"),
    "39744": ("int8", "float32", "float64"),
}
_DTYPE_STR_TO_DTYPE = {
    "float64": torch.float64,
    "float32": torch.float32,
    "int8": torch.int8,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", type=str, default=_PR_LIST[0], choices=_PR_LIST)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--test_variance", action="store_true")

    # (Implementation details)
    parser.add_argument("--DETAIL_context", type=str, choices=(_MAIN, _SUBPROCESS), default=_MAIN)
    parser.add_argument("--DETAIL_device", type=str, choices=(_CPU, _GPU), default=None)
    parser.add_argument("--DETAIL_env", type=str, default=None)
    parser.add_argument("--DETAIL_result_file", type=str, default=None)
    parser.add_argument("--DETAIL_seed", type=int, default=None)

    args = parser.parse_args()
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
    return args


_SUBPROCESS_CMD_TEMPLATE = (
    "source activate {source_env} && python -m examples.end_to_end "
    "--pr {pr} "
    "--DETAIL_context subprocess "
    "--DETAIL_device {device} "
    "--DETAIL_env {env} "
    "--DETAIL_result_file {result_file} "
    "--DETAIL_seed {seed}"
)


def construct_stmt_and_label(pr, params):
    if pr == "39850":
        k0, k1, k2, dim = [params[i] for i in ["k0", "k1", "k2", "dim"]]
        state = np.random.RandomState(params["random_value"])
        topk_dim = state.randint(low=0, high=dim)
        dim_size = [k0, k1, k2][topk_dim]
        k = max(int(np.floor(2 ** state.uniform(low=0, high=np.log2(dim_size)))), 1)

        return f"torch.topk(x, dim={topk_dim}, k={k})", "topk"

    if pr == "39967":
        return "torch.std(x)", "std"

    if pr == "39744":
        state = np.random.RandomState(params["random_value"])
        sort_dim = state.randint(low=0, high=params["dim"])
        return f"torch.sort(x, dim={sort_dim})", "sort"

    raise ValueError("Unknown PR")


def subprocess_main(args):
    seed = args.DETAIL_seed
    cuda = (args.DETAIL_device == _GPU)

    with open(args.DETAIL_result_file, "ab") as f:
        for dtype_str in _DTYPES_TO_TEST[args.pr]:
            dtype = _DTYPE_STR_TO_DTYPE[dtype_str]
            iterator = unary.UnaryOpFuzzer(
                seed=seed, dtype=dtype, cuda=cuda).take(_RUNS_PER_LOOP)
            for i, (tensors, tensor_parameters, params) in enumerate(iterator):
                params["dtype_str"] = dtype_str
                stmt, label = construct_stmt_and_label(args.pr, params)
                timer = Timer(
                    stmt=stmt,
                    globals=tensors,
                    label=label,
                    description=f"[{i}, seed={seed}] ({dtype_str}), stmt = {stmt}",
                    env=args.DETAIL_env,
                )

                measurement = timer.blocked_autorange(min_run_time=_MIN_RUN_SEC)
                measurement.metadata = {
                    "tensor_parameters": tensor_parameters,
                    "params": params,
                }
                print(measurement)
                pickle.dump(measurement, f)


def _main(args):
    pools, map_iters, finished_counts = {}, {}, {}
    pr = args.pr
    envs = (_REF_ENV_TEMPLATE.format(pr=pr), _PR_ENV_TEMPLATE.format(pr=pr))

    # We initialize both pools at the start so that they run simultaneously
    # if applicable
    if _DEVICES_TO_TEST[args.pr][_GPU]:
        finished_counts[_GPU] = 0
        for i in range(args.num_gpus):
            _AVAILABLE_GPUS.put(i)

        pools[_GPU] = multiprocessing.dummy.Pool(args.num_gpus)
        trials = [
            (seed, envs, pr, True, finished_counts, args.test_variance)
            for seed in range(_NUM_LOOPS[_GPU])] * _REPLICATES[_GPU]
        map_iters[_GPU] = pools[_GPU].imap(map_fn, trials)

    if _DEVICES_TO_TEST[args.pr][_CPU]:
        finished_counts[_CPU] = 0
        cpu_workers = int(multiprocessing.cpu_count() / 3)
        pools[_CPU] = multiprocessing.dummy.Pool(cpu_workers)
        trials = [
            (seed, envs, pr, False, finished_counts, args.test_variance)
            for seed in range(_NUM_LOOPS[_CPU])] * _REPLICATES[_CPU]
        map_iters[_CPU] = pools[_CPU].imap(map_fn, trials)

    results = []
    for map_iter in map_iters.values():
        for r in map_iter:
            results.append(r)
            progress = [
                f"{k}: {v} / {_NUM_LOOPS[k] * _REPLICATES[k]}"
                for k, v in finished_counts.items()]
            print(f"\r{(' ' * 10).join(progress)}", end="")
    print()

    for pool in pools.values():
        pool.close()

    process_results(results, args.test_variance)


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# == Data processing and string formatting ====================================
# /////////////////////////////////////////////////////////////////////////////
def merge(measurements):
    if not measurements:
        return None

    states = [m.__getstate__() for m in measurements]
    for k in states[0].keys():
        if k in ("number_per_run", "times", "metadata"):
            continue

        assert all(s[k] == states[0][k] for s in states)

    numbers_per_run = {m.number_per_run for m in measurements}
    n = numbers_per_run.pop() if len(numbers_per_run) == 1 else 1

    merged_state = states[0]
    times = [[t / m.number_per_run * n for t in m.times] for m in measurements]
    merged_state["times"] = list(it.chain(*times))
    merged_state["number_per_run"] = n
    merged_state["metadata"] = states[0]["metadata"]
    return Measurement(**merged_state)


def process_results(results, test_variance):
    paired_results: Dict[Tuple[str, str, int, bool, int], List] = {}
    for (seed, use_gpu), result_batch in results:
        for r in result_batch:
            key = (r.label, r.description, r.num_threads, use_gpu, seed)
            paired_results.setdefault(key, [[], []])
            index = 0 if r.env.startswith("ref") else 1
            paired_results[key][index].append(r)

    paired_results = {
        key: [merge(r_ref_list), merge(r_pr_list)]
        for key, (r_ref_list, r_pr_list) in paired_results.items()
    }

    flagged_for_removal = set()
    for key, (r_ref, r_pr) in paired_results.items():
        if any(r is None or r.has_warnings for r in (r_ref, r_pr)):
            flagged_for_removal.add(key)

    paired_results = {
        k: v for k, v in paired_results.items()
        if k not in flagged_for_removal
    }
    print(f"{len(flagged_for_removal)} samples were culled, {len(paired_results)} remain")

    gpu_results = [(k, v) for k, v in paired_results.items() if k[3]]
    cpu_results = [(k, v) for k, v in paired_results.items() if not k[3]]

    if cpu_results:
        construct_table(cpu_results, "CPU", test_variance)

    if gpu_results:
        construct_table(gpu_results, "GPU", test_variance)


def construct_table(results, device_str, test_variance):
    device_str = f"== {device_str} {' (Variance Test)' if test_variance else ''}  ".ljust(40, "=")
    print(f"{'=' * 40}\n{device_str}\n{'=' * 40}\n")
    results = sorted((
        (key, (r_ref, r_pr), r_pr.median / r_ref.median - 1)
        for key, (r_ref, r_pr) in results
    ), key=lambda i: i[2])

    n = len(results)
    n_regressed = len([i for i in results if i[2] > 0.05])
    n_improved = len([i for i in results if i[2] < -0.05])
    n_unchanged = n - n_improved - n_regressed
    legends = ["Improved  (>5%):", "Regressed (>5%):", "Within 5%:"]
    for legend, count in zip(legends, [n_improved, n_regressed, n_unchanged]):
        print(f"{legend:<17} {count:>6}  ({count / len(results) * 100:>3.0f}%)")

    keys_to_print = (
        {i[0] for i in results[20:30]} |
        {i[0] for i in results[int(n // 2 - 5):int(n // 2 + 5)]} |
        {i[0] for i in results[-30:-20]}
    )
    ellipsis_after = {results[29][0], results[int(n // 2 + 4)][0]}

    column_labels = (
        f"Relative Δ     Absolute Δ      |      numel{'':>8}dtype{'':>14}"
        f"shape{'':>10}steps{'':>10}layout{'':>7}task specific\n{'=' * 126}"
    )

    _, result_log_file = tempfile.mkstemp(suffix=".log")
    with open(result_log_file, "wt") as f:
        f.write(f"{device_str}\n\n{column_labels}\n")
        print(f"\n{column_labels}\n[First twenty omitted (these tend to be noisy) ]")
        for key, (r_ref, r_pr), rel_diff in results:
            row = row_str(rel_diff, r_pr.median - r_ref.median, r_ref)
            f.write(f"{row}\n")
            if key in keys_to_print:
                print(row)
            if key in ellipsis_after:
                print("...")
        print("[Last twenty omitted (these tend to be noisy) ]")

    print(textwrap.dedent("""
        steps:
            Indicates that `x` is sliced from a larger Tensor. For instance, if
            shape is [12, 4] and steps are [2, 1], then a larger Tensor of size
            [24, 4] was created, and then x = base_tensor[::2, ::1]. Omitted if
            all elements are ones.

        layout:
            Indicates that `x` is not contiguous due to permutation. Invoking
            `x.permute(layout)` (e.g. x.permute((2, 0, 1)) if layout = [2, 0, 1])
            would produce a Tensor with physical memory layout matching logical
            memory layout. (Though still not contiguous if `steps` contains
            non-one elements.)
        """))

    print(f"\nComplete results in: {result_log_file}")


def row_str(rel_diff, diff_seconds, measurement):
    params = measurement.metadata["params"]
    tensor_parameters = measurement.metadata["tensor_parameters"]

    dim = params["dim"]
    x_numel = tensor_parameters["x"]["numel"]
    steps = [params[f"x_step_{i}"] for i in range(dim)]
    order = tensor_parameters['x']["order"]
    order = str("" if all(i == j for i, j in zip(order, range(dim))) else order)

    task_specific = ""
    if measurement.stmt.startswith("torch.topk"):
        dim_str, k_str = measurement.stmt[:-1].replace("torch.topk(x, ", "").split(", ")
        task_specific = f"{dim_str}, {k_str:<8}"
    elif measurement.stmt.startswith("torch.std"):
        pass
    elif measurement.stmt.startswith("torch.sort"):
        task_specific = measurement.stmt[:-1].replace("torch.sort(x, ", "")

    return (
        f"{rel_diff * 100:>5.0f}%     {abs(diff_seconds) * 1e6:>11.1f} us{'':>6}|"
        f"{x_numel:>12}   {params['dtype_str']:>10}   "
        f"{str([params[f'k{i}'] for i in range(dim)]):>17}  "
        f"{str(steps) if not all(i == 1 for i in steps) else '':>12}  {order:>12}"
        f"{'':>8}{task_specific}"
    )


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# == Subprocess and environment management ====================================
# /////////////////////////////////////////////////////////////////////////////
def read_results(result_file: str):
    output = []
    with open(result_file, "rb") as f:
        while True:
            try:
                output.append(pickle.load(f))
            except EOFError:
                break
    return output


def run(cmd, cuda_visible_devices=""):
    return subprocess.run(
        cmd,
        env={
            "CUDA_VISIBLE_DEVICES": str(cuda_visible_devices),
            "PATH": os.getenv("PATH", ""),
        },
        stdout=subprocess.PIPE,
        shell=True
    )


def test_source(envs):
    """Ensure that subprocess"""
    for env in envs:
        result = run(f"source activate {env}")
        if result.returncode != 0:
            raise ValueError(f"Failed to source environment `{env}`")


def map_fn(args):
    seed, envs, pr, use_gpu, finished_counts, test_variance = args
    gpu = _AVAILABLE_GPUS.get() if use_gpu else None
    try:
        _, result_file = tempfile.mkstemp(suffix=".pkl")
        for env in envs:
            cmd = _SUBPROCESS_CMD_TEMPLATE.format(
                source_env=envs[0] if test_variance else env,
                env=env, pr=pr, device=_GPU if use_gpu else _CPU,
                result_file=result_file, seed=seed,
            )
            run(cmd=cmd, cuda_visible_devices=gpu if use_gpu else "")
        finished_counts[_GPU if use_gpu else _CPU] += 1
        return (seed, use_gpu), read_results(result_file)
    except KeyboardInterrupt:
        pass  # Handle ctrl-c gracefully.
    finally:
        if gpu is not None:
            _AVAILABLE_GPUS.put(gpu)
        if os.path.exists(result_file):
            os.remove(result_file)


def main(args):
    test_source([
        _REF_ENV_TEMPLATE.format(pr=args.pr),
        _PR_ENV_TEMPLATE.format(pr=args.pr),
    ])
    _main(args)


if __name__ == "__main__":
    args = parse_args()

    if args.DETAIL_context == "main":
        main(args)

    if args.DETAIL_context == "subprocess":
        try:
            subprocess_main(args)
        except KeyboardInterrupt:
            pass  # Handle ctrl-c gracefully.
