"""End-to-end example to test a PR for regressions:

$ python -m examples.end_to_end

NOTE:
  This example assumes that you have the following environments
  built from the following branches:
    master_1667aa6:   master
                      1667aa64515ff480af6aea091a6fc36609c9a388

    pr_38061:         https://github.com/pytorch/pytorch/pull/38061
                      a53fe434b8dd319f665e2321832d1f9833116341


  TODO(robieta): DO_NOT_SUBMIT until this is in a better state.

  As of 05/11/2020, there are merge conflicts between PR 38061 and HEAD.
  To build from the PR, activate the correct conda environment, and run:
    $ git checkout 1667aa64515ff480af6aea091a6fc36609c9a388
    $ wget https://github.com/pytorch/pytorch/pull/38061.diff
    $ git apply 38061.diff

    # Build PyTorch as normal.
"""

import argparse
import multiprocessing
import multiprocessing.dummy
import os
import pickle
import subprocess
import sys
import tempfile

import torch
from utils import FuzzedParameter, FuzzedTensor, Fuzzer, Timer


_MIN_RUN_SEC = 0.5
_RUNS_PER_LOOP = 10
_NUM_LOOPS = 50
_CONCURRENT_RUNS = max(int(multiprocessing.cpu_count() / 2), 1)

_MASTER = "master_1667aa6"
_BRANCH = "pr_38061"
_ENVS = (_MASTER, _BRANCH)
_DTYPE_STR_TO_DTYPE = {
    "float64": torch.float64,
    "float32": torch.float32,
    "int8": torch.int8,
}


# Whether a dimension should be size 1 and broadcast (True) or not (False).
_BROADCAST_DISTRIBUTION = {False: 0.8, True: 0.2}


class MaskedFullFuzzer(Fuzzer):
    def __init__(self, dtype, seed):
        self._dtype = dtype
        super(MaskedFullFuzzer, self).__init__(
            parameters=[
                # Dimensionality of x and mask. (e.g. 1D, 2D, or 3D.)
                FuzzedParameter("dim", distribution={1: 0.3, 2: 0.4, 3: 0.3}),

                # Shapes for x and mask. (Values may be discarded if dim < 3)
                FuzzedParameter("k0", minval=4, maxval=32 * 1024, distribution="loguniform"),
                FuzzedParameter("k1", minval=4, maxval=32 * 1024, distribution="loguniform"),
                FuzzedParameter("k2", minval=4, maxval=32 * 1024, distribution="loguniform"),

                # Should a dim be `1` for the mask.
                FuzzedParameter("broadcast_0", distribution=_BROADCAST_DISTRIBUTION),
                FuzzedParameter("broadcast_1", distribution=_BROADCAST_DISTRIBUTION),
                FuzzedParameter("broadcast_2", distribution=_BROADCAST_DISTRIBUTION),

                FuzzedParameter("mask_fraction", minval=0.0, maxval=1.0, distribution="uniform"),
            ],
            tensors=[
                FuzzedTensor(
                    name="x",
                    size=("k0", "k1", "k2"),
                    tensor_constructor=self._x_tensor_constructor,
                    dim_parameter="dim",
                    probability_contiguous=0.75,
                    max_elements=64 * 1024 ** 2,
                ),
                FuzzedTensor(
                    name="mask",
                    size=("k0", "k1", "k2"),
                    tensor_constructor=self._mask_tensor_constructor,
                    dim_parameter="dim",
                    probability_contiguous=0.75,
                    max_elements=64 * 1024 ** 2,
                ),
            ],
            seed=seed,
        )

    def _x_tensor_constructor(self, size, **kwargs):
        if self._dtype.is_floating_point:
            return torch.rand(size=size, dtype=self._dtype)
        return torch.randint(0, 127, size=size, dtype=self._dtype)

    def _mask_tensor_constructor(self, size, mask_fraction, **kwargs):
        # Force some dims to `1` if they are meant to be broadcast.
        size = tuple(
            1 if kwargs[f"broadcast_{i}"] else size[i] for i in range(len(size))
        )

        return torch.empty(size=size).uniform_(0, 1) <= mask_fraction


def subprocess_main(env: str, result_file: str, seed: int):
    with open(result_file, "wb") as f:
        for dtype_str, dtype in _DTYPE_STR_TO_DTYPE.items():
            iterator = enumerate(MaskedFullFuzzer(dtype, seed).take(_RUNS_PER_LOOP))
            for i, (tensors, tensor_parameters, params) in iterator:
                timer = Timer(
                    stmt="x.masked_fill_(mask, 1.)",
                    globals=tensors,
                    label=f"masked_fill",
                    description=f"[{i}, seed={seed}] ({dtype_str})",
                    env=env,
                )

                measurement = timer.blocked_autorange(min_run_time=_MIN_RUN_SEC)
                measurement.metadata = {
                    "tensor_parameters": tensor_parameters,
                    "params": params,
                }
                pickle.dump(measurement, f)


def _main():
    with multiprocessing.dummy.Pool(_CONCURRENT_RUNS) as pool:
        master_results, branch_results = [], []
        for i, batch_results in enumerate(pool.imap(map_fn, range(_NUM_LOOPS))):
            master_results.extend(batch_results[_MASTER])
            branch_results.extend(batch_results[_BRANCH])
            print(f"\r{i + 1:>2} / {_NUM_LOOPS}", end="")
            sys.stdout.flush()
        print()

    compare_times(master_results, branch_results)


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# == Data processing and string formatting ====================================
# /////////////////////////////////////////////////////////////////////////////
def compare_times(master_results, branch_results):
    results = list(zip(master_results, branch_results))
    relative_improvement = {}
    absolute_times = {_MASTER: {}, _BRANCH: {}}
    for m_master, m_branch in results:
        assert m_master.label == m_branch.label
        assert m_master.description == m_branch.description
        assert m_master.num_threads == m_branch.num_threads
        assert pretty_str(m_master) == pretty_str(m_branch)
        assert m_master.env == _MASTER
        assert m_branch.env == _BRANCH
        if m_master.has_warnings or m_branch.has_warnings:
            print(f"Skipping unreliable measurement: {m_master.description}")
            continue
        diff = m_master.median - m_branch.median
        relative_improvement[m_master.description] = diff / m_master.median
        absolute_times[_MASTER][m_master.description] = m_master.median
        absolute_times[_BRANCH][m_master.description] = m_branch.median

    def sort_by_value(x):
        return sorted(x.items(), key=lambda i: (i[1], i[0]))
    ordered_descriptions = [i[0] for i in sort_by_value(relative_improvement)]

    n_regressed = len([i for i in relative_improvement.values() if i < -0.1])
    n_improved = len([i for i in relative_improvement.values() if i > 0.1])
    n_similar = len(relative_improvement) - n_improved - n_regressed
    print()
    for label, n in [
        ("Improved  (>10%):", n_improved),
        ("Regressed (>10%):", n_regressed),
        ("Within 10%:", n_similar),
    ]:
        print(f"{label:<17} {n:>4}  ({n / len(relative_improvement) * 100:.0f}%)")

    print(f"\nImprovement    Absolute  |{'':>7}dtype{'':>6}numel   ", end="")
    print(f"mask_reuse     mask_true_pct      x_layout    mask_layout\n{'=' * 114}")

    def print_block(result_filter):
        print_result_block(
            master_results + branch_results, result_filter,
            relative_improvement, absolute_times,
        )
    print_block(ordered_descriptions[:15])
    print("\n...\n")
    print_block(ordered_descriptions[-15:])


def print_result_block(
    full_results, result_filter, relative_improvement, absolute_times
):
    results = [i for i in full_results if i.description in result_filter]
    labels = {r.description: pretty_str(r) for r in results}
    labels = sorted(labels.items(), key=lambda i: relative_improvement[i[0]])

    for key, value in labels:
        delta_t = absolute_times[_MASTER][key] - absolute_times[_BRANCH][key]
        print(
            f" {relative_improvement[key] * 100:>5.0f}%   "
            f"{abs(delta_t) * 1e6:>10.1f} us  |   "
            f"{key.split(']')[1][2:-1]:>9}  {value}"
        )


def pretty_str(m):
    tensor_parameters = m.metadata["tensor_parameters"]
    x_numel = tensor_parameters["x"]["numel"]
    mask_numel = tensor_parameters["mask"]["numel"]
    mask_reuse = int(x_numel / mask_numel)

    def order(name):
        return str("contiguous" if tensor_parameters[name]["is_contiguous"]
                   else tensor_parameters[name]["order"])
    return (
        f"{x_numel:>9}  {mask_reuse:>9}{'':>12}"
        f"{m.metadata['params']['mask_fraction'] * 100:>3.0f}%      "
        f"{order('x'):>12}   {order('mask'):>12}"
    )


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# == Subprocess and environment management ====================================
# /////////////////////////////////////////////////////////////////////////////
def read_results(result_file: str):
    output = []
    try:
        with open(result_file, "rb") as f:
            while True:
                try:
                    output.append(pickle.load(f))
                except EOFError:
                    break
    finally:
        os.remove(result_file)
    return output


def invoke_subprocess(env: str, seed: int):
    _, result_file = tempfile.mkstemp(suffix=".pkl")
    subprocess.run(
        f"source activate {env} && python -m examples.end_to_end "
        f"--context subprocess --env {env} --result_file {result_file} "
        f"--seed {seed}",
        stdout=subprocess.PIPE,
        shell=True,
    )

    return read_results(result_file)


def map_fn(seed):
    return {
        _MASTER: invoke_subprocess(_MASTER, seed),
        _BRANCH: invoke_subprocess(_BRANCH, seed),
    }


def main():
    for env in _ENVS:
        result = subprocess.run(
            f"source activate {env}", stdout=subprocess.PIPE, shell=True
        )
        if result.returncode != 0:
            raise ValueError(f"Failed to source environment `{env}`")

    _main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context", type=str, choices=("main", "subprocess"), default="main"
    )
    parser.add_argument("--env", type=str, choices=_ENVS, default=None)
    parser.add_argument("--result_file", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.context == "main":
        main()

    if args.context == "subprocess":
        subprocess_main(args.env, args.result_file, args.seed)
