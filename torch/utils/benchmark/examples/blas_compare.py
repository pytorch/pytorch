import argparse
import datetime
import itertools as it
import multiprocessing
import multiprocessing.dummy
import os
import queue
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Tuple, Dict

from . import blas_compare_setup


MIN_RUN_TIME = 1
NUM_REPLICATES = 20
NUM_THREAD_SETTINGS = (1, 2, 4)
RESULT_FILE = os.path.join(blas_compare_setup.WORKING_ROOT, "blas_results.pkl")
SCRATCH_DIR = os.path.join(blas_compare_setup.WORKING_ROOT, "scratch")


BLAS_CONFIGS = (
    ("MKL (2020.3)", blas_compare_setup.MKL_2020_3, None),
    ("MKL (2020.0)", blas_compare_setup.MKL_2020_0, None),
    ("OpenBLAS", blas_compare_setup.OPEN_BLAS, None)
)


_RESULT_FILE_LOCK = threading.Lock()
_WORKER_POOL: queue.Queue[Tuple[str, str, int]] = queue.Queue()
def clear_worker_pool():
    while not _WORKER_POOL.empty():
        _, result_file, _ = _WORKER_POOL.get_nowait()
        os.remove(result_file)

    if os.path.exists(SCRATCH_DIR):
        shutil.rmtree(SCRATCH_DIR)


def fill_core_pool(n: int):
    clear_worker_pool()
    os.makedirs(SCRATCH_DIR)

    # Reserve two cores so that bookkeeping does not interfere with runs.
    cpu_count = multiprocessing.cpu_count() - 2

    # Adjacent cores sometimes share cache, so we space out single core runs.
    step = max(n, 2)
    for i in range(0, cpu_count, step):
        core_str = f"{i}" if n == 1 else f"{i},{i + n - 1}"
        _, result_file = tempfile.mkstemp(suffix=".pkl", prefix=SCRATCH_DIR)
        _WORKER_POOL.put((core_str, result_file, n))


def _subprocess_main(seed=0, num_threads=1, sub_label="N/A", result_file=None, env=None):
    import torch
    from torch.utils.benchmark import Timer

    conda_prefix = os.getenv("CONDA_PREFIX")
    assert conda_prefix
    if not torch.__file__.startswith(conda_prefix):
        raise ValueError(
            f"PyTorch mismatch: `import torch` resolved to `{torch.__file__}`, "
            f"which is not in the correct conda env: {conda_prefix}"
        )

    torch.manual_seed(seed)
    results = []
    for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 7, 96, 150, 225]:
        dtypes = (("Single", torch.float32), ("Double", torch.float64))
        shapes = (
            # Square MatMul
            ((n, n), (n, n), "(n x n) x (n x n)", "Matrix-Matrix Product"),

            # Matrix-Vector product
            ((n, n), (n, 1), "(n x n) x (n x 1)", "Matrix-Vector Product"),
        )
        for (dtype_name, dtype), (x_shape, y_shape, shape_str, blas_type) in it.product(dtypes, shapes):
            t = Timer(
                stmt="torch.mm(x, y)",
                label=f"torch.mm {shape_str} {blas_type} ({dtype_name})",
                sub_label=sub_label,
                description=f"n = {n}",
                env=os.path.split(env or "")[1] or None,
                globals={
                    "x": torch.rand(x_shape, dtype=dtype),
                    "y": torch.rand(y_shape, dtype=dtype),
                },
                num_threads=num_threads,
            ).blocked_autorange(min_run_time=MIN_RUN_TIME)
            results.append(t)

    if result_file is not None:
        with open(result_file, "wb") as f:
            pickle.dump(results, f)


def run_subprocess(args):
    seed, env, sub_label, extra_env_vars = args
    core_str = None
    try:
        core_str, result_file, num_threads = _WORKER_POOL.get()
        with open(result_file, "wb"):
            pass

        env_vars: Dict[str, str] = {
            "PATH": os.getenv("PATH") or "",
            "PYTHONPATH": os.getenv("PYTHONPATH") or "",

            # NumPy
            "OMP_NUM_THREADS": str(num_threads),
            "MKL_NUM_THREADS": str(num_threads),
            "NUMEXPR_NUM_THREADS": str(num_threads),
        }
        env_vars.update(extra_env_vars or {})

        subprocess.run(
            f"source activate {env} && "
            f"taskset --cpu-list {core_str} "
            f"python {os.path.abspath(__file__)} "
            "--DETAIL_in_subprocess "
            f"--DETAIL_seed {seed} "
            f"--DETAIL_num_threads {num_threads} "
            f"--DETAIL_sub_label '{sub_label}' "
            f"--DETAIL_result_file {result_file} "
            f"--DETAIL_env {env}",
            env=env_vars,
            stdout=subprocess.PIPE,
            shell=True
        )

        with open(result_file, "rb") as f:
            result_bytes = f.read()

        with _RESULT_FILE_LOCK, \
             open(RESULT_FILE, "ab") as f:
            f.write(result_bytes)

    except KeyboardInterrupt:
        pass  # Handle ctrl-c gracefully.

    finally:
        if core_str is not None:
            _WORKER_POOL.put((core_str, result_file, num_threads))


def _compare_main():
    results = []
    with open(RESULT_FILE, "rb") as f:
        while True:
            try:
                results.extend(pickle.load(f))
            except EOFError:
                break

    from torch.utils.benchmark import Compare

    comparison = Compare(results)
    comparison.trim_significant_figures()
    comparison.colorize()
    comparison.print()


def main():
    with open(RESULT_FILE, "wb"):
        pass

    for num_threads in NUM_THREAD_SETTINGS:
        fill_core_pool(num_threads)
        workers = _WORKER_POOL.qsize()

        trials = []
        for seed in range(NUM_REPLICATES):
            for sub_label, env, extra_env_vars in BLAS_CONFIGS:
                env_path = os.path.join(blas_compare_setup.WORKING_ROOT, env)
                trials.append((seed, env_path, sub_label, extra_env_vars))

        n = len(trials)
        with multiprocessing.dummy.Pool(workers) as pool:
            start_time = time.time()
            for i, r in enumerate(pool.imap(run_subprocess, trials)):
                n_trials_done = i + 1
                time_per_result = (time.time() - start_time) / n_trials_done
                eta = int((n - n_trials_done) * time_per_result)
                print(f"\r{i + 1} / {n}    ETA:{datetime.timedelta(seconds=eta)}".ljust(80), end="")
                sys.stdout.flush()
        print(f"\r{n} / {n}  Total time: {datetime.timedelta(seconds=int(time.time() - start_time))}")
    print()

    # Any env will do, it just needs to have torch for benchmark utils.
    env_path = os.path.join(blas_compare_setup.WORKING_ROOT, BLAS_CONFIGS[0][1])
    subprocess.run(
        f"source activate {env_path} && "
        f"python {os.path.abspath(__file__)} "
        "--DETAIL_in_compare",
        shell=True
    )


if __name__ == "__main__":
    # These flags are for subprocess control, not controlling the main loop.
    parser = argparse.ArgumentParser()
    parser.add_argument("--DETAIL_in_subprocess", action="store_true")
    parser.add_argument("--DETAIL_in_compare", action="store_true")
    parser.add_argument("--DETAIL_seed", type=int, default=None)
    parser.add_argument("--DETAIL_num_threads", type=int, default=None)
    parser.add_argument("--DETAIL_sub_label", type=str, default="N/A")
    parser.add_argument("--DETAIL_result_file", type=str, default=None)
    parser.add_argument("--DETAIL_env", type=str, default=None)
    args = parser.parse_args()

    if args.DETAIL_in_subprocess:
        try:
            _subprocess_main(
                args.DETAIL_seed,
                args.DETAIL_num_threads,
                args.DETAIL_sub_label,
                args.DETAIL_result_file,
                args.DETAIL_env,
            )
        except KeyboardInterrupt:
            pass  # Handle ctrl-c gracefully.
    elif args.DETAIL_in_compare:
        _compare_main()
    else:
        main()
