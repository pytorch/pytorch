"""Basic runner for the instruction count microbenchmarks.

The contents of this file are placeholders, and will be replaced by more
expressive and robust components (e.g. better runner and result display
components) in future iterations. However this allows us to excercise the
underlying benchmark generation infrastructure in the mean time.
"""
import multiprocessing
import multiprocessing.dummy
import os
import pickle
import subprocess
from typing import Tuple

from core.api import AutoLabels, TimerArgs
from core.expand import materialize
from core.types import Label
from core.utils import get_temp_dir
from definitions.standard import BENCHMARKS
from worker.main import WORKER_PATH, WorkerFailure, WorkerOutput, WorkerTimerArgs, WorkerUnpickler


def call_worker(
    args: Tuple[int, Tuple[Label, AutoLabels, TimerArgs]]
) -> Tuple[Label, AutoLabels, int, WorkerOutput]:
    worker_id, (label, autolabels, timer_args) = args

    communication_file = os.path.join(get_temp_dir(), f"communication_file_{worker_id}.pkl")
    with open(communication_file, "wb") as f:
        pickle.dump(timer_args, f)

    subprocess.call(
        ["python", WORKER_PATH, "--communication_file", communication_file],
        shell=False,
    )

    with open(communication_file, "rb") as f:
        result = WorkerUnpickler(f).load_output()

    if isinstance(result, WorkerTimerArgs):
        raise RuntimeError("Benchmark worker failed without starting.")

    elif isinstance(result, WorkerFailure):
        raise RuntimeError(f"Worker failed: {label} {autolabels}\n{result.failure_trace}")

    assert isinstance(result, WorkerOutput)
    return label, autolabels, timer_args.num_threads, result


def main() -> None:
    with multiprocessing.dummy.Pool(multiprocessing.cpu_count() - 4) as pool:
        for label, autolabels, num_threads, result in pool.imap(call_worker, enumerate(materialize(BENCHMARKS)), 1):
            print(label, autolabels, num_threads, result.instructions)


if __name__ == "__main__":
    main()
