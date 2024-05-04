"""Handle the details of subprocess calls and retries for a given benchmark run."""
import dataclasses
import json
import os
import pickle
import signal
import subprocess
import time
import uuid
from typing import List, Optional, TYPE_CHECKING, Union

from core.api import AutoLabels
from core.types import Label
from core.utils import get_temp_dir
from worker.main import (
    WORKER_PATH,
    WorkerFailure,
    WorkerOutput,
    WorkerTimerArgs,
    WorkerUnpickler,
)

if TYPE_CHECKING:
    PopenType = subprocess.Popen[bytes]
else:
    PopenType = subprocess.Popen


# Mitigate https://github.com/pytorch/pytorch/issues/37377
_ENV = "MKL_THREADING_LAYER=GNU"
_PYTHON = "python"
PYTHON_CMD = f"{_ENV} {_PYTHON}"

# We must specify `bash` so that `source activate ...` always works
SHELL = "/bin/bash"


@dataclasses.dataclass(frozen=True)
class WorkOrder:
    """Spec to schedule work with the benchmark runner."""

    label: Label
    autolabels: AutoLabels
    timer_args: WorkerTimerArgs
    source_cmd: Optional[str] = None
    timeout: Optional[float] = None
    retries: int = 0

    def __hash__(self) -> int:
        return id(self)

    def __str__(self) -> str:
        return json.dumps(
            {
                "label": self.label,
                "autolabels": self.autolabels.as_dict,
                "num_threads": self.timer_args.num_threads,
            }
        )


class _BenchmarkProcess:
    """Wraps subprocess.Popen for a given WorkOrder."""

    _work_order: WorkOrder
    _cpu_list: Optional[str]
    _proc: PopenType

    # Internal bookkeeping
    _communication_file: str
    _start_time: float
    _end_time: Optional[float] = None
    _retcode: Optional[int]
    _result: Optional[Union[WorkerOutput, WorkerFailure]] = None

    def __init__(self, work_order: WorkOrder, cpu_list: Optional[str]) -> None:
        self._work_order = work_order
        self._cpu_list = cpu_list
        self._start_time = time.time()
        self._communication_file = os.path.join(get_temp_dir(), f"{uuid.uuid4()}.pkl")
        with open(self._communication_file, "wb") as f:
            pickle.dump(self._work_order.timer_args, f)

        self._proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            executable=SHELL,
        )

    def clone(self) -> "_BenchmarkProcess":
        return _BenchmarkProcess(self._work_order, self._cpu_list)

    @property
    def cmd(self) -> str:
        cmd: List[str] = []
        if self._work_order.source_cmd is not None:
            cmd.extend([self._work_order.source_cmd, "&&"])

        cmd.append(_ENV)

        if self._cpu_list is not None:
            cmd.extend(
                [
                    f"GOMP_CPU_AFFINITY={self._cpu_list}",
                    "taskset",
                    "--cpu-list",
                    self._cpu_list,
                ]
            )

        cmd.extend(
            [
                _PYTHON,
                WORKER_PATH,
                "--communication-file",
                self._communication_file,
            ]
        )
        return " ".join(cmd)

    @property
    def duration(self) -> float:
        return (self._end_time or time.time()) - self._start_time

    @property
    def result(self) -> Union[WorkerOutput, WorkerFailure]:
        self._maybe_collect()
        assert self._result is not None
        return self._result

    def poll(self) -> Optional[int]:
        self._maybe_collect()
        return self._retcode

    def interrupt(self) -> None:
        """Soft interrupt. Allows subprocess to cleanup."""
        self._proc.send_signal(signal.SIGINT)

    def terminate(self) -> None:
        """Hard interrupt. Immediately SIGTERM subprocess."""
        self._proc.terminate()

    def _maybe_collect(self) -> None:
        if self._result is not None:
            # We've already collected the results.
            return

        self._retcode = self._proc.poll()
        if self._retcode is None:
            # `_proc` is still running
            return

        with open(self._communication_file, "rb") as f:
            result = WorkerUnpickler(f).load_output()

        if isinstance(result, WorkerOutput) and self._retcode:
            # Worker managed to complete the designated task, but worker
            # process did not finish cleanly.
            result = WorkerFailure("Worker failed silently.")

        if isinstance(result, WorkerTimerArgs):
            # Worker failed, but did not write a result so we're left with the
            # original TimerArgs. Grabbing all of stdout and stderr isn't
            # ideal, but we don't have a better way to determine what to keep.
            proc_stdout = self._proc.stdout
            assert proc_stdout is not None
            result = WorkerFailure(failure_trace=proc_stdout.read().decode("utf-8"))

        self._result = result
        self._end_time = time.time()

        # Release communication file.
        os.remove(self._communication_file)


class InProgress:
    """Used by the benchmark runner to track outstanding jobs.
    This class handles bookkeeping and timeout + retry logic.
    """

    _proc: _BenchmarkProcess
    _timeouts: int = 0

    def __init__(self, work_order: WorkOrder, cpu_list: Optional[str]):
        self._work_order = work_order
        self._proc = _BenchmarkProcess(work_order, cpu_list)

    @property
    def work_order(self) -> WorkOrder:
        return self._proc._work_order

    @property
    def cpu_list(self) -> Optional[str]:
        return self._proc._cpu_list

    @property
    def proc(self) -> _BenchmarkProcess:
        # NB: For cleanup only.
        return self._proc

    @property
    def duration(self) -> float:
        return self._proc.duration

    def check_finished(self) -> bool:
        if self._proc.poll() is not None:
            return True

        timeout = self.work_order.timeout
        if timeout is None or self._proc.duration < timeout:
            return False

        self._timeouts += 1
        max_attempts = (self._work_order.retries or 0) + 1
        if self._timeouts < max_attempts:
            print(
                f"\nTimeout: {self._work_order.label}, {self._work_order.autolabels} "
                f"(Attempt {self._timeouts} / {max_attempts})"
            )
            self._proc.interrupt()
            self._proc = self._proc.clone()
            return False

        raise subprocess.TimeoutExpired(cmd=self._proc.cmd, timeout=timeout)

    @property
    def result(self) -> Union[WorkerOutput, WorkerFailure]:
        return self._proc.result

    def __hash__(self) -> int:
        return id(self)
