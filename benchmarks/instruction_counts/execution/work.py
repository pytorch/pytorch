import dataclasses
import os
import pickle
import signal
import subprocess
import time
from typing import List, Optional, Union, TYPE_CHECKING
import uuid

from core.api import AutoLabels
from core.types import Label
from core.utils import get_temp_dir
from worker.main import WORKER_PATH, WorkerFailure, WorkerOutput, WorkerTimerArgs, WorkerUnpickler

if TYPE_CHECKING:
    PopenType = subprocess.Popen[bytes]
else:
    PopenType = subprocess.Popen


_ENV = "MKL_THREADING_LAYER=GNU"
_PYTHON = "python"
PYTHON_CMD = f"{_ENV} {_PYTHON}"


@dataclasses.dataclass(frozen=True)
class WorkOrder:
    """Struct for scheduling work with the benchmark runner."""
    label: Label
    auto_labels: AutoLabels
    timer_args: WorkerTimerArgs
    source_cmd: Optional[str] = None
    timeout: Optional[float] = None
    retries: Optional[int] = None

    def __hash__(self) -> int:
        return id(self)


class _BenchmarkProcess:
    """Wraps subprocess.Popen for a given WorkOrder."""
    _work_order: WorkOrder
    _cpu_list: Optional[str]
    _proc: PopenType

    # Internal bookkeeping
    _communication_file: str
    _start_time: float
    _end_time: Optional[float] = None
    _returncode: Optional[int] = None
    _output: Optional[WorkerOutput] = None
    _worker_failure: Optional[WorkerFailure] = None

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
            executable="/bin/bash",
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
            cmd.extend(["taskset", "--cpu-list", self._cpu_list])

        cmd.extend([
            _PYTHON, WORKER_PATH,
            "--communication_file", self._communication_file,
        ])
        return " ".join(cmd)

    @property
    def ready(self) -> bool:
        return self.poll() is not None

    @property
    def duration(self) -> float:
        return (self._end_time or time.time()) - self._start_time

    @property
    def result(self) -> Union[WorkerOutput, WorkerFailure]:
        # Cannot both succeed and fail.
        assert self._output is None or self._worker_failure is None
        self._collect()
        result = self._output or self._worker_failure
        assert result is not None
        return result

    def poll(self) -> Optional[int]:
        return self._proc.poll()

    def interrupt(self) -> None:
        """Soft interrupt. Allows subprocess to cleanup."""
        self._proc.send_signal(signal.SIGINT)

    def terminate(self) -> None:
        """Hard interrupt. Immediately SIGTERM subprocess."""
        self._proc.terminate()

    def _collect(self) -> None:
        assert self.ready
        if self._output is not None or self._worker_failure is not None:
            return

        self._end_time = time.time()
        with open(self._communication_file, "rb") as f:
            result = WorkerUnpickler(f).load_from_worker()

        if isinstance(result, WorkerOutput):
            if self.poll():
                # Worker managed to complete the designated task, but worker
                # process did not finish cleanly.
                self._worker_failure = WorkerFailure(
                    "Worker failed, but did not return diagnostic information.")
            else:
                self._output = result

        elif isinstance(result, WorkerTimerArgs):
            # Worker failed, but did not write a result so we're left with the
            # original TimerArgs. Grabbing all of stdout and stderr isn't
            # ideal, but we don't have a better way to determine what to keep.
            proc_stdout = self._proc.stdout
            assert proc_stdout is not None
            self._worker_failure = WorkerFailure(
                failure_trace=proc_stdout.read().decode("utf-8"))

        else:
            assert isinstance(result, WorkerFailure)
            self._worker_failure = result

        # Release communication file.
        os.remove(self._communication_file)


class InProgress:
    """Used by the benchmark runner to track outstanding jobs.

    This class handles bookkeeping and timeout + retry logic.
    """
    _work_order: WorkOrder
    _cpu_list: Optional[str]
    _proc: _BenchmarkProcess
    _timeouts: int = 0

    def __init__(self, work_order: WorkOrder, cpu_list: Optional[str]):
        self._work_order = work_order
        self._cpu_list = cpu_list
        self._proc = _BenchmarkProcess(work_order, cpu_list)

    @property
    def work_order(self) -> WorkOrder:
        return self._work_order

    @property
    def cpu_list(self) -> Optional[str]:
        return self._cpu_list

    @property
    def proc(self) -> _BenchmarkProcess:
        # NB: For cleanup only.
        return self._proc

    @property
    def duration(self) -> float:
        return self._proc.duration

    @property
    def ready(self) -> bool:
        if self._proc.ready:
            return True

        timeout = self._work_order.timeout
        if timeout is None or self._proc.duration < timeout:
            return False

        self._timeouts += 1
        attempts = (self._work_order.retries or 0) + 1
        if self._timeouts < attempts:
            print(
                f"\nTimeout: {self._work_order.label}, {self._work_order.auto_labels} "
                f"(Attempt {self._timeouts} / {attempts})")
            self._proc.interrupt()
            self._proc = self._proc.clone()
            return False

        raise subprocess.TimeoutExpired(cmd=self._proc.cmd, timeout=timeout)

    @property
    def result(self) -> Union[WorkerOutput, WorkerFailure]:
        return self._proc.result

    def __hash__(self) -> int:
        return id(self)
