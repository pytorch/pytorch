from __future__ import annotations

import contextlib
import json
import operator
import os
import time
from contextlib import AbstractContextManager
from typing import Any, TYPE_CHECKING
from typing_extensions import TypeVar

import torch
from torch.profiler import profile, ProfilerActivity


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_R = TypeVar("_R")


def synchronize() -> None:
    pass


_CUDA_SYNC_SENTINEL = ["cuda"]


def _is_cuda_sync_sentinel(devices: list[str] | None) -> bool:
    return devices is None or devices == _CUDA_SYNC_SENTINEL


def _is_cpu_only_devices(devices: list[str]) -> bool:
    return len(devices) > 0 and all(spec == "cpu" for spec in devices)


def _is_device_process_label(labels: str) -> bool:
    if "GPU" in labels:
        return True
    acc = torch.accelerator.current_accelerator()
    return acc is not None and acc.type.upper() in labels


def _device_profiler_activity() -> ProfilerActivity:
    if not torch.accelerator.is_available():
        return ProfilerActivity.CUDA
    acc = torch.accelerator.current_accelerator()
    if acc is None or acc.type == "cuda":
        return ProfilerActivity.CUDA
    activity_name = acc.type.upper()
    activity = getattr(ProfilerActivity, activity_name, None)
    if activity is not None:
        return activity
    privateuse1_name = torch._C._get_privateuse1_backend_name()
    if acc.type == privateuse1_name:
        return ProfilerActivity.PrivateUse1
    raise RuntimeError(
        f"Profiler activity is not supported for accelerator {acc.type!r}"
    )


def _synchronize_for_devices(devices: list[str] | None) -> None:
    if devices is not None and len(devices) == 0:
        raise ValueError("devices must not be empty")
    if devices is not None and _is_cpu_only_devices(devices):
        return
    if not torch.accelerator.is_available():
        return
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return
    if _is_cuda_sync_sentinel(devices):
        torch.accelerator.synchronize()
        return
    targets: list[torch.device] = []
    mismatched: list[str] = []
    for spec in devices or []:
        if spec == "cpu":
            continue
        try:
            dev = torch.device(spec)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid device entry in devices: {spec!r}") from exc
        if dev.type != acc.type:
            mismatched.append(spec)
            continue
        targets.append(dev)
    if mismatched:
        raise ValueError(
            f"devices {mismatched!r} do not match current accelerator {acc.type!r}"
        )
    if not targets:
        raise ValueError(
            f"No accelerator entries in devices {devices!r} for current accelerator {acc.type!r}"
        )
    for dev in targets:
        torch.accelerator.synchronize(dev)


def dump_chrome_trace(
    f: Callable[[tuple[Any, ...]], _R],
    input_: tuple[Any, ...],
    trace_filename: str,
    optimize_ctx: AbstractContextManager[Any],
    activities: Sequence[ProfilerActivity],
    num_runs: int = 1,
    devices: list[str] | None = None,
    kwargs_for_f: dict[str, Any] | None = None,
    kwargs_for_profiler: dict[str, Any] | None = None,
) -> float:
    """
    Output the chrome trace of running f(input_, **kwargs_for_f) with [optimize_ctx]
    [num_runs] times to [trace_filename].

    [activities] are the activities that the profiler will record, e.g. ProfilerActivity.CUDA.
    Return total runtime without the profiler

    Outputs to trace_filename

    ``devices`` defaults to ``["cuda"]`` (a historical sentinel meaning "sync
    the current accelerator"). Any list containing only ``"cpu"`` skips
    accelerator sync. ``[]`` is invalid. Explicit device strings must match the
    current accelerator type or a ``ValueError`` is raised.
    """

    if devices is None:
        devices = ["cuda"]

    def _sync() -> None:
        _synchronize_for_devices(devices)

    if kwargs_for_f is None:
        kwargs_for_f = {}
    if kwargs_for_profiler is None:
        kwargs_for_profiler = {}

    with optimize_ctx:
        torch.manual_seed(1337)
        for _ in range(5):  # warmup runs
            f(input_, **kwargs_for_f)
            _sync()
        torch.manual_seed(1337)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            f(input_, **kwargs_for_f)
            _sync()
        t1 = time.perf_counter()
    timing = t1 - t0

    with profile(activities=activities, **kwargs_for_profiler) as prof:
        with optimize_ctx:
            _sync()
            torch.manual_seed(1337)
            for _ in range(num_runs):
                f(input_, **kwargs_for_f)
                _sync()
    prof.export_chrome_trace(trace_filename)

    return timing


def get_chrome_trace_events(filename: str) -> list[dict[str, Any]]:
    with open(filename) as f:
        data = json.load(f)
    events = data["traceEvents"]
    return events


def is_gpu_compute_event(event: dict[str, Any]) -> bool:
    global gpu_pids
    return (
        "pid" in event
        and event["pid"] in gpu_pids
        and "ph" in event
        and event["ph"] == "X"
    )


def get_sorted_gpu_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_gpu_events: list[dict[str, Any]] = []
    for event in events:
        if not is_gpu_compute_event(event):
            continue
        sorted_gpu_events.append(event)
    return sorted(sorted_gpu_events, key=operator.itemgetter("ts"))


def get_duration(sorted_gpu_events: list[dict[str, Any]]) -> int:
    if len(sorted_gpu_events) == 0:
        return 0
    event = sorted_gpu_events[0]
    current_end_time = event["ts"] + event["dur"]
    total_duration = event["dur"]
    for event in sorted_gpu_events[1:]:
        start_time = max(event["ts"], current_end_time)
        end_time = event["ts"] + event["dur"]
        total_duration = total_duration + max(end_time - start_time, 0)
        current_end_time = max(current_end_time, end_time)
    return total_duration


def get_sorted_gpu_mm_conv_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def is_mm_conv_event(event: dict[str, Any]) -> bool:
        return "name" in event and (
            "gemm" in event["name"]
            or "conv" in event["name"]
            or "cutlass" in event["name"]
            or "wgrad" in event["name"]
        )

    gpu_events = get_sorted_gpu_events(events)
    sorted_events: list[dict[str, Any]] = []
    for event in gpu_events:
        if not is_mm_conv_event(event):
            continue
        sorted_events.append(event)
    return sorted_events


gpu_pids: list[Any] = []


def compute_utilization(filename: str, total_length: float) -> tuple[float, float]:
    """
    Process the chrome traces outputs by the pytorch profiler to compute GPU Utilization
    and percent of times spent on matmul and convolution

    Args:
        filename(str): Name of chrome traces file produced by pytorch profiler

        total_length(float): total length of the process without profiler in second

    Return:
        tuple: (GPU Utilization, percent of time spent on matmul and convolution)
    """
    events = get_chrome_trace_events(filename)

    # get pids of GPU events
    global gpu_pids
    gpu_pids = []
    for event in events:
        if "name" not in event:
            continue
        if event["name"] == "process_labels" and _is_device_process_label(
            event["args"]["labels"]
        ):
            gpu_pids.append(event["pid"])

    total_length = total_length * 1e6
    sorted_gpu_events = get_sorted_gpu_events(events)
    utilization = get_duration(sorted_gpu_events) / total_length

    sorted_gpu_mm_conv_events = get_sorted_gpu_mm_conv_events(events)
    mm_conv_utilization = get_duration(sorted_gpu_mm_conv_events) / total_length

    return utilization, mm_conv_utilization


def benchmark_utilization(
    f: Callable[[tuple[Any, ...]], _R],
    input_: tuple[Any, ...],
    trace_folder: str,
    optimize_ctx: AbstractContextManager[Any] | None = None,
    trace_file_name: str = "tmp_chrome_trace",
    num_runs: int = 1,
) -> tuple[float, float]:
    """
    Benchmark the GPU Utilization and percent of time spent on matmul and convolution operations of
    running f(input_, **kwargs_for_f) with [optimize_ctx] [num_runs] times.
    It will produce a chrome trace file in trace_folder/trace_file_name.json

    Example:

    ```
    def f(a):
        return a.sum()


    a = torch.rand(2**20, device="cuda")
    utilization, mm_conv_utilization = benchmark_utilization(
        f, a, "tmp", trace_file_name="tmp_chrome_trace"
    )
    ```

    Args:
        f: function to benchmark

        input_: input to :attr:`f`

        trace_folder: name of the folder to store the chrome trace

        optimize_ctx: the context in which f will run

        trace_file_name: name of the dumped chrome trace file, default to "tmp_chrome_trace"

        num_runs: number of times to run f, excluding the warm-up runs, default to 1.

    Return:
        tuple: (GPU Utilization, percent of time spent on matmul and convolution)

    """
    isExist = os.path.exists(trace_folder)
    if not isExist:
        os.makedirs(trace_folder)
        print("create folder " + trace_folder)

    if optimize_ctx is None:
        optimize_ctx = contextlib.nullcontext()

    chrome_trace_file_name = os.path.join(trace_folder, trace_file_name + ".json")
    total_length = dump_chrome_trace(
        f,
        input_,
        chrome_trace_file_name,
        optimize_ctx,
        [_device_profiler_activity()],
        num_runs=num_runs,
        devices=["cuda"],
    )
    utilization, mm_conv_utilization = compute_utilization(
        chrome_trace_file_name, total_length
    )

    return utilization, mm_conv_utilization
