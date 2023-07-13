import dataclasses
import tempfile
from collections import defaultdict

import torch
from torch.autograd import DeviceType


@dataclasses.dataclass
class ProfileEvent:
    category: str
    key: str
    self_cuda_time_ms: float
    # the benchmark is run multiple times and we average the count across all the
    # runs. It should be an integer but define a float just in case.
    count: float


def parse_profile_event_list(benchmark_name, event_list, wall_time_ms, nruns):
    def get_self_cuda_time(ev):
        """
        ev.self_cuda_time_total is in microsecond. Convert to millisecond.
        """
        return ev.self_cuda_time_total / 1000 / nruns

    all_events = defaultdict(list)

    def add_event(ev, category):
        profile_ev = ProfileEvent(
            category=category,
            key=ev.key,
            self_cuda_time_ms=get_self_cuda_time(ev),
            count=ev.count / nruns,  # average across all runs
        )
        all_events[category].append(profile_ev)

    for ev in event_list:
        assert not ev.is_legacy, "Don't support the legacy profiler"
        if ev.device_type == DeviceType.CPU:
            # ignore the event on CPU side
            continue

        category = "unknown"
        if ev.key.startswith("triton_"):
            if ev.key.startswith("triton_poi"):
                category = "triton_pointwise"
            elif ev.key.startswith("triton_red"):
                category = "triton_reduction"
            elif ev.key.startswith("triton_per"):
                category = "triton_persistent_reduction"
            else:
                category = "triton_unknown"

        add_event(ev, category)

    def report_category(category, profile_events):
        from tabulate import tabulate

        profile_events.sort(key=lambda ev: ev.self_cuda_time_ms, reverse=True)

        rows = []
        total_time = 0.0
        print(f"\n  == {category} category kernels == ")
        for ev in profile_events:
            total_time += ev.self_cuda_time_ms
            percent = f"{ev.self_cuda_time_ms / wall_time_ms * 100:.2f}%"
            rows.append([ev.key[:120], ev.self_cuda_time_ms, ev.count, percent])
        rows.append(
            ["Total", total_time, "", f"{total_time / wall_time_ms * 100:.2f}%"]
        )
        print(
            tabulate(
                rows, headers=["Kernel", "Self CUDA TIME (ms)", "Count", "Percent"]
            )
        )
        return total_time

    def report():
        category_list = [
            "triton_pointwise",
            "triton_reduction",
            "triton_persistent_reduction",
            "triton_unknown",
            "unknown",
        ]
        assert set(all_events.keys()).issubset(
            set(category_list)
        ), f"{list(all_events.keys())}"

        per_category_wall_time = {}
        total_cuda_ms = 0.0
        for category in category_list:
            if category in all_events:
                _time = report_category(category, all_events[category])
                per_category_wall_time[category] = _time
                total_cuda_ms += _time

        gpu_busy_percent = f"{total_cuda_ms / wall_time_ms * 100:.2f}%"
        print(f"\nPercent of time when GPU is busy: {gpu_busy_percent}")
        print(f"Total wall time {wall_time_ms:.3f} ms")

        # output such a line so we can gather such line from all compiled modules from all
        # benchmarks and tabulate it!
        # Columns: benchmark_name, pointwise_percent, reduction_percent, persistent_reduction_percent,
        #   unknown_category_percent, GPU_busy_percent, wall_time_ms
        tabulate_line = f"Output for tabulate: {benchmark_name}"
        for category in category_list:
            percent = (
                f"{per_category_wall_time.get(category, 0.0) / wall_time_ms * 100:.2f}%"
            )
            tabulate_line += f", {percent}"
        tabulate_line += f", {gpu_busy_percent}, {wall_time_ms:.3f}ms"

        print(tabulate_line)

    report()


def compiled_module_main(benchmark_name, benchmark_compiled_module_fn):
    """
    This is the function called in __main__ block of a compiled module.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-kernels",
        "-k",
        action="store_true",
        help="Whether to benchmark each individual kernels",
    )
    parser.add_argument(
        "--benchmark-all-configs",
        "-c",
        action="store_true",
        help="Whether to benchmark each individual config for a kernel",
    )
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Whether to profile the compiled module",
    )
    args = parser.parse_args()

    if args.benchmark_kernels:
        benchmark_all_kernels(benchmark_name, args.benchmark_all_configs)
    else:
        times = 10
        repeat = 10
        wall_time_ms = (
            benchmark_compiled_module_fn(times=times, repeat=repeat) / times * 1000
        )

        if not args.profile:
            return

        with torch.profiler.profile(record_shapes=True) as p:
            benchmark_compiled_module_fn(times=times, repeat=repeat)

        path = f"{tempfile.gettempdir()}/compiled_module_profile.json"
        p.export_chrome_trace(path)
        print(f"Profiling result for a compiled module of benchmark {benchmark_name}:")
        print(f"Chrome trace for the profile is written to {path}")
        event_list = p.key_averages(group_by_input_shape=True)
        print(event_list.table(sort_by="self_cuda_time_total", row_limit=10))
        parse_profile_event_list(
            benchmark_name, event_list, wall_time_ms, times * repeat
        )
