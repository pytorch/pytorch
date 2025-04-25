import dataclasses
import datetime
import tempfile
from collections import defaultdict
from types import ModuleType
from typing import Any, Optional, Protocol

import torch
from torch.autograd import DeviceType
from torch.utils._ordered_set import OrderedSet

from .runtime.benchmarking import benchmarker
from .runtime.runtime_utils import create_bandwidth_info_str, get_num_bytes


class BenchmarkCallableType(Protocol):
    def __call__(self, times: int, repeat: int) -> float: ...


_kernel_category_choices = [
    "foreach",
    "persistent_reduction",
    "pointwise",
    "reduction",
    "split_scan",
    "template",
]


def get_kernel_category_by_source_code(src_code: str) -> str:
    """
    Similar to get_kernel_category but use the source code. Call this API
    if we have not compile the src_code to module yet.
    """
    choices = [
        ch for ch in _kernel_category_choices if f"@triton_heuristics.{ch}" in src_code
    ]
    if len(choices) == 1:
        return choices[0]
    else:
        return "unknown"


def get_kernel_category(kernel_mod: ModuleType) -> str:
    """
    Given the module defining a triton kernel, return the category of the kernel.
    Category can be one of:
    - pointwise
    - reduction
    - persistent_reduction

    Currently we simply decide the category depending on what decorator is imported
    by the kernel.
    """
    choices = [ch for ch in _kernel_category_choices if ch in kernel_mod.__dict__]
    if len(choices) == 1:
        return choices[0]
    else:
        return "unknown"


def get_triton_kernel(mod: ModuleType):  # type: ignore[no-untyped-def]
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    cand_list = [
        v
        for k, v in mod.__dict__.items()
        if k.startswith("triton_") and isinstance(v, CachingAutotuner)
    ]
    assert len(cand_list) == 1
    return cand_list[0]


def benchmark_all_kernels(
    benchmark_name: str, benchmark_all_configs: Optional[dict[Any, Any]]
) -> None:
    """
    An experimental API used only when config.benchmark_kernel is true.

    Run the kernel benchmarks for all the kernels cached in PyCodeCache.
    Used in the compiled modules.

    Put this method here rather than codegen it for convenience since its implementation
    does not change based on different graph modules being compiled.
    """
    from torch._inductor.codecache import PyCodeCache

    nfound = 0
    for kernel_mod in PyCodeCache.modules:
        kernel_key = kernel_mod.key
        if not hasattr(kernel_mod, "get_args") or not hasattr(kernel_mod, "call"):
            continue

        triton_kernel = get_triton_kernel(kernel_mod)
        kernel_category = get_kernel_category(kernel_mod)
        args = kernel_mod.get_args()
        num_in_out_ptrs = len(
            [
                arg_name
                for arg_name in triton_kernel.fn.arg_names
                if arg_name.startswith("in_out_ptr")
            ]
        )
        num_gb = triton_kernel.inductor_meta.get("kernel_num_gb", None)
        if num_gb is None:
            num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9

        def get_info_str(
            ms: float,
            n_regs: Optional[Any],
            n_spills: Optional[Any],
            shared: Optional[Any],
            prefix: str = "",
        ) -> str:
            if not any(x is None for x in [n_regs, n_spills, shared]):
                kernel_detail_str = (
                    f"  {n_regs:3} regs  {n_spills:3} spills  {shared:8} shared mem"
                )
            else:
                kernel_detail_str = ""

            gb_per_s = num_gb / (ms / 1e3)
            return create_bandwidth_info_str(
                ms, num_gb, gb_per_s, prefix=prefix, suffix=kernel_detail_str
            )

        kernel_desc = (
            f"{benchmark_name:20} {kernel_category[:3].upper()} {kernel_key[:10]}"
        )
        if benchmark_all_configs:
            assert hasattr(kernel_mod, "benchmark_all_configs")
            bench_result = kernel_mod.benchmark_all_configs(args)
            print(kernel_desc)
            for launcher, ms in bench_result.items():
                print(
                    f"  {get_info_str(ms, launcher.n_regs, launcher.n_spills, launcher.shared)} @ {launcher.config}"
                )
        else:
            ms = benchmarker.benchmark_gpu(lambda: kernel_mod.call(args), rep=40)
            assert len(triton_kernel.launchers) == 1, (
                "Autotuner should have selected the best config"
            )
            launcher = triton_kernel.launchers[0]
            print(
                get_info_str(
                    ms,
                    launcher.n_regs,
                    launcher.n_spills,
                    launcher.shared,
                    prefix=f"{kernel_desc} ",
                )
            )

        nfound += 1
    if nfound == 0:
        print(
            "No kernel with benchmark functionality found. Make sure you run inductor with config.benchmark_kernel being True"
        )


@dataclasses.dataclass
class ProfileEvent:
    category: str
    key: str
    self_device_time_ms: float
    # the benchmark is run multiple times and we average the count across all the
    # runs. It should be an integer but define a float just in case.
    count: float


def parse_profile_event_list(
    benchmark_name: str,
    event_list: torch.autograd.profiler_util.EventList,
    wall_time_ms: float,
    nruns: int,
    device_name: str,
) -> None:
    def get_self_device_time(
        ev: torch.autograd.profiler_util.EventList,
    ) -> float:
        """
        ev.self_device_time_total is in microsecond. Convert to millisecond.
        """
        return ev.self_device_time_total / 1000 / nruns  # type: ignore[attr-defined]

    all_events: dict[str, list[ProfileEvent]] = defaultdict(list)

    def add_event(
        ev: torch.autograd.profiler_util.EventList,
        category: str,
    ) -> None:
        profile_ev = ProfileEvent(
            category=category,
            key=ev.key,  # type: ignore[attr-defined]
            self_device_time_ms=get_self_device_time(ev),
            count=ev.count / nruns,  # type: ignore[operator] # average across all runs
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

    def report_category(category: str, profile_events: list[ProfileEvent]) -> float:
        if not device_name:
            return 0.0

        from tabulate import tabulate

        profile_events.sort(key=lambda ev: ev.self_device_time_ms, reverse=True)

        rows = []
        total_time = 0.0
        print(f"\n  == {category} category kernels == ")
        for ev in profile_events:
            total_time += ev.self_device_time_ms
            percent = f"{ev.self_device_time_ms / wall_time_ms * 100:.2f}%"
            rows.append([ev.key[:120], ev.self_device_time_ms, ev.count, percent])
        rows.append(
            ["Total", total_time, "", f"{total_time / wall_time_ms * 100:.2f}%"]
        )
        print(
            tabulate(
                rows,
                headers=[
                    "Kernel",
                    f"Self {device_name.upper()} TIME (ms)",
                    "Count",
                    "Percent",
                ],
            )
        )
        return total_time

    def report() -> None:
        category_list = [
            "triton_pointwise",
            "triton_reduction",
            "triton_persistent_reduction",
            "triton_unknown",
            "unknown",
        ]
        assert OrderedSet(all_events.keys()).issubset(OrderedSet(category_list)), (
            f"{list(all_events.keys())}"
        )

        per_category_wall_time = {}
        total_device_ms = 0.0
        for category in category_list:
            if category in all_events:
                _time = report_category(category, all_events[category])
                per_category_wall_time[category] = _time
                total_device_ms += _time

        device_busy_percent = f"{total_device_ms / wall_time_ms * 100:.2f}%"
        if device_name:
            print(
                f"\nPercent of time when {device_name.upper()} is busy: {device_busy_percent}"
            )
        else:
            print("No device detected")

        print(f"Total wall time {wall_time_ms:.3f} ms")

        # output such a line so we can gather such line from all compiled modules from all
        # benchmarks and tabulate it!
        # Columns: benchmark_name, pointwise_percent, reduction_percent, persistent_reduction_percent,
        #   unknown_category_percent, device_busy_percent, wall_time_ms
        tabulate_line = f"Output for tabulate: {benchmark_name}"
        for category in category_list:
            percent = (
                f"{per_category_wall_time.get(category, 0.0) / wall_time_ms * 100:.2f}%"
            )
            tabulate_line += f", {percent}"
        tabulate_line += f", {device_busy_percent}, {wall_time_ms:.3f}ms"

        print(tabulate_line)

    report()


def perf_profile(
    wall_time_ms: float,
    times: int,
    repeat: int,
    benchmark_name: str,
    benchmark_compiled_module_fn: BenchmarkCallableType,
) -> None:
    with torch.profiler.profile(record_shapes=True) as p:
        benchmark_compiled_module_fn(times=times, repeat=repeat)

    path = f"{tempfile.gettempdir()}/compiled_module_profile.json"
    p.export_chrome_trace(path)
    print(f"Profiling result for a compiled module of benchmark {benchmark_name}:")
    print(f"Chrome trace for the profile is written to {path}")
    event_list = p.key_averages(group_by_input_shape=True)
    print(event_list.table(sort_by="self_device_time_total", row_limit=10))
    parse_profile_event_list(
        benchmark_name, event_list, wall_time_ms, times * repeat, p.use_device
    )


def ncu_analyzer(
    benchmark_name: str, benchmark_compiled_module_fn: BenchmarkCallableType
) -> None:
    import inspect
    import os
    import subprocess

    module_file = inspect.getfile(benchmark_compiled_module_fn)
    module_dir = os.path.dirname(module_file)
    module_name = os.path.splitext(os.path.basename(module_file))[0]

    ncu_dir = tempfile.gettempdir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ncu_output = os.path.join(ncu_dir, f"ncu_output_{timestamp}.ncu-rep")
    python_cmd = (
        f"""import sys; sys.path.insert(0, '{module_dir}'); """
        f"""from {module_name} import benchmark_compiled_module; """
        """benchmark_compiled_module(times=1, repeat=1)"""
    )

    ncu_cmd = [
        "ncu",
        "--target-processes",
        "all",
        "--replay-mode",
        "kernel",
        "--kernel-name-base",
        "function",
        "--print-units",
        "base",
        "--set",
        "full",
        "--import-source",
        "yes",
        "--force-overwrite",
        "--export",
        ncu_output,
        "python",
        "-c",
        python_cmd,
    ]

    try:
        subprocess.run(ncu_cmd, check=True)
        print(f"\nNCU profiling results for benchmark {benchmark_name}:")
        print(f"NCU report has been written to {ncu_output}")

    except subprocess.CalledProcessError as e:
        print(f"NCU profiling failed with error: {e}")
        return


def collect_memory_snapshot(
    benchmark_compiled_module_fn: BenchmarkCallableType,
) -> None:
    assert torch.cuda.is_available()

    torch.cuda.memory._record_memory_history(max_entries=100000)
    benchmark_compiled_module_fn(times=10, repeat=1)  # run 10 times
    snapshot_path = f"{tempfile.gettempdir()}/memory_snapshot.pickle"
    torch.cuda.memory._dump_snapshot(snapshot_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"The collect memory snapshot has been written to {snapshot_path}")


def compiled_module_main(
    benchmark_name: str, benchmark_compiled_module_fn: BenchmarkCallableType
) -> None:
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
    parser.add_argument(
        "--cuda-memory-snapshot",
        action="store_true",
        help="""
            Whether to collect CUDA memory snapshot. Refer to
            "https://pytorch.org/blog/understanding-gpu-memory-1/
            for details about how to visualize the collected snapshot
        """,
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Whether to run ncu analysis",
    )
    args = parser.parse_args()

    if args.benchmark_kernels:
        benchmark_all_kernels(benchmark_name, args.benchmark_all_configs)
    else:
        times = 10
        repeat = 10

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        wall_time_ms = benchmark_compiled_module_fn(times=times, repeat=repeat) * 1000

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated()
            print(f"Peak GPU memory usage {peak_mem / 1e6:.3f} MB")

        if torch.cuda.is_available() and args.cuda_memory_snapshot:
            collect_memory_snapshot(benchmark_compiled_module_fn)

        if args.profile:
            perf_profile(
                wall_time_ms,
                times,
                repeat,
                benchmark_name,
                benchmark_compiled_module_fn,
            )
        if args.ncu:
            ncu_analyzer(benchmark_name, benchmark_compiled_module_fn)
