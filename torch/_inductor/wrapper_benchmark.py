import dataclasses
import tempfile
from collections import defaultdict

import torch
from torch.autograd import DeviceType
from .utils import create_bandwidth_info_str, do_bench, get_num_bytes

_kernel_category_choices = [
    "pointwise",
    "reduction",
    "persistent_reduction",
    "template",
    "foreach",
]


def get_kernel_category_by_source_code(src_code):
    """
    Similar to get_kernel_category but use the source code. Call this API
    if we have not compile the src_code to module yet.
    """
    choices = [ch for ch in _kernel_category_choices if f"@{ch}" in src_code]
    if len(choices) == 1:
        return choices[0]
    else:
        return "unknown"


def get_kernel_category(kernel_mod):
    """
    Given the module defining a triton kernel, return the category of the kernel.
    Cateogry can be one of:
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


def benchmark_all_kernels(benchmark_name, benchmark_all_configs):
    """
    An experimental API used only when config.benchmark_kernel is true.

    Run the kernel benchmarks for all the kernels cached in PyCodeCache.
    Used in the compiled modules.

    Put this method here rather than codegen it for convenience since its implementation
    does not change based on different graph modules being compiled.
    """
    from torch._inductor.codecache import PyCodeCache

    def get_triton_kernel(mod):
        from torch._inductor.triton_heuristics import CachingAutotuner

        cand_list = [
            v
            for k, v in mod.__dict__.items()
            if k.startswith("triton_") and isinstance(v, CachingAutotuner)
        ]
        assert len(cand_list) == 1
        return cand_list[0]

    nfound = 0
    for kernel_key, kernel_mod in PyCodeCache.cache.items():
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
        num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9

        def get_info_str(ms, n_regs, n_spills, shared, prefix=""):
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
            ms = do_bench(lambda: kernel_mod.call(args), rep=40, fast_flush=True)
            assert (
                len(triton_kernel.launchers) == 1
            ), "Autotuner should have selected the best config"
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
