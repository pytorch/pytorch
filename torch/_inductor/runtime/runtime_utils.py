from __future__ import annotations

import functools
import getpass
import operator
import os
import re
import tempfile
import time

import torch


def conditional_product(*args):
    return functools.reduce(operator.mul, [x for x in args if x])


def ceildiv(numer: int, denom: int) -> int:
    return -(numer // -denom)


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def get_num_bytes(*args: torch.Tensor, num_in_out_args: int = 0) -> int:
    """
    Return the total number of bytes the arguments of tensor type takes.

    For in/out args, tensor sizes are counted twice: once for reading and
    once for writing.

    The first num_in_out_args arguments are in out tensors.
    """
    return sum(
        arg.numel() * arg.element_size() * (1 + int(i < num_in_out_args))
        for i, arg in enumerate(args)
        if isinstance(arg, torch.Tensor)
    )


def triton_config_to_hashable(cfg):
    """
    Convert triton config to a tuple that can uniquely identify it. We can use
    the return value as a dictionary key.
    """
    items = sorted(cfg.kwargs.items())
    items.append(("num_warps", cfg.num_warps))
    items.append(("num_stages", cfg.num_stages))
    return tuple(items)


def create_bandwidth_info_str(ms, num_gb, gb_per_s, prefix="", suffix="", color=True):
    info_str = f"{prefix}{ms:.3f}ms    \t{num_gb:.3f} GB \t {gb_per_s:7.2f}GB/s{suffix}"
    slow = ms > 0.012 and gb_per_s < 650
    return red_text(info_str) if color and slow else info_str


def get_max_y_grid():
    return 65535


def do_bench(fn, fn_args, fn_kwargs, **kwargs):
    from torch._inductor.utils import is_cpu_device

    args = list(fn_args)
    args.extend(fn_kwargs.values())
    if is_cpu_device(args):
        return do_bench_cpu(lambda: fn(*fn_args, **fn_kwargs), **kwargs)
    else:
        return do_bench_gpu(lambda: fn(*fn_args, **fn_kwargs), **kwargs)


def do_bench_gpu(
    fn, warmup=25, rep=100, fast_flush=True, quantiles=(0.5,), return_mode="mean"
):
    @functools.lru_cache(None)
    def get_cache_size():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return properties.l2CacheSize

    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    estimation_iters = 5
    event_pairs = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(estimation_iters)
    ]
    for start_event, end_event in event_pairs:
        start_event.record()
        cache.zero_()
        fn()
        end_event.record()
    torch.cuda.synchronize()
    estimate_ms = min(
        [event_pair[0].elapsed_time(event_pair[1]) for event_pair in event_pairs]
    )

    if fast_flush:
        cache = torch.empty(int(get_cache_size() // 4), dtype=torch.int, device="cuda")

    warmup_iters = max(1, int(warmup / estimate_ms))
    repeat_iters = max(1, int(rep / estimate_ms))

    event_pairs = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(repeat_iters)
    ]

    for _ in range(warmup_iters):
        fn()

    for start_event, end_event in event_pairs:
        cache.zero_()
        start_event.record()
        fn()
        end_event.record()
    torch.cuda.synchronize()

    timings = torch.tensor(
        [event_pair[0].elapsed_time(event_pair[1]) for event_pair in event_pairs],
        dtype=torch.float,
    )
    if quantiles is not None:
        timing_quantiles = torch.quantile(
            timings, torch.tensor(quantiles, dtype=torch.float)
        ).tolist()
        if len(timing_quantiles) == 1:
            timing_quantiles = timing_quantiles[0]
        return timing_quantiles
    return getattr(torch, return_mode)(timings).item()


def do_bench_cpu(fn, warmup=5, times=20):
    assert times > 0
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(times):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000)
    # return the median time
    sorted_durations = sorted(durations)
    if times % 2 == 0:
        return (sorted_durations[times // 2 - 1] + sorted_durations[times // 2]) / 2
    else:
        return sorted_durations[times // 2]


def cache_dir() -> str:
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir is None:
        sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir = os.path.join(
            tempfile.gettempdir(),
            "torchinductor_" + sanitized_username,
        )
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


HAS_COLORAMA = True
try:
    import colorama
except ImportError:
    HAS_COLORAMA = False


def _color_text(msg, color):
    if not HAS_COLORAMA:
        return msg

    return getattr(colorama.Fore, color.upper()) + msg + colorama.Fore.RESET


def green_text(msg):
    return _color_text(msg, "green")


def yellow_text(msg):
    return _color_text(msg, "yellow")


def red_text(msg):
    return _color_text(msg, "red")


def blue_text(msg):
    return _color_text(msg, "blue")


def get_first_attr(obj, *attrs):
    """
    Return the first available attribute or throw an exception if none is present.
    """
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)

    raise AssertionError(f"{obj} does not has any of the attributes: {attrs}")


try:
    dynamo_timed = torch._dynamo.utils.dynamo_timed
except AttributeError:  # Compile workers only have a mock version of torch

    def dynamo_timed(original_function=None, phase_name=None):
        if original_function:
            return original_function
        return dynamo_timed
