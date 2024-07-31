import collections
import gc
from typing import Callable, Tuple

import torch
import torch.utils.benchmark as benchmark
from torch.profiler import profile, ProfilerActivity, record_function

# copied from https://github.com/pytorch/ao/pull/629/files
# TODO: reuse instead of copy
def profiler_output_to_filtered_time_by_kernel_name(
    prof,
    num_iter: int,
    num_leaf_tensors: int,
):
    """ 
    Input: 
      * `prof`: a profiler with captured events
      * `num_iter`: number of iterations used to capture `prof`
      * `num_leaf_tensors`: number of leaf tensors to accumulate gradients to
    Output: a deduplicated list of GPU time in nanoseconds grouped by CPU kernel name,
      with the microbenchmark overhead filtered out

    Currently assumes that `prof` captured events from a microbenchmark which was
    set up as follows:

        #
        # Forward pass 
        #

        # Expected GPU kernel overhead: none
        y = func(...)

        # Convenient way to set up the backward pass without caring about shapes
        y_sum = y.sum()

        # Expected GPU kernel overhead:
        # * the call to `sum`

        #
        # Backward pass
        #
        y_sum.backward()

        # Expected GPU kernel overhead:
        # * the call to `aten.fill_` to put a tensor with a single 1.0 value as the input to the backward
        # * the call to `aten.copy_` to fill the first `grad_output` tensor with 1.0
        # * the call to `aten.add_` to accumulate grads, once per leaf tensor

    Note that if there are user_annotations in the captured events, `torch.profiler`
    will include their time in the total GPU time displayed at the bottom of
    `key_averages.table()`. The filter below excludes them to prevent double
    counting.
    """
    key_averages = prof.key_averages()
    thresh = 1e-10
    kernel_name_to_gpu_time_us = collections.defaultdict(float)
    for e in key_averages:

        # manually filter top-level CPU events with attributed CUDA time
        # example CPU event row from printing `key_averages`:
        #                                               aten::addmm         0.83%      76.554us         0.98%      90.846us      90.846us       1.022ms        31.82%       1.022ms       1.022ms             1
        # and it maps to this CUDA event:
        #   sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize256x64...         0.00%       0.000us         0.00%       0.000us       0.000us       1.022ms        31.82%       1.022ms       1.022ms             1
        if not (e.self_cpu_time_total > thresh and e.self_device_time_total > thresh):
            continue

        # manually filter expected microbenchmarking overhead, in order of execution
        if e.key == 'aten::sum':
            # forward pass sum
            assert e.count == num_iter, f'unexpected number of iter for {e.key}'
            continue
        elif e.key == 'aten::fill_':
            # filling the forward pass sum with 1.0
            assert e.count == num_iter, f'unexpected number of iter for {e.key}'
            continue
        elif e.key == 'aten::copy_':
            # copying 1.0 from grad_out of `sum` to grad_out of next op
            # assert e.count == num_iter, f'unexpected number of iter for {e.key}'
            # continue
            # note: uncommented above for subgraph 12, TODO debug why we need it
            pass
        elif e.key == 'aten::add_':
            # accumulating gradients into leaf tensors
            assert e.count == (num_iter * num_leaf_tensors), f'unexpected number of iter for {e.key}'
            continue
        elif e.key == 'cudaDeviceSynchronize':
            continue

        kernel_name_to_gpu_time_us[e.key] = e.self_device_time_total
    return kernel_name_to_gpu_time_us

def prof_to_gemm_vs_non_gemm_time(prof, num_iter, num_leaf_tensors):
    kernel_name_to_gpu_time_us = profiler_output_to_filtered_time_by_kernel_name(prof, num_iter=3, num_leaf_tensors=num_leaf_tensors)
    trace_gemm_time_us, trace_non_gemm_time_us = 0., 0.
    for k, v in kernel_name_to_gpu_time_us.items():
        v = v / num_iter
        # print(k, v)
        if k in ('aten::mm', 'aten::addmm', 'aten::_scaled_mm'):
            trace_gemm_time_us += v
        else:
            trace_non_gemm_time_us += v
    return trace_gemm_time_us, trace_non_gemm_time_us

def benchmark_torch_function_in_microseconds(
    func: Callable,
    *args,
    **kwargs,
) -> float:

    if True:
        # warmup
        for _ in range(2):
            func(*args, **kwargs)
        t0 = benchmark.Timer(
            stmt="func(*args, **kwargs)",
            globals={"args": args, "kwargs": kwargs, "func": func},
        )
        return t0.blocked_autorange().median * 1e6

    if False:
        n_warmup = 3
        n_iter = 10

        for _ in range(n_warmup):
            func(*args, **kwargs)

        t0 = time.time()
        for _ in range(n_iter):
            func(*args, **kwargs)
        t1 = time.time()
        return (t1 - t0) / n_iter * 1e6


def profile_to_file(target_file, func, *args, **kwargs):
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    # warm up
    for _ in range(2):
        func(*args, **kwargs)
        torch.cuda.synchronize()
    with profile(activities=activities) as prof:
        for _ in range(3):
            func(*args, **kwargs)
            torch.cuda.synchronize()
    prof.export_chrome_trace(target_file)
    return prof

def reset_memory():
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=None)

def get_cuda_mem_allocated_gb():
    return torch.cuda.max_memory_allocated() / 1e9
