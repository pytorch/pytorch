# not for land

import collections
import copy
import csv
import gc
import os
import time
from typing import Callable, Tuple

import pandas as pd

import torch
import torch.utils.benchmark as benchmark

from torchao.float8 import (
    convert_to_float8_training,
    ScalingType,
    Float8LinearConfig,
    CastConfig,
    sync_float8_amax_and_scale_history,
)

from .vasiliy_debug_extract_subgraphs import summary_headers
from .vasiliy_debug_analyze_subgraphs_utils import (
    profiler_output_to_filtered_time_by_kernel_name,
    prof_to_gemm_vs_non_gemm_time,
    benchmark_torch_function_in_microseconds,
    profile_to_file,
    reset_memory,
    get_cuda_mem_allocated_gb,
)

# don't truncate long fields
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_columns', None)  

bytes_in_gb = 1024 * 1024 * 1024


def fwd_and_bwd(m, args):
    outs = m(*args)
    # TODO: maybe remove this cat if it contributes meaningfully to microbenchmarks
    if isinstance(outs, tuple):
        outs = torch.cat([*outs], dim=0)
    outs.sum().backward()
    torch.cuda.synchronize()

# TODO reuse input and weight to save memory
@torch.inference_mode
def bench_fwd_bwd_gemms(M, K, N):
    # fwd: in (M, K) @ w_t (K, N) -> out (M, N)
    # bwd 1: grad_out (M, N) @ w (N, K) -> grad_in (M, K)
    # bwd 2: grad_out_t (N, M) @ in (M, K) -> grad_w (N, K)

    input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    grad_out = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')

    input_fp8 = input.to(torch.float8_e4m3fn)
    weight_fp8 = weight.to(torch.float8_e4m3fn)
    grad_out_fp8 = grad_out.to(torch.float8_e5m2)

    scale_a = torch.tensor([1.0], device='cuda')
    scale_b = torch.tensor([1.0], device='cuda')
    
    fwd_time_bf16 = benchmark_torch_function_in_microseconds(
        torch.mm,
        input, weight.t()
    )

    fwd_time_fp8 = benchmark_torch_function_in_microseconds(
        torch._scaled_mm,
        input_fp8, weight_fp8.t(),
        scale_a, scale_b, out_dtype=torch.bfloat16, use_fast_accum=True,
    )

    grad_in_time_bf16 = benchmark_torch_function_in_microseconds(
        torch.mm,
        grad_out, weight,
    )

    grad_in_time_fp8 = benchmark_torch_function_in_microseconds(
        torch._scaled_mm,
        grad_out_fp8, weight_fp8.t().contiguous().t(),
        scale_a, scale_b, out_dtype=torch.bfloat16, use_fast_accum=False,
    )

    grad_w_time_bf16 = benchmark_torch_function_in_microseconds(
        torch.mm,
        grad_out.t(), input,
    )

    grad_w_time_fp8 = benchmark_torch_function_in_microseconds(
        torch._scaled_mm,
        grad_out_fp8.t().contiguous(), input_fp8.t().contiguous().t(),
        scale_a, scale_b, out_dtype=torch.bfloat16, use_fast_accum=False,
    )

    total_bf16 = fwd_time_bf16 + grad_in_time_bf16 + grad_w_time_bf16
    total_fp8 = fwd_time_fp8 + grad_in_time_fp8 + grad_w_time_fp8
    del input, weight, grad_out, input_fp8, weight_fp8, grad_out_fp8
    return total_bf16, total_fp8


def get_mkn(inputs: Tuple[torch.Tensor], m: torch.nn.Module):
    # hack: assume that the first input with rank 2 is the linear input
    # TODO fix it! 
    first_linear_input = None
    for input in inputs:
        if len(input.size()) == 2:
            first_linear_input = input
            break
    assert first_linear_input is not None, 'unsupported'
    M1, K1 = first_linear_input.shape
    # We know m.0 is the first linear because of how we constructed this 
    # subgraph in the extraction code.
    linear_mod = getattr(m, '0')
    K1_extracted, N1 = linear_mod.in_features, linear_mod.out_features
    assert K1 == K1_extracted, 'unexpected K'
    mkn1 = M1, K1, N1

    mkn2 = None
    # hacky, but use the knowledge of how we constructed the sugraph
    # to check for presence of dual linear
    dual_linear_mod = getattr(m, 'dual_linear', None)
    if dual_linear_mod is not None:
        # assume output of linear1 feeds into linear2, we know this is ture
        # from how we extracted the subgraphs
        # linear1: (M1, K1) @ (K1, N1) -> (M1, N1)
        # linear2: (M1, N1) @ (K2, N2) -> (M1, N2)
        #               K2 == N1
        assert N1 == dual_linear_mod.in_features, 'unexpected K'
        mkn2 = M1, dual_linear_mod.in_features, dual_linear_mod.out_features

    return mkn1, mkn2

def bench_single_or_dual_gemms(inputs, m):
    mkn1, mkn2 = get_mkn(inputs, m)
    M1, K1, N1 = mkn1
    gemm_time_bf16, gemm_time_fp8 = bench_fwd_bwd_gemms(M1, K1, N1)
    if mkn2 is not None:
        M2, K2, N2 = mkn2
        gemm_time_bf16_2, gemm_time_fp8_2 = bench_fwd_bwd_gemms(M2, K2, N2)
        gemm_time_bf16 += gemm_time_bf16_2
        gemm_time_fp8 += gemm_time_fp8_2
    gemm_time_speedup = gemm_time_bf16 / gemm_time_fp8
    return gemm_time_bf16, gemm_time_fp8, gemm_time_speedup


# adjust each input's bsz to target_bsz
# enable grad
def resize_input_and_enable_grad(t, extracted_bsz, target_bsz):
    if len(t.shape) > 1:
        old_first_dim, old_rest = t.size()[0], t.size()[1:]
        new_first_dim = old_first_dim // extracted_bsz * target_bsz
        new_shape = (new_first_dim, *old_rest)
        t = torch.randn(*new_shape, dtype=t.dtype, device=t.device, requires_grad=True)
    else:
        # assume that rank 1 tensors do not depend on batch size
        t.requires_grad_(True)
        pass
    return t

def inputs_zero_grad(inputs):
    # need to manually delete grad from inputs, otherwise it would survive
    # and eventually OOM for large problem sizes
    for inp in inputs:
        del inp.grad

def profile_performance(
    func,
    model,
    inputs,
    target_folder,
    name_suffix,
):
    # save torch.compile logs to a file specific to this benchmark run
    # TODO: can we hack torch.compile to print to file only and not stdout?
    # or maybe just use tlparse?
    logs_out = os.path.join(target_folder, f'torch_logs_{name_suffix}.txt')
    torch._logging.set_logs(output_code=True)
    torch._logging._init_logs(log_file_name=logs_out)

    num_leaf_tensors = len(inputs) + len(list(model.parameters()))
    time_compile_us = benchmark_torch_function_in_microseconds(fwd_and_bwd, model, inputs)
    profile_file = os.path.join(target_folder, f'profile_{name_suffix}.json')
    prof = profile_to_file(profile_file, fwd_and_bwd, model, inputs)
    trace_gemm_time_us, trace_non_gemm_time_us = prof_to_gemm_vs_non_gemm_time(
        prof, num_iter=3, num_leaf_tensors=num_leaf_tensors) 
    trace_total_time_us = trace_gemm_time_us + trace_non_gemm_time_us

    # undo custom log settings
    torch._logging.set_logs(output_code=False)
    torch._logging._init_logs(log_file_name=None)

    return time_compile_us, trace_gemm_time_us, trace_non_gemm_time_us, trace_total_time_us


def analyze_subgraphs(
    target_folder: str,
    extracted_bsz: int,
    target_bsz: int,
) -> None:
    """
    Assumes folder structure:

        target_folder/
          debug_logs.txt
          summary.csv
          subgraph_with_inputs_0.pt
          ...
          subgraph_with_inputs_(n-1).pt

    Writes new files as a part of the analysis:
        
        target_folder/
          profile_0_eager.json 
          profile_0_compile.json 
          profile_0_float8_compile.json 
          profile_0_float8_d_compile.json 
          torch_logs_0_compile.txt
          torch_logs_0_float8_compile.txt
          torch_logs_0_float8_d_compile.txt
          ...
          analysis.csv 

    Does the following:
    * load each subgraph and example inputs in bf16
    * increase batch size of inputs to target_batch_size
    * benchmark the following data for each subgraph:
    * gemm time (3 gemms across fwd and bwd) in bf16 vs float8
    * for each in compile, float8_compile, float8_delayed_compile
      * benchmark e2e time of fwd + bwd (includes benchmarking overhead)
      * benchmark gpu kernel time of fwd + bwd, separated by gemm vs non-gemm (excludes benchmarking overhead)
      * record peak GPU memory usage
      * save a profiling trace to profile_{...}.json
      * record TORCH_LOGS="output_code" to torch_logs_{...}.txt
    * finally, outputs `analysis.csv` with a summary of all the above data
    """
    summary_filename = os.path.join(target_folder, 'summary.csv')

    summary_rows = []
    with open(summary_filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            summary_rows.append(row)

    # add names of metrics we are about to collect to headers
    # TODO: adding new metrics is brittle because we need to hand track the
    # indices, there is a definitely a better way to do this
    # Note: not loading directly in a dataframe here because we are going to
    # modify rows inplace and that seemed annoying with dataframes, but there
    # is probably a solution. If someone figures that out, doing `df` throughout
    # this file is cleaner.
    summary_rows[0].extend([
        'inp_shapes',
        'gemm_time_bf16', 'gemm_time_fp8', 'gemm_time_speedup', 
        'time_eager_us', 
        'time_compile_us', 
        'c_trace_gemm_time_us', 'c_trace_non_gemm_time_us', 'c_trace_total_time_us',
        'mem_compile_gb',
        'time_f8_compile_us',
        'f8_c_trace_gemm_time_us', 'f8_c_trace_non_gemm_time_us', 'f8_c_trace_total_time_us',
        'mem_f8_c_gb',
        'time_f8_d_compile_us',
        'f8_c_d_trace_gemm_time_us', 'f8_c_d_trace_non_gemm_time_us', 'f8_c_d_trace_total_time_us',
        'mem_f8_c_d_gb',
    ])

    # [1:] to skip header row
    for row_idx, row in enumerate(summary_rows[1:]):
        # if row_idx > 1:
        #     continue
        subgraph_idx = row[1]
        subgraph_fname = f'subgraph_with_inputs_{subgraph_idx}.pt'
        print(f'benchmarking {subgraph_fname}')
        subgraph_fname = os.path.join(target_folder, subgraph_fname)
        m, inputs = torch.load(subgraph_fname, weights_only=False)

        inputs = [resize_input_and_enable_grad(t, extracted_bsz, target_bsz) for t in inputs]
        input_shapes_str = ', '.join(str(tuple(inp.shape)) for inp in inputs)
        row.append(input_shapes_str)

        # estimate memory used by inputs, params, grads
        input_gb = 0
        for inp in inputs:
            input_gb += (inp.numel() * inp.element_size()) / bytes_in_gb
        model_gb = sum(p.numel() * p.element_size() / bytes_in_gb for p in m.parameters())
        grad_gb = input_gb + model_gb
        total_gb = input_gb + model_gb + grad_gb
        # print(f'param mem estimate (GB): input {input_gb} model {model_gb} grad {grad_gb} total {total_gb}')

        # benchmark gemm time in bf16 vs fp8
        bench_gemms = True
        if bench_gemms:
            gemm_time_bf16, gemm_time_fp8, gemm_time_speedup = \
                bench_single_or_dual_gemms(inputs, m)
            row.extend([gemm_time_bf16, gemm_time_fp8, gemm_time_speedup])
        else:
            row.extend([0., 0., 0.])

        time_eager_us = benchmark_torch_function_in_microseconds(fwd_and_bwd, m, inputs)
        profile_file_eager = os.path.join(target_folder, f'profile_{subgraph_idx}_eager.json')
        prof = profile_to_file(profile_file_eager, fwd_and_bwd, m, inputs)
        row.extend([time_eager_us])
        inputs_zero_grad(inputs)

        bench_compile = True
        if bench_compile:
            reset_memory()

            m_c = torch.compile(m)
            time_compile_us, trace_gemm_time_us, trace_non_gemm_time_us, trace_total_time_us = \
                profile_performance(fwd_and_bwd, m_c, inputs, target_folder, f'{subgraph_idx}_compile')
            mem_usage_gb = get_cuda_mem_allocated_gb()
            print('b16 trace times', trace_gemm_time_us, trace_non_gemm_time_us, trace_total_time_us)
            print('b16 mem', mem_usage_gb)
            
            row.extend([time_compile_us, trace_gemm_time_us, trace_non_gemm_time_us, trace_total_time_us, mem_usage_gb])
            inputs_zero_grad(inputs)
            del m_c
        else:
            row.extend([0., 0., 0., 0., 0.])

        # TODO: figure out why setting this to True influences the fusions in fp8 delayed scaling
        bench_fp8 = True
        if bench_fp8:
            reset_memory()
            f8_config = Float8LinearConfig()
            m_f8 = convert_to_float8_training(copy.deepcopy(m), config=f8_config)
            m_f8_c = torch.compile(m_f8)
            time_compile_us, trace_gemm_time_us, trace_non_gemm_time_us, trace_total_time_us = \
                profile_performance(fwd_and_bwd, m_f8_c, inputs, target_folder, f'{subgraph_idx}_f8_compile')
            mem_usage_gb = get_cuda_mem_allocated_gb()
            print('fp8 trace times', trace_gemm_time_us, trace_non_gemm_time_us, trace_total_time_us)
            print('fp8 mem', mem_usage_gb)
            row.extend([time_compile_us, trace_gemm_time_us, trace_non_gemm_time_us, trace_total_time_us, mem_usage_gb])
            inputs_zero_grad(inputs)
            del m_f8, m_f8_c
        else:
            row.extend([0., 0., 0., 0., 0.])

        bench_fp8_delayed = True
        if bench_fp8_delayed:
            reset_memory()
            # upper bound on perf - all-delayed
            # TODO(future): add weight-only delayed
            f8_config = Float8LinearConfig(
                enable_amax_init=False,
                enable_pre_and_post_forward=False,
                cast_config_input=CastConfig(scaling_type=ScalingType.DELAYED),
                cast_config_weight=CastConfig(scaling_type=ScalingType.DELAYED),
                cast_config_grad_output=CastConfig(scaling_type=ScalingType.DELAYED),
            )
            m_f8_d = convert_to_float8_training(m, config=f8_config)
            m_f8_c_d = torch.compile(m_f8_d)
            time_compile_us, trace_gemm_time_us, trace_non_gemm_time_us, trace_total_time_us = \
                profile_performance(fwd_and_bwd, m_f8_c_d, inputs, target_folder, f'{subgraph_idx}_f8_d_compile')
            mem_usage_gb = get_cuda_mem_allocated_gb()
            print('fp8_d trace times', trace_gemm_time_us, trace_non_gemm_time_us, trace_total_time_us)
            print('fp8_d mem', mem_usage_gb)
            row.extend([time_compile_us, trace_gemm_time_us, trace_non_gemm_time_us, trace_total_time_us, mem_usage_gb])
            inputs_zero_grad(inputs)
            del m_f8_d, m_f8_c_d
        else:
            row.extend([0., 0., 0., 0., 0.])

        del m, inputs
        gc.collect()
        torch.cuda.empty_cache()

    # convert to pandas df for easy printing and aggregate manipulations
    summary_df = pd.DataFrame(summary_rows[1:], columns=summary_rows[0])

    # calculate total subgraph time and each row's contribution to it
    total_time_us = summary_df['time_compile_us'].sum()
    summary_df['time_compile_pct'] = summary_df['time_compile_us'] / total_time_us
    print(summary_df)

    analysis_filename = os.path.join(target_folder, 'analysis.csv')
    summary_df.to_csv(analysis_filename)

    print('done')
