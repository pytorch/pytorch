import argparse
import itertools
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn.attention._flex_attention import _flex_attention

torch._dynamo.config.automatic_dynamic_shapes = False
# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


from triton.testing import do_bench





def benchmark_torch_function_in_microseconds(func: Callable, cuda_graph: bool = True, *args, **kwargs) -> float:
    def run_func():
        func(*args, **kwargs)

    if cuda_graph:
        func(*args, **kwargs)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run_func()
        def run_func():
            g.replay()

    # warmup
    for _ in range(5):
        run_func()
    return do_bench(lambda: run_func()) * 1e3

import csv
def read_benchmark_results_from_csv(filename: str):
    results = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape = tuple(map(int, eval(row['shape'])))
            dtype = eval(row['dtype'])
            key = (*shape, dtype)
            value = float(row['fwd_compiled_time'])
            results[key] = value
    return results


def write_results_to_csv(results, filename: str):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['shape', 'score_mod', 'dtype', 'baseline_time', 'fwd_eager_time', 'fwd_compiled_time', 'fwd_speedup', 'fwd_bw (TB/s)', 'GFlops/s'])
        for shape, score_mod, dtype, baseline_time, fwd_eager_time, fwd_compiled_time, fwd_speedup, fwd_bw, gflops in zip(results['shape'], results['score_mod'], results['dtype'], results['baseline_time'], results['fwd_eager_time'], results['fwd_compiled_time'], results['fwd_speedup'], results['fwd_bw (TB/s)'], results['GFlops/s']):
            writer.writerow([str(shape), score_mod, str(dtype), str(baseline_time), str(fwd_eager_time), str(fwd_compiled_time), str(fwd_speedup), str(fwd_bw), str(gflops)])
    print("Results are written to score_mod_decoding_results.csv")


@dataclass(frozen=True)
class ExperimentConfig:
    shape: Tuple[int]
    score_mod: Callable
    dtype: torch.dtype
    calculate_bwd_time: bool
    bench_xformers: bool
    baseline_time: Optional[float]

    def __post_init__(self):
        assert len(self.shape) == 5, "Shape must be of length 5" #[B, H, Q, D, KV]

    def asdict(self):
        # Convert the dataclass instance to a dictionary
        d = asdict(self)
        # Remove the 'calculate_bwd_time' key
        d.pop("calculate_bwd_time", None)
        d.pop("bench_xformers", None)
        if not self.baseline_time: 
            d.pop("baseline_time")
        d['shape(B,Hkv,Hq//Hkv,M,D,N)'] = d.pop('shape')
        return d


@dataclass(frozen=True)
class Times:
    baseline_time: float
    optimized_time: float
    xformers_time: Optional[float]


@dataclass(frozen=True)
class ExperimentResults:
    fwd_times: Times
    bwd_times: Optional[Times]


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def asdict(self):
        dict1 = self.config.asdict()
        dict2 = asdict(self.results)
        return {**dict1, **dict2}


def generate_inputs(
    batch_size: int,
    num_heads: int,
    q_sequence_length: int,
    kv_sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
):
    q_shape = (batch_size, num_heads, q_sequence_length, head_dim)
    kv_shape = (batch_size, num_heads, kv_sequence_length, head_dim)
    query = torch.rand(q_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    key = torch.rand(kv_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    value = torch.rand(kv_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    return query, key, value



def run_single_experiment(config: ExperimentConfig, dynamic=False) -> ExperimentResults:
    device = torch.device("cuda")
    batch_size, num_heads, q_seq_len, head_dim, kv_seq_len = config.shape
    query, key, value = generate_inputs(
        batch_size = batch_size,
        num_heads = num_heads,
        q_sequence_length = q_seq_len,
        kv_sequence_length = kv_seq_len,
        head_dim = head_dim,
        dtype = config.dtype,
        device = device,
        requires_grad=config.calculate_bwd_time,
    )

    def eager_sdpa(query, key, value, _):
        return F.scaled_dot_product_attention(query, key, value)


    score_mod = config.score_mod

    forward_eager_time = benchmark_torch_function_in_microseconds(
        eager_sdpa, False, query, key, value, score_mod
    )

    compiled_flex_attention = torch.compile(_flex_attention, dynamic=dynamic)
    forward_flex_decoding_time = benchmark_torch_function_in_microseconds(
        compiled_flex_attention, False, query, key, value, score_mod
    )


    if config.bench_xformers and config.dtype in {torch.bfloat16, torch.float16}:
        import xformers.ops as xops
        from xformers.attn_bias_utils import create_attn_bias
        def gen_xformers_input(Config: ExperimentConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            B, H, M, K, N = Config.shape
            Hkv = H
            Hq = Hkv * M
            Mq = 1
            Mkv = N

            dtype = Config.dtype

            q = torch.randn(
                [B, Mq, Hkv, Hq // Hkv, K], device="cuda", dtype=dtype, requires_grad=False
            )
            k = torch.randn(
                [B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype, requires_grad=False
            ).expand(-1, -1, -1, Hq // Hkv, -1)
            v = torch.randn(
                [B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype, requires_grad=False
            ).expand(-1, -1, -1, Hq // Hkv, -1)


            if Hq == Hkv:
                q = q[:, :, :, 0]
                k = k[:, :, :, 0]
                v = v[:, :, :, 0]
            if Hkv == 1:
                q = q[:, :, 0]
                k = k[:, :, 0]
                v = v[:, :, 0]
            
            return q, k, v
 

        def xformers_sdpa(query, key, value, _):
            return xops.memory_efficient_attention_forward(query, key, value, op=xops.fmha.triton_splitk.FwOp, attn_bias=None, scale=1)
        
        xformers_q, xformers_k, xformers_v = gen_xformers_input(config)
        forward_xformers_time = benchmark_torch_function_in_microseconds(
            xformers_sdpa,  True, xformers_q, xformers_k, xformers_v, score_mod
        )

        return ExperimentResults(
        fwd_times=Times(forward_eager_time, forward_flex_decoding_time, forward_xformers_time),
        bwd_times=None,
        )


    return ExperimentResults(
        fwd_times=Times(forward_eager_time, forward_flex_decoding_time, float('nan')),
        bwd_times=None,
    )


def calculate_speedup(config: ExperimentConfig, results: ExperimentResults, type: str) -> float:
    if type == "fwd":
        if config.baseline_time is None:
            return results.fwd_times.baseline_time / results.fwd_times.optimized_time
        else:
            return config.baseline_time / results.fwd_times.optimized_time
    elif type == "bwd":
        assert results.bwd_times is not None
        return results.bwd_times.baseline_time / results.bwd_times.optimized_time
    else:
        raise ValueError(f"Invalid type {type}")



def calculate_speedup_wrt_xformers(results: ExperimentResults, type: str) -> float:
    if type == "fwd":
        return results.fwd_times.xformers_time / results.fwd_times.optimized_time
    else:
        raise ValueError(f"Invalid type {type}")

def calculate_bandwidth(config: ExperimentConfig, results: ExperimentResults, type: str) -> float:
    if type == "fwd":
        query_size = config.shape[0] * config.shape[1] * config.shape[2] * config.shape[3] * torch.finfo(config.dtype).bits / 8
        kv_size =config.shape[0] * config.shape[1] * config.shape[4] * config.shape[3] * torch.finfo(config.dtype).bits / 8 * 2
        output_size = config.shape[0] * config.shape[1] * config.shape[2] * config.shape[3] * torch.finfo(config.dtype).bits / 8
        total_size = (query_size + kv_size + output_size) / 1024 / 1024 / 1024 # In GB 
        time_in_seconds = results.fwd_times.optimized_time / 1e6
        return total_size / time_in_seconds / 1024
    else:
        raise ValueError(f"Invalid type {type}")

def calculate_gflops(config: ExperimentConfig, results: ExperimentResults) -> float:
    B = config.shape[0]
    H = config.shape[1]
    M = config.shape[2]
    D = config.shape[3]
    N = config.shape[4]
    qk_flops = M * N * D * 2
    softmax_flops = M * N * 2 # Not counting online softmax overhead
    o_flops = M * D * N * 2
    # Not counting split k overhead
    total_flops = B * H * (qk_flops + softmax_flops + o_flops)
    return total_flops/ results.fwd_times.optimized_time / 1e3 # in GFLOPs/s


def get_func_name(func):
    return func.__name__.split("<locals>.")[-1].split(" at ")[0]


def get_average_speedups(results: List[Experiment], speedups:List[float], bw:List[float], gflops: List[float], type: str):


    # Find indices of max and min speedups
    max_speedup_index = np.nanargmax(speedups)
    min_speedup_index = np.nanargmin(speedups)

    # Get the config dictionaries
    max_config_dict = results[max_speedup_index].config.asdict()
    min_config_dict = results[min_speedup_index].config.asdict()

    # Extract function names from score_mod strings
    max_config_dict["score_mod"] = (
        max_config_dict["score_mod"].__name__.split("<locals>.")[-1].split(" at ")[0]
    )
    min_config_dict["score_mod"] = (
        min_config_dict["score_mod"].__name__.split("<locals>.")[-1].split(" at ")[0]
    )

    # Create table data
    table_data = [
        {
            "Type": "Average",
            "Speedup": np.nanmean(speedups),
            **dict.fromkeys(max_config_dict),
        },
        {"Type": "Max", "Speedup": speedups[max_speedup_index], "BandWidth (TB/s)":  bw[max_speedup_index], "GFlops/s": gflops[max_speedup_index],  **max_config_dict},
        {"Type": "Min", "Speedup": speedups[min_speedup_index], "BandWidth (TB/s)": bw[min_speedup_index], "GFlops/s": gflops[min_speedup_index], **min_config_dict},
    ]

    return table_data


def print_results(results: List[Experiment]):
    table_data = defaultdict(list)
    for experiment in results:
        for key, value in experiment.asdict().items():
            if key == "fwd_times":
                for name, time in value.items():
                    if time: 
                        table_data[f"fwd_{name}"].append(float(time))
            elif key == "bwd_times":
                if experiment.config.calculate_bwd_time:
                    for name, time in value.items():
                        if time: 
                            table_data[f"bwd_{name}"].append(float(time))
            else:
                table_data[key].append(value)

    # Calculate speedups
    fwd_speedups = [calculate_speedup(r.config, r.results, type="fwd") for r in results]
    table_data["fwd_speedup"] = fwd_speedups
    if results[0].config.calculate_bwd_time:
        bwd_speedups = [calculate_speedup(r.config, r.results, type="bwd") for r in results]
        table_data["bwd_speedup"] = bwd_speedups
    
    if results[0].config.bench_xformers:
        xformers_speedup = [calculate_speedup_wrt_xformers(r.results, type="fwd" )for r in results]
        table_data["speedup_wrt_xformers"] = xformers_speedup
        
    
    # calculate theoretical bandwidth
    fwd_bandwidth = [calculate_bandwidth(r.config, r.results, type="fwd") for r in results]
    table_data["fwd_bw (TB/s)"] = fwd_bandwidth
    fwd_gflops = [calculate_gflops(r.config, r.results) for r in results]
    table_data["GFlops/s"] = fwd_gflops
 
    table_data["score_mod"] = [get_func_name(func) for func in table_data["score_mod"]]
    print(tabulate(table_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    write_results_to_csv(table_data, "score_mod_decoding_results.csv")

    print("\n")
    print("FWD Speedups".center(125, "="))
    print("\n")
    average_data = get_average_speedups(results, fwd_speedups, fwd_bandwidth, fwd_gflops, type="fwd")
    print(tabulate(average_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    if results[0].config.bench_xformers:
        print("\n")
        print("Speedups w.r.t Xformers SplitK Kernel".center(125, "="))
        print("\n")
        average_data = get_average_speedups(results, xformers_speedup, fwd_bandwidth, fwd_gflops, type="fwd")
        print(tabulate(average_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    if results[0].config.calculate_bwd_time:
        print("\n")
        print("BWD Speedups".center(125, "="))
        print("\n")
        average_data = get_average_speedups(results, bwd_speedups, bwd_bandwidth, bwd_gflops, type="bwd")
        print(tabulate(average_data, headers="keys", tablefmt="github", floatfmt=".3f"))


def generate_score_mods() -> List[Callable]:
    def noop(score, b, h, m, n):
        return score

    def causal_mask(score, b, h, token_q, token_kv):
        return torch.where(token_q >= token_kv, score, float("-inf"))

    def relative_bias(score, b, h, m, n):
        return score + (m - n)

    def head_bias(score, b, h, m, n):
        return score + 2 * h

    # return [noop, causal_mask, relative_bias, head_bias]
    return [noop]


def generate_experiment_configs(calculate_bwd: bool, bench_xformers: bool, baseline_performance) -> List[ExperimentConfig]:
    kv_cache_sizes = [
        (128, 512), 
        (64, 1024), 
        (32, 2048), 
        (16, 4096), 
        (8, 8192), 
        (4, 16384), 
        (2, 32768), 
        (1, 65536), 
        (1, 131072)
    ]
    n_heads = [
        (16, 1), 
        (16, 2),
        (16, 16)
    ] # (Hq, Hkv)
    # head_dims = [64, 128, 256]
    head_dims = [128]
    dtypes = [
        torch.bfloat16,
        torch.float16,
        torch.float32
    ]
    score_mods = generate_score_mods()
    all_configs = []
    for (
        (Hq, Hkv),
        (bsz, kv_seq_len),
        head_dim,
        score_mod,
        dtype,
    ) in itertools.product(
        n_heads, kv_cache_sizes, head_dims, score_mods, dtypes
    ):
       
        n_heads = Hkv
        q_seq_len = Hq // Hkv
        assert Hq % Hkv == 0
        if baseline_performance is None: 
            all_configs.append(
                ExperimentConfig(
                    shape=(bsz, n_heads, q_seq_len, head_dim, kv_seq_len),
                    score_mod=score_mod,
                    dtype=dtype,
                    calculate_bwd_time=calculate_bwd,
                    bench_xformers=bench_xformers,
                    baseline_time=None,
                )
            )
        else: 
            baseline_time = baseline_performance[(bsz, n_heads, q_seq_len, head_dim, kv_seq_len, dtype)] 
            if not baseline_time: 
                baseline_time = float('nan')
            all_configs.append(
                ExperimentConfig(
                    shape=(bsz, n_heads, q_seq_len, head_dim, kv_seq_len),
                    score_mod=score_mod,
                    dtype=dtype,
                    calculate_bwd_time=calculate_bwd,
                    bench_xformers=bench_xformers,
                    baseline_time=baseline_time,
                )
            )

    return all_configs



def main(dynamic: bool, calculate_bwd: bool, bench_xformers, baseline: str):
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    if baseline: 
        baseline_performance = read_benchmark_results_from_csv(baseline)
    else: 
        baseline_performance = None
    results = []
    for config in tqdm(generate_experiment_configs(calculate_bwd, bench_xformers, baseline_performance)):
        results.append(
            Experiment(config, run_single_experiment(config, dynamic=dynamic))
        )

    print_results(results)





if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Run sweep over sizes and score mods for flex attention"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Runs a dynamic shapes version of compiled flex attention.",
    )
    parser.add_argument(
        "--calculate-bwd", action="store_true", help="Calculate backward pass times"
    )

    parser.add_argument(
        "--baseline", type=str, help="Baseline imported from csv"
    )
    parser.add_argument(
        "--xformers", action="store_true", help="Benchmark against xformers splitK kernel"
    )

    # Parse arguments
    args = parser.parse_args()

    main(args.dynamic, args.calculate_bwd, args.xformers, args.baseline)
