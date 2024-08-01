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
from torch.nn.attention.flex_attention import (
    _create_empty_block_mask,
    create_block_mask,
    flex_attention,
)


torch._dynamo.config.automatic_dynamic_shapes = False
# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


from triton.testing import do_bench


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    return do_bench(lambda: func(*args, **kwargs)) * 1e3


@dataclass(frozen=True)
class ExperimentConfig:
    shape: Tuple[int]
    score_mod: Callable
    mask_mod: Callable
    dtype: torch.dtype
    calculate_bwd_time: bool
    cal_bandwidth: bool

    def __post_init__(self):
        assert (
            len(self.shape) == 6
        ), "Shape must be of length 6"  # [B, Hq, M, Hkv, N, D]

    def asdict(self):
        # Convert the dataclass instance to a dictionary
        d = asdict(self)
        # Remove the 'calculate_bwd_time' and `cal_bandwidth` key
        d.pop("calculate_bwd_time", None)
        d.pop("cal_bandwidth", None)
        d["shape(B,Hq,M,Hkv,N,D)"] = d.pop("shape")
        return d


@dataclass(frozen=True)
class Times:
    eager_time: float
    compiled_time: float


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
    q_heads: int,
    q_sequence_length: int,
    kv_heads: int,
    kv_sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
):
    q_shape = (batch_size, q_sequence_length, q_heads * head_dim)
    kv_shape = (batch_size, kv_sequence_length, kv_heads * head_dim)

    assert q_heads % kv_heads == 0

    num_h_groups = q_heads // kv_heads

    make_q = partial(
        torch.rand, q_shape, device=device, dtype=dtype, requires_grad=requires_grad
    )
    make_kv = partial(
        torch.rand, kv_shape, device=device, dtype=dtype, requires_grad=requires_grad
    )
    query = (
        make_q()
        .view(batch_size, num_h_groups * q_sequence_length, kv_heads, head_dim)
        .transpose(1, 2)
    )
    key = (
        make_kv()
        .view(batch_size, kv_sequence_length, kv_heads, head_dim)
        .transpose(1, 2)
    )
    value = (
        make_kv()
        .view(batch_size, kv_sequence_length, kv_heads, head_dim)
        .transpose(1, 2)
    )
    return query, key, value


def run_single_experiment(
    config: ExperimentConfig,
    dynamic=False,
    max_autotune=False,
) -> ExperimentResults:
    device = torch.device("cuda")
    batch_size, q_heads, q_seq_len, kv_heads, kv_seq_len, head_dim = config.shape
    query, key, value = generate_inputs(
        batch_size,
        q_heads,
        q_seq_len,
        kv_heads,
        kv_seq_len,
        head_dim,
        config.dtype,
        device,
        requires_grad=config.calculate_bwd_time,
    )

    def eager_sdpa(query, key, value, _):
        return F.scaled_dot_product_attention(query, key, value)

    if max_autotune:
        compiled_sdpa = torch.compile(
            flex_attention, dynamic=dynamic, mode="max-autotune-no-cudagraphs"
        )
    else:
        compiled_sdpa = torch.compile(flex_attention, dynamic=dynamic)

    score_mod = config.score_mod
    mask_mod = config.mask_mod

    if mask_mod:
        block_mask = create_block_mask(
            mask_mod, 1, 1, q_seq_len * (q_heads // kv_heads), kv_seq_len, query.device
        )
    else:
        block_mask = _create_empty_block_mask(query, key)

    forward_eager_time = benchmark_torch_function_in_microseconds(
        eager_sdpa, query, key, value, score_mod
    )
    forward_compiled_time = benchmark_torch_function_in_microseconds(
        compiled_sdpa, query, key, value, score_mod, block_mask
    )

    if config.calculate_bwd_time:
        out_eager = eager_sdpa(query, key, value, score_mod)
        dOut = torch.randn_like(out_eager)
        backward_eager_time = benchmark_torch_function_in_microseconds(
            out_eager.backward, dOut, retain_graph=True
        )

        out_compile = compiled_sdpa(query, key, value, score_mod)
        dOut = torch.randn_like(out_eager)
        backward_compile_time = benchmark_torch_function_in_microseconds(
            out_compile.backward, dOut, retain_graph=True
        )

        return ExperimentResults(
            fwd_times=Times(forward_eager_time, forward_compiled_time),
            bwd_times=Times(backward_eager_time, backward_compile_time),
        )
    else:
        return ExperimentResults(
            fwd_times=Times(forward_eager_time, forward_compiled_time),
            bwd_times=None,
        )


def calculate_speedup(results: ExperimentResults, type: str) -> float:
    if type == "fwd":
        return results.fwd_times.eager_time / results.fwd_times.compiled_time
    elif type == "bwd":
        assert results.bwd_times is not None
        return results.bwd_times.eager_time / results.bwd_times.compiled_time
    else:
        raise ValueError(f"Invalid type {type}")


def calculate_bandwidth(
    config: ExperimentConfig, results: ExperimentResults, type: str
) -> float:
    if type == "fwd":
        batch_size, q_heads, q_seq_len, kv_heads, kv_seq_len, head_dim = config.shape
        query_size = (
            batch_size
            * q_heads
            * q_seq_len
            * head_dim
            * torch.finfo(config.dtype).bits
            / 8
        )
        kv_size = (
            batch_size
            * kv_heads
            * kv_seq_len
            * head_dim
            * torch.finfo(config.dtype).bits
            / 8
            * 2
        )
        output_size = query_size
        total_size = (query_size + kv_size + output_size) / 1e9  # In GB
        time_in_seconds = results.fwd_times.compiled_time / 1e6
        return total_size / time_in_seconds / 1e3
    else:
        raise ValueError(f"Invalid type {type}")


def calculate_tflops(config: ExperimentConfig, results: ExperimentResults) -> float:
    (B, Hq, M, Hkv, N, D) = config.shape
    qk_flops = M * N * D * 2
    softmax_flops = M * N * 2  # Not counting online softmax overhead
    o_flops = M * D * N * 2
    # Not counting split k overhead
    total_flops = B * Hq * (qk_flops + softmax_flops + o_flops)
    return total_flops / results.fwd_times.compiled_time / 1e6  # in TFLOPs/


def get_func_name(func):
    return func.__name__.split("<locals>.")[-1].split(" at ")[0]


def set_func_name(func, name):
    func.__name__ = name


def get_average_speedups(results: List[Experiment], type: str):
    # Calculate speedups
    speedups = [calculate_speedup(r.results, type) for r in results]

    # Find indices of max and min speedups
    max_speedup_index = np.argmax(speedups)
    min_speedup_index = np.argmin(speedups)

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
            "Speedup": np.mean(speedups),
            **dict.fromkeys(max_config_dict),
        },
        {"Type": "Max", "Speedup": speedups[max_speedup_index], **max_config_dict},
        {"Type": "Min", "Speedup": speedups[min_speedup_index], **min_config_dict},
    ]

    return table_data


def print_results(results: List[Experiment]):
    table_data = defaultdict(list)
    for experiment in results:
        for key, value in experiment.asdict().items():
            if key == "fwd_times":
                for name, time in value.items():
                    table_data[f"fwd_{name}"].append(float(time))
            elif key == "bwd_times":
                if experiment.config.calculate_bwd_time:
                    for name, time in value.items():
                        table_data[f"bwd_{name}"].append(float(time))
            else:
                table_data[key].append(value)

    # Calculate speedups
    fwd_speedups = [calculate_speedup(r.results, type="fwd") for r in results]
    table_data["fwd_speedup"] = fwd_speedups

    # Calculate mem + computational throughput
    if results[0].config.cal_bandwidth:
        fwd_bandwidth = [
            calculate_bandwidth(r.config, r.results, type="fwd") for r in results
        ]
        table_data["fwd_mem_bw (TB/s)"] = fwd_bandwidth
        fwd_tflops = [calculate_tflops(r.config, r.results) for r in results]
        table_data["TFlops/s"] = fwd_tflops

    if results[0].config.calculate_bwd_time:
        bwd_speedups = [calculate_speedup(r.results, type="bwd") for r in results]
        table_data["bwd_speedup"] = bwd_speedups

    table_data["score_mod"] = [get_func_name(func) for func in table_data["score_mod"]]
    print(tabulate(table_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    print("\n")
    print("FWD Speedups".center(125, "="))
    print("\n")
    average_data = get_average_speedups(results, type="fwd")
    print(tabulate(average_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    if results[0].config.calculate_bwd_time:
        print("\n")
        print("BWD Speedups".center(125, "="))
        print("\n")
        average_data = get_average_speedups(results, type="bwd")
        print(tabulate(average_data, headers="keys", tablefmt="github", floatfmt=".3f"))


def generate_score_mods(score_mods: List[str]) -> List[Callable | None]:
    def noop(score, b, h, m, n):
        return score

    def causal_mask(score, b, h, token_q, token_kv):
        return torch.where(token_q >= token_kv, score, float("-inf"))

    def relative_bias(score, b, h, m, n):
        return score + (m - n)

    def head_bias(score, b, h, m, n):
        return score + 2 * h

    function_dict = {
        "noop": noop,
        "causal": None,
        "rel": relative_bias,
        "head_bias": head_bias,
    }
    return [function_dict[name] for name in score_mods]


def generate_mask_mods(score_mods: List[str]) -> List[Callable | None]:
    def noop(b, h, m, n):
        return True

    def causal(b, h, m, n):
        return m >= n

    mask_mod_dict = {
        "noop": None,
        "causal": causal,
        "rel": None,
        "head_bias": None,
    }
    return [mask_mod_dict[name] for name in score_mods]


def get_gqa_score_mod(score_mod, G, q_seq_len):
    if score_mod is None:
        return None

    def score_mod_gqa(score, b, hkv, m, n):
        g = m // q_seq_len
        new_m = m % q_seq_len
        hq = hkv * G + g
        return score_mod(score, b, hq, new_m, n)

    score_mod_name = get_func_name(score_mod)
    set_func_name(score_mod_gqa, score_mod_name + "_gqa")
    return score_mod_gqa


def get_gqa_mask_mod(mask_mod, G, q_seq_len):
    if mask_mod is None:
        return None

    def mask_mod_gqa(b, h, m, n):
        g = m // q_seq_len
        new_m = m % q_seq_len
        hq = h * G + g
        return mask_mod(b, hq, new_m, n)

    mask_mod_name = get_func_name(mask_mod)
    set_func_name(mask_mod_gqa, mask_mod_name + "_gqa")
    return mask_mod_gqa


def generate_experiment_configs(
    calculate_bwd: bool,
    dtype: torch.dtype,
    batch_sizes: List[int],
    num_heads: List[Tuple[int, int]],
    seq_lens: List[int],
    head_dims: List[int],
    score_mods_str: List[str],
    decoding: bool,
    kv_cache_size: List[int],
    cal_bandwidth: bool,
) -> List[ExperimentConfig]:
    assert not (calculate_bwd and decoding), "Decoding does not support backward"

    if decoding:
        q_kv_seq_lens = [(1, i) for i in seq_lens]  # only testing query length == 1
    else:
        q_kv_seq_lens = [(i, i) for i in seq_lens]  # only testing q_len == kv_len
    dtypes = [dtype]
    score_mods = generate_score_mods(score_mods_str)
    mask_mods = generate_mask_mods(score_mods_str)
    all_configs = []
    for (
        bsz,
        (q_heads, kv_heads),
        (q_seq_len, kv_seq_len),
        head_dim,
        score_mod,
        mask_mod,
        dtype,
    ) in itertools.product(
        kv_cache_size if kv_cache_size else batch_sizes,
        num_heads,
        q_kv_seq_lens,
        head_dims,
        score_mods,
        mask_mods,
        dtypes,
    ):
        if kv_cache_size:
            head_size_bytes = torch.finfo(dtype).bits / 8 * head_dim
            bsz = int(
                (bsz * 1024 * 1024) // (kv_heads * kv_seq_len * head_size_bytes * 2)
            )
            if bsz <= 0:
                continue

        if q_heads != kv_heads:  # GQA work around before it's explicitly supported
            assert q_heads % kv_heads == 0
            G = q_heads // kv_heads
            score_mod = get_gqa_score_mod(score_mod, G, q_seq_len)
            mask_mod = get_gqa_mask_mod(mask_mod, G, q_seq_len)

        all_configs.append(
            ExperimentConfig(
                shape=(bsz, q_heads, q_seq_len, kv_heads, kv_seq_len, head_dim),
                score_mod=score_mod,
                mask_mod=mask_mod,
                dtype=dtype,
                calculate_bwd_time=calculate_bwd,
                cal_bandwidth=cal_bandwidth,
            )
        )

    return all_configs


def main(args):
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    results = []
    for config in tqdm(
        generate_experiment_configs(
            args.calculate_bwd,
            args.dtype,
            args.b,
            args.nh,
            args.s,
            args.d,
            args.mods,
            args.decoding,
            args.kv_cache_size,
            args.cal_bandwidth,
        )
    ):
        results.append(
            Experiment(
                config,
                run_single_experiment(
                    config,
                    dynamic=args.dynamic,
                    max_autotune=args.max_autotune,
                ),
            )
        )

    print_results(results)


def heads_input_type(s):
    try:
        hq, hkv = map(int, s.split(","))
        return hq, hkv
    except Exception as e:
        raise argparse.ArgumentTypeError("Heads must be Hq,Hkv") from e


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

    parser.add_argument("-dtype", type=str, help="dtype", default="bfloat16")

    parser.add_argument(
        "-b", type=int, nargs="+", help="batch sizes", default=[2, 8, 16]
    )
    parser.add_argument(
        "-nh",
        type=heads_input_type,
        nargs="+",
        help="# of q-heads,kv-heads",
        default=[(16, 16), (16, 2)],
    )
    parser.add_argument(
        "-s", type=int, nargs="+", help="sequence lengths", default=[512, 1024, 4096]
    )
    parser.add_argument("-d", type=int, nargs="+", help="head dims", default=[64, 128])
    parser.add_argument(
        "-mods",
        type=str,
        nargs="+",
        help="score mods",
        default=["noop", "causal", "rel", "head_bias"],
    )
    parser.add_argument(
        "--max-autotune", action="store_true", help="Turn on max-autotune"
    )
    parser.add_argument(
        "--decoding",
        action="store_true",
        help="Benchmark Decoding (query sequence length = 1)",
    )
    parser.add_argument(
        "--kv-cache-size",
        type=int,
        nargs="+",
        required=False,
        help="""
key/value cache size in MiB.
Ignores -b batch size and calculate batch size from kv_cache size instead when specified.
""",
    )
    parser.add_argument(
        "--cal-bandwidth",
        action="store_true",
        help="Calculate kernel memory bandwidth & computational throughput. ",
    )

    # Parse arguments
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)

    main(args)
