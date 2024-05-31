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


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    return do_bench(lambda: func(*args, **kwargs)) * 1e3


@dataclass(frozen=True)
class ExperimentConfig:
    shape: Tuple[int]
    score_mod: Callable
    dtype: torch.dtype
    calculate_bwd_time: bool

    def __post_init__(self):
        assert len(self.shape) == 4, "Shape must be of length 4"

    def asdict(self):
        # Convert the dataclass instance to a dictionary
        d = asdict(self)
        # Remove the 'calculate_bwd_time' key
        d.pop("calculate_bwd_time", None)
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
    num_heads: int,
    q_sequence_length: int,
    kv_sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
):
    q_shape = (batch_size, q_sequence_length, num_heads * head_dim)
    kv_shape = (batch_size, kv_sequence_length, num_heads * head_dim)

    make_q = partial(
        torch.rand, q_shape, device=device, dtype=dtype, requires_grad=requires_grad
    )
    make_kv = partial(
        torch.rand, kv_shape, device=device, dtype=dtype, requires_grad=requires_grad
    )
    query = (
        make_q()
        .view(batch_size, q_sequence_length, num_heads, head_dim)
        .transpose(1, 2)
    )
    key = (
        make_kv()
        .view(batch_size, kv_sequence_length, num_heads, head_dim)
        .transpose(1, 2)
    )
    value = (
        make_kv()
        .view(batch_size, kv_sequence_length, num_heads, head_dim)
        .transpose(1, 2)
    )
    return query, key, value


def run_single_experiment(config: ExperimentConfig, dynamic=False) -> ExperimentResults:
    device = torch.device("cuda")
    batch_size, num_heads, q_seq_len, head_dim = config.shape
    query, key, value = generate_inputs(
        batch_size,
        num_heads,
        q_seq_len,
        q_seq_len,
        head_dim,
        config.dtype,
        device,
        requires_grad=config.calculate_bwd_time,
    )

    def eager_sdpa(query, key, value, _):
        return F.scaled_dot_product_attention(query, key, value)

    compiled_sdpa = torch.compile(_flex_attention, dynamic=dynamic)

    score_mod = config.score_mod

    forward_eager_time = benchmark_torch_function_in_microseconds(
        eager_sdpa, query, key, value, score_mod
    )
    forward_compiled_time = benchmark_torch_function_in_microseconds(
        compiled_sdpa, query, key, value, score_mod
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


def get_func_name(func):
    return func.__name__.split("<locals>.")[-1].split(" at ")[0]


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


def generate_score_mods() -> List[Callable]:
    def noop(score, b, h, m, n):
        return score

    def causal_mask(score, b, h, token_q, token_kv):
        return torch.where(token_q >= token_kv, score, float("-inf"))

    def relative_bias(score, b, h, m, n):
        return score + (m - n)

    def head_bias(score, b, h, m, n):
        return score + 2 * h

    return [noop, causal_mask, relative_bias, head_bias]


def generate_experiment_configs(calculate_bwd: bool) -> List[ExperimentConfig]:
    batch_sizes = [2, 8, 16]
    num_heads = [16]
    q_kv_seq_lens = [(512, 512), (1024, 1024), (4096, 4096)]
    head_dims = [64, 128, 256]
    dtypes = [
        torch.bfloat16,
    ]
    score_mods = generate_score_mods()
    all_configs = []
    for (
        bsz,
        n_heads,
        (q_seq_len, kv_seq_len),
        head_dim,
        score_mod,
        dtype,
    ) in itertools.product(
        batch_sizes, num_heads, q_kv_seq_lens, head_dims, score_mods, dtypes
    ):
        assert q_seq_len == kv_seq_len, "Only equal length inputs supported for now."
        all_configs.append(
            ExperimentConfig(
                shape=(bsz, n_heads, q_seq_len, head_dim),
                score_mod=score_mod,
                dtype=dtype,
                calculate_bwd_time=calculate_bwd,
            )
        )

    return all_configs


def main(dynamic: bool, calculate_bwd: bool):
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    results = []
    for config in tqdm(generate_experiment_configs(calculate_bwd)):
        results.append(
            Experiment(config, run_single_experiment(config, dynamic=dynamic))
        )
    for config in tqdm(generate_experiment_configs(calculate_bwd)):
        results.append(Experiment(config, run_single_experiment(config)))

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

    # Parse arguments
    args = parser.parse_args()

    main(args.dynamic, args.calculate_bwd)
