import itertools
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate
from torch.nn.attention._templated_attention import _templated_attention
from tqdm import tqdm

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
    batch_size: int
    num_heads: int
    q_seq_len: int
    k_seq_len: int
    head_dim: int
    score_mod: Callable
    dtype: torch.dtype

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResults:
    eager_time: float
    compiled_time: float

    def get_entries(self) -> List:
        return [
            f"{self.eager_time:2f}",
            f"{self.compiled_time:2f}",
        ]


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def get_entries(self) -> List:
        return self.config.get_entries() + self.results.get_entries()

    def asdict(self):
        dict1 = asdict(self.config)
        dict2 = asdict(self.results)
        return {**dict1, **dict2}


def generate_inputs(
    batch_size,
    num_heads,
    q_sequence_length,
    kv_sequence_length,
    head_dim,
    dtype,
    device,
):
    q_shape = (batch_size, q_sequence_length, num_heads * head_dim)
    kv_shape = (batch_size, kv_sequence_length, num_heads * head_dim)

    make_q = partial(torch.rand, q_shape, device=device, dtype=dtype)
    make_kv = partial(torch.rand, kv_shape, device=device, dtype=dtype)
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


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    device = torch.device("cuda")
    query, key, value = generate_inputs(
        config.batch_size,
        config.num_heads,
        config.q_seq_len,
        config.k_seq_len,
        config.head_dim,
        config.dtype,
        device,
    )

    def eager_sdpa(query, key, value, _):
        return F.scaled_dot_product_attention(query, key, value)

    compiled_sdpa = torch.compile(_templated_attention)

    score_mod = config.score_mod

    forward_eager_time = benchmark_torch_function_in_microseconds(
        eager_sdpa, query, key, value, score_mod
    )
    forward_compiled_time = benchmark_torch_function_in_microseconds(
        compiled_sdpa, query, key, value, score_mod
    )

    return ExperimentResults(
        eager_time=forward_eager_time,
        compiled_time=forward_compiled_time,
    )


def calculate_speedup(results: ExperimentResults) -> float:
    return results.eager_time / results.compiled_time


def get_func_name(func):
    return func.__name__.split("<locals>.")[-1].split(" at ")[0]


def get_average_speedups(results: List[Experiment]):
    # Calculate speedups
    speedups = [calculate_speedup(r.results) for r in results]

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
            if key == "eager_time" or key == "compiled_time":
                value = float(value)
            table_data[key].append(value)

    # Calculate speedups
    speedups = [calculate_speedup(r.results) for r in results]
    table_data["speedup"] = speedups

    table_data["score_mod"] = [get_func_name(func) for func in table_data["score_mod"]]
    print(tabulate(table_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    average_data = get_average_speedups(results)
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


def generate_experiment_configs() -> List[ExperimentConfig]:
    batch_sizes = [1, 8, 16]
    num_heads = [16]
    q_kv_seq_lens = [(512, 512), (1024, 1024), (4096, 4096)]
    head_dims = [64]
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
        all_configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=n_heads,
                q_seq_len=q_seq_len,
                k_seq_len=kv_seq_len,
                head_dim=head_dim,
                score_mod=score_mod,
                dtype=dtype,
            )
        )

    return all_configs


def main():
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    results = []
    for config in tqdm(generate_experiment_configs()):
        results.append(Experiment(config, run_single_experiment(config)))

    print_results(results)


if __name__ == "__main__":
    main()
