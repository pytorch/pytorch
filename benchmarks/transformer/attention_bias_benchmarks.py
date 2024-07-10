import itertools
from dataclasses import asdict, dataclass
from enum import auto, Enum
from functools import partial
from typing import Callable, List, Optional, Union

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch.nn.attention.bias import CausalBias, CausalVariant, SlidingWindowBias
from torch.nn.parameter import Parameter


class BiasType(Enum):
    CAUSAL_SUBCLASS = auto()
    SLIDING_WINDOW = auto()


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    return t0.adaptive_autorange(min_run_time=0.1).median * 1e6


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int
    num_heads: int
    q_seq_len: int
    k_seq_len: int
    embed_dim: int
    dtype: torch.dtype
    bias_type: BiasType
    window_size_left: Optional[int] = None
    window_size_right: Optional[int] = None

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def asdict(self):
        dict_obj = asdict(self)
        dict_obj["head_dim"] = self.head_dim
        dict_obj["bias_type"] = self.bias_type.name
        return dict_obj


@dataclass(frozen=True)
class ExperimentResults:
    materialized_time: float
    bias_time: float

    def get_entries(self) -> List[float]:
        return [self.materialized_time, self.bias_time]

    def get_speedup(self) -> float:
        return (
            self.materialized_time / self.bias_time
            if self.bias_time != 0
            else float("inf")
        )


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def get_entries(self) -> List:
        return (
            [*self.config.asdict().values()]
            + self.results.get_entries()
            + [self.results.get_speedup()]
        )


class CompositeMHA(torch.nn.Module):
    def __init__(self, num_heads, embed_dim, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj_weight = Parameter(
            torch.empty((embed_dim, embed_dim), **factory_kwargs)
        )
        self.k_proj_weight = Parameter(
            torch.empty((embed_dim, embed_dim), **factory_kwargs)
        )
        self.v_proj_weight = Parameter(
            torch.empty((embed_dim, embed_dim), **factory_kwargs)
        )
        self.out_proj = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.num_heads = num_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Union[torch.Tensor, CausalBias, SlidingWindowBias],
    ):
        query_projected = F.linear(query, self.q_proj_weight)
        key_projected = F.linear(key, self.k_proj_weight)
        value_projected = F.linear(value, self.v_proj_weight)

        query = query.view(
            query_projected.size(0), -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            key_projected.size(0), -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            value_projected.size(0), -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
        )

        attn = attn.transpose(1, 2).reshape(query.size(0), -1, self.embed_dim)
        return F.linear(attn, self.out_proj)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.constant_(self.out_proj, 0.0)


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    device = torch.device("cuda")
    composite_mha = CompositeMHA(
        config.num_heads, config.embed_dim, device, config.dtype
    )
    composite_mha.reset_parameters()
    query, key, value = generate_inputs(
        config.batch_size,
        config.q_seq_len,
        config.k_seq_len,
        config.embed_dim,
        config.dtype,
        device,
    )

    if config.bias_type == BiasType.CAUSAL_SUBCLASS:
        bias_mask = CausalBias(
            CausalVariant.LOWER_RIGHT, config.q_seq_len, config.k_seq_len
        )
    elif config.bias_type == BiasType.SLIDING_WINDOW:
        bias_mask = SlidingWindowBias(
            CausalVariant.LOWER_RIGHT,
            config.window_size_left,
            config.window_size_right,
            config.q_seq_len,
            config.k_seq_len,
        )
    else:
        raise ValueError(f"Unknown bias type: {config.bias_type}")

    materialized_mask = bias_mask._materialize(device)

    materialized_time = benchmark_torch_function_in_microseconds(
        composite_mha, query, key, value, materialized_mask
    )
    bias_time = benchmark_torch_function_in_microseconds(
        composite_mha, query, key, value, bias_mask
    )

    return ExperimentResults(materialized_time=materialized_time, bias_time=bias_time)


def generate_inputs(
    batch_size, q_sequence_length, kv_sequence_length, embed_dim, dtype, device
):
    q_shape = (batch_size, q_sequence_length, embed_dim)
    kv_shape = (batch_size, kv_sequence_length, embed_dim)

    make_q = partial(torch.rand, q_shape, device=device, dtype=dtype)
    make_kv = partial(torch.rand, kv_shape, device=device, dtype=dtype)
    return make_q(), make_kv(), make_kv()


def generate_experiment_configs() -> List[ExperimentConfig]:
    batch_sizes = [1, 16]
    num_heads = [16, 32]
    q_kv_seq_lens = [(512, 4097), (1024, 2048), (1, 2048)]
    embed_dims = [2048, 4096]
    dtypes = [torch.bfloat16]
    bias_types = [BiasType.CAUSAL_SUBCLASS, BiasType.SLIDING_WINDOW]
    window_sizes = [(1024, 0), (4096, 32)]

    all_configs = []
    for (
        bsz,
        heads,
        (q_seq_len, kv_seq_len),
        embed_dim,
        dtype,
        bias_type,
    ) in itertools.product(
        batch_sizes, num_heads, q_kv_seq_lens, embed_dims, dtypes, bias_types
    ):
        if bias_type == BiasType.SLIDING_WINDOW:
            for window_left, window_right in window_sizes:
                all_configs.append(
                    ExperimentConfig(
                        batch_size=bsz,
                        num_heads=heads,
                        q_seq_len=q_seq_len,
                        k_seq_len=kv_seq_len,
                        embed_dim=embed_dim,
                        dtype=dtype,
                        bias_type=bias_type,
                        window_size_left=window_left,
                        window_size_right=window_right,
                    )
                )
        else:
            all_configs.append(
                ExperimentConfig(
                    batch_size=bsz,
                    num_heads=heads,
                    q_seq_len=q_seq_len,
                    k_seq_len=kv_seq_len,
                    embed_dim=embed_dim,
                    dtype=dtype,
                    bias_type=bias_type,
                )
            )

    return all_configs


def print_results(results: List[Experiment]):
    # Group results by bias type
    grouped_results = {}
    for experiment in results:
        bias_type = experiment.config.bias_type
        if bias_type not in grouped_results:
            grouped_results[bias_type] = []
        grouped_results[bias_type].append(experiment)

    # Calculate and print statistics for each bias type
    for bias_type, experiments in grouped_results.items():
        print(f"\nResults for {bias_type.name}:")
        speedups = [exp.results.get_speedup() for exp in experiments]
        avg_speedup = np.mean(speedups)
        min_speedup = np.min(speedups)
        max_speedup = np.max(speedups)

        min_config = experiments[np.argmin(speedups)].config.asdict()
        max_config = experiments[np.argmax(speedups)].config.asdict()

        table_data = [
            {
                "Metric": "Average Speedup",
                "Value": f"{avg_speedup:.2f}x",
                **dict.fromkeys(min_config),
            },
            {"Metric": "Min Speedup", "Value": f"{min_speedup:.2f}x", **min_config},
            {"Metric": "Max Speedup", "Value": f"{max_speedup:.2f}x", **max_config},
        ]

        print(tabulate(table_data, headers="keys", tablefmt="pretty"))

    # Print detailed results
    print("\nDetailed Results:")
    headers = list(results[0].config.asdict().keys()) + [
        "Materialized Time (µs)",
        "Bias Time (µs)",
        "Speedup",
    ]
    table_data = [exp.get_entries() for exp in results]
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))


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
