import itertools
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, List, Tuple
from enum import Enum

from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

import torch
import torch.utils.benchmark as benchmark
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    return do_bench(lambda: func(*args, **kwargs)) * 1e3


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int
    q_num_heads: int
    kv_num_heads: int
    q_seq_len: int
    kv_seq_len: int
    embed_dim: int
    is_causal: bool
    dtype: torch.dtype
    backend: SDPBackend
    device: torch.device = torch.device("cuda")
    enable_gqa: bool = False
    tensor_repeat_interleave: bool = False

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.q_num_heads

    def asdict(self):
        dict_obj = asdict(self)
        dict_obj["head_dim"] = self.head_dim
        return dict_obj


@dataclass(frozen=True)
class ExperimentResults:
    forward_time: float
    backward_time: float

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def asdict(self):
        dict1 = asdict(self.config)
        dict2 = asdict(self.results)
        return {**dict1, **dict2}


def get_input(
    config: ExperimentConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(
        (config.batch_size, config.q_num_heads, config.q_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    k = torch.randn(
        (config.batch_size, config.kv_num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    v = torch.randn(
        (config.batch_size, config.kv_num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    return q, k, v


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    q, k, v = get_input(config)
    is_causal = config.is_causal
    enable_gqa = config.enable_gqa
    context = (
        sdpa_kernel(config.backend) if config.backend is not None else nullcontext()
    )
    with context:
        forward_time = benchmark_torch_function_in_microseconds(
            scaled_dot_product_attention,
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=None,
            enable_gqa=enable_gqa,
        )
        out_torch = scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, attn_mask=None, enable_gqa=enable_gqa
        )
        dOut = torch.randn_like(out_torch)
        backward_time = benchmark_torch_function_in_microseconds_(
            out_torch.backward, dOut, retain_graph=True
        )
    return ExperimentResults(
        forward_time=forward_time + kv_repeat_time,
        backward_time=backward_time,
    )


def generate_experiment_configs(experiment: ExperimentName) -> List[ExperimentConfig]:
    batch_sizes = [
        1,
        8,
    ]
    num_heads = [16]
    q_kv_seq_lens = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    embed_dims = [2048]
    backends = [None]  # If set to None, all backends are enabled
    dtypes = [
        torch.bfloat16,
    ]
    is_causal = [True, False]
    all_configs = []
    if experiment == ExperimentName.SDPA:
        for (
            bsz,
            heads,
            (q_seq_len, kv_seq_len),
            embed_dim,
            causal,
            dtype,
            backend,
        ) in itertools.product(
            batch_sizes,
            num_heads,
            q_kv_seq_lens,
            embed_dims,
            is_causal,
            dtypes,
            backends,
        ):
            all_configs.append(
                ExperimentConfig(
                    batch_size=bsz,
                    q_num_heads=heads,
                    kv_num_heads=heads,
                    q_seq_len=q_seq_len,
                    kv_seq_len=kv_seq_len,
                    embed_dim=embed_dim,
                    is_causal=causal,
                    dtype=dtype,
                    backend=backend,
                )
            )

    elif experiment == ExperimentName.GQA:
        # Added GQA test-cases
        gqa_batch_sizes = [1, 8, 16, 32]
        gqa_num_heads = [(32, 8), (64, 4), (128, 16), (256, 32), (512, 64), (1024, 128)]
        gqa_backends = [SDPBackend.FLASH_ATTENTION]
        gqa_q_kv_seq_lens = [(2048, 2048)]
        for (
            bsz,
            (q_heads, kv_heads),
            (q_seq_len, kv_seq_len),
            embed_dim,
            dtype,
            backend,
        ) in itertools.product(
            gqa_batch_sizes,
            gqa_num_heads,
            gqa_q_kv_seq_lens,
            embed_dims,
            dtypes,
            gqa_backends,
        ):
            all_configs.append(
                ExperimentConfig(
                    batch_size=bsz,
                    q_num_heads=q_heads,
                    kv_num_heads=kv_heads,
                    q_seq_len=q_seq_len,
                    kv_seq_len=kv_seq_len,
                    embed_dim=embed_dim,
                    is_causal=True,
                    dtype=dtype,
                    backend=backend,
                    enable_gqa=True,
                )
            )
            all_configs.append(
                ExperimentConfig(
                    batch_size=bsz,
                    q_num_heads=q_heads,
                    kv_num_heads=kv_heads,
                    q_seq_len=q_seq_len,
                    kv_seq_len=kv_seq_len,
                    embed_dim=embed_dim,
                    is_causal=True,
                    dtype=dtype,
                    backend=backend,
                    enable_gqa=False,
                    tensor_repeat_interleave=True,
                )
            )
    return all_configs


def print_results(experiments: List[Experiment]):
    table_data = defaultdict(list)
    for experiment in experiments:
        for key, value in experiment.asdict().items():
            table_data[key].append(value)
    del table_data["device"]
    if table_data["backend"][0] is None:
        del table_data["backend"]
    print(tabulate(table_data, headers="keys", tablefmt="pretty", floatfmt=".3f"))


def main():
    seed = 123
    torch.manual_seed(seed)
    print("Experiment 1: SDPA experiments")
    results_exp1 = []
    for config in tqdm(generate_experiment_configs(ExperimentName.SDPA)):
        results_exp1.append(Experiment(config, run_single_experiment(config)))
    print_results(results_exp1)
    print("Experiment 2: GQA experiments")
    results_exp2 = []
    for config in tqdm(generate_experiment_configs(ExperimentName.GQA)):
        results_exp2.append(Experiment(config, run_single_experiment(config)))
    print_results(results_exp2)


if __name__ == "__main__":
    main()
