import torch
import itertools
import numpy as np
import random
import argparse
from pathlib import Path
import torch.utils.benchmark as benchmark
from dataclasses import dataclass
from typing import Optional, List
from pprint import pprint
from torch.backends.cuda import sdp_kernel
from tqdm import tqdm
from prettytable import PrettyTable

import warnings

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int
    num_heads: int
    max_sequence_len: int
    embed_dimension: int
    dtype: torch.dtype
    pad_percentage: Optional[float]
    enable_math: bool
    enable_flash: bool
    enable_mem_efficient: bool

    def get_entries(self) -> List:
        return [
            self.batch_size,
            self.num_heads,
            self.max_sequence_len,
            self.embed_dimension,
            self.dtype,
            self.pad_percentage,
            self.enable_math,
            self.enable_flash,
            self.enable_mem_efficient,
        ]

    @classmethod
    def get_entry_names(cls) -> List[str]:
        return [
            "batch_size",
            "num_heads",
            "max_sequence_len",
            "embed_dimension",
            "dtype",
            "pad_percentage",
            "enable_math",
            "enable_flash",
            "enable_mem_efficient",
        ]


@dataclass(frozen=True)
class ExperimentResults:
    nn_mha_time: float
    compiled_nn_mha_time: Optional[float]
    composite_mha_time: float
    compiled_composite_mha_time: Optional[float]

    def get_entries(self) -> List:
        return [
            f"{self.nn_mha_time:2f}",
            f"{self.compiled_nn_mha_time:2f}" if self.compiled_nn_mha_time else None,
            f"{self.composite_mha_time:2f}",
            f"{self.compiled_composite_mha_time:2f}" if self.compiled_composite_mha_time else None,
        ]

    @classmethod
    def get_entry_names(cls) -> List[str]:
        return [
            "nn_mha_time (μs)",
            "compiled_nn_mha_time (μs)",
            "composite_mha_time (μs)",
            "compiled_composite_mha_time (μs)",
        ]


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def get_entries(self) -> List:
        return self.config.get_entries() + self.results.get_entries()


class CompositeMHA(torch.nn.Module):
    def __init__(self, num_heads, in_proj_weight, in_proj_bias, out_proj):
        super().__init__()
        self.in_proj_weight = in_proj_weight
        self.in_proj_bias = in_proj_bias
        self.out_proj = out_proj
        self.num_heads = num_heads

    def forward(self, query, key, value, mask):
        if not (query is key and key is value):
            raise NotImplementedError(
                "query, key and value must be the same Tensor for now."
            )
        if mask is not None:
            raise NotImplementedError("mask is currently not supported.")

        query_projected = torch.nn.functional.linear(
            query, self.in_proj_weight, self.in_proj_bias
        )

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)

        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        attn = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        attn = attn.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)
        # Match return signature of nn.MHA
        return self.out_proj(attn), None


def build_composite_mha_from_nn_mha(pt):
    assert pt._qkv_same_embed_dim
    in_proj_weight = pt.in_proj_weight
    assert in_proj_weight is not None
    assert pt.batch_first
    return CompositeMHA(pt.num_heads, pt.in_proj_weight, pt.in_proj_bias, pt.out_proj)


def generate_rand_batch(
    batch_size,
    max_sequence_len,
    embed_dimension,
    pad_percentage=None,
    dtype=torch.float16,
    device="cuda",
):
    if not pad_percentage:
        return (
            torch.randn(
                batch_size,
                max_sequence_len,
                embed_dimension,
                dtype=dtype,
                device=device,
            ),
            None,
        )
    # Really slow but should work
    seq_len_list = [
        int(max_sequence_len * (1 - random.gauss(pad_percentage, 0.01)))
        for _ in range(batch_size)
    ]
    # Make random ele max length
    seq_len_list[random.randint(0, batch_size - 1)] = max_sequence_len
    # print(f"Theoretical padding: {pad_percentage} actual: {1 - (sum(seq_len_list) / (batch_size * max_sequence_len))}")
    return (
        torch.nested.nested_tensor(
            [
                torch.randn(seq_len, embed_dimension, dtype=dtype, device=device)
                for seq_len in seq_len_list
            ]
        ),
        seq_len_list,
    )


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


def assert_close_tensors(tensor_a, tensor_b):
    # First order sanity check. Not a replacement for rigorous tests.
    if tensor_a.is_nested and tensor_b.is_nested:
        for a, b in zip(tensor_a.unbind(), tensor_b.unbind()):
            assert torch.allclose(a, b, atol=1e-2, rtol=1e-2)
    else:
        assert torch.allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3)


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    with sdp_kernel(
        enable_math=config.enable_math,
        enable_flash=config.enable_flash,
        enable_mem_efficient=config.enable_mem_efficient,
    ) as kernel_choice, torch.inference_mode() as inference_mode:
        dropout_p = 0.0
        mask = None

        nn_mha = torch.nn.MultiheadAttention(
            embed_dim=config.embed_dimension,
            num_heads=config.num_heads,
            batch_first=True,
            dropout=dropout_p,
        )
        nn_mha = nn_mha.eval().to("cuda", config.dtype)
        composite_mha = build_composite_mha_from_nn_mha(nn_mha)
        qkv, lengths = generate_rand_batch(
            config.batch_size,
            config.max_sequence_len,
            config.embed_dimension,
            config.pad_percentage,
            config.dtype,
        )
        nn_mha_output, _ = nn_mha(qkv, qkv, qkv, mask)
        composite_mha_output, _ = composite_mha(qkv, qkv, qkv, mask)

        # First order sanity check
        assert_close_tensors(nn_mha_output, composite_mha_output)

        nn_mha_time = benchmark_torch_function_in_microseconds(
            nn_mha, qkv, qkv, qkv, mask
        )
        composite_mha_time = benchmark_torch_function_in_microseconds(
            composite_mha, qkv, qkv, qkv, mask
        )

        # TorchDynamo will error on NestedTensors
        if config.pad_percentage is None:
            compiled_nn_mha = torch.compile(nn_mha)
            compiled_composite_mha = torch.compile(composite_mha)

            compiled_nn_mha_time = benchmark_torch_function_in_microseconds(
                compiled_nn_mha, qkv, qkv, qkv, mask
            )

            compiled_composite_mha_time = benchmark_torch_function_in_microseconds(
                compiled_composite_mha, qkv, qkv, qkv, mask,
            )
        else:
            compiled_nn_mha_time = None
            compiled_composite_mha_time = None

        results = ExperimentResults(
            nn_mha_time,
            compiled_nn_mha_time,
            composite_mha_time,
            compiled_composite_mha_time,
        )
        return Experiment(config, results)


# Could return generator
def generate_experiments(
    batch_sizes, num_heads, max_seq_lens, embed_dims, dtypes, pad_percentages
) -> List[ExperimentConfig]:
    configs = []
    for bsz, n_heads, seq_len, embed_dim, dtype, padding in itertools.product(
        batch_sizes, num_heads, max_seq_lens, embed_dims, dtypes, pad_percentages
    ):
        configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=n_heads,
                max_sequence_len=seq_len,
                embed_dimension=embed_dim,
                dtype=dtype,
                pad_percentage=padding,
                enable_math=False,
                enable_flash=True,
                enable_mem_efficient=True,
            )
        )
    return configs


def main(save_path: Optional[Path]):
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Run one timing experiment comparing nn_mha vs composite_mha
    config = ExperimentConfig(
        batch_size=128,
        num_heads=8,
        max_sequence_len=512,
        embed_dimension=512,
        dtype=torch.float16,
        pad_percentage=None,
        enable_math=False,
        enable_flash=True,
        enable_mem_efficient=True,
    )

    experiment = run_single_experiment(config)
    pprint(experiment)

    table = PrettyTable()
    table.float_format = ".3"
    table.field_names = (
        ExperimentConfig.get_entry_names() + ExperimentResults.get_entry_names()
    )

    # Run a bunch of experiments
    batch_sizes = [256]
    num_heads = [32]
    max_seq_lens = [256]
    embed_dims = [512]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    pad_percentages = [None, 0.9]

    experiment_configs = generate_experiments(
        batch_sizes, num_heads, max_seq_lens, embed_dims, dtypes, pad_percentages
    )

    experiments: List[Experiment] = []
    for experiment_config in tqdm(experiment_configs):
        experiment = run_single_experiment(experiment_config)
        experiments.append(experiment)
        table.add_row(experiment.get_entries())

    print(table)

    csv_string = table.get_csv_string()
    if save_path is not None:
        with open(save_path, "w") as csvfile:
            csvfile.write(csv_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", "--save_path", type=str, help="Path to save the results")

    args = parser.parse_args()
    save_path = Path(args.save_path) if args.save_path else None
    main(save_path)
