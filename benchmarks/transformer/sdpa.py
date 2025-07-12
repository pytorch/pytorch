import itertools
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable

from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.utils.benchmark as benchmark
from torch._inductor.utils import do_bench_using_profiling
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention


def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""

    def no_args():
        func(*args, **kwargs)

    time = do_bench_using_profiling(no_args)
    return time * 1e3


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
    kv_seq_len: int
    embed_dim: int
    is_causal: bool
    dtype: torch.dtype
    backend: SDPBackend
    device: torch.device = torch.device("cuda")

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def asdict(self):
        dict_obj = asdict(self)
        dict_obj["head_dim"] = self.head_dim
        return dict_obj


@dataclass(frozen=True)
class ExperimentResults:
    forward_time: float  # microseconds
    backward_time: float  # microseconds
    forward_tflops: float
    backward_tflops: float

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def asdict(self):
        dict1 = self.config.asdict()
        dict2 = self.results.asdict()
        return {**dict1, **dict2}


def calculate_tflops(
    config: ExperimentConfig,
    time_us: float,
    is_backward: bool = False,
    sparsity: float = 0.0,
) -> float:
    """
    Calculate TFLOPS for scaled dot product attention.

    Parameters:
    - config: The experiment configuration
    - time_us: The execution time in microseconds
    - is_backward: Whether to calculate for backward pass (includes gradient computation)
    - sparsity: Sparsity factor between 0.0 and 1.0, where 0.0 means no sparsity and 1.0 means fully sparse

    Returns:
    - TFLOPS value
    """
    B = config.batch_size
    H = config.num_heads
    M = config.q_seq_len
    N = config.kv_seq_len
    D = config.head_dim

    # Calculate density factor (1.0 - sparsity)
    density = 1.0 - sparsity

    # Forward pass FLOPs
    qk_flops = (
        M * N * D * 2
    )  # Q*K^T matmul: (M,D) @ (D,N) with 2 FLOPs per multiply-add
    softmax_flops = M * N * 2  # Softmax operations (exp and div)
    av_flops = (
        M * N * D * 2
    )  # Attention @ V: (M,N) @ (N,D) with 2 FLOPs per multiply-add

    total_flops = B * H * (qk_flops + softmax_flops + av_flops)

    # Apply density factor to account for sparsity
    total_flops *= density

    # For backward pass flash uses 2.5x more flops will use this
    if is_backward:
        total_flops *= 2.5

    # Convert to TFLOPS: flops / (time_us * 1e-6) / 1e12
    tflops = total_flops / (time_us * 1e-6) / 1e12

    return tflops


def get_input(
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(
        (config.batch_size, config.num_heads, config.q_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    k = torch.randn(
        (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    v = torch.randn(
        (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    return q, k, v


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    q, k, v = get_input(config)
    is_causal = config.is_causal
    context = (
        sdpa_kernel(config.backend) if config.backend is not None else nullcontext()
    )
    with context:
        forward_time = benchmark_cuda_function_in_microseconds(
            scaled_dot_product_attention,
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=None,
        )
        out_torch = scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, attn_mask=None
        )
        d_out = torch.randn_like(out_torch)
        backward_time = benchmark_cuda_function_in_microseconds(
            out_torch.backward, d_out, retain_graph=True
        )

    # Calculate TFLOPS for forward and backward passes
    sparsity = 0.5 if is_causal else 0.0
    forward_tflops = calculate_tflops(config, forward_time, sparsity=sparsity)
    backward_tflops = calculate_tflops(
        config, backward_time, is_backward=True, sparsity=sparsity
    )

    return ExperimentResults(
        forward_time=forward_time,
        backward_time=backward_time,
        forward_tflops=forward_tflops,
        backward_tflops=backward_tflops,
    )


def print_results(experiments: list[Experiment]):
    table_data = defaultdict(list)
    for experiment in experiments:
        for key, value in experiment.asdict().items():
            table_data[key].append(value)
    del table_data["device"]
    if table_data["backend"][0] is None:
        del table_data["backend"]
    print(tabulate(table_data, headers="keys", tablefmt="pretty", floatfmt=".3f"))


def write_results_to_csv(
    experiments: list[Experiment], output_dir: str = "benchmark_results"
):
    """
    Write experiment results to a CSV file in the specified directory.
    The filename includes a timestamp for uniqueness.
    """
    import csv
    import os
    from datetime import datetime

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"benchmark_results_{timestamp}.csv")

    # Get all fields from the first experiment
    if not experiments:
        return

    fieldnames = list(experiments[0].asdict().keys())
    if "device" in fieldnames:
        fieldnames.remove("device")  # Remove device field as it's always cuda

    # Write results to CSV
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in experiments:
            row = experiment.asdict()
            if "device" in row:
                del row["device"]  # Remove device field
            writer.writerow(row)

    print(f"Results written to: {filename}")


def generate_experiment_configs() -> list[ExperimentConfig]:
    batch_sizes = [1, 8, 16]
    num_heads = [16]
    q_kv_seq_lens = [(128, 128), (256, 256), (512, 512), (1024, 1024), (8192, 8192)]
    embed_dims = [2048]
    backends = [None]  # If set to None, all backends are enabled
    dtypes = [
        torch.bfloat16,
    ]
    is_causal = [True, False]
    all_configs = []
    for (
        bsz,
        heads,
        (q_seq_len, kv_seq_len),
        embed_dim,
        causal,
        dtype,
        backend,
    ) in itertools.product(
        batch_sizes, num_heads, q_kv_seq_lens, embed_dims, is_causal, dtypes, backends
    ):
        all_configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=heads,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
                embed_dim=embed_dim,
                is_causal=causal,
                dtype=dtype,
                backend=backend,
            )
        )

    return all_configs


def main():
    seed = 123
    torch.manual_seed(seed)
    results = []
    for config in tqdm(generate_experiment_configs()):
        results.append(Experiment(config, run_single_experiment(config)))

    print_results(results)
    write_results_to_csv(results, "../benchmark_results")


if __name__ == "__main__":
    main()
