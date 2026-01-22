"""
Benchmark comparing Flash Attention 3 vs Flash Attention 2 for SDPA.

This is an experimental benchmark for the FA3 backend.
Requires: Hopper GPU (SM90), flash_attn_interface module installed.

Usage:
    python benchmarks/transformer/sdpa_fa3.py
"""

import itertools
from collections import defaultdict
from dataclasses import asdict, dataclass

from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.utils.benchmark as benchmark
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
    sdpa_kernel,
    SDPBackend,
)
from torch.nn.functional import scaled_dot_product_attention


def benchmark_torch_function_in_microseconds(func, *args, **kwargs) -> float:
    """Benchmark a function and return the median time in microseconds."""
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
    seq_len: int
    head_dim: int
    dtype: torch.dtype
    is_causal: bool
    backend: str  # "FA2" or "FA3"
    device: torch.device = torch.device("cuda")

    def asdict(self):
        d = asdict(self)
        d["dtype"] = str(self.dtype).split(".")[-1]
        return d


@dataclass(frozen=True)
class ExperimentResults:
    forward_time_us: float
    tflops: float

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def asdict(self):
        return {**self.config.asdict(), **self.results.asdict()}


def calculate_tflops(config: ExperimentConfig, time_us: float) -> float:
    """Calculate TFLOPS for attention forward pass."""
    B = config.batch_size
    H = config.num_heads
    M = config.seq_len
    N = config.seq_len
    D = config.head_dim

    # Forward pass FLOPs: Q@K^T + softmax + attn@V
    qk_flops = M * N * D * 2
    softmax_flops = M * N * 2
    av_flops = M * N * D * 2
    total_flops = B * H * (qk_flops + softmax_flops + av_flops)

    # Apply causal sparsity (roughly 50% of attention computed)
    if config.is_causal:
        total_flops *= 0.5

    return total_flops / (time_us * 1e-6) / 1e12


def get_inputs(config: ExperimentConfig):
    """Generate Q, K, V tensors for the experiment."""
    shape = (config.batch_size, config.num_heads, config.seq_len, config.head_dim)
    q = torch.randn(shape, dtype=config.dtype, device=config.device)
    k = torch.randn(shape, dtype=config.dtype, device=config.device)
    v = torch.randn(shape, dtype=config.dtype, device=config.device)
    return q, k, v


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """Run a single benchmark experiment."""
    q, k, v = get_inputs(config)

    # Set up the backend
    if config.backend == "FA3":
        activate_flash_attention_impl("FA3")
    else:
        restore_flash_attention_impl()

    with torch.no_grad():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            forward_time = benchmark_torch_function_in_microseconds(
                scaled_dot_product_attention,
                q,
                k,
                v,
                is_causal=config.is_causal,
            )

    tflops = calculate_tflops(config, forward_time)
    return ExperimentResults(forward_time_us=forward_time, tflops=tflops)


def generate_configs() -> list[ExperimentConfig]:
    """Generate experiment configurations."""
    batch_sizes = [1, 8, 16]
    num_heads = [16, 32]
    seq_lens = [512, 1024, 2048, 4096, 8192]
    head_dims = [64, 128]
    dtypes = [torch.float16, torch.bfloat16]
    is_causal = [True, False]
    backends = ["FA2", "FA3"]

    configs = []
    for bsz, heads, seq_len, head_dim, dtype, causal, backend in itertools.product(
        batch_sizes, num_heads, seq_lens, head_dims, dtypes, is_causal, backends
    ):
        configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=heads,
                seq_len=seq_len,
                head_dim=head_dim,
                dtype=dtype,
                is_causal=causal,
                backend=backend,
            )
        )
    return configs


def print_results(experiments: list[Experiment]):
    """Print results as a formatted table."""
    table_data = defaultdict(list)
    for exp in experiments:
        for key, value in exp.asdict().items():
            if key != "device":
                table_data[key].append(value)
    print(tabulate(table_data, headers="keys", tablefmt="pretty", floatfmt=".2f"))


def print_comparison(experiments: list[Experiment]):
    """Print FA3 vs FA2 comparison showing speedup."""
    # Group by config (excluding backend)
    grouped = defaultdict(dict)
    for exp in experiments:
        key = (
            exp.config.batch_size,
            exp.config.num_heads,
            exp.config.seq_len,
            exp.config.head_dim,
            exp.config.dtype,
            exp.config.is_causal,
        )
        grouped[key][exp.config.backend] = exp.results

    # Build comparison table
    rows = []
    for key, backends in grouped.items():
        if "FA2" in backends and "FA3" in backends:
            fa2_time = backends["FA2"].forward_time_us
            fa3_time = backends["FA3"].forward_time_us
            speedup = fa2_time / fa3_time
            rows.append(
                {
                    "batch": key[0],
                    "heads": key[1],
                    "seq_len": key[2],
                    "head_dim": key[3],
                    "dtype": str(key[4]).split(".")[-1],
                    "causal": key[5],
                    "FA2 (us)": fa2_time,
                    "FA3 (us)": fa3_time,
                    "speedup": speedup,
                }
            )

    if rows:
        print("\n=== FA3 vs FA2 Comparison ===")
        table_data = defaultdict(list)
        for row in rows:
            for k, v in row.items():
                table_data[k].append(v)
        print(tabulate(table_data, headers="keys", tablefmt="pretty", floatfmt=".2f"))


def check_dependencies() -> bool:
    """Check if FA3 dependencies are available."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False

    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major != 9:
        print(f"FA3 requires Hopper (SM90), got SM{major}0")
        return False

    try:
        import flash_attn_interface  # noqa: F401
    except ImportError:
        print("flash_attn_interface not available")
        return False

    return True


def main():
    if not check_dependencies():
        return

    torch.manual_seed(42)

    configs = generate_configs()
    experiments = []

    print(f"Running {len(configs)} experiments...")
    for config in tqdm(configs):
        try:
            results = run_experiment(config)
            experiments.append(Experiment(config, results))
        except Exception as e:
            print(f"Failed: {config} - {e}")

    print_results(experiments)
    print_comparison(experiments)


if __name__ == "__main__":
    main()
