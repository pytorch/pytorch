#!/usr/bin/env python3
"""
Benchmark script for static dispatch POC using TorchBench models.

This script measures the performance difference between dynamic and static
dispatch modes when using make_fx for tracing with real TorchBench models.

Usage:
    # Run with default models
    python benchmarks/static_dispatch_benchmark.py

    # Run with specific model
    python benchmarks/static_dispatch_benchmark.py --only resnet50

    # Run with all available torchbench models
    python benchmarks/static_dispatch_benchmark.py --torchbench-all

Metrics measured:
    - Cold start time: First trace of a model
    - Warm start time: Subsequent traces
    - Ops dispatched: Number of ops traced
    - Time per op: Average time per traced op
"""

import argparse
import gc
import importlib
import os
import sys
import time
from collections import namedtuple
from os.path import abspath, exists
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx


# ============================================================================
# TorchBench Integration
# ============================================================================

def setup_torchbench_cwd() -> str:
    """Setup TorchBench working directory."""
    original_dir = abspath(os.getcwd())

    os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
    for torchbench_dir in (
        "./torchbenchmark",
        "../torchbenchmark",
        "../torchbench",
        "../benchmark",
        "../../torchbenchmark",
        "../../torchbench",
        "../../benchmark",
        "../../../torchbenchmark",
        "../../../torchbench",
        "../../../benchmark",
    ):
        if exists(torchbench_dir):
            break

    if exists(torchbench_dir):
        torchbench_dir = abspath(torchbench_dir)
        os.chdir(torchbench_dir)
        sys.path.append(torchbench_dir)

    return original_dir


def load_torchbench_model(
    model_name: str,
    device: str = "cpu",
    batch_size: Optional[int] = None,
) -> Tuple[nn.Module, Tuple[Any, ...]]:
    """
    Load a model from TorchBench.

    Args:
        model_name: Name of the model to load
        device: Device to load the model on
        batch_size: Optional batch size override

    Returns:
        Tuple of (model, example_inputs)
    """
    candidates = [
        f"torchbenchmark.models.{model_name}",
        f"torchbenchmark.canary_models.{model_name}",
        f"torchbenchmark.models.fb.{model_name}",
    ]
    for c in candidates:
        try:
            module = importlib.import_module(c)
            break
        except ModuleNotFoundError as e:
            if e.name != c:
                raise
    else:
        raise ImportError(f"Could not import any of {candidates}")

    benchmark_cls = getattr(module, "Model", None)
    if benchmark_cls is None:
        raise NotImplementedError(f"{model_name}.Model is None")

    if not hasattr(benchmark_cls, "name"):
        benchmark_cls.name = model_name

    # Create benchmark in eval mode
    benchmark = benchmark_cls(
        test="eval",
        device=device,
        batch_size=batch_size,
    )

    model, example_inputs = benchmark.get_module()
    model.eval()

    return model, example_inputs


def list_torchbench_models() -> List[str]:
    """List available TorchBench models."""
    try:
        from torchbenchmark import _list_model_paths
        return [os.path.basename(p) for p in _list_model_paths()]
    except ImportError:
        # Fallback to common models
        return [
            "resnet50",
            "resnet18",
            "alexnet",
            "vgg16",
            "mobilenet_v2",
            "shufflenet_v2_x1_0",
            "squeezenet1_1",
        ]


# ============================================================================
# Fallback Models (when TorchBench is not available)
# ============================================================================

def simple_mlp(x: torch.Tensor) -> torch.Tensor:
    """Simple MLP for basic benchmarking."""
    x = torch.relu(torch.matmul(x, torch.randn(64, 128)))
    x = torch.relu(torch.matmul(x, torch.randn(128, 64)))
    x = torch.matmul(x, torch.randn(64, 10))
    return x


class ResNetBlock(nn.Module):
    """Simple ResNet-style block for benchmarking."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)


def create_simple_resnet() -> Tuple[nn.Module, Tuple[torch.Tensor]]:
    """Create a simple ResNet-like model for benchmarking."""
    class SimpleResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.block1 = ResNetBlock(64, 64)
            self.block2 = ResNetBlock(64, 128)
            self.block3 = ResNetBlock(128, 256)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.pool(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x

    model = SimpleResNet()
    model.eval()
    inputs = (torch.randn(1, 3, 64, 64),)
    return model, inputs


class MultiHeadAttention(nn.Module):
    """Multi-head attention module similar to BERT."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    """Transformer block similar to BERT."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleBERT(nn.Module):
    """Simple BERT-like model for benchmarking."""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.pooler = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        # Pool the first token (CLS token)
        pooled = torch.tanh(self.pooler(x[:, 0]))
        return pooled


def create_simple_bert(
    num_layers: int = 12,
    hidden_size: int = 768,
    seq_len: int = 128,
    batch_size: int = 1,
) -> Tuple[nn.Module, Tuple[torch.Tensor]]:
    """Create a simple BERT-like model for benchmarking."""
    model = SimpleBERT(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=12,
        max_seq_len=512,
    )
    model.eval()
    inputs = (torch.randint(0, 30522, (batch_size, seq_len)),)
    return model, inputs


# ============================================================================
# Benchmarking Functions
# ============================================================================

BenchmarkResult = namedtuple("BenchmarkResult", [
    "cold_time",
    "warm_time",
    "avg_time",
    "min_time",
    "max_time",
    "op_count",
    "times",
])


def benchmark_trace(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    static_dispatch: bool,
    num_runs: int = 5,
    tracing_mode: str = "fake",
) -> BenchmarkResult:
    """
    Benchmark make_fx tracing with the given model and inputs.

    Args:
        model: Model to trace
        inputs: Input tensors
        static_dispatch: Whether to use static dispatch mode
        num_runs: Number of benchmark runs
        tracing_mode: Tracing mode ("real", "fake", "symbolic")

    Returns:
        BenchmarkResult with timing data
    """
    times = []
    op_counts = []

    def forward_fn(*args):
        return model(*args)

    for i in range(num_runs):
        # Clear caches and run GC
        gc.collect()
        torch._dynamo.reset()

        start = time.perf_counter()
        gm = make_fx(
            forward_fn,
            tracing_mode=tracing_mode,
            static_dispatch=static_dispatch,
            _allow_non_fake_inputs=True,
        )(*inputs)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        op_counts.append(len([n for n in gm.graph.nodes if n.op == "call_function"]))

    # First run is cold, rest are warm
    return BenchmarkResult(
        cold_time=times[0],
        warm_time=sum(times[1:]) / max(len(times) - 1, 1),
        avg_time=sum(times) / len(times),
        min_time=min(times),
        max_time=max(times),
        op_count=op_counts[0],
        times=times,
    )


def run_single_benchmark(
    model_name: str,
    model: nn.Module,
    inputs: Tuple[Any, ...],
    num_runs: int = 5,
    tracing_mode: str = "fake",
) -> Dict[str, Any]:
    """Run benchmark for a single model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    # Benchmark with dynamic dispatch (default)
    print("\nDynamic dispatch (default):")
    dynamic_results = benchmark_trace(
        model, inputs, static_dispatch=False,
        num_runs=num_runs, tracing_mode=tracing_mode,
    )
    print(f"  Cold start: {dynamic_results.cold_time*1000:.2f} ms")
    print(f"  Warm start: {dynamic_results.warm_time*1000:.2f} ms")
    print(f"  Avg time:   {dynamic_results.avg_time*1000:.2f} ms")
    print(f"  Op count:   {dynamic_results.op_count}")

    # Benchmark with static dispatch
    print("\nStatic dispatch (experimental):")
    static_results = benchmark_trace(
        model, inputs, static_dispatch=True,
        num_runs=num_runs, tracing_mode=tracing_mode,
    )
    print(f"  Cold start: {static_results.cold_time*1000:.2f} ms")
    print(f"  Warm start: {static_results.warm_time*1000:.2f} ms")
    print(f"  Avg time:   {static_results.avg_time*1000:.2f} ms")
    print(f"  Op count:   {static_results.op_count}")

    # Calculate improvement
    if dynamic_results.cold_time > 0:
        cold_improvement = (dynamic_results.cold_time - static_results.cold_time) / dynamic_results.cold_time * 100
    else:
        cold_improvement = 0.0

    if dynamic_results.warm_time > 0:
        warm_improvement = (dynamic_results.warm_time - static_results.warm_time) / dynamic_results.warm_time * 100
    else:
        warm_improvement = 0.0

    print(f"\nImprovement:")
    print(f"  Cold start: {cold_improvement:+.1f}%")
    print(f"  Warm start: {warm_improvement:+.1f}%")

    return {
        "model": model_name,
        "dynamic": dynamic_results._asdict(),
        "static": static_results._asdict(),
        "cold_improvement": cold_improvement,
        "warm_improvement": warm_improvement,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark static dispatch POC")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only this model (e.g., resnet50)",
    )
    parser.add_argument(
        "--torchbench-all",
        action="store_true",
        help="Run all available TorchBench models",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use fallback models (no TorchBench required)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of benchmark runs per model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to benchmark on",
    )
    parser.add_argument(
        "--tracing-mode",
        type=str,
        default="fake",
        choices=["real", "fake", "symbolic"],
        help="Tracing mode for make_fx",
    )
    args = parser.parse_args()

    print("Static Dispatch POC Benchmark")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {args.device}")
    print(f"Tracing mode: {args.tracing_mode}")
    print(f"Num runs: {args.num_runs}")

    results = []

    # Try to use TorchBench first
    torchbench_available = False
    if not args.fallback:
        original_dir = setup_torchbench_cwd()
        try:
            import torchbenchmark
            torchbench_available = True
            print("TorchBench: Available")
        except ImportError:
            print("TorchBench: Not available, using fallback models")
            os.chdir(original_dir)

    if torchbench_available and not args.fallback:
        # Determine which models to run
        if args.only:
            models_to_run = [args.only]
        elif args.torchbench_all:
            models_to_run = list_torchbench_models()
        else:
            # Default set of models
            models_to_run = [
                "resnet50",
                "resnet18",
                "mobilenet_v2",
            ]

        for model_name in models_to_run:
            try:
                print(f"\nLoading model: {model_name}")
                model, inputs = load_torchbench_model(
                    model_name,
                    device=args.device,
                )

                # Move inputs to device if needed
                if args.device == "cuda":
                    inputs = tuple(
                        x.cuda() if isinstance(x, torch.Tensor) else x
                        for x in inputs
                    )

                result = run_single_benchmark(
                    model_name,
                    model,
                    inputs,
                    num_runs=args.num_runs,
                    tracing_mode=args.tracing_mode,
                )
                results.append(result)

            except Exception as e:
                print(f"\nError loading/running {model_name}: {e}")
                continue

        os.chdir(original_dir)

    else:
        # Use fallback models
        print("\nUsing fallback models (no TorchBench)")

        # Determine which model to run
        if args.only:
            models_to_run = [args.only.lower()]
        else:
            models_to_run = ["simple_bert", "simple_resnet", "simple_mlp"]

        for model_name in models_to_run:
            try:
                if model_name in ("bert", "simple_bert", "bert_pytorch"):
                    # SimpleBERT - 12 layer transformer
                    print("\nLoading model: SimpleBERT (12 layers, 768 hidden)")
                    model, inputs = create_simple_bert(num_layers=12, hidden_size=768, seq_len=128)
                    result = run_single_benchmark(
                        "SimpleBERT",
                        model,
                        inputs,
                        num_runs=args.num_runs,
                        tracing_mode=args.tracing_mode,
                    )
                    results.append(result)

                elif model_name in ("bert_small", "simple_bert_small"):
                    # Smaller BERT for faster iteration
                    print("\nLoading model: SimpleBERT-Small (6 layers, 512 hidden)")
                    model, inputs = create_simple_bert(num_layers=6, hidden_size=512, seq_len=64)
                    result = run_single_benchmark(
                        "SimpleBERT-Small",
                        model,
                        inputs,
                        num_runs=args.num_runs,
                        tracing_mode=args.tracing_mode,
                    )
                    results.append(result)

                elif model_name in ("resnet", "simple_resnet"):
                    # Simple ResNet
                    print("\nLoading model: SimpleResNet")
                    model, inputs = create_simple_resnet()
                    result = run_single_benchmark(
                        "SimpleResNet",
                        model,
                        inputs,
                        num_runs=args.num_runs,
                        tracing_mode=args.tracing_mode,
                    )
                    results.append(result)

                elif model_name in ("mlp", "simple_mlp"):
                    # Simple MLP
                    print("\nLoading model: SimpleMLP")
                    x = torch.randn(32, 64)
                    result = run_single_benchmark(
                        "SimpleMLP",
                        nn.Sequential(
                            nn.Linear(64, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 10),
                        ),
                        (x,),
                        num_runs=args.num_runs,
                        tracing_mode=args.tracing_mode,
                    )
                    results.append(result)

                else:
                    print(f"\nUnknown model: {model_name}")
                    print("Available fallback models: simple_bert, simple_bert_small, simple_resnet, simple_mlp")

            except Exception as e:
                print(f"\nError loading/running {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if not results:
        print("No benchmarks completed successfully.")
        return

    for result in results:
        print(f"\n{result['model']}:")
        print(f"  Cold improvement: {result['cold_improvement']:+.1f}%")
        print(f"  Warm improvement: {result['warm_improvement']:+.1f}%")

    # Average improvement
    avg_cold = sum(r['cold_improvement'] for r in results) / len(results)
    avg_warm = sum(r['warm_improvement'] for r in results) / len(results)
    print(f"\nOverall average improvement:")
    print(f"  Cold start: {avg_cold:+.1f}%")
    print(f"  Warm start: {avg_warm:+.1f}%")

    # Write results to JSON for further analysis
    import json
    results_file = "static_dispatch_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
