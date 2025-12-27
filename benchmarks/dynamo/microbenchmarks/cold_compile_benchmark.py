"""
Benchmark measuring end-to-end cold compile time for torch.compile.

This benchmark measures the total wall clock time for the first compilation
of various functions, with a fresh cache. It also reports phase breakdown:
- dynamo tracing
- AOT autograd (forward/backward decomposition)
- inductor lowering and codegen
- autotuning (if enabled)

Usage:
    python cold_compile_benchmark.py [--device cpu|cuda] [--repeat N] [--json output.json]

Examples:
    # Simple run with default settings
    python cold_compile_benchmark.py

    # Run on CUDA with 3 repetitions
    python cold_compile_benchmark.py --device cuda --repeat 3

    # Output results to JSON
    python cold_compile_benchmark.py --json results.json
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Optional


@dataclasses.dataclass
class BenchmarkResult:
    name: str
    device: str
    cold_compile_time_s: float
    dynamo_tracing_s: Optional[float] = None
    aot_autograd_s: Optional[float] = None
    inductor_lowering_s: Optional[float] = None
    code_generation_s: Optional[float] = None
    autotuning_s: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


def _create_benchmark_script(
    benchmark_name: str,
    benchmark_code: str,
    device: str,
) -> str:
    return f'''
import json
import sys
import time

import torch
import torch._dynamo
import torch._dynamo.utils as dynamo_utils

torch._dynamo.reset()
dynamo_utils.reset_frame_count()

device = "{device}"
if device == "cuda" and not torch.cuda.is_available():
    print(json.dumps({{"error": "CUDA not available"}}))
    sys.exit(1)

{benchmark_code}

timing = dynamo_utils.calculate_time_spent()

result = {{
    "name": "{benchmark_name}",
    "device": device,
    "cold_compile_time_s": timing.get("total_wall_time", 0),
    "dynamo_tracing_s": timing.get("entire_frame_compile", 0) - timing.get("backend_compile", 0),
    "backend_compile_s": timing.get("backend_compile", 0),
    "inductor_lowering_s": timing.get("code_gen", 0),
    "code_generation_s": timing.get("code_gen", 0),
}}

print("BENCHMARK_RESULT:" + json.dumps(result))
'''


BENCHMARKS = {
    "simple_function": '''
@torch.compile
def f(x):
    return x.sin() + x.cos()

x = torch.randn(100, device=device)
start = time.perf_counter()
f(x)
end = time.perf_counter()
''',
    "linear_layer": '''
model = torch.nn.Linear(512, 512).to(device)
compiled_model = torch.compile(model)

x = torch.randn(32, 512, device=device)
start = time.perf_counter()
compiled_model(x)
end = time.perf_counter()
''',
    "mlp": '''
model = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512),
).to(device)
compiled_model = torch.compile(model)

x = torch.randn(32, 512, device=device)
start = time.perf_counter()
compiled_model(x)
end = time.perf_counter()
''',
    "attention": '''
class SimpleAttention(torch.nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

model = SimpleAttention().to(device)
compiled_model = torch.compile(model)

x = torch.randn(2, 128, 512, device=device)
start = time.perf_counter()
compiled_model(x)
end = time.perf_counter()
''',
    "pointwise_fusion": '''
@torch.compile
def f(x, y, z):
    a = x + y
    b = a * z
    c = torch.relu(b)
    d = c.sin()
    e = d.cos()
    return e + a

x = torch.randn(1024, 1024, device=device)
y = torch.randn(1024, 1024, device=device)
z = torch.randn(1024, 1024, device=device)
start = time.perf_counter()
f(x, y, z)
end = time.perf_counter()
''',
    "reduction": '''
@torch.compile
def f(x):
    return x.sum(dim=-1).mean()

x = torch.randn(1024, 1024, device=device)
start = time.perf_counter()
f(x)
end = time.perf_counter()
''',
}


def run_cold_compile_benchmark(
    benchmark_name: str,
    benchmark_code: str,
    device: str,
) -> Optional[BenchmarkResult]:
    script = _create_benchmark_script(benchmark_name, benchmark_code, device)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as script_file:
        script_file.write(script)
        script_path = script_file.name

    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            env = os.environ.copy()
            env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
            env["TRITON_CACHE_DIR"] = os.path.join(cache_dir, "triton")

            start = time.perf_counter()
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                env=env,
            )
            wall_time = time.perf_counter() - start

            if result.returncode != 0:
                print(f"Benchmark {benchmark_name} failed: {result.stderr}")
                return None

            for line in result.stdout.splitlines():
                if line.startswith("BENCHMARK_RESULT:"):
                    data = json.loads(line[len("BENCHMARK_RESULT:") :])
                    if "error" in data:
                        print(f"Benchmark {benchmark_name} error: {data['error']}")
                        return None

                    return BenchmarkResult(
                        name=data["name"],
                        device=data["device"],
                        cold_compile_time_s=wall_time,  # Use wall time from subprocess
                        dynamo_tracing_s=data.get("dynamo_tracing_s"),
                        aot_autograd_s=data.get("aot_autograd_s"),
                        inductor_lowering_s=data.get("inductor_lowering_s"),
                        code_generation_s=data.get("code_generation_s"),
                        autotuning_s=data.get("autotuning_s"),
                    )

            print(f"Benchmark {benchmark_name}: No result found in output")
            print(f"stdout: {result.stdout}")
            return None

    finally:
        os.unlink(script_path)


def run_all_benchmarks(
    device: str = "cpu",
    repeat: int = 1,
    benchmarks: Optional[list[str]] = None,
) -> list[BenchmarkResult]:
    results = []

    benchmark_items = BENCHMARKS.items()
    if benchmarks:
        benchmark_items = [(k, v) for k, v in benchmark_items if k in benchmarks]

    for name, code in benchmark_items:
        print(f"Running benchmark: {name} on {device}")
        times = []
        for i in range(repeat):
            result = run_cold_compile_benchmark(name, code, device)
            if result:
                times.append(result.cold_compile_time_s)
                if i == repeat - 1:
                    result.cold_compile_time_s = min(times)
                    results.append(result)
                    print(f"  {name}: {result.cold_compile_time_s:.3f}s (min of {repeat})")
            else:
                print(f"  {name}: FAILED")
                break

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    if not results:
        print("No results to display")
        return

    print("\n" + "=" * 70)
    print("Cold Compile Benchmark Results")
    print("=" * 70)
    print(f"{'Benchmark':<25} {'Device':<8} {'Time (s)':<12} {'Breakdown'}")
    print("-" * 70)

    for r in results:
        breakdown_parts = []
        if r.dynamo_tracing_s:
            breakdown_parts.append(f"dynamo={r.dynamo_tracing_s:.2f}s")
        if r.inductor_lowering_s:
            breakdown_parts.append(f"inductor={r.inductor_lowering_s:.2f}s")
        breakdown = ", ".join(breakdown_parts) if breakdown_parts else "N/A"
        print(f"{r.name:<25} {r.device:<8} {r.cold_compile_time_s:<12.3f} {breakdown}")

    print("=" * 70)


def write_json_results(results: list[BenchmarkResult], output_path: str) -> None:
    records = []
    for result in results:
        records.append(
            {
                "benchmark": {
                    "name": "cold_compile_benchmark",
                    "mode": "inference",
                    "extra_info": {
                        "device": result.device,
                    },
                },
                "model": {
                    "name": result.name,
                    "type": "microbenchmark",
                    "backend": "inductor",
                },
                "metric": {
                    "name": "cold_compile_time_s",
                    "benchmark_values": [result.cold_compile_time_s],
                },
            }
        )
        if result.dynamo_tracing_s is not None:
            records.append(
                {
                    "benchmark": {"name": "cold_compile_benchmark"},
                    "model": {"name": result.name},
                    "metric": {
                        "name": "dynamo_tracing_s",
                        "benchmark_values": [result.dynamo_tracing_s],
                    },
                }
            )

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cold compile times for torch.compile"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run benchmarks on",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat each benchmark (reports min)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=list(BENCHMARKS.keys()),
        default=None,
        help="Specific benchmarks to run (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = run_all_benchmarks(
        device=args.device,
        repeat=args.repeat,
        benchmarks=args.benchmarks,
    )

    print_results(results)

    if args.json:
        write_json_results(results, args.json)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
