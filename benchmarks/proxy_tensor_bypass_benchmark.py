"""
Benchmark script to measure the dispatching time improvement from
bypassing proxy mode and directly calling fake tensor dispatch.

Usage:
    # Compare bypass vs no-bypass in a single run (recommended)
    python benchmarks/proxy_tensor_bypass_benchmark.py --compare-both

    # Save results to file for later comparison
    python benchmarks/proxy_tensor_bypass_benchmark.py --save bypass.json

    # Compare two saved files
    python benchmarks/proxy_tensor_bypass_benchmark.py --compare baseline.json bypass.json

    # Just run and print results
    python benchmarks/proxy_tensor_bypass_benchmark.py
"""

import contextlib
import argparse
import json
import os
import subprocess
import sys
import time
import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx


@contextlib.contextmanager
def _disable_proxy_bypass():
    """Temporarily disable the bypass by patching _can_bypass_proxy_dispatch to False."""
    patched = []
    for ns_name in dir(torch.ops):
        op_ns = getattr(torch.ops, ns_name, None)
        if op_ns is None:
            continue
        for op_name in dir(op_ns):
            op = getattr(op_ns, op_name, None)
            if op is None or not hasattr(op, "overloads"):
                continue
            for overload_name in op.overloads():
                overload = getattr(op, overload_name, None)
                if overload is not None and getattr(overload, "_can_bypass_proxy_dispatch", False):
                    overload._can_bypass_proxy_dispatch = False
                    patched.append(overload)
    try:
        yield
    finally:
        for overload in patched:
            overload._can_bypass_proxy_dispatch = True


def benchmark_make_fx(fn, args, tracing_mode="fake", warmup=3, iterations=10, name=""):
    """Benchmark make_fx tracing time."""
    # Warmup
    for _ in range(warmup):
        try:
            traced = make_fx(fn, tracing_mode=tracing_mode)(*args)
            del traced
        except Exception as e:
            print(f"  {name}: Error during warmup - {e}")
            return None

    # Timed runs
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        traced = make_fx(fn, tracing_mode=tracing_mode)(*args)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append(end - start)
        del traced

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    return {"avg": avg_time, "min": min_time, "max": max_time, "times": times}


def simple_add(x, y):
    return x + y


def matmul_chain(x, y, z):
    return torch.matmul(torch.matmul(x, y), z)


def conv_bn_relu(x, weight, bn_weight, bn_bias, bn_mean, bn_var):
    out = torch.nn.functional.conv2d(x, weight, padding=1)
    out = torch.nn.functional.batch_norm(
        out, bn_mean, bn_var, bn_weight, bn_bias, training=False
    )
    out = torch.nn.functional.relu(out)
    return out


def transformer_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def mlp_block(x, w1, w2, b1, b2):
    h = torch.nn.functional.linear(x, w1, b1)
    h = torch.nn.functional.gelu(h)
    return torch.nn.functional.linear(h, w2, b2)


def resnet_block(x, conv1_w, conv2_w, bn1_w, bn1_b, bn1_mean, bn1_var, bn2_w, bn2_b, bn2_mean, bn2_var):
    identity = x
    out = torch.nn.functional.conv2d(x, conv1_w, padding=1)
    out = torch.nn.functional.batch_norm(out, bn1_mean, bn1_var, bn1_w, bn1_b, training=False)
    out = torch.nn.functional.relu(out)
    out = torch.nn.functional.conv2d(out, conv2_w, padding=1)
    out = torch.nn.functional.batch_norm(out, bn2_mean, bn2_var, bn2_w, bn2_b, training=False)
    out = out + identity
    out = torch.nn.functional.relu(out)
    return out


def many_small_ops(x):
    """Many small operations to stress test dispatch overhead."""
    for _ in range(50):
        x = x + 1
        x = x * 0.99
        x = torch.relu(x)
    return x


def run_benchmarks():
    print("=" * 70)
    print("Proxy Tensor Bypass Benchmark")
    print("=" * 70)
    print()

    device = "cpu"
    tracing_mode = "fake"  # Use "symbolic" to test symbolic tracing

    benchmarks = []

    # Benchmark 1: Simple Add
    print("1. Simple Add (2 tensors)...")
    x = torch.randn(64, 64, device=device)
    y = torch.randn(64, 64, device=device)
    result = benchmark_make_fx(simple_add, (x, y), tracing_mode=tracing_mode, name="Simple Add")
    if result:
        benchmarks.append(("Simple Add", result))
        print(f"   Avg: {result['avg']*1000:.3f} ms, Min: {result['min']*1000:.3f} ms, Max: {result['max']*1000:.3f} ms")

    # Benchmark 2: MatMul Chain
    print("2. MatMul Chain (3 matmuls)...")
    x = torch.randn(32, 64, device=device)
    y = torch.randn(64, 128, device=device)
    z = torch.randn(128, 32, device=device)
    result = benchmark_make_fx(matmul_chain, (x, y, z), tracing_mode=tracing_mode, name="MatMul Chain")
    if result:
        benchmarks.append(("MatMul Chain", result))
        print(f"   Avg: {result['avg']*1000:.3f} ms, Min: {result['min']*1000:.3f} ms, Max: {result['max']*1000:.3f} ms")

    # Benchmark 3: Conv + BN + ReLU
    print("3. Conv + BatchNorm + ReLU...")
    x = torch.randn(1, 64, 32, 32, device=device)
    weight = torch.randn(64, 64, 3, 3, device=device)
    bn_weight = torch.randn(64, device=device)
    bn_bias = torch.randn(64, device=device)
    bn_mean = torch.randn(64, device=device)
    bn_var = torch.abs(torch.randn(64, device=device)) + 0.1
    result = benchmark_make_fx(
        conv_bn_relu, (x, weight, bn_weight, bn_bias, bn_mean, bn_var),
        tracing_mode=tracing_mode, name="Conv+BN+ReLU"
    )
    if result:
        benchmarks.append(("Conv+BN+ReLU", result))
        print(f"   Avg: {result['avg']*1000:.3f} ms, Min: {result['min']*1000:.3f} ms, Max: {result['max']*1000:.3f} ms")

    # Benchmark 4: Transformer Attention
    print("4. Transformer Attention...")
    batch, seq, heads, dim = 2, 128, 8, 64
    q = torch.randn(batch, heads, seq, dim, device=device)
    k = torch.randn(batch, heads, seq, dim, device=device)
    v = torch.randn(batch, heads, seq, dim, device=device)
    result = benchmark_make_fx(transformer_attention, (q, k, v), tracing_mode=tracing_mode, name="Attention")
    if result:
        benchmarks.append(("Transformer Attention", result))
        print(f"   Avg: {result['avg']*1000:.3f} ms, Min: {result['min']*1000:.3f} ms, Max: {result['max']*1000:.3f} ms")

    # Benchmark 5: MLP Block
    print("5. MLP Block (2 linear + GELU)...")
    x = torch.randn(32, 256, device=device)
    w1 = torch.randn(512, 256, device=device)
    w2 = torch.randn(256, 512, device=device)
    b1 = torch.randn(512, device=device)
    b2 = torch.randn(256, device=device)
    result = benchmark_make_fx(mlp_block, (x, w1, w2, b1, b2), tracing_mode=tracing_mode, name="MLP Block")
    if result:
        benchmarks.append(("MLP Block", result))
        print(f"   Avg: {result['avg']*1000:.3f} ms, Min: {result['min']*1000:.3f} ms, Max: {result['max']*1000:.3f} ms")

    # Benchmark 6: ResNet Block
    print("6. ResNet Block...")
    x = torch.randn(1, 64, 32, 32, device=device)
    conv1_w = torch.randn(64, 64, 3, 3, device=device)
    conv2_w = torch.randn(64, 64, 3, 3, device=device)
    bn1_w = torch.randn(64, device=device)
    bn1_b = torch.randn(64, device=device)
    bn1_mean = torch.randn(64, device=device)
    bn1_var = torch.abs(torch.randn(64, device=device)) + 0.1
    bn2_w = torch.randn(64, device=device)
    bn2_b = torch.randn(64, device=device)
    bn2_mean = torch.randn(64, device=device)
    bn2_var = torch.abs(torch.randn(64, device=device)) + 0.1
    result = benchmark_make_fx(
        resnet_block,
        (x, conv1_w, conv2_w, bn1_w, bn1_b, bn1_mean, bn1_var, bn2_w, bn2_b, bn2_mean, bn2_var),
        tracing_mode=tracing_mode, name="ResNet Block"
    )
    if result:
        benchmarks.append(("ResNet Block", result))
        print(f"   Avg: {result['avg']*1000:.3f} ms, Min: {result['min']*1000:.3f} ms, Max: {result['max']*1000:.3f} ms")

    # Benchmark 7: Many Small Ops (stress test dispatch overhead)
    print("7. Many Small Ops (150 ops, stress test)...")
    x = torch.randn(64, 64, device=device)
    result = benchmark_make_fx(many_small_ops, (x,), tracing_mode=tracing_mode, name="Many Small Ops")
    if result:
        benchmarks.append(("Many Small Ops (150)", result))
        print(f"   Avg: {result['avg']*1000:.3f} ms, Min: {result['min']*1000:.3f} ms, Max: {result['max']*1000:.3f} ms")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Benchmark':<25} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 70)
    for name, result in benchmarks:
        print(f"{name:<25} {result['avg']*1000:<12.3f} {result['min']*1000:<12.3f} {result['max']*1000:<12.3f}")

    # Calculate total
    if benchmarks:
        total_avg = sum(r['avg'] for _, r in benchmarks)
        total_min = sum(r['min'] for _, r in benchmarks)
        total_max = sum(r['max'] for _, r in benchmarks)
        print("-" * 70)
        print(f"{'TOTAL':<25} {total_avg*1000:<12.3f} {total_min*1000:<12.3f} {total_max*1000:<12.3f}")

    return {name: result for name, result in benchmarks}


def analyze_op_counts():
    """
    For each benchmark, run make_fx once and report bypass/normal dispatch counts.
    This explains why certain models benefit more than others from the bypass.
    """
    from torch.utils._stats import simple_call_counter

    device = "cpu"

    workloads = [
        ("Simple Add",         simple_add,            (torch.randn(64, 64), torch.randn(64, 64))),
        ("MatMul Chain",       matmul_chain,           (torch.randn(32, 64), torch.randn(64, 128), torch.randn(128, 32))),
        ("Many Small Ops",     many_small_ops,         (torch.randn(64, 64),)),
        ("MLP Block",          mlp_block,              (torch.randn(32, 256), torch.randn(512, 256), torch.randn(256, 512), torch.randn(512), torch.randn(256))),
        ("Transformer Attn",   transformer_attention,  (torch.randn(2, 8, 128, 64), torch.randn(2, 8, 128, 64), torch.randn(2, 8, 128, 64))),
    ]

    print("=" * 75)
    print("Op Count Analysis (single make_fx trace per workload)")
    print("=" * 75)
    print(f"{'Workload':<22} {'Total':>7} {'Bypassed':>10} {'Rate':>7} {'Failed':>8} {'Normal':>8}")
    print("-" * 75)

    for name, fn, args in workloads:
        simple_call_counter.clear()
        try:
            make_fx(fn, tracing_mode="fake")(*args)
        except Exception as e:
            print(f"  {name}: Error - {e}")
            continue
        bypassed = simple_call_counter.get("proxy_call.bypass_succeeded", 0)
        failed   = simple_call_counter.get("proxy_call.bypass_failed", 0)
        normal   = simple_call_counter.get("proxy_call.normal_dispatch", 0)
        total    = bypassed + failed + normal
        rate     = bypassed / total * 100 if total > 0 else 0.0
        print(f"{name:<22} {total:>7} {bypassed:>10} {rate:>6.1f}% {failed:>8} {normal:>8}")

    print()
    print("'Normal' = op was ineligible (_can_bypass_proxy_dispatch=False)")
    print("'Failed' = bypass attempted but raised an exception (fell back to normal)")
    print()


# ---------------------------------------------------------------------------
# torch.compile benchmarks using real nn.Module models
# ---------------------------------------------------------------------------

class _SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(256, 512)
        self.fc2 = torch.nn.Linear(512, 256)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _SimpleResBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn1   = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2   = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        return torch.nn.functional.relu(self.bn2(self.conv2(out)) + x)


class _SimpleTransformerLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, batch_first=True
        )

    def forward(self, x):
        return self.layer(x)


def benchmark_compile(model, example_input, warmup=1, iterations=5, name=""):
    """Benchmark torch.compile first-compile and subsequent calls."""
    times = []
    for i in range(warmup + iterations):
        compiled = torch.compile(model, backend="aot_eager")
        torch.compiler.reset()
        start = time.perf_counter()
        _ = compiled(*example_input) if isinstance(example_input, tuple) else compiled(example_input)
        end = time.perf_counter()
        if i >= warmup:
            times.append(end - start)
    avg = sum(times) / len(times)
    return {"avg": avg, "min": min(times), "max": max(times)}


def run_compile_benchmarks():
    print("=" * 70)
    print("torch.compile (aot_eager backend) Benchmarks")
    print("=" * 70)
    print()

    benchmarks = []

    print("1. MLP (256->512->256)...")
    model = _SimpleMLP().eval()
    x = torch.randn(32, 256)
    result = benchmark_compile(model, (x,), name="MLP")
    benchmarks.append(("MLP (compile)", result))
    print(f"   Avg: {result['avg']*1000:.3f} ms")

    print("2. ResNet Block...")
    model = _SimpleResBlock().eval()
    x = torch.randn(1, 64, 32, 32)
    result = benchmark_compile(model, (x,), name="ResBlock")
    benchmarks.append(("ResBlock (compile)", result))
    print(f"   Avg: {result['avg']*1000:.3f} ms")

    print("3. TransformerEncoderLayer...")
    model = _SimpleTransformerLayer().eval()
    x = torch.randn(2, 128, 256)
    result = benchmark_compile(model, (x,), name="TransformerLayer")
    benchmarks.append(("Transformer (compile)", result))
    print(f"   Avg: {result['avg']*1000:.3f} ms")

    print()
    print("=" * 70)
    print(f"{'Benchmark':<25} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 70)
    for name, r in benchmarks:
        print(f"{name:<25} {r['avg']*1000:<12.3f} {r['min']*1000:<12.3f} {r['max']*1000:<12.3f}")

    return {name: result for name, result in benchmarks}


def compare_both():
    """Run make_fx and torch.compile benchmarks with and without bypass and print comparison."""
    print("=" * 70)
    print("Proxy Tensor Bypass: Side-by-Side Comparison")
    print("=" * 70)
    print()

    print("--- Running WITH bypass ---")
    bypass_results = run_benchmarks()
    compile_bypass = run_compile_benchmarks()

    print()
    print("--- Running WITHOUT bypass (patching _can_bypass_proxy_dispatch=False) ---")
    with _disable_proxy_bypass():
        baseline_results = run_benchmarks()
        compile_baseline = run_compile_benchmarks()

    baseline_results.update(compile_baseline)
    bypass_results.update(compile_bypass)

    print()
    print("=" * 80)
    print("Comparison: No-Bypass (baseline) vs Bypass")
    print("=" * 80)
    print(f"{'Benchmark':<25} {'Baseline (ms)':<15} {'Bypass (ms)':<15} {'Speedup':<10} {'Change'}")
    print("-" * 80)

    total_baseline = 0
    total_bypass = 0

    for name in bypass_results:
        if name in baseline_results:
            base_avg = baseline_results[name]['avg'] * 1000
            byp_avg = bypass_results[name]['avg'] * 1000
            total_baseline += base_avg
            total_bypass += byp_avg
            speedup = base_avg / byp_avg if byp_avg > 0 else float('inf')
            change_pct = ((base_avg - byp_avg) / base_avg) * 100 if base_avg > 0 else 0
            indicator = "FASTER" if change_pct > 5 else ("SLOWER" if change_pct < -5 else "~SAME")
            print(f"{name:<25} {base_avg:<15.3f} {byp_avg:<15.3f} {speedup:<10.2f}x {change_pct:>+7.1f}% {indicator}")

    print("-" * 80)
    if total_baseline > 0 and total_bypass > 0:
        total_speedup = total_baseline / total_bypass
        total_change = ((total_baseline - total_bypass) / total_baseline) * 100
        print(f"{'TOTAL':<25} {total_baseline:<15.3f} {total_bypass:<15.3f} {total_speedup:<10.2f}x {total_change:>+7.1f}%")


def save_results(results, filepath):
    """Save benchmark results to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")


def load_results(filepath):
    """Load benchmark results from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_results(baseline_path, bypass_path):
    """Compare two benchmark result files and print the difference."""
    baseline = load_results(baseline_path)
    bypass = load_results(bypass_path)

    print("=" * 80)
    print("Comparison: Baseline vs Bypass")
    print("=" * 80)
    print(f"{'Benchmark':<25} {'Baseline (ms)':<15} {'Bypass (ms)':<15} {'Speedup':<12} {'Change':<10}")
    print("-" * 80)

    total_baseline = 0
    total_bypass = 0

    for name in baseline:
        if name in bypass:
            base_avg = baseline[name]['avg'] * 1000
            byp_avg = bypass[name]['avg'] * 1000
            total_baseline += base_avg
            total_bypass += byp_avg

            speedup = base_avg / byp_avg if byp_avg > 0 else float('inf')
            change_pct = ((base_avg - byp_avg) / base_avg) * 100 if base_avg > 0 else 0

            # Color coding (if supported)
            if change_pct > 5:
                indicator = "✓ FASTER"
            elif change_pct < -5:
                indicator = "✗ SLOWER"
            else:
                indicator = "~ SAME"

            print(f"{name:<25} {base_avg:<15.3f} {byp_avg:<15.3f} {speedup:<12.2f}x {change_pct:>+7.1f}% {indicator}")

    print("-" * 80)
    if total_baseline > 0:
        total_speedup = total_baseline / total_bypass if total_bypass > 0 else float('inf')
        total_change = ((total_baseline - total_bypass) / total_baseline) * 100
        print(f"{'TOTAL':<25} {total_baseline:<15.3f} {total_bypass:<15.3f} {total_speedup:<12.2f}x {total_change:>+7.1f}%")

    print()
    print("Summary:")
    if total_bypass < total_baseline:
        print(f"  Bypass is {total_change:.1f}% FASTER overall ({total_speedup:.2f}x speedup)")
    else:
        print(f"  Bypass is {-total_change:.1f}% SLOWER overall")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark proxy tensor bypass")
    parser.add_argument("--save", type=str, help="Save results to JSON file (e.g., baseline.json)")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "BYPASS"),
                        help="Compare two result files (baseline.json bypass.json)")
    parser.add_argument("--compare-both", action="store_true",
                        help="Run with and without bypass in one invocation and compare")
    parser.add_argument("--analyze-ops", action="store_true",
                        help="Show bypass/normal dispatch op counts per workload")
    parser.add_argument("--compile", action="store_true",
                        help="Benchmark torch.compile (aot_eager) on real nn.Module models")

    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    elif args.compare_both:
        compare_both()
    elif args.analyze_ops:
        analyze_op_counts()
    elif args.compile:
        run_compile_benchmarks()
    else:
        results = run_benchmarks()
        if args.save and results:
            save_results(results, args.save)
