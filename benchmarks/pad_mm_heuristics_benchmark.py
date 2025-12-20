#!/usr/bin/env python
"""
Benchmark to validate pad_mm heuristics against actual measurements.

This script compares:
1. Actual measured runtime for padded vs unpadded matmul
2. Roofline heuristic predictions
3. nvMatmulHeuristics predictions

It can also fit alignment efficiency coefficients based on measurements.

Usage:
    # Run full benchmark on current GPU
    python benchmarks/pad_mm_heuristics_benchmark.py -o results.csv

    # Quick test with fewer cases
    python benchmarks/pad_mm_heuristics_benchmark.py --quick

    # Fit efficiency coefficients from measurements
    python benchmarks/pad_mm_heuristics_benchmark.py --fit

    # Load existing results and analyze
    python benchmarks/pad_mm_heuristics_benchmark.py --load-csv results.csv --fit

Running on different architectures:
    1. Run on each target GPU to collect measurements:
       python benchmarks/pad_mm_heuristics_benchmark.py -o h100_results.csv
       python benchmarks/pad_mm_heuristics_benchmark.py -o a100_results.csv

    2. Use --fit to generate efficiency tables for each architecture:
       python benchmarks/pad_mm_heuristics_benchmark.py --load-csv h100_results.csv --fit

    3. Add the generated efficiency tables to pad_mm.py's _get_alignment_efficiency_table()
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.utils.benchmark as benchmark


@dataclass
class BenchmarkResult:
    m: int
    k: int
    n: int
    m_pad: int
    k_pad: int
    n_pad: int
    dtype: str
    layout: str  # 'NN', 'NT', 'TN', 'TT'
    # Actual measurements (in microseconds)
    unpadded_time_us: float
    padded_time_us: float
    padding_overhead_us: float  # Time to do the padding operations
    # Whether padding actually helps
    actual_should_pad: bool
    actual_speedup: float  # unpadded_time / (padded_time + padding_overhead)
    # Heuristic predictions
    roofline_should_pad: bool
    nvmatmul_should_pad: bool
    # nvMatmulHeuristics estimated times (in microseconds)
    nvmatmul_unpadded_us: float
    nvmatmul_padded_us: float


def measure_matmul_time(
    m: int, k: int, n: int, dtype: torch.dtype,
    trans_a: bool = False, trans_b: bool = False,
    device: str = "cuda"
) -> float:
    """Measure matmul execution time in microseconds."""
    if trans_a:
        a = torch.randn(k, m, dtype=dtype, device=device).t()
    else:
        a = torch.randn(m, k, dtype=dtype, device=device)

    if trans_b:
        b = torch.randn(n, k, dtype=dtype, device=device).t()
    else:
        b = torch.randn(k, n, dtype=dtype, device=device)

    # Warmup
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()

    # Benchmark
    timer = benchmark.Timer(
        stmt="torch.mm(a, b)",
        globals={"torch": torch, "a": a, "b": b},
        num_threads=1,
    )
    measurement = timer.blocked_autorange(min_run_time=0.5)
    return measurement.median * 1e6  # Convert to microseconds


def measure_padding_overhead(
    m: int, k: int, n: int,
    m_pad: int, k_pad: int, n_pad: int,
    dtype: torch.dtype,
    trans_a: bool = False, trans_b: bool = False,
    device: str = "cuda"
) -> float:
    """Measure the overhead of padding operations in microseconds."""
    if trans_a:
        a = torch.randn(k, m, dtype=dtype, device=device).t()
    else:
        a = torch.randn(m, k, dtype=dtype, device=device)

    if trans_b:
        b = torch.randn(n, k, dtype=dtype, device=device).t()
    else:
        b = torch.randn(k, n, dtype=dtype, device=device)

    m_padded = m + m_pad
    k_padded = k + k_pad
    n_padded = n + n_pad

    def pad_tensors():
        # Pad A: need to handle transpose
        if trans_a:
            # A is (k, m).t() -> logical (m, k), pad to (m+m_pad, k+k_pad)
            a_padded = torch.nn.functional.pad(a.t(), (0, k_pad, 0, m_pad)).t()
        else:
            a_padded = torch.nn.functional.pad(a, (0, k_pad, 0, m_pad))

        # Pad B: need to handle transpose
        if trans_b:
            # B is (n, k).t() -> logical (k, n), pad to (k+k_pad, n+n_pad)
            b_padded = torch.nn.functional.pad(b.t(), (0, n_pad, 0, k_pad)).t()
        else:
            b_padded = torch.nn.functional.pad(b, (0, n_pad, 0, k_pad))

        return a_padded, b_padded

    # Warmup
    for _ in range(10):
        pad_tensors()
    torch.cuda.synchronize()

    # Benchmark
    timer = benchmark.Timer(
        stmt="pad_tensors()",
        globals={"pad_tensors": pad_tensors},
        num_threads=1,
    )
    measurement = timer.blocked_autorange(min_run_time=0.2)
    padding_time = measurement.median * 1e6

    # Also measure output slicing if needed
    slice_time = 0.0
    if m_pad > 0 or n_pad > 0:
        c_padded = torch.randn(m_padded, n_padded, dtype=dtype, device=device)

        def slice_output():
            return c_padded[:m, :n].contiguous()

        for _ in range(10):
            slice_output()
        torch.cuda.synchronize()

        timer = benchmark.Timer(
            stmt="slice_output()",
            globals={"slice_output": slice_output},
            num_threads=1,
        )
        measurement = timer.blocked_autorange(min_run_time=0.2)
        slice_time = measurement.median * 1e6

    return padding_time + slice_time


def get_roofline_prediction(
    m: int, k: int, n: int,
    m_pad: int, k_pad: int, n_pad: int,
    dtype: torch.dtype
) -> bool:
    """Get roofline heuristic prediction."""
    try:
        from torch._inductor.fx_passes.pad_mm import should_pad_heuristic
        return should_pad_heuristic(m, k, n, m_pad, k_pad, n_pad, dtype)
    except Exception as e:
        print(f"Warning: roofline heuristic failed: {e}")
        return False


def get_nvmatmul_prediction(
    m: int, k: int, n: int,
    m_pad: int, k_pad: int, n_pad: int,
    dtype: torch.dtype,
    trans_a: bool = False,
    trans_b: bool = False,
    hardware_descriptor=None,
    heuristics_interface=None,
) -> tuple[bool, float, float]:
    """Get nvMatmulHeuristics prediction and estimated times."""
    try:
        from torch._inductor.fx_passes.pad_mm import (
            _nvmatmul_heuristics_available,
            _dtype_to_nvmatmul_precision,
            _estimate_copy_time_ns,
        )

        if not _nvmatmul_heuristics_available():
            return False, 0.0, 0.0

        from nvMatmulHeuristics import (
            NvMatmulHeuristicsMatmulLayout,
            boolsToNvMatmulHeuristicsLayout,
        )

        # Determine layout
        layout = boolsToNvMatmulHeuristicsLayout(trans_a, trans_b)

        precision = _dtype_to_nvmatmul_precision(dtype)

        # Use provided interface or create new one
        if heuristics_interface is None:
            from nvMatmulHeuristics import (
                NvMatmulHeuristicsInterface,
                NvMatmulHeuristicsTarget,
            )
            heuristics_interface = NvMatmulHeuristicsInterface(
                backend=NvMatmulHeuristicsTarget.GENERIC,
                precision=precision,
            )

        # Query for best kernel configuration
        # Note: nvMatmulHeuristics uses (M, N, K) order
        configs = heuristics_interface.get_with_mnk(
            m, n, k, layout, count=1, hardware_descriptor=hardware_descriptor
        )
        unpadded_runtime = float(configs[0]["runtime"]) * 1e6 if configs else float("inf")

        # Padded dimensions
        m_padded = m + m_pad
        k_padded = k + k_pad
        n_padded = n + n_pad

        configs_padded = heuristics_interface.get_with_mnk(
            m_padded, n_padded, k_padded, layout, count=1, hardware_descriptor=hardware_descriptor
        )
        padded_runtime = float(configs_padded[0]["runtime"]) * 1e6 if configs_padded else float("inf")

        # Estimate padding copy overhead
        copy_time = _estimate_copy_time_ns(
            m_padded * k_padded + k_padded * n_padded, dtype
        ) / 1000  # ns to us

        if m_pad > 0 or n_pad > 0:
            copy_time += _estimate_copy_time_ns(m * n, dtype) / 1000

        # Total time comparison with 5% safety margin
        unpadded_total = unpadded_runtime
        padded_total = padded_runtime + copy_time
        should_pad = padded_total * 1.05 < unpadded_total

        return should_pad, unpadded_runtime, padded_runtime
    except Exception as e:
        print(f"Warning: nvmatmul heuristic failed: {e}")
        return False, 0.0, 0.0


def run_benchmark(
    m: int, k: int, n: int,
    m_pad: int, k_pad: int, n_pad: int,
    dtype: torch.dtype,
    trans_a: bool = False,
    trans_b: bool = False,
    device: str = "cuda",
    hardware_descriptor=None,
    heuristics_interface=None,
) -> BenchmarkResult:
    """Run a single benchmark case."""
    dtype_str = str(dtype).replace("torch.", "")
    layout = ('T' if trans_a else 'N') + ('T' if trans_b else 'N')

    # Measure actual times
    unpadded_time = measure_matmul_time(m, k, n, dtype, trans_a, trans_b, device)
    padded_time = measure_matmul_time(m + m_pad, k + k_pad, n + n_pad, dtype, trans_a, trans_b, device)
    padding_overhead = measure_padding_overhead(m, k, n, m_pad, k_pad, n_pad, dtype, trans_a, trans_b, device)

    # Calculate actual speedup
    total_padded_time = padded_time + padding_overhead
    actual_speedup = unpadded_time / total_padded_time if total_padded_time > 0 else 1.0
    actual_should_pad = actual_speedup > 1.05  # 5% threshold

    # Get heuristic predictions
    roofline_should_pad = get_roofline_prediction(m, k, n, m_pad, k_pad, n_pad, dtype)
    nvmatmul_should_pad, nvmatmul_unpadded, nvmatmul_padded = get_nvmatmul_prediction(
        m, k, n, m_pad, k_pad, n_pad, dtype, trans_a, trans_b,
        hardware_descriptor, heuristics_interface
    )

    return BenchmarkResult(
        m=m, k=k, n=n,
        m_pad=m_pad, k_pad=k_pad, n_pad=n_pad,
        dtype=dtype_str,
        layout=layout,
        unpadded_time_us=unpadded_time,
        padded_time_us=padded_time,
        padding_overhead_us=padding_overhead,
        actual_should_pad=actual_should_pad,
        actual_speedup=actual_speedup,
        roofline_should_pad=roofline_should_pad,
        nvmatmul_should_pad=nvmatmul_should_pad,
        nvmatmul_unpadded_us=nvmatmul_unpadded,
        nvmatmul_padded_us=nvmatmul_padded,
    )


def generate_test_cases() -> list[tuple[int, int, int, int, int, int, torch.dtype, bool, bool]]:
    """Generate test cases with various dimensions, misalignments, and layouts."""
    cases = []

    # Different dtypes
    dtypes = [torch.float16, torch.bfloat16, torch.float32]

    # Different layouts: (trans_a, trans_b)
    layouts = [
        (False, False),  # NN - both row major
        (False, True),   # NT - A row major, B column major
        (True, False),   # TN - A column major, B row major
        (True, True),    # TT - both column major
    ]

    # Different sizes and misalignments
    # Format: (m, k, n, m_pad, k_pad, n_pad)
    dimension_cases = [
        # Small matrices - K misaligned by 1
        (128, 127, 128, 0, 1, 0),
        (256, 255, 256, 0, 1, 0),
        (512, 511, 512, 0, 1, 0),

        # Medium matrices - K misaligned by 1
        (1024, 1023, 1024, 0, 1, 0),
        (2048, 2047, 2048, 0, 1, 0),

        # Large matrices - K misaligned by 1
        (4096, 4095, 4096, 0, 1, 0),

        # K misaligned by 7 (worst case for 8-element alignment)
        (1024, 1017, 1024, 0, 7, 0),
        (2048, 2041, 2048, 0, 7, 0),

        # M misaligned
        (1023, 1024, 1024, 1, 0, 0),
        (2047, 2048, 2048, 1, 0, 0),

        # N misaligned
        (1024, 1024, 1023, 0, 0, 1),
        (2048, 2048, 2047, 0, 0, 1),

        # Multiple dimensions misaligned
        (1023, 1023, 1023, 1, 1, 1),
        (2047, 2047, 2047, 1, 1, 1),

        # Non-square matrices
        (512, 2047, 512, 0, 1, 0),
        (1024, 4095, 256, 0, 1, 0),

        # Transformer-like shapes
        (2048, 767, 768, 0, 1, 0),
        (4096, 1023, 1024, 0, 1, 0),
    ]

    for dtype in dtypes:
        for m, k, n, m_pad, k_pad, n_pad in dimension_cases:
            # Skip already-aligned cases (no padding needed)
            if m_pad == 0 and k_pad == 0 and n_pad == 0:
                continue

            # Test all layouts for key cases, NN only for others to save time
            if (m, k, n) in [(1024, 1023, 1024), (2048, 2047, 2048)]:
                for trans_a, trans_b in layouts:
                    cases.append((m, k, n, m_pad, k_pad, n_pad, dtype, trans_a, trans_b))
            else:
                cases.append((m, k, n, m_pad, k_pad, n_pad, dtype, False, False))

    return cases


def compute_confusion_matrix(results: list[BenchmarkResult], heuristic: str) -> dict:
    """Compute confusion matrix for a heuristic."""
    tp = tn = fp = fn = 0

    for r in results:
        actual = r.actual_should_pad
        if heuristic == "roofline":
            predicted = r.roofline_should_pad
        else:
            predicted = r.nvmatmul_should_pad

        if actual and predicted:
            tp += 1
        elif not actual and not predicted:
            tn += 1
        elif not actual and predicted:
            fp += 1
        else:  # actual and not predicted
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total, "accuracy": accuracy,
        "precision": precision, "recall": recall, "f1": f1
    }


def analyze_nvmatmul_error(results: list[BenchmarkResult]) -> dict:
    """Analyze nvMatmulHeuristics prediction error patterns."""
    errors = []

    for r in results:
        if r.nvmatmul_unpadded_us > 0 and r.unpadded_time_us > 0:
            ratio = r.unpadded_time_us / r.nvmatmul_unpadded_us
            error_pct = (r.unpadded_time_us - r.nvmatmul_unpadded_us) / r.unpadded_time_us * 100
            errors.append({
                'dtype': r.dtype,
                'layout': r.layout,
                'm': r.m, 'k': r.k, 'n': r.n,
                'problem_size': r.m * r.k * r.n,
                'actual_us': r.unpadded_time_us,
                'predicted_us': r.nvmatmul_unpadded_us,
                'ratio': ratio,
                'error_pct': error_pct,
            })

    if not errors:
        return {}

    # Group by dtype
    dtype_stats = {}
    for e in errors:
        dtype = e['dtype']
        if dtype not in dtype_stats:
            dtype_stats[dtype] = []
        dtype_stats[dtype].append(e)

    analysis = {
        'by_dtype': {},
        'overall': {}
    }

    # Per-dtype analysis
    for dtype, errs in dtype_stats.items():
        ratios = [e['ratio'] for e in errs]
        analysis['by_dtype'][dtype] = {
            'count': len(ratios),
            'mean_ratio': sum(ratios) / len(ratios),
            'min_ratio': min(ratios),
            'max_ratio': max(ratios),
            'median_ratio': sorted(ratios)[len(ratios) // 2],
        }

    # Overall analysis
    all_ratios = [e['ratio'] for e in errors]
    analysis['overall'] = {
        'count': len(all_ratios),
        'mean_ratio': sum(all_ratios) / len(all_ratios),
        'min_ratio': min(all_ratios),
        'max_ratio': max(all_ratios),
        'median_ratio': sorted(all_ratios)[len(all_ratios) // 2],
        'correction_factor': sum(all_ratios) / len(all_ratios),  # Multiply nvmatmul estimate by this
    }

    # Correlation with problem size
    sizes = [e['problem_size'] for e in errors]
    ratios = [e['ratio'] for e in errors]

    # Simple linear regression for correlation
    n = len(sizes)
    if n > 1:
        mean_size = sum(sizes) / n
        mean_ratio = sum(ratios) / n

        numerator = sum((s - mean_size) * (r - mean_ratio) for s, r in zip(sizes, ratios))
        denom_size = sum((s - mean_size) ** 2 for s in sizes)
        denom_ratio = sum((r - mean_ratio) ** 2 for r in ratios)

        if denom_size > 0 and denom_ratio > 0:
            correlation = numerator / ((denom_size * denom_ratio) ** 0.5)
            analysis['size_correlation'] = correlation

    return analysis


def compute_alignment_efficiency(results: list[BenchmarkResult]) -> dict:
    """
    Compute alignment efficiency coefficients from benchmark results.
    """
    # Group results by dtype
    dtype_results = {}
    for r in results:
        dtype = r.dtype
        if dtype not in dtype_results:
            dtype_results[dtype] = []
        dtype_results[dtype].append(r)

    fitted_params = {}

    for dtype, results_for_dtype in dtype_results.items():
        efficiencies = []

        for r in results_for_dtype:
            if r.padded_time_us > 0:
                # Raw GEMM efficiency (excluding padding overhead)
                gemm_efficiency = r.padded_time_us / r.unpadded_time_us

                # Compute misalignment
                alignment = 8 if dtype in ('float16', 'bfloat16') else 4
                k_misalign = r.k % alignment
                m_misalign = r.m % alignment
                n_misalign = r.n % alignment

                problem_size = r.m * r.k * r.n

                efficiencies.append({
                    'm': r.m, 'k': r.k, 'n': r.n,
                    'm_pad': r.m_pad, 'k_pad': r.k_pad, 'n_pad': r.n_pad,
                    'layout': r.layout,
                    'm_misalign': m_misalign,
                    'k_misalign': k_misalign,
                    'n_misalign': n_misalign,
                    'gemm_efficiency': gemm_efficiency,
                    'actual_speedup': r.actual_speedup,
                    'problem_size': problem_size,
                })

        fitted_params[dtype] = efficiencies

    return fitted_params


def fit_efficiency_table(results: list[BenchmarkResult]) -> dict:
    """
    Fit alignment efficiency lookup table from benchmark results.
    """
    efficiency_data = compute_alignment_efficiency(results)

    fitted_tables = {}

    for dtype, efficiencies in efficiency_data.items():
        misalign_groups = {}

        for e in efficiencies:
            k_mis = e['k_misalign']
            if k_mis not in misalign_groups:
                misalign_groups[k_mis] = []
            misalign_groups[k_mis].append(e)

        efficiency_table = {}
        for misalign, group in misalign_groups.items():
            total_weight = sum(e['problem_size'] for e in group)
            if total_weight > 0:
                weighted_eff = sum(
                    e['gemm_efficiency'] * e['problem_size']
                    for e in group
                ) / total_weight
            else:
                weighted_eff = 1.0

            efficiency_table[misalign] = weighted_eff

        efficiency_table[0] = 1.0
        fitted_tables[dtype] = efficiency_table

    return fitted_tables


def generate_efficiency_table_code(fitted_tables: dict, compute_capability: tuple) -> str:
    """Generate Python code for the fitted efficiency table."""
    cc_str = f"({compute_capability[0]}, {compute_capability[1]})"

    lines = [
        f"    # Fitted from benchmark measurements on compute capability {cc_str}",
        f"    # GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'Unknown'}",
        f"    cc_{compute_capability[0]}_{compute_capability[1]}_efficiency = {{",
    ]

    unified_table = {}
    for dtype, table in fitted_tables.items():
        if dtype in ('float16', 'bfloat16'):
            for misalign, eff in table.items():
                if misalign not in unified_table:
                    unified_table[misalign] = eff
                else:
                    unified_table[misalign] = min(unified_table[misalign], eff)

    for misalign in sorted(unified_table.keys()):
        eff = unified_table[misalign]
        lines.append(f"        {misalign}: {eff:.4f},")

    lines.append("    }")

    return "\n".join(lines)


def print_results(results: list[BenchmarkResult], output_file: str = None):
    """Print results in a readable format and optionally save to CSV."""

    print("\n" + "=" * 140)
    print("PAD_MM HEURISTICS BENCHMARK RESULTS")
    print("=" * 140)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print()

    # Print table
    header = (
        f"{'Dimensions':<25} | {'Layout':<6} | {'dtype':<8} | "
        f"{'Unpad(us)':<10} | {'Pad(us)':<10} | {'Ovrhd(us)':<10} | "
        f"{'Speedup':<8} | {'Actual':<7} | {'Roofline':<8} | {'nvMatmul':<8}"
    )
    print(header)
    print("-" * 140)

    for r in results:
        dims = f"M={r.m} K={r.k}+{r.k_pad} N={r.n}"
        row = (
            f"{dims:<25} | {r.layout:<6} | {r.dtype:<8} | "
            f"{r.unpadded_time_us:<10.2f} | {r.padded_time_us:<10.2f} | {r.padding_overhead_us:<10.2f} | "
            f"{r.actual_speedup:<8.3f} | {str(r.actual_should_pad):<7} | {str(r.roofline_should_pad):<8} | {str(r.nvmatmul_should_pad):<8}"
        )
        print(row)

    # Confusion matrices
    print("\n" + "=" * 140)
    print("CONFUSION MATRICES")
    print("=" * 140)

    for heuristic in ["roofline", "nvmatmul"]:
        cm = compute_confusion_matrix(results, heuristic)
        print(f"\n{heuristic.upper()} Heuristic:")
        print("-" * 60)
        print(f"                    Predicted Pad  |  Predicted No-Pad")
        print(f"  Actual Pad     |      TP={cm['tp']:<4}    |      FN={cm['fn']:<4}")
        print(f"  Actual No-Pad  |      FP={cm['fp']:<4}    |      TN={cm['tn']:<4}")
        print(f"\n  Accuracy:  {cm['accuracy']*100:.1f}% ({cm['tp']+cm['tn']}/{cm['total']})")
        print(f"  Precision: {cm['precision']*100:.1f}% (of predicted pads, how many were correct)")
        print(f"  Recall:    {cm['recall']*100:.1f}% (of actual pads, how many were detected)")
        print(f"  F1 Score:  {cm['f1']*100:.1f}%")

    # Detailed error analysis
    print("\n" + "=" * 140)
    print("DETAILED ERROR CASES")
    print("=" * 140)

    for heuristic in ["roofline", "nvmatmul"]:
        print(f"\n{heuristic.upper()} False Positives (predicted pad, but shouldn't):")
        for r in results:
            predicted = r.roofline_should_pad if heuristic == "roofline" else r.nvmatmul_should_pad
            if predicted and not r.actual_should_pad:
                print(f"  M={r.m} K={r.k}+{r.k_pad} N={r.n} {r.layout} {r.dtype}: speedup={r.actual_speedup:.3f}")

        print(f"\n{heuristic.upper()} False Negatives (predicted no pad, but should):")
        for r in results:
            predicted = r.roofline_should_pad if heuristic == "roofline" else r.nvmatmul_should_pad
            if not predicted and r.actual_should_pad:
                print(f"  M={r.m} K={r.k}+{r.k_pad} N={r.n} {r.layout} {r.dtype}: speedup={r.actual_speedup:.3f}")

    # nvMatmulHeuristics error analysis
    print("\n" + "=" * 140)
    print("nvMatmulHeuristics TIMING ERROR ANALYSIS")
    print("=" * 140)

    nvmatmul_analysis = analyze_nvmatmul_error(results)

    if nvmatmul_analysis:
        print("\nPer-dtype timing ratio (actual / predicted):")
        for dtype, stats in nvmatmul_analysis.get('by_dtype', {}).items():
            print(f"  {dtype}:")
            print(f"    Mean ratio:   {stats['mean_ratio']:.2f}x")
            print(f"    Median ratio: {stats['median_ratio']:.2f}x")
            print(f"    Range:        {stats['min_ratio']:.2f}x - {stats['max_ratio']:.2f}x")

        overall = nvmatmul_analysis.get('overall', {})
        print(f"\nOverall:")
        print(f"  Mean ratio:   {overall.get('mean_ratio', 0):.2f}x (actual is this much slower than predicted)")
        print(f"  Median ratio: {overall.get('median_ratio', 0):.2f}x")
        print(f"  Suggested correction factor: {overall.get('correction_factor', 1):.2f}x")

        if 'size_correlation' in nvmatmul_analysis:
            print(f"  Correlation with problem size: {nvmatmul_analysis['size_correlation']:.3f}")
            if nvmatmul_analysis['size_correlation'] > 0.5:
                print("  -> Strong positive correlation: larger problems have bigger estimation errors")
            elif nvmatmul_analysis['size_correlation'] < -0.5:
                print("  -> Strong negative correlation: smaller problems have bigger estimation errors")

    print("\nDetailed timing comparison:")
    print(f"{'Dimensions':<25} | {'Layout':<6} | {'dtype':<8} | {'Actual(us)':<12} | {'nvMatmul(us)':<12} | {'Ratio':<8}")
    print("-" * 100)
    for r in results:
        if r.nvmatmul_unpadded_us > 0:
            dims = f"M={r.m} K={r.k} N={r.n}"
            ratio = r.unpadded_time_us / r.nvmatmul_unpadded_us if r.nvmatmul_unpadded_us > 0 else 0
            print(f"{dims:<25} | {r.layout:<6} | {r.dtype:<8} | {r.unpadded_time_us:<12.2f} | {r.nvmatmul_unpadded_us:<12.2f} | {ratio:<8.2f}")

    # Save to CSV
    if output_file:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'M', 'K', 'N', 'M_pad', 'K_pad', 'N_pad', 'dtype', 'layout',
                'unpadded_time_us', 'padded_time_us', 'padding_overhead_us',
                'actual_should_pad', 'actual_speedup',
                'roofline_should_pad', 'nvmatmul_should_pad',
                'nvmatmul_unpadded_us', 'nvmatmul_padded_us',
                'roofline_correct', 'nvmatmul_correct'
            ])
            for r in results:
                writer.writerow([
                    r.m, r.k, r.n, r.m_pad, r.k_pad, r.n_pad, r.dtype, r.layout,
                    r.unpadded_time_us, r.padded_time_us, r.padding_overhead_us,
                    r.actual_should_pad, r.actual_speedup,
                    r.roofline_should_pad, r.nvmatmul_should_pad,
                    r.nvmatmul_unpadded_us, r.nvmatmul_padded_us,
                    r.roofline_should_pad == r.actual_should_pad,
                    r.nvmatmul_should_pad == r.actual_should_pad,
                ])
        print(f"\nResults saved to: {output_file}")


def print_fitted_results(fit_results: dict, compute_capability: tuple):
    """Print the fitted results and generated code."""

    print("\n" + "=" * 100)
    print("FITTED ALIGNMENT EFFICIENCY TABLES")
    print("=" * 100)

    fitted_tables = fit_results['fitted_tables']

    for dtype, table in fitted_tables.items():
        print(f"\n{dtype}:")
        print("-" * 50)
        for misalign in sorted(table.keys()):
            eff = table[misalign]
            status = "PADDING HELPS" if eff < 0.95 else "NO BENEFIT"
            print(f"  Misalignment {misalign}: efficiency = {eff:.4f} ({status})")

    print("\n" + "=" * 100)
    print("GENERATED CODE FOR pad_mm.py")
    print("=" * 100)
    print("\n# Add this to _get_alignment_efficiency_table() in pad_mm.py:\n")
    print(generate_efficiency_table_code(fitted_tables, compute_capability))

    print("\n# Recommendation based on measurements:")
    for dtype, table in fitted_tables.items():
        avg_eff = sum(table.values()) / len(table) if table else 1.0
        if avg_eff > 0.95:
            print(f"# {dtype}: Padding provides NO benefit - consider disabling")
        else:
            print(f"# {dtype}: Padding provides significant benefit (avg efficiency = {avg_eff:.4f})")


def fit_and_evaluate(results: list[BenchmarkResult]) -> dict:
    """Fit efficiency coefficients and evaluate their accuracy."""
    fitted_tables = fit_efficiency_table(results)

    return {
        'fitted_tables': fitted_tables,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pad_mm heuristics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV file path")
    parser.add_argument("--quick", action="store_true",
                        help="Run a quick subset of tests")
    parser.add_argument("--fit", action="store_true",
                        help="Fit alignment efficiency coefficients from measurements")
    parser.add_argument("--load-csv", type=str, default=None,
                        help="Load results from CSV instead of running benchmarks")
    parser.add_argument("--use-hardware-descriptor", action="store_true",
                        help="Use nvMatmulHeuristics hardware descriptor for the current GPU")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        sys.exit(1)

    compute_capability = torch.cuda.get_device_capability()

    # Setup hardware descriptor if requested
    hardware_descriptor = None
    heuristics_interface = None

    if args.use_hardware_descriptor:
        try:
            from nvMatmulHeuristics import (
                NvMatmulHeuristicsInterface,
                NvMatmulHeuristicsTarget,
                NvMatmulHeuristicsNvidiaGpu,
            )

            # Map compute capability to predefined GPU
            cc_to_gpu = {
                (8, 0): NvMatmulHeuristicsNvidiaGpu.A100_SXM_80GB,
                (8, 6): NvMatmulHeuristicsNvidiaGpu.A40_PCIE,
                (8, 9): NvMatmulHeuristicsNvidiaGpu.L40S,
                (9, 0): NvMatmulHeuristicsNvidiaGpu.H100_SXM,
                (10, 0): NvMatmulHeuristicsNvidiaGpu.B200,
            }

            gpu = cc_to_gpu.get(compute_capability)
            if gpu:
                heuristics_interface = NvMatmulHeuristicsInterface(
                    backend=NvMatmulHeuristicsTarget.GENERIC,
                    precision='HSS',
                )
                hardware_descriptor = heuristics_interface.createHardwareDescriptor()
                heuristics_interface.setHardwarePredefinedGpu(hardware_descriptor, gpu)
                print(f"Using nvMatmulHeuristics hardware descriptor for {gpu.name}")
            else:
                print(f"No predefined GPU for compute capability {compute_capability}")
        except Exception as e:
            print(f"Warning: Could not setup hardware descriptor: {e}")

    if args.load_csv:
        # Load results from CSV
        results = []
        with open(args.load_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(BenchmarkResult(
                    m=int(row['M']),
                    k=int(row['K']),
                    n=int(row['N']),
                    m_pad=int(row['M_pad']),
                    k_pad=int(row['K_pad']),
                    n_pad=int(row['N_pad']),
                    dtype=row['dtype'],
                    layout=row.get('layout', 'NN'),
                    unpadded_time_us=float(row['unpadded_time_us']),
                    padded_time_us=float(row['padded_time_us']),
                    padding_overhead_us=float(row['padding_overhead_us']),
                    actual_should_pad=row['actual_should_pad'] == 'True',
                    actual_speedup=float(row['actual_speedup']),
                    roofline_should_pad=row['roofline_should_pad'] == 'True',
                    nvmatmul_should_pad=row['nvmatmul_should_pad'] == 'True',
                    nvmatmul_unpadded_us=float(row['nvmatmul_unpadded_us']),
                    nvmatmul_padded_us=float(row['nvmatmul_padded_us']),
                ))
        print(f"Loaded {len(results)} results from {args.load_csv}")
    else:
        # Generate test cases
        test_cases = generate_test_cases()

        if args.quick:
            test_cases = test_cases[:15]

        print(f"Running {len(test_cases)} benchmark cases...")
        print("This may take several minutes.\n")

        results = []
        for i, (m, k, n, m_pad, k_pad, n_pad, dtype, trans_a, trans_b) in enumerate(test_cases):
            layout = ('T' if trans_a else 'N') + ('T' if trans_b else 'N')
            print(f"[{i+1}/{len(test_cases)}] M={m} K={k}+{k_pad} N={n} {layout} dtype={dtype}...")
            result = run_benchmark(
                m, k, n, m_pad, k_pad, n_pad, dtype, trans_a, trans_b,
                hardware_descriptor=hardware_descriptor,
                heuristics_interface=heuristics_interface,
            )
            results.append(result)

    print_results(results, args.output)

    if args.fit:
        fit_results = fit_and_evaluate(results)
        print_fitted_results(fit_results, compute_capability)


if __name__ == "__main__":
    main()
