"""
Example: GEMM FLOPS and Bandwidth Utilization Analysis

This script demonstrates how the PyTorch profiler automatically computes
and annotates FLOPS and bandwidth utilization for Triton GEMM kernels.

It shows two types of GEMMs:
1. FLOPS-bound (large square matrices): High arithmetic intensity
2. Bandwidth-bound (tall-skinny matrices): Low arithmetic intensity

Usage:
    python examples/gemm_utilization_example.py

Requirements:
    - CUDA-capable GPU
    - PyTorch with torch.compile support
"""

import json
import tempfile
import torch
from torch.profiler import profile, ProfilerActivity


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        return

    device_name = torch.cuda.get_device_name()
    print(f"Device: {device_name}")
    print()

    # Configuration for max-autotune with Triton backend
    torch._inductor.config.force_disable_caches = True
    torch._inductor.config.max_autotune_gemm_backends = "TRITON"

    # Define compiled functions for different GEMM types
    @torch.compile(mode="max-autotune-no-cudagraphs")
    def large_square_mm(a, b):
        """FLOPS-bound: 4096x4096 @ 4096x4096

        FLOPS = 2 * 4096^3 = 137 GFLOPS
        Memory = 3 * 4096^2 * 4 bytes = 201 MB
        Arithmetic Intensity = 341 FLOP/byte (very high)
        """
        return torch.mm(a, b)

    @torch.compile(mode="max-autotune-no-cudagraphs")
    def tall_skinny_mm(a, b):
        """Bandwidth-bound: 65536x64 @ 64x64

        FLOPS = 2 * 65536 * 64 * 64 = 536 MFLOPS
        Memory = (65536*64 + 64*64 + 65536*64) * 4 bytes = 33.6 MB
        Arithmetic Intensity = 8 FLOP/byte (low)
        """
        return torch.mm(a, b)

    # Create test tensors
    print("Creating tensors...")
    large_a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    large_b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    skinny_a = torch.randn(65536, 64, device="cuda", dtype=torch.float32)
    skinny_b = torch.randn(64, 64, device="cuda", dtype=torch.float32)

    # Warmup (triggers compilation and autotuning)
    print("Warming up (includes autotuning)...")
    for _ in range(3):
        _ = large_square_mm(large_a, large_b)
        _ = tall_skinny_mm(skinny_a, skinny_b)
    torch.cuda.synchronize()

    # Profile - utilization annotations are added automatically on export
    print("Profiling...")
    trace_path = "/tmp/gemm_utilization_trace.json"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(3):
            _ = large_square_mm(large_a, large_b)
        torch.cuda.synchronize()

        for _ in range(3):
            _ = tall_skinny_mm(skinny_a, skinny_b)
        torch.cuda.synchronize()

    # Export trace - utilization annotations added automatically
    prof.export_chrome_trace(trace_path)

    # Load and analyze results
    with open(trace_path) as f:
        data = json.load(f)

    print()
    print("=" * 75)
    print("GEMM UTILIZATION ANALYSIS")
    print("=" * 75)

    # Find kernel events with utilization data
    kernel_events = [
        e
        for e in data["traceEvents"]
        if e.get("cat") == "kernel"
        and "achieved_flops_percent" in e.get("args", {})
    ]

    # Separate large and small GEMMs by FLOPS count
    large_gemms = [e for e in kernel_events if e["args"].get("kernel_flop", 0) > 1e10]
    small_gemms = [e for e in kernel_events if e["args"].get("kernel_flop", 0) < 1e10]

    def print_gemm_stats(events, label, description):
        if not events:
            print(f"\nNo {label} kernels found")
            return

        # Compute averages
        avg_flops = sum(e["args"]["achieved_flops_percent"] for e in events) / len(events)
        avg_bw = sum(e["args"]["achieved_bandwidth_percent"] for e in events) / len(events)
        avg_dur = sum(e.get("dur", 0) for e in events) / len(events)
        avg_roofline = sum(e["args"].get("roofline_efficiency_percent", 0) for e in events) / len(events)

        args = events[0]["args"]
        kernel_flop = args.get("kernel_flop", 0)
        kernel_num_gb = args.get("kernel_num_gb", 0)
        arith_intensity = args.get("arithmetic_intensity", 0)
        ridge_point = args.get("ridge_point", 0)
        roofline_ceiling = args.get("roofline_ceiling_tflops", 0)
        roofline_bound = args.get("roofline_bound", "unknown")

        print(f"\n{label}")
        print(f"  {description}")
        print("-" * 60)
        print(f"  Invocations:          {len(events)}")
        print(f"  Avg Duration:         {avg_dur:,.0f} us")
        print(f"  FLOPS per call:       {kernel_flop / 1e9:.2f} GFLOPS")
        print(f"  Memory per call:      {kernel_num_gb * 1000:.2f} MB")
        print()
        print(f"  Arithmetic Intensity: {arith_intensity:.1f} FLOP/byte")
        print(f"  Ridge Point:          {ridge_point:.1f} FLOP/byte")
        print(f"  Roofline Ceiling:     {roofline_ceiling:.2f} TFLOPS")
        print(f"  Roofline Bound:       {roofline_bound.upper()}")
        print()
        print(f"  Achieved FLOPS:       {avg_flops:6.2f}%")
        print(f"  Achieved Bandwidth:   {avg_bw:6.2f}%")
        print(f"  Roofline Efficiency:  {avg_roofline:6.2f}%")

    print_gemm_stats(
        large_gemms,
        "Large Square GEMM",
        "4096x4096 @ 4096x4096 - High arithmetic intensity",
    )
    print_gemm_stats(
        small_gemms,
        "Tall-Skinny GEMM",
        "65536x64 @ 64x64 - Low arithmetic intensity",
    )

    print()
    print("=" * 75)
    print()
    print("ROOFLINE MODEL EXPLANATION")
    print("-" * 75)
    print()
    print("  The roofline model visualizes performance limits:")
    print()
    print("  Performance")
    print("       ^")
    print("       |         _______________  <- Peak FLOPS (compute ceiling)")
    print("       |        /")
    print("       |       /")
    print("       |      /  <- Memory bandwidth ceiling")
    print("       |     /")
    print("       |    /")
    print("       +---/-----------------> Arithmetic Intensity (FLOP/byte)")
    print("           ^")
    print("           Ridge Point")
    print()
    print("  Key metrics:")
    print("    - Arithmetic Intensity: FLOPS / bytes accessed")
    print("    - Ridge Point: Where memory and compute ceilings meet")
    print("    - Roofline Ceiling: Max achievable performance for this kernel")
    print("    - Roofline Efficiency: Actual / Ceiling (how close to theoretical max)")
    print()
    print("  If AI < Ridge Point -> MEMORY-BOUND (limited by bandwidth)")
    print("  If AI > Ridge Point -> COMPUTE-BOUND (limited by FLOPS)")
    print()
    print("=" * 75)
    print()
    print(f"Trace saved to: {trace_path}")
    print("View in chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
