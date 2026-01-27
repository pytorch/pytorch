"""
Example: Automatic FLOPS and Bandwidth Utilization in PyTorch Profiler

This example demonstrates how the profiler automatically adds utilization
annotations to traces. No setup or callbacks required - just use the profiler
normally and the trace will include achieved_flops_percent and
achieved_bandwidth_percent for each kernel.

This helps identify whether kernels are:
- FLOPS-bound (compute limited): high FLOPS%, low bandwidth%
- Bandwidth-bound (memory limited): low FLOPS%, high bandwidth%

Note: Utilization annotations work for cuBLAS kernels (mm, addmm, conv, etc.)
which are captured with full metadata by the profiler. Operations that use
ATen ops will show utilization. torch.compile uses cuBLAS by default for
matrix operations.
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

    # Create test tensors
    # FLOPS-bound: large square matrices (high arithmetic intensity)
    print("Creating tensors...")
    large_a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    large_b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)

    # Bandwidth-bound: tall-skinny matrices (low arithmetic intensity)
    skinny_a = torch.randn(16384, 64, device="cuda", dtype=torch.float32)
    skinny_b = torch.randn(64, 64, device="cuda", dtype=torch.float32)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = torch.mm(large_a, large_b)
        _ = torch.mm(skinny_a, skinny_b)
    torch.cuda.synchronize()

    # Profile - just use the profiler normally!
    print("Profiling...")
    trace_path = tempfile.mktemp(suffix=".json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        # FLOPS-bound: 4096x4096 x 4096x4096
        for _ in range(3):
            c1 = torch.mm(large_a, large_b)
        torch.cuda.synchronize()

        # Bandwidth-bound: 16384x64 x 64x64
        for _ in range(3):
            c2 = torch.mm(skinny_a, skinny_b)
        torch.cuda.synchronize()

    # Export - utilization annotations added automatically!
    prof.export_chrome_trace(trace_path)
    print(f"Trace saved to: {trace_path}")
    print()

    # Load and display results
    with open(trace_path) as f:
        data = json.load(f)

    print("=" * 70)
    print("KERNEL UTILIZATION RESULTS (automatically computed)")
    print("=" * 70)

    # Find kernel events with utilization data
    kernel_events = [
        e for e in data["traceEvents"]
        if e.get("cat") == "kernel"
        and ("achieved_flops_percent" in e.get("args", {})
             or "achieved_bandwidth_percent" in e.get("args", {}))
    ]

    for i, event in enumerate(kernel_events):
        args = event.get("args", {})
        name = event.get("name", "unknown")[:50]
        dur_us = event.get("dur", 0)

        kernel_flop = args.get("kernel_flop", 0)
        kernel_num_gb = args.get("kernel_num_gb", 0)
        achieved_flops = args.get("achieved_flops_percent", 0)
        achieved_bw = args.get("achieved_bandwidth_percent", 0)

        print(f"\nKernel {i + 1}: {name}")
        print(f"  Duration:           {dur_us:,.1f} us")
        print(f"  FLOPS:              {kernel_flop / 1e9:,.2f} GFLOPS")
        print(f"  Memory accessed:    {kernel_num_gb * 1000:,.2f} MB")
        print(f"  Achieved FLOPS:     {achieved_flops:6.2f}%")
        print(f"  Achieved Bandwidth: {achieved_bw:6.2f}%")

        # Classify the kernel
        if achieved_flops > achieved_bw and achieved_flops > 5:
            print("  --> FLOPS-BOUND (compute limited)")
        elif achieved_bw > achieved_flops and achieved_bw > 5:
            print("  --> BANDWIDTH-BOUND (memory limited)")

    print()
    print("=" * 70)
    print()
    print("The trace file can be viewed in chrome://tracing or Perfetto.")
    print("Each kernel event now has 'achieved_flops_percent' and")
    print("'achieved_bandwidth_percent' in its args.")

    # Cleanup
    import os
    os.unlink(trace_path)


def torch_compile_example():
    """Example with torch.compile - uses cuBLAS by default for matmul."""
    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        return

    print("\n" + "=" * 70)
    print("TORCH.COMPILE EXAMPLE")
    print("=" * 70 + "\n")

    @torch.compile
    def compiled_matmul(a, b):
        return torch.mm(a, b)

    # Large GEMM
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)

    # Warmup/compile
    for _ in range(3):
        _ = compiled_matmul(a, b)
    torch.cuda.synchronize()

    trace_path = tempfile.mktemp(suffix=".json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(3):
            c = compiled_matmul(a, b)
        torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)

    with open(trace_path) as f:
        data = json.load(f)

    # Find kernel events
    kernel_events = [
        e for e in data["traceEvents"]
        if e.get("cat") == "kernel"
        and ("achieved_flops_percent" in e.get("args", {})
             or "achieved_bandwidth_percent" in e.get("args", {}))
    ]

    if kernel_events:
        print("torch.compile GEMM kernels:")
        for event in kernel_events[:3]:
            args = event.get("args", {})
            print(f"  {event['name'][:50]}")
            print(f"    FLOPS: {args.get('achieved_flops_percent', 0):.1f}%")
            print(f"    BW:    {args.get('achieved_bandwidth_percent', 0):.1f}%")
    else:
        print("No kernels with utilization data found")

    import os
    os.unlink(trace_path)


if __name__ == "__main__":
    main()
    torch_compile_example()
