"""Benchmark copy_to_mps_stride_contig (CPU->MPS): zero-copy fast path vs blit (PR #182749).

On Apple Silicon UMA, CPU and GPU share physical memory. This PR adds a fast path in
copy_to_mps_stride_contig: for contiguous, same-dtype copies, write directly to the
MTLBuffer's CPU-visible memory via [destBuffer contents] + memcpy. No Metal blit needed.

Fast path: [destBuffer contents] + memcpy  (bandwidth-bound, no GPU command)
Blit path: encode blit command + GPU staging copy (command overhead dominates at small sizes)

Usage:
    python benchmarks/bench_mps_copy_to_mps.py
"""
import torch
from torch.utils.benchmark import Timer

sizes_kb = [64, 256, 1024, 4096, 16384]

print(f"torch={torch.__version__}  device=mps (Apple Silicon UMA)")
print(f"Methodology: torch.utils.benchmark.Timer.blocked_autorange (median +/- IQR)")
print()
print(f"{'Size':>10} {'to(mps) (us)':>14}  {'GB/s':>6}")
print("-" * 36)

for kb in sizes_kb:
    n = kb * 1024 // 4  # float32 = 4 bytes
    setup = f"import torch; x = torch.randn({n})"  # CPU tensor
    stmt  = "x.to('mps'); torch.mps.synchronize()"
    t = Timer(stmt=stmt, setup=setup).blocked_autorange(min_run_time=1.0)
    bw = (kb / 1024) / (t.median * 1e3)  # GB/s
    print(f"{str(kb)+'KB':>10} {t.median*1e6:>10.2f} +/- {t.iqr*1e6:<6.2f}  {bw:>5.1f}")
