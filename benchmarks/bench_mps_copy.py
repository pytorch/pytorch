"""Benchmark copy_from_mps_ (MPS->CPU): zero-copy via shared buffer vs blit (PR #182736).

On Apple Silicon UMA, MPS tensors share physical memory with CPU. This PR adds a
fast path in copy_from_mps_ that reads directly from the shared MTLBuffer via
getSharedBufferPtr(), avoiding a Metal blit command.

Fast path: getSharedBufferPtr() + COMMIT_AND_WAIT + memcpy  (bandwidth-bound)
Blit path: encode blit + commit + wait + staging copy        (command overhead dominates at small sizes)

Usage:
    python benchmarks/bench_mps_copy.py
"""
import torch
from torch.utils.benchmark import Timer

sizes_kb = [64, 256, 1024, 4096, 16384]

print(f"torch={torch.__version__}  device=mps (Apple Silicon UMA)")
print(f"Methodology: torch.utils.benchmark.Timer.blocked_autorange (median +/- IQR)")
print()
print(f"{'Size':>10} {'to(cpu) (us)':>14}  {'GB/s':>6}")
print("-" * 36)

for kb in sizes_kb:
    n = kb * 1024 // 4  # float32 = 4 bytes
    setup = f"import torch; x = torch.randn({n}, device='mps')"
    stmt  = "x.to('cpu'); torch.mps.synchronize()"
    t = Timer(stmt=stmt, setup=setup).blocked_autorange(min_run_time=1.0)
    bw = (kb / 1024) / (t.median * 1e3)  # GB/s
    print(f"{str(kb)+'KB':>10} {t.median*1e6:>10.2f} +/- {t.iqr*1e6:<6.2f}  {bw:>5.1f}")
