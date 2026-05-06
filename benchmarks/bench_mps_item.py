"""Benchmark _local_scalar_dense_mps (.item()) on Apple Silicon UMA (PR #182731).

On Apple Silicon, MPS tensors use shared CPU/GPU memory. This PR adds a fast
path that reads directly from the shared buffer after GPU sync, avoiding the
Metal blit command that the previous implementation used.

Fast path: getSharedBufferPtr() + COMMIT_AND_WAIT + direct read  (~0.5 us)
Blit path: encode blit command + commit + wait + CPU staging copy (~100 us)

Usage:
    python benchmarks/bench_mps_item.py
"""
import torch
from torch.utils.benchmark import Timer

print(f"torch={torch.__version__}  device=mps (Apple Silicon UMA)")
print(f"Methodology: torch.utils.benchmark.Timer.blocked_autorange (median +/- IQR)")
print()
print(f"{'Tensor':>16} {'item() (us)':>13}")
print("-" * 32)

for n in [1, 64, 256, 1024, 4096]:
    setup = f"import torch; x = torch.randn({n}, device='mps')"
    stmt  = "x[-1].item(); torch.mps.synchronize()"
    t = Timer(stmt=stmt, setup=setup).blocked_autorange(min_run_time=1.0)
    print(f"{str(n)+' elem':>16} {t.median*1e6:>9.2f} +/- {t.iqr*1e6:.2f}")
