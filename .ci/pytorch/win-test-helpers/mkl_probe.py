"""Temporary diagnostics for the Windows MKL 2026.2 LBFGS NaN. Not for landing.

Prints one GitHub ::notice:: line per fact so that it is readable from the check-run
annotations without admin access to the job logs.
"""

import os
import re
import subprocess
import sys


def notice(msg):
    print(f"::notice title=mkl-probe::{msg}", flush=True)


def child(tag):
    import psutil

    import torch

    mods = sorted(
        {os.path.basename(m.path).lower() for m in psutil.Process().memory_maps()}
    )
    mods = [m for m in mods if re.search(r"mkl|iomp|omp|svml", m)]
    notice(
        f"[{tag}] torch={torch.__version__} mkl={torch.backends.mkl.is_available()} dlls={mods}"
    )

    torch.manual_seed(0)
    bad_dot = []
    for n in range(1, 130):
        a, b = torch.randn(n), torch.randn(n)
        got, ref = torch.dot(a, b).item(), torch.dot(a.double(), b.double()).item()
        if not (abs(got - ref) <= 1e-3 * (1 + abs(ref))):
            bad_dot.append(n)

    bad_mv = []
    for m in (1, 2, 3, 5, 10, 50):
        for n in (1, 2, 3, 5, 10, 50):
            A, x = torch.randn(m, n), torch.randn(n)
            if not torch.allclose(
                A.mv(x).double(), A.double().mv(x.double()), atol=1e-3
            ):
                bad_mv.append((m, n))

    losses = []
    p = torch.zeros(50, requires_grad=True)
    target = torch.arange(50, dtype=torch.float32)
    opt = torch.optim.LBFGS([p], lr=1.0)

    def closure():
        opt.zero_grad()
        loss = ((p - target) ** 2).sum()
        loss.backward()
        return loss

    for _ in range(5):
        losses.append(float(opt.step(closure)))
    notice(
        f"[{tag}] bad_dot={bad_dot[:12]}({len(bad_dot)}) bad_mv={bad_mv[:8]}({len(bad_mv)}) lbfgs_losses={losses}"
    )


def main():
    if len(sys.argv) > 1:
        try:
            child(sys.argv[1])
        except Exception as e:
            notice(f"[{sys.argv[1]}] probe failed: {type(e).__name__}: {e}")
        return
    variants = {
        "default": {},
        "avx2": {"MKL_ENABLE_INSTRUCTIONS": "AVX2"},
        "sse42": {"MKL_ENABLE_INSTRUCTIONS": "SSE4_2"},
        "1thread": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    }
    for tag, env in variants.items():
        subprocess.run(
            [sys.executable, __file__, tag], env={**os.environ, **env}, check=False
        )


if __name__ == "__main__":
    main()
