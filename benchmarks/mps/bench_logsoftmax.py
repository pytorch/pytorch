"""MPS log_softmax: native Metal kernels (IS_LOG variant) vs the kept MPSGraph path.

Same-build A/B. log_softmax reuses the softmax kernels (IS_LOG=true) and routes
through the same canUseMetalSoftmax gate, so the softmax escape hatch controls it:
mode A = forced MPSGraph (PYTORCH_MPS_FORCE_MPSGRAPH_SOFTMAX=1), mode B = Metal
(env unset). Toggle read once per process -> run each MODE in its own process.
Methodology: amortized many-iter / one-sync (dodges the MPS sync floor).
Requires this checkout importable as torch (pip install -e . or PYTHONPATH).
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F


if not torch.backends.mps.is_available():
    raise RuntimeError("MPS not available")
DEV = torch.device("mps")
MODE = os.environ.get("BENCH_MODE", "B")
DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
SHAPES = [
    ("8x2048x4096_d-1", (8, 2048, 4096), -1),
    ("8x2048x4096_d0", (8, 2048, 4096), 0),
    ("128x65536_d-1", (128, 65536), -1),
    ("65536x128_d0", (65536, 128), 0),
    ("1024x1024_d-1", (1024, 1024), -1),
    ("4x65536_d-1", (4, 65536), -1),
]
ITERS = int(os.environ.get("ITERS", "200"))


def bench(shape, dim, dt, backward):
    x = torch.randn(shape, device=DEV, dtype=dt, requires_grad=backward)
    for _ in range(20):
        y = F.log_softmax(x, dim=dim)
        if backward:
            x.grad = None
            y.backward(torch.ones_like(y))
    torch.mps.synchronize()
    s = time.perf_counter()
    for _ in range(ITERS):
        y = F.log_softmax(x, dim=dim)
        if backward:
            x.grad = None
            y.backward(torch.ones_like(y))
    torch.mps.synchronize()
    return round((time.perf_counter() - s) / ITERS * 1e6, 1)


forced = os.environ.get("PYTORCH_MPS_FORCE_MPSGRAPH_SOFTMAX")
sys.stderr.write(f"MODE={MODE} FORCE_ENV={forced!r} torch={torch.__file__}\n")
out = {}
for nm, sh, dim in SHAPES:
    for dn, dt in DTYPES.items():
        for kind in ("fwd", "fwdbwd"):
            out[f"{nm}|{dn}|{kind}"] = bench(sh, dim, dt, kind == "fwdbwd")
print(json.dumps(out))
