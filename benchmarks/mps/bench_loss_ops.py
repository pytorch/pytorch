"""
Benchmark for MPS loss ops: MSE, BCE, SmoothL1, Huber, NLL, CrossEntropy.
Uses torch.utils.benchmark.Timer.blocked_autorange — no hand-rolled timing.

For reduction=none, fwd+bwd uses .backward(rand_weights) to model the real
use case (per-sample importance weighting). reduction=mean/sum use .sum().backward().

Covers:
  - All ops replaced in this PR (forward + backward)
  - Standard shapes + LLM training workloads (LLaMA / GPT-4 scale)
  - float32, float16, bfloat16

Run: python benchmarks/mps/bench_loss_ops.py
"""

import itertools

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer


DEVICE = "mps"
MIN_RUN = 1.0  # seconds per config for blocked_autorange

# Warm up the MPS device + PSO cache
for _ in range(30):
    torch.randn(1024, device=DEVICE).sum()
torch.mps.synchronize()

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
REDUCTIONS = ["none", "mean", "sum"]

# Pointwise (MSE / SmoothL1 / Huber / BCE) shapes.
# Includes standard small/medium/large + LLM-scale hidden-state tensors.
POINTWISE_SHAPES = [
    (4096,),  # 1D medium
    (1 << 20,),  # 1D large (1M)
    (32, 4096),  # batch=32, hidden=4096 (LLaMA hidden)
    (16, 512, 4096),  # batch=16, seq=512, hidden=4096 (LLM activations)
    (8, 2048, 4096),  # batch=8, seq=2048, hidden=4096
]

# (N, C) = (batch*seq_len, vocab_size) for NLL / CrossEntropy.
NLL_SHAPES = [
    (1024, 1000),  # standard CIFAR-scale
    (1024, 32000),  # LLaMA-2 vocab, 1K tokens
    (4096, 32000),  # LLaMA-2 vocab, 4K tokens
    (512, 128256),  # LLaMA-3 vocab, 512 tokens
    (2048, 128256),  # LLaMA-3 vocab, 2K tokens
]


# ── helpers ────────────────────────────────────────────────────────────────


def hdr(title):
    print(f"\n{'─' * 100}")
    print(f"  {title}")
    print(f"{'─' * 100}")
    print(
        f"  {'shape':<28} {'dtype':<10} {'red':<6} {'fwd median µs':>15} {'fwd+bwd median µs':>19}"
    )
    print(f"  {'─' * 28} {'─' * 10} {'─' * 6} {'─' * 15} {'─' * 19}")


def row(shape, dtype, red, fwd, fwdbwd):
    dt = str(dtype).split(".")[-1]
    bwd_s = f"{fwdbwd:>19.2f}" if fwdbwd is not None else f"{'—':>19}"
    print(f"  {str(shape):<28} {dt:<10} {red:<6} {fwd:>15.2f} {bwd_s}")


def time_fwd(stmt, g):
    try:
        m = Timer(stmt=stmt, globals=g).blocked_autorange(min_run_time=MIN_RUN)
        return m.median * 1e6
    except Exception:
        return None


def time_fwdbwd(stmt, g):
    try:
        m = Timer(stmt=stmt, globals=g).blocked_autorange(min_run_time=MIN_RUN)
        return m.median * 1e6
    except Exception:
        return None


# ── MSE Loss ──────────────────────────────────────────────────────────────

hdr("MSELoss — forward   |   forward+backward")
for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
    g = dict(
        F=F,
        x=torch.randn(shape, device=DEVICE, dtype=dtype),
        y=torch.randn(shape, device=DEVICE, dtype=dtype),
        red=red,
    )
    g["xg"] = g["x"].detach().requires_grad_(True)
    g["w"] = torch.rand_like(g["x"])
    fwd = time_fwd("F.mse_loss(x, y, reduction=red)", g)
    if fwd is None:
        continue
    bwd_stmt = (
        "xg.grad=None; F.mse_loss(xg, y, reduction='none').backward(w)"
        if red == "none"
        else "xg.grad=None; F.mse_loss(xg, y, reduction=red).sum().backward()"
    )
    fwdbwd = time_fwdbwd(bwd_stmt, g)
    row(shape, dtype, red, fwd, fwdbwd)


# ── SmoothL1 Loss ─────────────────────────────────────────────────────────

hdr("SmoothL1Loss — forward   |   forward+backward")
for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
    g = dict(
        F=F,
        x=torch.randn(shape, device=DEVICE, dtype=dtype),
        y=torch.randn(shape, device=DEVICE, dtype=dtype),
        red=red,
    )
    g["xg"] = g["x"].detach().requires_grad_(True)
    g["w"] = torch.rand_like(g["x"])
    fwd = time_fwd("F.smooth_l1_loss(x, y, reduction=red)", g)
    if fwd is None:
        continue
    bwd_stmt = (
        "xg.grad=None; F.smooth_l1_loss(xg, y, reduction='none').backward(w)"
        if red == "none"
        else "xg.grad=None; F.smooth_l1_loss(xg, y, reduction=red).sum().backward()"
    )
    fwdbwd = time_fwdbwd(bwd_stmt, g)
    row(shape, dtype, red, fwd, fwdbwd)


# ── Huber Loss ────────────────────────────────────────────────────────────

hdr("HuberLoss (delta=1.0) — forward   |   forward+backward")
for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
    g = dict(
        F=F,
        x=torch.randn(shape, device=DEVICE, dtype=dtype),
        y=torch.randn(shape, device=DEVICE, dtype=dtype),
        red=red,
    )
    g["xg"] = g["x"].detach().requires_grad_(True)
    g["w"] = torch.rand_like(g["x"])
    fwd = time_fwd("F.huber_loss(x, y, reduction=red)", g)
    if fwd is None:
        continue
    bwd_stmt = (
        "xg.grad=None; F.huber_loss(xg, y, reduction='none').backward(w)"
        if red == "none"
        else "xg.grad=None; F.huber_loss(xg, y, reduction=red).sum().backward()"
    )
    fwdbwd = time_fwdbwd(bwd_stmt, g)
    row(shape, dtype, red, fwd, fwdbwd)


# ── BCE Loss ──────────────────────────────────────────────────────────────

hdr("BCELoss — forward   |   forward+backward")
for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
    try:
        xb = torch.sigmoid(torch.randn(shape, device=DEVICE, dtype=dtype))
        yb = torch.randint(0, 2, shape, device=DEVICE).to(dtype)
    except Exception:
        continue
    g = dict(
        F=F,
        x=xb,
        y=yb,
        red=red,
        xg=xb.detach().requires_grad_(True),
        w=torch.rand_like(xb),
    )
    fwd = time_fwd("F.binary_cross_entropy(x, y, reduction=red)", g)
    if fwd is None:
        continue
    bwd_stmt = (
        "xg.grad=None; F.binary_cross_entropy(xg, y, reduction='none').backward(w)"
        if red == "none"
        else "xg.grad=None; F.binary_cross_entropy(xg, y, reduction=red).sum().backward()"
    )
    fwdbwd = time_fwdbwd(bwd_stmt, g)
    row(shape, dtype, red, fwd, fwdbwd)


# ── NLL Loss ──────────────────────────────────────────────────────────────

hdr("NLLLoss — forward   |   forward+backward  (LLM-scale vocab included)")
for (N, C), dtype, red in itertools.product(NLL_SHAPES, DTYPES, REDUCTIONS):
    try:
        lp = F.log_softmax(torch.randn(N, C, device=DEVICE, dtype=dtype), dim=1)
        t = torch.randint(0, C, (N,), device=DEVICE)
    except Exception:
        continue
    g = dict(
        F=F,
        lp=lp,
        t=t,
        red=red,
        lpg=lp.detach().requires_grad_(True),
        w=torch.rand(N, device="mps", dtype=dtype),
    )
    fwd = time_fwd("F.nll_loss(lp, t, reduction=red)", g)
    if fwd is None:
        continue
    bwd_stmt = (
        "lpg.grad=None; F.nll_loss(lpg, t, reduction='none').backward(w)"
        if red == "none"
        else "lpg.grad=None; F.nll_loss(lpg, t, reduction=red).sum().backward()"
    )
    fwdbwd = time_fwdbwd(bwd_stmt, g)
    row((N, C), dtype, red, fwd, fwdbwd)


# ── CrossEntropy Loss (fused log_softmax + NLL) ───────────────────────────

hdr("CrossEntropyLoss (fused) — forward   |   forward+backward")
for (N, C), dtype, red in itertools.product(NLL_SHAPES, DTYPES, REDUCTIONS):
    try:
        x = torch.randn(N, C, device=DEVICE, dtype=dtype)
        t = torch.randint(0, C, (N,), device=DEVICE)
    except Exception:
        continue
    g = dict(
        F=F,
        x=x,
        t=t,
        red=red,
        xg=x.detach().requires_grad_(True),
        w=torch.rand(N, device="mps", dtype=dtype),
    )
    fwd = time_fwd("F.cross_entropy(x, t, reduction=red)", g)
    if fwd is None:
        continue
    bwd_stmt = (
        "xg.grad=None; F.cross_entropy(xg, t, reduction='none').backward(w)"
        if red == "none"
        else "xg.grad=None; F.cross_entropy(xg, t, reduction=red).sum().backward()"
    )
    fwdbwd = time_fwdbwd(bwd_stmt, g)
    row((N, C), dtype, red, fwd, fwdbwd)


print()
