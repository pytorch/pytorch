#!/usr/bin/env python3
"""Benchmark _scaled_dot_product_flash_attention_varlen_for_mps on MPS.

Measures forward and backward pass latency across Llama-3-style configs
(H=32, D=128, GQA=4) and a small reference (H=8, D=64, GQA=1).

Usage:
    PYTHONPATH=~/pytorch python3 bench_flash_attn_varlen.py

Requires:
    PyTorch built from pytorch/pytorch#181744 (MPS FlashAttention-2 varlen).
"""
import math
import time

import torch

assert torch.backends.mps.is_available(), "MPS not available"

DEVICE  = "mps"
WARMUP  = 30
ITERS   = 100

_fa_op = torch.ops.aten._scaled_dot_product_flash_attention_varlen_for_mps


def make_tensors(B, H, S, D, dtype, gqa=1):
    kH = H // gqa
    q = torch.randn(B * S, H,  D, device=DEVICE, dtype=dtype) * 0.1
    k = torch.randn(B * S, kH, D, device=DEVICE, dtype=dtype) * 0.1
    v = torch.randn(B * S, kH, D, device=DEVICE, dtype=dtype) * 0.1
    sl = torch.tensor([0] + [S] * B, device=DEVICE, dtype=torch.int32)
    cu = torch.cumsum(sl, dim=0, dtype=torch.int32)
    return q, k, v, cu, S


def bench_fwd(q, k, v, cu, ms, scale):
    for _ in range(WARMUP):
        _fa_op(q, k, v, cu, cu, ms, ms, 0.0, True, scale=scale)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _fa_op(q, k, v, cu, cu, ms, ms, 0.0, True, scale=scale)
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e3


def bench_fwd_bwd(q, k, v, cu, ms, scale):
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    for _ in range(WARMUP):
        out, _ = _fa_op(q, k, v, cu, cu, ms, ms, 0.0, True, scale=scale)
        out.backward(torch.randn_like(out))
        q.grad = k.grad = v.grad = None
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out, _ = _fa_op(q, k, v, cu, cu, ms, ms, 0.0, True, scale=scale)
        out.backward(torch.randn_like(out))
        q.grad = k.grad = v.grad = None
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e3


configs = [
    # label,                                         B,  H,  S,    D,  gqa
    ("B4  S=512  H32 D128 GQA4  (Llama train)  ",   4, 32,  512, 128, 4),
    ("B2  S=1024 H32 D128 GQA4  (Llama train)  ",   2, 32, 1024, 128, 4),
    ("B1  S=2048 H32 D128 GQA4  (Llama train)  ",   1, 32, 2048, 128, 4),
    ("B1  S=4096 H32 D128 GQA4  (Llama prefill)",   1, 32, 4096, 128, 4),
    ("B1  S=8192 H32 D128 GQA4  (long context) ",   1, 32, 8192, 128, 4),
    ("B4  S=512  H8  D64  GQA1  (reference)    ",   4,  8,  512,  64, 1),
]

print(f"\n{'Config':<48} {'fwd(ms)':>9} {'bwd(ms)':>9} {'fwd+bwd':>9}")
print("-" * 80)
for label, B, H, S, D, gqa in configs:
    q, k, v, cu, ms = make_tensors(B, H, S, D, torch.float16, gqa=gqa)
    sc = 1.0 / math.sqrt(D)
    fwd_ms = bench_fwd(q, k, v, cu, ms, sc)
    fb_ms  = bench_fwd_bwd(q, k, v, cu, ms, sc)
    bwd_ms = fb_ms - fwd_ms
    print(f"{label:<48} {fwd_ms:8.3f}ms {bwd_ms:8.3f}ms {fb_ms:8.3f}ms")
