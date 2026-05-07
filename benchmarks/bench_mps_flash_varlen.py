"""
bench_mps_flash_varlen.py — MPS FlashAttention-2 varlen vs padded benchmark

Compares attention paths on Apple Silicon:
  nn_mha   : nn.MultiheadAttention + to_dense_batch  (padded baseline)
  sdpa_pad : F.scaled_dot_product_attention on padded [B,H,max_S,D], no mask
             (lower bound on padded — faster than masked; varlen beating this is strong)
  varlen   : _scaled_dot_product_flash_attention_varlen_for_mps on [total,H,D]

Run:
  cd ~/Documents/pytorch-dev
  PYTHONPATH=. /opt/homebrew/bin/python3.11 benchmarks/bench_mps_flash_varlen.py
"""

import random
import sys
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

_varlen_op = torch.ops.aten._scaled_dot_product_flash_attention_varlen_for_mps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cu_seqlens(seqlens: List[int], device="mps") -> torch.Tensor:
    return torch.cat([
        torch.zeros(1, dtype=torch.int32),
        torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), 0),
    ]).to(device)


def to_dense_batch_manual(x_packed, seqlens):
    """Minimal to_dense_batch: [total, C] -> [B, max_S, C], mask [B, max_S]."""
    B = len(seqlens)
    max_s = max(seqlens)
    C = x_packed.size(1)
    out  = torch.zeros(B, max_s, C, device=x_packed.device, dtype=x_packed.dtype)
    mask = torch.zeros(B, max_s, dtype=torch.bool, device=x_packed.device)
    offset = 0
    for b, L in enumerate(seqlens):
        out[b, :L] = x_packed[offset:offset + L]
        mask[b, :L] = True
        offset += L
    return out, mask


def run_nn_mha(x_packed, seqlens, mha_module):
    """nn.MultiheadAttention + to_dense_batch (padded baseline)."""
    x_dense, mask = to_dense_batch_manual(x_packed, seqlens)
    out, _ = mha_module(x_dense, x_dense, x_dense, key_padding_mask=~mask)
    return out[mask]   # unpack back to [total, H*D]


def run_sdpa_padded(q_pad, k_pad, v_pad, causal, scale):
    """F.sdpa on zero-padded [B,H,max_S,D] WITHOUT masking.
    Lower bound on padded cost — faster than masked, so beating it is strong."""
    return F.scaled_dot_product_attention(q_pad, k_pad, v_pad,
                                          is_causal=causal, scale=scale)


def run_varlen(q, k, v, cu, max_s, causal, scale):
    out, _ = _varlen_op(q, k, v, cu, cu, max_s, max_s, 0.0, causal, scale=scale)
    return out


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class Config:
    label: str
    B: int
    H: int
    D: int
    seqlens: List[int]
    dtype: torch.dtype
    causal: bool

    @property
    def max_s(self):   return max(self.seqlens)
    @property
    def total(self):   return sum(self.seqlens)
    @property
    def embed(self):   return self.H * self.D
    @property
    def waste(self):   return 1.0 - self.total / (self.B * self.max_s)
    @property
    def dtype_str(self):
        return {torch.float16: "fp16", torch.bfloat16: "bf16",
                torch.float32: "fp32"}[self.dtype]


def make_configs() -> List[Config]:
    random.seed(7)
    configs = []

    # realistic workload: B=16, seqlens in [50,358]
    realistic_seqlens = [random.randint(50, 358) for _ in range(16)]
    configs.append(Config(
        label="B=16 S=50-358 H=8 D=64 (variable length)",
        B=16, H=8, D=64, seqlens=realistic_seqlens,
        dtype=torch.float16, causal=False))

    # Uniform padding waste sweep
    for waste_pct in [20, 50, 70]:
        max_s, B = 256, 8
        mean_s = max(1, int(max_s * (1 - waste_pct / 100)))
        seqlens = [max(1, min(max_s, int(random.gauss(mean_s, mean_s * 0.15))))
                   for _ in range(B)]
        configs.append(Config(
            label=f"B={B} max={max_s} ~{waste_pct}% waste H=8 D=64",
            B=B, H=8, D=64, seqlens=seqlens,
            dtype=torch.float16, causal=False))

    # D=128
    seqlens = [random.randint(32, 256) for _ in range(8)]
    configs.append(Config(
        label="B=8 S=32-256 H=8 D=128 fp16",
        B=8, H=8, D=128, seqlens=seqlens,
        dtype=torch.float16, causal=False))

    # Causal
    seqlens = [random.randint(64, 256) for _ in range(8)]
    configs.append(Config(
        label="B=8 S=64-256 H=8 D=64 causal",
        B=8, H=8, D=64, seqlens=seqlens,
        dtype=torch.float16, causal=True))

    # Long sequences
    seqlens = [random.randint(200, 512) for _ in range(4)]
    configs.append(Config(
        label="B=4 S=200-512 H=8 D=64",
        B=4, H=8, D=64, seqlens=seqlens,
        dtype=torch.float16, causal=False))

    return configs


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench_fn(fn, warmup=10, min_run_time=0.5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t = benchmark.Timer(
        stmt="fn(); torch.mps.synchronize()",
        globals={"fn": fn, "torch": torch})
    return t.adaptive_autorange(min_run_time=min_run_time).median * 1e3  # ms


def run_all():
    if not torch.backends.mps.is_available():
        print("MPS not available — run on Apple Silicon")
        sys.exit(1)

    print(f"\nPyTorch {torch.__version__}")
    print("Columns: nn_mha = padded baseline | sdpa_pad = unmasked padded SDPA "
          "(lower bound) | varlen = new kernel\n")

    col_w = 48
    header = (f"{'Config':<{col_w}}  {'waste':>6}  {'nn_mha':>10}  "
              f"{'sdpa_pad':>10}  {'varlen':>10}  {'vs nn_mha':>10}  {'vs sdpa_pad':>12}")
    print(header)
    print("-" * len(header))

    rows_for_table = []
    configs = make_configs()

    for cfg in configs:
        scale = 1.0 / (cfg.D ** 0.5)
        torch.manual_seed(42)

        # Packed inputs [total, H*D] for nn.MultiheadAttention
        x_packed = torch.randn(cfg.total, cfg.embed, device="mps", dtype=cfg.dtype)
        mha = nn.MultiheadAttention(cfg.embed, cfg.H, batch_first=True,
                                     dtype=cfg.dtype, device="mps")
        mha.eval()

        # Padded inputs [B, H, max_S, D] for sdpa
        q_pad = torch.randn(cfg.B, cfg.H, cfg.max_s, cfg.D, device="mps", dtype=cfg.dtype)
        k_pad = torch.randn(cfg.B, cfg.H, cfg.max_s, cfg.D, device="mps", dtype=cfg.dtype)
        v_pad = torch.randn(cfg.B, cfg.H, cfg.max_s, cfg.D, device="mps", dtype=cfg.dtype)

        # Varlen inputs [total, H, D]
        q_vl = torch.randn(cfg.total, cfg.H, cfg.D, device="mps", dtype=cfg.dtype)
        k_vl = torch.randn(cfg.total, cfg.H, cfg.D, device="mps", dtype=cfg.dtype)
        v_vl = torch.randn(cfg.total, cfg.H, cfg.D, device="mps", dtype=cfg.dtype)
        cu = make_cu_seqlens(cfg.seqlens)

        t_mha  = bench_fn(lambda: run_nn_mha(x_packed, cfg.seqlens, mha))
        t_pad  = bench_fn(lambda: run_sdpa_padded(q_pad, k_pad, v_pad, cfg.causal, scale))
        t_vl   = bench_fn(lambda: run_varlen(q_vl, k_vl, v_vl, cu, cfg.max_s, cfg.causal, scale))

        row = (f"{cfg.label:<{col_w}}  {cfg.waste*100:>5.0f}%  "
               f"{t_mha:>9.3f}ms  {t_pad:>9.3f}ms  {t_vl:>9.3f}ms  "
               f"{t_mha/t_vl:>9.2f}x  {t_pad/t_vl:>10.2f}x")
        print(row)
        rows_for_table.append((cfg.label, cfg.waste, t_mha, t_pad, t_vl))

    print()

    # fwd+bwd for variable-length config
    cfg = configs[0]
    scale = 1.0 / (cfg.D ** 0.5)
    print(f"--- fwd+bwd timing: {cfg.label} ---")
    torch.manual_seed(42)

    x_packed = torch.randn(cfg.total, cfg.embed, device="mps", dtype=cfg.dtype,
                            requires_grad=True)
    mha = nn.MultiheadAttention(cfg.embed, cfg.H, batch_first=True,
                                 dtype=cfg.dtype, device="mps")

    q_vl = torch.randn(cfg.total, cfg.H, cfg.D, device="mps", dtype=cfg.dtype,
                        requires_grad=True)
    k_vl = torch.randn(cfg.total, cfg.H, cfg.D, device="mps", dtype=cfg.dtype,
                        requires_grad=True)
    v_vl = torch.randn(cfg.total, cfg.H, cfg.D, device="mps", dtype=cfg.dtype,
                        requires_grad=True)
    cu = make_cu_seqlens(cfg.seqlens)

    def fwd_bwd_mha():
        out = run_nn_mha(x_packed, cfg.seqlens, mha)
        out.sum().backward()
        x_packed.grad = None

    def fwd_bwd_varlen():
        out, _ = _varlen_op(q_vl, k_vl, v_vl, cu, cu, cfg.max_s, cfg.max_s,
                            0.0, False, scale=scale)
        out.sum().backward()
        q_vl.grad = None; k_vl.grad = None; v_vl.grad = None

    t_mha_fb = bench_fn(fwd_bwd_mha)
    t_vl_fb  = bench_fn(fwd_bwd_varlen)
    print(f"  nn_mha  fwd+bwd : {t_mha_fb:.3f} ms")
    print(f"  varlen  fwd+bwd : {t_vl_fb:.3f} ms")
    print(f"  speedup         : {t_mha_fb/t_vl_fb:.2f}x\n")


if __name__ == "__main__":
    run_all()
