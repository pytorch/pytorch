#!/usr/bin/env python3
"""Benchmark Qwen3-style training step on Vulkan vs CPU.

Measures: forward, backward, optimizer step, and total iteration time.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch_vulkan


# ── Model ────────────────────────────────────────────────────────

class MiniQwen3(nn.Module):
    """Minimal Qwen3-style decoder for benchmarking."""

    def __init__(self, vocab=1024, d_model=128, n_heads=4, n_kv_heads=2,
                 intermediate=384, n_layers=2, seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.seq_len = seq_len
        self.vocab = vocab

        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "ln1_w": nn.ParameterDict({"w": nn.Parameter(torch.ones(d_model))}),
                "ln2_w": nn.ParameterDict({"w": nn.Parameter(torch.ones(d_model))}),
                "q_proj": nn.Linear(d_model, n_heads * self.head_dim, bias=False),
                "k_proj": nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False),
                "v_proj": nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False),
                "o_proj": nn.Linear(n_heads * self.head_dim, d_model, bias=False),
                "gate_proj": nn.Linear(d_model, intermediate, bias=False),
                "up_proj": nn.Linear(d_model, intermediate, bias=False),
                "down_proj": nn.Linear(intermediate, d_model, bias=False),
            }))
        self.ln_f_w = nn.Parameter(torch.ones(d_model))
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def rms_norm(self, x, w, use_vulkan):
        if use_vulkan:
            return torch_vulkan.rms_norm(x, w, 1e-6)
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + 1e-6) * w

    def forward(self, input_ids, use_vulkan=False):
        B, S = input_ids.shape
        h = self.embed(input_ids)

        # Cache causal mask expanded to 4D (avoids broadcast expand per layer)
        if not hasattr(self, '_causal_mask') or self._causal_mask.device != h.device or self._causal_mask.size(-1) != S:
            mask_2d = torch.triu(torch.full((S, S), float('-inf'), device=h.device), diagonal=1)
            self._causal_mask = mask_2d.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, S, S).contiguous()

        for layer in self.layers:
            # Pre-attention RMSNorm
            h_normed = self.rms_norm(h, layer["ln1_w"]["w"], use_vulkan)

            # GQA
            q = layer["q_proj"](h_normed).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            k = layer["k_proj"](h_normed).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = layer["v_proj"](h_normed).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

            # Repeat KV
            k = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, S, self.head_dim)
            k = k.reshape(B, self.n_heads, S, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, S, self.head_dim)
            v = v.reshape(B, self.n_heads, S, self.head_dim)

            # Attention
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = F.softmax(attn + self._causal_mask, dim=-1)
            attn_out = (attn @ v).transpose(1, 2).reshape(B, S, self.d_model)
            attn_out = layer["o_proj"](attn_out)
            h = h + attn_out

            # Pre-MLP RMSNorm
            h_normed2 = self.rms_norm(h, layer["ln2_w"]["w"], use_vulkan)

            # SwiGLU MLP
            if use_vulkan:
                gate_up = torch_vulkan.swiglu(layer["gate_proj"](h_normed2),
                                               layer["up_proj"](h_normed2))
            else:
                gate_up = F.silu(layer["gate_proj"](h_normed2)) * layer["up_proj"](h_normed2)
            h = h + layer["down_proj"](gate_up)

        h = self.rms_norm(h, self.ln_f_w, use_vulkan)
        return self.lm_head(h)


# ── Benchmark ────────────────────────────────────────────────────

def sync_device(device):
    """Ensure all GPU work is complete."""
    if device == "vulkan":
        torch_vulkan._C._flush()
    elif device == "cuda":
        torch.cuda.synchronize()


def benchmark_training(device, model_kwargs, batch_size=4, n_warmup=3, n_iter=20,
                       use_momentum=True, label=""):
    torch.manual_seed(42)
    model = MiniQwen3(**model_kwargs)
    seq_len = model_kwargs.get("seq_len", 64)
    vocab = model_kwargs.get("vocab", 1024)
    use_vulkan = (device == "vulkan")

    if device == "vulkan":
        model = model.vulkan()
    elif device == "cuda":
        model = model.cuda()

    if use_vulkan and use_momentum:
        opt = torch_vulkan.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif use_vulkan:
        opt = torch_vulkan.SGD(model.parameters(), lr=0.01)
    elif use_momentum:
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=False)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)

    # Warmup
    for _ in range(n_warmup):
        ids = torch.randint(0, vocab, (batch_size, seq_len))
        tgt = torch.randint(0, vocab, (batch_size, seq_len))
        if device == "vulkan":
            ids, tgt_flat = ids.vulkan(), tgt.view(-1).vulkan()
        elif device == "cuda":
            ids, tgt_flat = ids.cuda(), tgt.view(-1).cuda()
        else:
            tgt_flat = tgt.view(-1)

        opt.zero_grad()
        logits = model(ids, use_vulkan=use_vulkan)
        loss = F.cross_entropy(logits.view(-1, vocab), tgt_flat)
        loss.backward()
        opt.step()
    sync_device(device)

    # Timed runs
    fwd_times, bwd_times, opt_times, total_times = [], [], [], []

    for i in range(n_iter):
        ids = torch.randint(0, vocab, (batch_size, seq_len))
        tgt = torch.randint(0, vocab, (batch_size, seq_len))
        if device == "vulkan":
            ids, tgt_flat = ids.vulkan(), tgt.view(-1).vulkan()
        elif device == "cuda":
            ids, tgt_flat = ids.cuda(), tgt.view(-1).cuda()
        else:
            tgt_flat = tgt.view(-1)

        t0 = time.perf_counter()

        opt.zero_grad()
        sync_device(device)

        t1 = time.perf_counter()
        logits = model(ids, use_vulkan=use_vulkan)
        loss = F.cross_entropy(logits.view(-1, vocab), tgt_flat)
        sync_device(device)

        t2 = time.perf_counter()
        loss.backward()
        sync_device(device)

        t3 = time.perf_counter()
        opt.step()
        sync_device(device)

        t4 = time.perf_counter()

        fwd_times.append(t2 - t1)
        bwd_times.append(t3 - t2)
        opt_times.append(t4 - t3)
        total_times.append(t4 - t0)

    # Report
    def ms(times):
        return sum(times) / len(times) * 1000

    print(f"\n{'='*60}")
    print(f"  {label or device.upper()} — MiniQwen3 "
          f"(B={batch_size}, S={seq_len}, D={model_kwargs.get('d_model', 128)}, "
          f"L={model_kwargs.get('n_layers', 2)}, V={vocab})")
    print(f"{'='*60}")
    print(f"  Forward:    {ms(fwd_times):8.2f} ms")
    print(f"  Backward:   {ms(bwd_times):8.2f} ms")
    print(f"  Optimizer:  {ms(opt_times):8.2f} ms")
    print(f"  Total:      {ms(total_times):8.2f} ms")
    print(f"  Throughput: {batch_size * seq_len / (ms(total_times) / 1000):.0f} tokens/sec")

    return {
        "fwd_ms": ms(fwd_times),
        "bwd_ms": ms(bwd_times),
        "opt_ms": ms(opt_times),
        "total_ms": ms(total_times),
    }


def main():
    # Small model — quick benchmark
    small = dict(vocab=1024, d_model=64, n_heads=4, n_kv_heads=2,
                 intermediate=192, n_layers=1, seq_len=32)

    # Medium model — closer to real use
    medium = dict(vocab=4096, d_model=128, n_heads=4, n_kv_heads=2,
                  intermediate=384, n_layers=2, seq_len=64)

    # Larger model
    large = dict(vocab=4096, d_model=256, n_heads=8, n_kv_heads=4,
                 intermediate=768, n_layers=4, seq_len=64)

    configs = [
        ("Small (1L, D=64)", small, 8),
        ("Medium (2L, D=128)", medium, 4),
        ("Large (4L, D=256)", large, 2),
    ]

    for name, kwargs, bs in configs:
        print(f"\n{'#'*60}")
        print(f"  Config: {name}")
        print(f"{'#'*60}")

        results = {}
        for dev in ["cpu", "vulkan"]:
            try:
                r = benchmark_training(dev, kwargs, batch_size=bs,
                                       label=f"{dev.upper()} {name}")
                results[dev] = r
            except Exception as e:
                print(f"  {dev.upper()}: FAILED — {e}")

        if "cpu" in results and "vulkan" in results:
            speedup = results["cpu"]["total_ms"] / results["vulkan"]["total_ms"]
            print(f"\n  Vulkan speedup vs CPU: {speedup:.2f}x")


if __name__ == "__main__":
    main()
