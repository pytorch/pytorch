"""Standalone Qwen3-style tensor-parallel transformer block for benchmarking.

Mirrors the vLLM Qwen3 architecture (GQA, QK-norm, RoPE, fused QKV,
fused gate-up, RMSNorm with fused residual) using only plain PyTorch,
so that allreduce sync overhead measurements reflect realistic
compute-to-communication ratios.
"""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor,
                residual: torch.Tensor | None = None):
        """If residual is provided, fuse residual add + norm:
        new_residual = residual + x; output = norm(new_residual).
        Returns (output, new_residual) when residual is given,
        or just output when residual is None."""
        if residual is not None:
            x = residual + x
            return self._norm(x.float()).to(x.dtype) * self.weight, x
        return self._norm(x.float()).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int = 131072,
                 theta: float = 1000000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = max_position

    def forward(self, positions: torch.Tensor,
                q: torch.Tensor, k: torch.Tensor):
        """Apply rotary embeddings to q and k.
        positions: (batch, seq_len)
        q: (batch, seq_len, q_dim)  k: (batch, seq_len, kv_dim)
        """
        head_dim = self.inv_freq.shape[0] * 2
        batch, seq_len = positions.shape
        pos = positions.reshape(-1).float()  # (batch * seq_len,)
        freqs = torch.outer(pos, self.inv_freq)  # (batch*seq_len, head_dim/2)
        cos = freqs.cos()  # (batch*seq_len, head_dim/2)
        sin = freqs.sin()

        def rotate(x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, n_heads * head_dim)
            orig_dtype = x.dtype
            n_heads = x.shape[-1] // head_dim
            # (batch, seq_len, n_heads, head_dim) -> (batch*seq_len, n_heads, head_dim)
            x = x.view(batch * seq_len, n_heads, head_dim).float()
            x1 = x[..., : head_dim // 2]
            x2 = x[..., head_dim // 2 :]
            # cos/sin: (batch*seq_len, 1, head_dim/2) — broadcast over heads
            c = cos.unsqueeze(1)
            s = sin.unsqueeze(1)
            out = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
            return out.view(batch, seq_len, -1).to(orig_dtype)

        return rotate(q), rotate(k)


class Qwen3Attention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, n_kv_heads: int,
                 world_size: int,
                 rms_norm_eps: float = 1e-6,
                 device=None, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.world_size = world_size

        self.head_dim = hidden_dim // n_heads
        # Per-rank head counts
        self.n_local_heads = n_heads // world_size
        self.n_local_kv_heads = max(1, n_kv_heads // world_size)

        q_dim = self.n_local_heads * self.head_dim
        kv_dim = self.n_local_kv_heads * self.head_dim
        # Fused QKV projection (column-parallel: no allreduce)
        self.qkv_proj = nn.Linear(
            hidden_dim, q_dim + 2 * kv_dim,
            bias=False, device=device, dtype=dtype,
        )
        self.q_size = q_dim
        self.kv_size = kv_dim

        # Row-parallel output projection (allreduce after)
        self.o_proj = nn.Linear(
            q_dim, hidden_dim,
            bias=False, device=device, dtype=dtype,
        )

        # QK-norm: per-head RMSNorm
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps,
                              device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps,
                              device=device, dtype=dtype)

        self.rotary_emb = RotaryEmbedding(self.head_dim, device=device)

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                out: torch.Tensor | None = None):
        batch, seq_len, _ = x.shape

        # Fused QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK-norm (per-head)
        q = self.q_norm(
            q.view(batch, seq_len, self.n_local_heads, self.head_dim)
        ).view(batch, seq_len, -1)
        k = self.k_norm(
            k.view(batch, seq_len, self.n_local_kv_heads, self.head_dim)
        ).view(batch, seq_len, -1)

        # RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Reshape for SDPA: (batch, n_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.n_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_local_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_local_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: expand KV heads to match Q heads
        if self.n_local_kv_heads < self.n_local_heads:
            repeat = self.n_local_heads // self.n_local_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=(seq_len > 1))
        attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)

        # Row-parallel o_proj: write directly into out buffer if provided
        if out is not None:
            return F.linear(attn, self.o_proj.weight, out=out)
        return self.o_proj(attn)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_size: int,
                 world_size: int,
                 device=None, dtype=torch.bfloat16):
        super().__init__()
        shard = intermediate_size // world_size

        # Fused gate + up projection (column-parallel: no allreduce)
        self.gate_up_proj = nn.Linear(
            hidden_dim, shard * 2,
            bias=False, device=device, dtype=dtype,
        )
        # Row-parallel down projection (allreduce after)
        self.down_proj = nn.Linear(
            shard, hidden_dim,
            bias=False, device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up
        # Row-parallel down_proj: write directly into out buffer if provided
        if out is not None:
            return F.linear(h, self.down_proj.weight, out=out)
        return self.down_proj(h)


class Qwen3DecoderLayer(nn.Module):
    """Standalone Qwen3 decoder layer matching vLLM's architecture.

    Tensor parallelism: column-parallel QKV + gate_up, row-parallel o_proj +
    down_proj.  Allreduce after each row-parallel matmul (2 per layer).

    allreduce_fn(inp, out): copies inp into out after allreduce.
    If allreduce_buf is provided, the row-parallel matmuls write directly into
    it (zero-copy) and allreduce_fn operates in-place on that buffer.
    """

    def __init__(self, hidden_dim: int, n_heads: int, n_kv_heads: int,
                 intermediate_size: int, world_size: int,
                 allreduce_fn: Callable,
                 max_tokens: int = 1,
                 allreduce_buf: torch.Tensor | None = None,
                 rms_norm_eps: float = 1e-6,
                 device=None, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.allreduce_fn = allreduce_fn

        self.input_layernorm = RMSNorm(hidden_dim, eps=rms_norm_eps,
                                       device=device, dtype=dtype)
        self.self_attn = Qwen3Attention(
            hidden_dim, n_heads, n_kv_heads, world_size,
            rms_norm_eps=rms_norm_eps, device=device, dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(hidden_dim, eps=rms_norm_eps,
                                                device=device, dtype=dtype)
        self.mlp = Qwen3MLP(
            hidden_dim, intermediate_size, world_size,
            device=device, dtype=dtype,
        )

        if allreduce_buf is not None:
            self._allreduce_buf = allreduce_buf
        else:
            self._allreduce_buf = torch.empty(
                max_tokens * hidden_dim, dtype=dtype, device=device,
            )

        # Whether matmuls write directly into the allreduce buffer
        self.direct = allreduce_buf is not None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                residual: torch.Tensor | None = None):
        # Fused residual + RMSNorm
        if residual is None:
            residual = x
            h = self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)

        n = h.shape[0] * h.shape[1] * self.hidden_dim
        buf = self._allreduce_buf[:n]
        buf_3d = buf.view(x.shape)

        if self.direct:
            # Matmul writes directly into symm_mem buffer — zero copy
            self.self_attn(h, positions, out=buf_3d)
            self.allreduce_fn(buf, buf)
        else:
            attn_out = self.self_attn(h, positions).view(-1)
            self.allreduce_fn(attn_out, buf)

        # Fused residual + RMSNorm
        h, residual = self.post_attention_layernorm(buf_3d, residual)

        if self.direct:
            self.mlp(h, out=buf_3d)
            self.allreduce_fn(buf, buf)
        else:
            mlp_out = self.mlp(h).view(-1)
            self.allreduce_fn(mlp_out, buf)

        return buf_3d, residual
