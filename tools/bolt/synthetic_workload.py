"""Synthetic PyTorch workload for LLVM BOLT profile collection.

Each variant exercises a slice of the libtorch call tree (dispatch, autograd, cuBLAS,
cuDNN, NCCL, TorchScript, Dynamo, model blocks). Run on BOLT-instrumented libraries
from tools/bolt/collect_profiles.sh, which collects their profiles at normal exit.
"""

from __future__ import annotations

import argparse
import io
import os
import time
import traceback
from typing import TYPE_CHECKING

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend


if TYPE_CHECKING:
    from collections.abc import Callable


def _log(msg: str) -> None:
    print(f"[bolt-synth] {msg}", flush=True)


def _eager_variant(device: torch.device, dtype: torch.dtype):
    """Op dispatch, allocator churn, views, indexing, copies and small CPU ops."""
    shapes = [(16, 256), (32, 256), (16, 512), (8, 1024), (64, 128), (16, 16, 64)]
    shapes += [(4, 32, 256), (1, 256, 512), (16, 1024), (32, 512), (8, 2048)]
    shapes += [(128, 64), (4, 4, 512)]
    weight = torch.randn(1024, 1024, device=device, dtype=dtype)
    bias = torch.randn(1024, device=device, dtype=dtype)
    layer_norm = nn.LayerNorm(1024, device=device, dtype=dtype)
    # Same-dtype CPU copies take a per-dtype direct_copy_kernel specialization.
    cpu_copy_dtypes = [torch.bool, torch.uint8, torch.float8_e4m3fn, torch.float16]
    cpu_copy_dtypes += [torch.bfloat16, torch.int64, torch.int32, torch.float32]
    cpu_src_buffers = [torch.zeros(64, 1024, dtype=dt) for dt in cpu_copy_dtypes]
    cpu_dst_buffers = [torch.empty(64, 1024, dtype=dt) for dt in cpu_copy_dtypes]
    cpu_idx_src = torch.randn(256, 64, dtype=torch.float32)
    cpu_idx_idx = torch.tensor([0, 5, 10, 20, 50, 100, 200], dtype=torch.long)
    np_buffer = np.random.randn(128, 128).astype(np.float32)
    adv_idx = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35], device=device)
    idx_put_target = torch.zeros(2048, 256, device=device, dtype=torch.float32)
    idx_put_idx_row = torch.randint(0, 2048, (64,), device=device, dtype=torch.long)
    idx_put_idx_col = torch.randint(0, 256, (64,), device=device, dtype=torch.long)
    idx_put_vals = torch.randn(64, device=device, dtype=torch.float32)
    # fp32 operands: the other elementwise ops here run in the requested dtype.
    fp32_a = torch.randn(256, 1024, device=device, dtype=torch.float32)
    fp32_b = torch.randn(256, 1024, device=device, dtype=torch.float32)
    small_alloc_shapes = [(16,), (64,), (128,), (256,), (512,), (1024,), (16, 16)]
    small_alloc_shapes += [(32, 32), (64, 64), (16, 256), (32, 128), (1, 128)]
    small_alloc_shapes += [(1, 512), (2, 64), (4, 32, 64)]
    chain_tensor = torch.empty(128, device=device, dtype=dtype)
    silu_buf = torch.empty(256, 512, device=device, dtype=dtype)
    d2d_dst = torch.empty(64, 1024, device=device, dtype=dtype)
    big2 = torch.randn(64, 1024, device=device, dtype=dtype)
    cpu_lengths = torch.randint(1, 64, (64,), dtype=torch.long)
    cpu_ids = torch.randint(0, 4096, (1, 512), dtype=torch.long)
    cpu_gather_idx = torch.randint(0, 256, (64,), dtype=torch.long)
    ckpt_state = {"weight": torch.randn(128, 128), "bias": torch.randn(128)}
    # Rare ops run every Nth step to keep the instrumented step time low.
    rotate = [0]
    rare_every = 8

    def step():
        for s in shapes:
            x = torch.empty(s, device=device, dtype=dtype)
            x.uniform_(-1.0, 1.0)
            y = x.contiguous()
            _ = (y.size(), y.stride())
            z = y.view(-1, s[-1])
            z = z + 1.0
            z = z * 2.0
            z = F.gelu(z)
            z = z.contiguous()
            del x, y, z

        a, b = (torch.randn(64, 1024, device=device, dtype=dtype) for _ in range(2))
        c = a + b
        c = c * a
        c = torch.addmm(bias, c, weight)
        c = layer_norm(c)
        c = F.relu(c)
        c = c.contiguous()
        c = c.transpose(0, 1).contiguous()
        host = torch.empty(c.shape, dtype=dtype, pin_memory=True)
        host.copy_(c, non_blocking=True)
        torch.cuda.synchronize()
        i = rotate[0]
        di = i % len(cpu_dst_buffers)
        cpu_dst_buffers[di].copy_(cpu_src_buffers[di])

        big = torch.randn(64, 1024, device=device, dtype=dtype)
        _ = big[:, 100:300]
        _ = big[:, ::2]
        _ = big[adv_idx]
        _ = big[big > 0]
        _ = (big.dtype, big.shape, big.size(0), big.data_ptr(), big.storage_offset())
        if i % rare_every == 0:
            _ = torch.index_select(cpu_idx_src, 0, cpu_idx_idx)
            _ = cpu_idx_idx.tolist()
            _ = torch.randint(0, 100, (64,), device=device)
            _ = torch.from_numpy(np_buffer)
            _ = torch.ops.aten.add(big, big)
        rotate[0] = i + 1
        big_fp32 = big.to(torch.float32)
        big_bf16 = big_fp32.to(torch.bfloat16)
        _ = big_bf16.to(torch.float32)
        for _ in range(4):
            _ = torch.cat([big, big, big, big], dim=0)
        _ = fp32_a + fp32_b
        _ = fp32_a * fp32_b
        idx_put_target[idx_put_idx_row, idx_put_idx_col] = idx_put_vals
        _ = idx_put_target[idx_put_idx_row, idx_put_idx_col]
        # An explicit int list is boxed into a vector<long> IValue.
        _ = torch.split(big, [200, 200, 200, 200, 224], dim=1)

        # Allocator malloc / free churn and per-op glue: the cost scales with
        # the number of ops fired, not with their size.
        for _ in range(8):
            for sh in small_alloc_shapes:
                _ = torch.empty(sh, device=device, dtype=dtype)
        ct = chain_tensor
        for _ in range(48):
            ct = ct + 1.0
        big4d = big.view(2, 32, 4, 256)  # 4-D, so the channels_last check runs
        for _ in range(8):
            _ = big.is_contiguous()
            _ = big4d.is_contiguous(memory_format=torch.channels_last)
            _ = big.stride()
            _ = big.size()
        for _ in range(4):
            _ = big.as_strided((32, 512), (1024, 1), 0)
            _ = big.as_strided((16, 16, 64), (1024, 64, 1), 0)
        for _ in range(4):
            _ = big.unsqueeze(0).squeeze(0)
            _ = big.unsqueeze(-1).squeeze(-1)
            _ = big.unsqueeze(1).squeeze(1)
        silu_buf.uniform_(-1.0, 1.0)
        _ = F.silu(silu_buf, inplace=True)
        _ = big.clone()
        d2d_dst.copy_(big)
        # Transposed / permuted operands defeat the contiguous fast path.
        _ = big.t() + big2.t()
        big3d = big.view(2, 32, 1024)
        _ = big3d.permute(2, 0, 1) * 2.0
        for t in (big, big3d, c, silu_buf):
            for _ in range(4):
                _ = t.shape

        # CPU-side int64 index preprocessing, as in recommender models.
        pos = torch.arange(cpu_lengths.numel())
        offs = torch.arange(0, 512, 8)
        sorted_lengths, order = torch.sort(cpu_lengths, stable=True)
        ids = torch.clamp(cpu_ids.squeeze() - 1, min=0, max=4095)
        for k in range(4):
            _ = ids[k]
            _ = order[k]
        _ = torch.repeat_interleave(pos, sorted_lengths)
        _ = offs.sub(1)
        _ = cpu_idx_src[cpu_gather_idx]
        _ = big.type()
        if i % 32 == 0:
            torch.save(ckpt_state, io.BytesIO())
        return c

    return step


def _gemm_variant(device: torch.device, dtype: torch.dtype):
    """cuBLASLt heuristics and dispatch over varied (M, N, K) and transposes."""
    shapes = [(4096, 4096, 4096), (1024, 4096, 4096), (4096, 1024, 4096)]
    shapes += [(256, 1024, 1024), (8192, 1024, 1024), (1024, 8192, 1024)]
    shapes += [(2048, 2048, 8192), (768, 3072, 768), (3072, 768, 768)]
    shapes += [(128, 128, 16384)]
    matrices = []
    for m, n, k in shapes:
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)
        matrices.append((a, b, b.t().contiguous()))
    # Small fp16 matmuls take the legacy cuBLAS Hgemm path, not cuBLASLt.
    hgemm_shapes = [(1, 80, 512), (1, 256, 256), (16, 128, 256), (32, 64, 512)]
    hgemm_shapes += [(64, 32, 256)]
    small_fp16_matrices = [
        (
            torch.randn(m, k, device=device, dtype=torch.float16),
            torch.randn(k, n, device=device, dtype=torch.float16),
        )
        for m, n, k in hgemm_shapes
    ]

    def step():
        out = None
        for a, b, bt in matrices:
            c = torch.mm(a, b)
            c = torch.mm(a, bt.t())
            out = c
        ba = torch.randn(8, 256, 1024, device=device, dtype=dtype)
        bb = torch.randn(8, 1024, 512, device=device, dtype=dtype)
        bc = torch.bmm(ba, bb)
        for fa, fb in small_fp16_matrices:
            _ = torch.mm(fa, fb)
        torch.cuda.synchronize()
        return out, bc

    return step


def _cnn_variant(device: torch.device, dtype: torch.dtype):
    """cuDNN convolutions in NCHW and NHWC, depthwise conv2d and a conv1d block."""
    n, c_in, h, w = 8, 64, 56, 56
    # (out_channels, kernel, stride, padding); not all powers of two on purpose.
    spec = [(64, 3, 1, 1), (128, 3, 2, 1), (96, 1, 1, 0), (256, 3, 1, 1)]
    spec += [(160, 5, 1, 2), (256, 1, 1, 0), (512, 3, 2, 1)]
    convs_nchw: list[nn.Module] = []
    convs_nhwc: list[nn.Module] = []
    layouts = [(convs_nchw, torch.contiguous_format), (convs_nhwc, torch.channels_last)]
    for oc, k, s, p in spec:
        for convs, fmt in layouts:
            m = nn.Conv2d(c_in, oc, kernel_size=k, stride=s, padding=p, bias=False)
            # pyrefly: ignore [unexpected-keyword]
            convs.append(m.to(device=device, dtype=dtype, memory_format=fmt))
        c_in = oc
    bn_nchw = nn.BatchNorm2d(c_in).to(device=device, dtype=dtype)
    bn_nhwc = nn.BatchNorm2d(c_in).to(device=device, dtype=dtype)
    bn_nhwc = bn_nhwc.to(memory_format=torch.channels_last)
    dw_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False).to(
        device=device, dtype=dtype
    )
    # Speech-style conv1d block; the explicit F.pad drives constant_pad_nd.
    conv1d_ch, conv1d_k = 256, 31
    conv1d_pw1 = nn.Conv1d(conv1d_ch, 2 * conv1d_ch, 1).to(device=device, dtype=dtype)
    conv1d_dw = nn.Conv1d(conv1d_ch, conv1d_ch, conv1d_k, groups=conv1d_ch).to(
        device=device, dtype=dtype
    )
    conv1d_pw2 = nn.Conv1d(conv1d_ch, conv1d_ch, 1).to(device=device, dtype=dtype)
    conv1d_x = torch.randn(1, conv1d_ch, 128, device=device, dtype=dtype)

    def _run_chain(x, convs, bn):
        for m in convs:
            x = m(x)
            x = F.relu(x, inplace=True)
        return bn(x)

    def step():
        x_nchw = torch.randn(n, 64, h, w, device=device, dtype=dtype)
        out_nchw = _run_chain(x_nchw, convs_nchw, bn_nchw)
        x_nhwc = torch.randn(n, 64, h, w, device=device, dtype=dtype).to(
            memory_format=torch.channels_last
        )
        out_nhwc = _run_chain(x_nhwc, convs_nhwc, bn_nhwc)
        _ = dw_conv(x_nchw)
        with torch.no_grad():
            y = F.glu(conv1d_pw1(conv1d_x), dim=1)
            y = F.pad(y, (conv1d_k // 2, conv1d_k // 2))
            _ = conv1d_pw2(F.silu(conv1d_dw(y)))
        torch.cuda.synchronize()
        return out_nchw, out_nhwc

    return step


def _attention_variant(device: torch.device, dtype: torch.dtype):
    """SDPA backend selection, plus each backend's launch path forced in turn."""
    # (batch, heads, seq, head_dim)
    configs = [(1, 16, 512, 64), (1, 16, 1024, 64), (1, 32, 2048, 64)]
    configs += [(4, 16, 512, 128), (2, 8, 4096, 64), (1, 16, 8192, 64)]
    tensors = [
        tuple(torch.randn(b, h, s, d, device=device, dtype=dtype) for _ in range(3))
        for b, h, s, d in configs
    ]
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    backends += [SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]
    # (backend, config index, causal) combinations the backend rejected.
    unsupported: set[tuple[SDPBackend, int, bool]] = set()
    rotate = [0]

    def step():
        # Cycle through every config x backend pair, then flip the mask.
        i = rotate[0]
        rotate[0] = i + 1
        ci = i % len(tensors)
        backend = backends[(i // len(tensors)) % len(backends)]
        is_causal = (i // (len(tensors) * len(backends))) % 2 == 0
        q, k, v = tensors[ci]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        if (backend, ci, is_causal) not in unsupported:
            try:
                with sdpa_kernel(backend):
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            except RuntimeError as e:
                unsupported.add((backend, ci, is_causal))
                _log(
                    f"attention: {backend.name} does not support config "
                    f"{configs[ci]} causal={is_causal}: {str(e).splitlines()[0]}"
                )
        torch.cuda.synchronize()
        return out

    return step


class _IdentityWithGrad(torch.autograd.Function):
    """Crosses THPFunction_apply and unpacks a saved tensor in backward."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clone()

    @staticmethod
    def backward(ctx, *grad_outputs):
        (saved_x,) = ctx.saved_tensors
        # Read saved_x so the SavedVariable unpack path actually runs.
        return grad_outputs[0] * (1.0 + 0.0 * saved_x.sum())


def _training_variant(device: torch.device, dtype: torch.dtype):
    """Autograd engine, AMP GradScaler and multi-tensor (foreach) Adam."""
    # fp32 params under autocast(fp16): GradScaler refuses fp16 gradients and
    # its unscale kernel does not support bf16.
    del dtype
    f32 = torch.float32
    conv_blocks: list[nn.Module] = []
    ch = 64
    for i in range(6):
        out_ch = 128 if i >= 3 else 64
        conv = nn.Conv2d(ch, out_ch, 3, stride=2 if i == 3 else 1, padding=1)
        conv_blocks += [conv, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        ch = out_ch
    head: list[nn.Module] = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 256)]
    for _ in range(3):
        head += [nn.GELU(), nn.Linear(256, 256)]
    model = nn.Sequential(*conv_blocks, *head).to(device=device, dtype=f32)
    params = list(model.parameters())
    emb_w = nn.Parameter(torch.randn(256, 256, device=device, dtype=f32))
    mm_w = nn.Parameter(torch.randn(256, 256, device=device, dtype=f32))
    tok_ids = torch.randint(0, 4096, (64,), device=device)
    tok_mod = torch.full((64,), 1000, device=device, dtype=torch.long)
    side_stream = torch.cuda.Stream()
    # fused=True would skip multi_tensor_apply entirely.
    optim = torch.optim.Adam([*params, emb_w, mm_w], lr=1e-3, foreach=True, fused=False)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    # For the list + list _foreach_add_, which Adam does not use.
    foreach_a = [torch.randn(128, 128, device=device, dtype=f32) for _ in range(8)]
    foreach_b = [torch.randn(128, 128, device=device, dtype=f32) for _ in range(8)]
    cuda_event_a = torch.cuda.Event(enable_timing=True)
    cuda_event_b = torch.cuda.Event(enable_timing=True)

    def step():
        # Small spatial size: autograd cost scales with graph nodes, not FLOPs.
        x = torch.randn(32, 64, 16, 16, device=device, dtype=f32, requires_grad=True)
        target = torch.randn(32, 256, device=device, dtype=f32)
        x.register_hook(lambda g: g)
        x = _IdentityWithGrad.apply(x)
        cuda_event_a.record()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x)
            # Using `out` twice makes InputBuffer::add accumulate.
            first, second = out.chunk(2, dim=1)
            mixed = torch.cat([second, first], dim=1)
            loss = F.mse_loss(mixed, target) + 1e-3 * out.float().pow(2).mean()
            tok = torch.remainder(torch.remainder(tok_ids, 1000), tok_mod)
            pos = tok.clone()
            pos.div_(4, rounding_mode="floor")
            rows = emb_w[pos]
            proj = torch.mm(out, mm_w)
            for _ in range(4):
                proj = _IdentityWithGrad.apply(proj)
            loss = loss + 1e-3 * (
                rows.float().pow(2).mean() + proj.float().pow(2).mean()
            )
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        cuda_event_b.record()
        for p in params:
            _ = p.data
        with torch.cuda.stream(side_stream):
            _ = foreach_b[0] * 1.0
        foreach_b[0].record_stream(side_stream)
        _ = loss.detach()
        torch._foreach_add_(foreach_a, foreach_b, alpha=0.01)
        torch.cuda.synchronize()
        return loss

    return step


def _distributed_variant(device: torch.device, dtype: torch.dtype):
    """Host-side ProcessGroupNCCL dispatch of the usual collectives on one rank."""
    # Flight recorder with C++ stacks and collective timing, as large training
    # jobs enable it. Must be set before init_process_group.
    os.environ.setdefault("TORCH_FR_BUFFER_SIZE", "2000")
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    os.environ.setdefault("TORCH_NCCL_TRACE_CPP_STACK", "1")
    os.environ.setdefault("TORCH_NCCL_ENABLE_TIMING", "1")
    if not dist.is_initialized():
        # A HashStore avoids binding a TCP port; one rank needs no rendezvous.
        dist.init_process_group(
            backend="nccl", store=dist.HashStore(), world_size=1, rank=0
        )
    dump_nccl_trace = getattr(torch._C._distributed_c10d, "_dump_nccl_trace", None)

    inp = torch.randn(1024, device=device, dtype=dtype)
    ag_in = torch.randn(256, device=device, dtype=dtype)
    ag_out = torch.empty(256, device=device, dtype=dtype)
    rs_in = torch.randn(256, device=device, dtype=dtype)
    rs_out = torch.empty(256, device=device, dtype=dtype)
    a2a_in = torch.randn(256, device=device, dtype=dtype)
    a2a_out = torch.empty(256, device=device, dtype=dtype)
    uneq_in = torch.randn(200, device=device, dtype=dtype)
    uneq_out = torch.empty(200, device=device, dtype=dtype)
    a2a_list_in = [torch.randn(128, device=device, dtype=dtype)]
    a2a_list_out = [torch.empty(128, device=device, dtype=dtype)]
    ag_coal_pairs = [
        (
            torch.randn(128, device=device, dtype=dtype),
            torch.empty(128, device=device, dtype=dtype),
        )
        for _ in range(4)
    ]
    ag_rs = [
        lambda: dist.all_gather_into_tensor(ag_out, ag_in, async_op=True),
        lambda: dist.reduce_scatter_tensor(rs_out, rs_in, async_op=True),
    ]
    collectives = [
        lambda: dist.all_reduce(inp.clone(), async_op=True),
        *(ag_rs * 4),
        lambda: dist.broadcast(inp.clone(), src=0, async_op=True),
        lambda: dist.all_to_all_single(a2a_out, a2a_in, async_op=True),
        # Explicit (output, input) split sizes take the unequal-split path.
        lambda: dist.all_to_all_single(uneq_out, uneq_in, [200], [200], async_op=True),
        lambda: dist.all_to_all(a2a_list_out, a2a_list_in, async_op=True),
    ]
    dump_counter = [0]

    def step():
        for _ in range(4):
            for issue in collectives:
                w = issue()
                if w is None:
                    raise RuntimeError("async collective returned no Work")
                w.wait()
            with dist._coalescing_manager():
                for in_t, out_t in ag_coal_pairs:
                    dist.all_gather_into_tensor(out_t, in_t)
        dist.barrier()
        if dump_nccl_trace is not None and dump_counter[0] % 16 == 0:
            try:
                _ = dump_nccl_trace()
            except Exception:
                pass
        dump_counter[0] += 1
        torch.cuda.synchronize()

    return step


class _AttentionDecoderStep(nn.Module):
    """One step of a seq2seq attention decoder, run through torch.jit.script."""

    def __init__(self, dim: int):
        super().__init__()
        self.prenet = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.memory = nn.Linear(dim, dim, bias=False)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor, gate: float
    ) -> tuple[torch.Tensor, bool]:
        h = F.dropout(F.relu(self.prenet(x)), p=0.5, training=True)
        energies = self.score(
            torch.tanh(self.query(h).unsqueeze(1) + self.memory(memory))
        ).squeeze(-1)
        energies = energies.masked_fill(mask == 0, -1e4)
        align = F.softmax(energies, dim=1)
        ctx = torch.bmm(align.unsqueeze(1), memory).squeeze(1)
        done = bool(torch.sigmoid(ctx.mean()) > gate)
        if not done:
            ctx = ctx + h
        return ctx, done


def _torchscript_variant(device: torch.device, dtype: torch.dtype):
    """TorchScript interpreter and NNC fuser, pinned to fp16 like JIT inference."""
    del dtype
    fp16 = torch.float16

    @torch.jit.script
    def fused_chain(x: torch.Tensor, scale: float) -> torch.Tensor:
        y = x * scale
        y = y + 1.0
        y = torch.tanh(y)
        y = y * x
        y = y + scale
        y = torch.relu(y)
        return y

    # Unlike fused_chain, these list ops stay in the interpreter.
    @torch.jit.script
    def interp_step(
        x: torch.Tensor,
        h: torch.Tensor,
        w: torch.Tensor,
        ranks: list[int],
        ln_w: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [x, h]
        y = torch.cat(parts, dim=1)
        for t in parts:
            if t.dim() in ranks:
                y = y + 0.0
        y = torch.matmul(y, w)
        y = F.layer_norm(y, [y.size(1)], ln_w, None)
        return y.reshape(y.size(0), -1)

    interp_x = torch.randn(8, 512, device=device, dtype=fp16)
    interp_h = torch.randn(8, 512, device=device, dtype=fp16)
    interp_w = torch.randn(1024, 1024, device=device, dtype=fp16)
    interp_ln_w = torch.ones(1024, device=device, dtype=fp16)
    interp_ranks = [2, 3]
    decoder = torch.jit.script(_AttentionDecoderStep(256).to(device=device, dtype=fp16))
    dec_memory = torch.randn(4, 64, 256, device=device, dtype=fp16)
    dec_mask = torch.ones(4, 64, device=device, dtype=torch.int32)
    dec_mask[:, 48:] = 0
    dec_x0 = torch.randn(4, 256, device=device, dtype=fp16)
    # Rotating batch sizes make the JIT re-infer shapes every few calls.
    jit_xs = [torch.randn(n, 1024, device=device, dtype=fp16) for n in (64, 48, 80, 96)]
    jit_rotate = [0]

    def step():
        with torch.no_grad():
            j = jit_rotate[0]
            x_in = jit_xs[j % len(jit_xs)]
            for _ in range(8):
                _ = fused_chain(x_in, 0.5)
                _ = fused_chain(x_in, 0.25)
            jit_rotate[0] = j + 1
            for _ in range(8):
                _ = interp_step(interp_x, interp_h, interp_w, interp_ranks, interp_ln_w)
            dec_x = dec_x0
            for _ in range(8):
                dec_x, _ = decoder(dec_x, dec_memory, dec_mask, 0.5)
        torch.cuda.synchronize()

    return step


def _rnn_variant(device: torch.device, dtype: torch.dtype):
    """fp16 RNN inference: native LSTMCell kernels and cuDNN LSTM / GRU."""
    del dtype
    fp16 = torch.float16
    lstm_cell_a = nn.LSTMCell(512, 1024).to(device=device, dtype=fp16)
    lstm_cell_b = nn.LSTMCell(1024, 1024).to(device=device, dtype=fp16)
    lstm = nn.LSTM(512, 512, num_layers=2, batch_first=True).to(
        device=device, dtype=fp16
    )
    gru = nn.GRU(512, 512, num_layers=2, batch_first=True).to(device=device, dtype=fp16)
    seq_input = torch.randn(8, 64, 512, device=device, dtype=fp16)
    drop_input = torch.randn(8, 1024, device=device, dtype=fp16)
    mask_input = torch.randn(8, 1024, device=device, dtype=fp16)

    def step():
        with torch.no_grad():
            h_a, c_a, h_b, c_b = (
                torch.zeros(8, 1024, device=device, dtype=fp16) for _ in range(4)
            )
            x_step = torch.randn(8, 512, device=device, dtype=fp16)
            for _ in range(8):
                h_a, c_a = lstm_cell_a(x_step, (h_a, c_a))
                h_b, c_b = lstm_cell_b(h_a, (h_b, c_b))
            _ = lstm(seq_input)
            _ = gru(seq_input)
            _ = F.dropout(drop_input, p=0.1, training=True)
            mask = mask_input > 0.0
            _ = mask_input.masked_fill(mask, -float("inf"))
        torch.cuda.synchronize()

    return step


def _activations_variant(device: torch.device, dtype: torch.dtype):
    """bf16 SiLU, GLU, eval-mode BatchNorm2d and masked_fill kernels."""
    # Kept out of other variants so the shared glue code's branch profile is
    # not skewed towards any single caller.
    del dtype
    bf16 = torch.bfloat16
    act_input = torch.randn(8, 2048, device=device, dtype=bf16)
    glu_input = torch.randn(8, 2048, device=device, dtype=bf16)
    bn_input = torch.randn(8, 256, 32, 32, device=device, dtype=bf16)
    bn_layer = nn.BatchNorm2d(256).to(device=device, dtype=bf16)
    bn_layer.eval()

    def step():
        with torch.no_grad():
            _ = F.silu(act_input)
            _ = F.glu(glu_input, dim=-1)
            _ = bn_layer(bn_input)
            mask = act_input > 0.0
            _ = act_input.masked_fill(mask, 0.0)
        torch.cuda.synchronize()

    return step


def _transformer_variant(device: torch.device, dtype: torch.dtype):
    """bf16 transformer MLP blocks at batch 1, plus matmul batch folding on 3-D."""
    del dtype
    bf16 = torch.bfloat16
    hidden = 512
    n_layers = 16
    linears_a = nn.ModuleList(
        nn.Linear(hidden, hidden * 2, device=device, dtype=bf16)
        for _ in range(n_layers)
    )
    linears_b = nn.ModuleList(
        nn.Linear(hidden * 2, hidden, device=device, dtype=bf16)
        for _ in range(n_layers)
    )
    norms = nn.ModuleList(
        nn.LayerNorm(hidden, device=device, dtype=bf16) for _ in range(n_layers)
    )
    x_buf = torch.randn(1, 64, hidden, device=device, dtype=bf16)
    w_fold = torch.randn(hidden * 2, hidden, device=device, dtype=bf16)

    def step():
        with torch.no_grad():
            h = x_buf
            for la, lb, ln in zip(linears_a, linears_b, norms, strict=True):
                ff = lb(F.silu(la(h)))
                h = ln(h + ff)
            # Bias-less linear and matmul on 3-D input fold the batch dims
            # (nn.Linear with bias takes addmm instead).
            for _ in range(4):
                y = F.linear(h, w_fold)
                y = torch.matmul(y, w_fold)
                _ = y.reshape(-1, hidden)
                _ = y.transpose(1, 2).reshape(-1, 64)
        torch.cuda.synchronize()
        return h

    return step


def _moe_variant(device: torch.device, dtype: torch.dtype):
    """Top-2 of 8 mixture-of-experts: routing, token dispatch and expert MLPs."""
    num_experts, top_k = 8, 2
    tokens, hidden, ffn = 2048, 1024, 2048
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    router = nn.Linear(hidden, num_experts, bias=False, device=device, dtype=dtype)
    w_gate_up = (
        torch.randn(num_experts, hidden, 2 * ffn, device=device, dtype=dtype) * 0.02
    )
    w_down = torch.randn(num_experts, ffn, hidden, device=device, dtype=dtype) * 0.02

    def step():
        with torch.no_grad():
            # Fresh router noise varies the per-expert GEMM shapes every step.
            logits = router(x).float()
            logits = logits + torch.randn_like(logits)
            probs = F.softmax(logits, dim=-1)
            weights, experts = torch.topk(probs, top_k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            flat_experts = experts.flatten()
            order = torch.argsort(flat_experts, stable=True)
            counts = torch.bincount(flat_experts, minlength=num_experts)
            _ = torch.cumsum(counts, dim=0)
            token_idx = order // top_k
            dispatched = x.index_select(0, token_idx)
            expert_outs = []
            for e, chunk in enumerate(torch.split(dispatched, counts.tolist())):
                if chunk.shape[0] == 0:
                    continue
                gate, up = (chunk @ w_gate_up[e]).chunk(2, dim=-1)
                expert_outs.append((F.silu(gate) * up) @ w_down[e])
            expert_out = torch.cat(expert_outs)
            expert_out = expert_out * weights.flatten()[order].unsqueeze(-1).to(dtype)
            out = torch.zeros_like(x)
            out.index_add_(0, token_idx, expert_out)
        torch.cuda.synchronize()
        return out

    return step


def _compile_variant(device: torch.device, dtype: torch.dtype):
    """torch.compile guards, eval-frame shim and generated-code size asserts."""
    del dtype
    bf16 = torch.bfloat16

    def _fn(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
        z = x * scale + y
        z = F.gelu(z)
        z = z @ y.t()
        return z

    compiled = torch.compile(_fn, dynamic=False)
    x = torch.randn(64, 512, device=device, dtype=bf16)
    y = torch.randn(64, 512, device=device, dtype=bf16)

    def step():
        with torch.no_grad():
            _ = compiled(x, y, 0.5)
        torch.cuda.synchronize()

    return step


VARIANTS: dict[str, Callable[[torch.device, torch.dtype], Callable[[], object]]] = {
    "eager": _eager_variant,
    "gemm": _gemm_variant,
    "cnn": _cnn_variant,
    "attention": _attention_variant,
    "training": _training_variant,
    "distributed": _distributed_variant,
    "torchscript": _torchscript_variant,
    "rnn": _rnn_variant,
    "activations": _activations_variant,
    "transformer": _transformer_variant,
    "moe": _moe_variant,
    "compile": _compile_variant,
}
ALL_VARIANTS = tuple(VARIANTS)


def _build_round_robin(
    device: torch.device, dtype: torch.dtype, strict: bool
) -> tuple[Callable[[], object], dict[str, list[float]]]:
    names = []
    steps = []
    for name in ALL_VARIANTS:
        # Run one step up front so a variant that cannot run here is dropped
        # (unless --strict) before the timed loop.
        try:
            step_fn = VARIANTS[name](device, dtype)
            step_fn()
        except Exception:
            if strict:
                raise
            _log(f"WARNING: dropping variant {name}:\n{traceback.format_exc()}")
            continue
        names.append(name)
        steps.append(step_fn)
    if not steps:
        raise SystemExit("[bolt-synth] no variant could run")
    _log(f"variants={','.join(names)}")

    counter = 0
    # Per-variant [seconds, steps]; main() prints and resets them every window.
    stats: dict[str, list[float]] = {name: [0.0, 0] for name in names}

    def round_robin():
        nonlocal counter
        idx = counter % len(steps)
        t0 = time.perf_counter()
        result = steps[idx]()
        torch.cuda.synchronize()
        stats[names[idx]][0] += time.perf_counter() - t0
        stats[names[idx]][1] += 1
        counter += 1
        return result

    return round_robin, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", required=True, choices=[*sorted(VARIANTS), "all"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=50)
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    p.add_argument("--dtype", default="bf16", choices=list(dtypes))
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="stop the timed loop early after this many seconds (0: no limit)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="with --variant all, fail instead of dropping variants that cannot run",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    device = torch.device("cuda")
    dtype = dtypes[args.dtype]
    _log(
        f"variant={args.variant} dtype={args.dtype} "
        f"steps={args.steps} warmup={args.warmup}"
    )
    _log(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(0)}"
    )

    stats: dict[str, list[float]] = {}
    if args.variant == "all":
        step_fn, stats = _build_round_robin(device, dtype, args.strict)
    else:
        step_fn = VARIANTS[args.variant](device, dtype)

    # Untimed warmup lets cuDNN / cuBLAS heuristics and the allocator settle.
    for i in range(args.warmup):
        step_fn()
        if (i + 1) % max(1, args.warmup // 5) == 0:
            _log(f"warmup {i + 1}/{args.warmup}")

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    last_log_t = t_start
    last_log_step = 0
    done_steps = 0
    for i in range(args.steps):
        step_fn()
        done_steps = i + 1
        if (i + 1) % args.log_every == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            window = now - last_log_t
            window_iters = (i + 1) - last_log_step
            window_ips = window_iters / window if window > 0 else 0.0
            window_ms = (window / window_iters) * 1000.0 if window_iters > 0 else 0.0
            _log(
                f"step={i + 1}/{args.steps} iters_per_second = {window_ips:.4f}; "
                f"step_time_ms = {window_ms:.4f};"
            )
            if stats:
                parts = [
                    f"{name}={vtime / vcount * 1000.0:.1f}"
                    for name, (vtime, vcount) in stats.items()
                    if vcount > 0
                ]
                _log(f"per_variant_ms step={i + 1}: " + " ".join(parts))
                for stat in stats.values():
                    stat[:] = [0.0, 0]
            last_log_t = now
            last_log_step = i + 1
        if args.max_seconds and time.perf_counter() - t_start > args.max_seconds:
            _log(
                f"stopping after {done_steps}/{args.steps} steps: "
                f"--max-seconds={args.max_seconds:g} reached"
            )
            break

    torch.cuda.synchronize()
    total = time.perf_counter() - t_start
    overall_ips = done_steps / total if total > 0 else 0.0
    overall_ms = (total / done_steps) * 1000.0 if done_steps > 0 else 0.0
    _log(
        f"done variant={args.variant} steps={done_steps} "
        f"iters_per_second = {overall_ips:.4f}; step_time_ms = {overall_ms:.4f};"
    )
    # Exit normally: the instrumented libraries write profiles from fini hooks.
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
