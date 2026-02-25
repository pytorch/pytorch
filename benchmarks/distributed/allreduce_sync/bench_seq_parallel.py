"""Sequence-parallel fused GEMM+comm benchmark.

Compares fused vs unfused approaches to sequence parallelism, where
allreduce is decomposed into reduce_scatter + all_gather and fused
with adjacent GEMM operations.

Level 2 optimization path (vs Level 1 allreduce):
  Column-parallel GEMM: all_gather(input) @ weight  ->  fused_all_gather_matmul
  Row-parallel GEMM:    input @ weight -> reduce_scatter  ->  fused_matmul_reduce_scatter

Backends:
  nccl:           unfused NCCL all_gather/reduce_scatter + torch.mm
  symm_fused:     PyTorch symmetric memory fused ops (micro-pipelined)
  allreduce_nccl: Level 1 baseline (matmul + NCCL allreduce, no seq parallelism)

Usage:
  torchrun --nproc_per_node=N bench_seq_parallel.py [options]

Examples:
  torchrun --nproc_per_node=8 bench_seq_parallel.py --backend all
  torchrun --nproc_per_node=8 bench_seq_parallel.py --skip-raw --seq-len 512
"""

import argparse
import statistics

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory  # registers torch.ops.symm_mem.*
import torch.nn as nn
import torch.nn.functional as F

from qwen3_block import Qwen3DecoderLayer, RMSNorm, RotaryEmbedding


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
rank = 0
world_size = 1
device = None
group_name = None

BACKENDS = ["nccl", "symm_fused", "allreduce_nccl"]


# ---------------------------------------------------------------------------
# Backend functions
#
# ag_mm(x_shard, weight): all_gather(x_shard) @ weight.T
#   x_shard: (local_M, K), weight: Parameter(N, K) -> (global_M, N)
#
# mm_rs(x, weight): reduce_scatter(x @ weight.T)
#   x: (global_M, K), weight: Parameter(N, K) -> (local_M, N)
# ---------------------------------------------------------------------------

def nccl_ag_mm(x_shard, weight):
    M_full = x_shard.shape[0] * world_size
    buf = torch.empty(M_full, x_shard.shape[1], dtype=x_shard.dtype, device=device)
    dist.all_gather_into_tensor(buf, x_shard.contiguous())
    return F.linear(buf, weight)


def nccl_mm_rs(x, weight):
    full = F.linear(x, weight)
    M_local = full.shape[0] // world_size
    buf = torch.empty(M_local, full.shape[1], dtype=full.dtype, device=device)
    dist.reduce_scatter_tensor(buf, full)
    return buf


def symm_ag_mm(x_shard, weight):
    _, results = torch.ops.symm_mem.fused_all_gather_matmul(
        x_shard.contiguous(), [weight.t()], 0, group_name, return_A=False
    )
    return results[0]


def symm_mm_rs(x, weight):
    return torch.ops.symm_mem.fused_matmul_reduce_scatter(
        x.contiguous(), weight.t(), "sum", 0, group_name
    )


BACKEND_FNS = {
    "nccl": (nccl_ag_mm, nccl_mm_rs),
    "symm_fused": (symm_ag_mm, symm_mm_rs),
}


# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------
class SeqParallelTimer:
    """Wraps ag_mm/mm_rs and records CUDA events per call."""

    def __init__(self, ag_mm_fn, mm_rs_fn):
        self._ag_mm = ag_mm_fn
        self._mm_rs = mm_rs_fn
        self.events = []
        self.recording = False

    def start_recording(self):
        self.recording = True
        self.events = []

    def stop_recording(self):
        self.recording = False

    def ag_mm(self, x_shard, weight):
        if self.recording:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            result = self._ag_mm(x_shard, weight)
            e.record()
            self.events.append((s, e))
            return result
        return self._ag_mm(x_shard, weight)

    def mm_rs(self, x, weight):
        if self.recording:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            result = self._mm_rs(x, weight)
            e.record()
            self.events.append((s, e))
            return result
        return self._mm_rs(x, weight)

    def per_call_times(self, calls_per_iter):
        n_iters = len(self.events) // calls_per_iter
        slot_times = [[] for _ in range(calls_per_iter)]
        for i in range(n_iters):
            for j in range(calls_per_iter):
                s, e = self.events[i * calls_per_iter + j]
                slot_times[j].append(s.elapsed_time(e) * 1000.0)
        return [sorted(t)[len(t) // 2] for t in slot_times]


class AllreduceTimer:
    """Records CUDA events around each allreduce call (for baseline)."""

    def __init__(self, fn):
        self.fn = fn
        self.events = []
        self.recording = False

    def __call__(self, inp, out):
        if self.recording:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            self.fn(inp, out)
            e.record()
            self.events.append((s, e))
        else:
            self.fn(inp, out)

    def start_recording(self):
        self.recording = True
        self.events = []

    def stop_recording(self):
        self.recording = False

    def per_call_times(self, calls_per_iter):
        n_iters = len(self.events) // calls_per_iter
        slot_times = [[] for _ in range(calls_per_iter)]
        for i in range(n_iters):
            for j in range(calls_per_iter):
                s, e = self.events[i * calls_per_iter + j]
                slot_times[j].append(s.elapsed_time(e) * 1000.0)
        return [sorted(t)[len(t) // 2] for t in slot_times]


# ---------------------------------------------------------------------------
# Seq-parallel transformer blocks
# ---------------------------------------------------------------------------
class Qwen3SeqParallelAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_kv_heads, world_size,
                 rms_norm_eps=1e-6, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.world_size = world_size
        self.head_dim = hidden_dim // n_heads
        self.n_local_heads = n_heads // world_size
        self.n_local_kv_heads = max(1, n_kv_heads // world_size)

        q_dim = self.n_local_heads * self.head_dim
        kv_dim = self.n_local_kv_heads * self.head_dim
        self.q_size = q_dim
        self.kv_size = kv_dim

        self.qkv_proj = nn.Linear(
            hidden_dim, q_dim + 2 * kv_dim,
            bias=False, device=device, dtype=dtype,
        )
        self.o_proj = nn.Linear(
            q_dim, hidden_dim,
            bias=False, device=device, dtype=dtype,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps,
                              device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps,
                              device=device, dtype=dtype)
        self.rotary_emb = RotaryEmbedding(self.head_dim, device=device)

    def forward(self, h_shard, positions, ag_mm, mm_rs):
        """h_shard: (batch, local_seq, hidden) -> (batch, local_seq, hidden)"""
        batch, local_seq, _ = h_shard.shape
        full_seq = local_seq * self.world_size

        # All-gather + QKV projection
        h_flat = h_shard.reshape(-1, self.hidden_dim)
        qkv_flat = ag_mm(h_flat, self.qkv_proj.weight)
        qkv = qkv_flat.view(batch, full_seq, -1)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(
            q.view(batch, full_seq, self.n_local_heads, self.head_dim)
        ).view(batch, full_seq, -1)
        k = self.k_norm(
            k.view(batch, full_seq, self.n_local_kv_heads, self.head_dim)
        ).view(batch, full_seq, -1)

        q, k = self.rotary_emb(positions, q, k)

        q = q.view(batch, full_seq, self.n_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, full_seq, self.n_local_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, full_seq, self.n_local_kv_heads, self.head_dim).transpose(1, 2)

        if self.n_local_kv_heads < self.n_local_heads:
            repeat = self.n_local_heads // self.n_local_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=(full_seq > 1))
        attn = attn.transpose(1, 2).reshape(batch, full_seq, -1)

        # O projection + reduce-scatter
        attn_flat = attn.reshape(-1, self.q_size)
        out_flat = mm_rs(attn_flat, self.o_proj.weight)
        return out_flat.view(batch, local_seq, self.hidden_dim)


class Qwen3SeqParallelMLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_size, world_size,
                 device=None, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.world_size = world_size
        shard = intermediate_size // world_size
        self.shard = shard

        self.gate_up_proj = nn.Linear(
            hidden_dim, shard * 2,
            bias=False, device=device, dtype=dtype,
        )
        self.down_proj = nn.Linear(
            shard, hidden_dim,
            bias=False, device=device, dtype=dtype,
        )

    def forward(self, h_shard, ag_mm, mm_rs):
        batch, local_seq, _ = h_shard.shape
        full_seq = local_seq * self.world_size

        h_flat = h_shard.reshape(-1, self.hidden_dim)
        gate_up_flat = ag_mm(h_flat, self.gate_up_proj.weight)
        gate_up = gate_up_flat.view(batch, full_seq, -1)

        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up

        h_flat = h.reshape(-1, self.shard)
        out_flat = mm_rs(h_flat, self.down_proj.weight)
        return out_flat.view(batch, local_seq, self.hidden_dim)


class Qwen3SeqParallelDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_kv_heads, intermediate_size,
                 world_size, timer,
                 rms_norm_eps=1e-6, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.timer = timer
        self.input_layernorm = RMSNorm(hidden_dim, eps=rms_norm_eps,
                                       device=device, dtype=dtype)
        self.self_attn = Qwen3SeqParallelAttention(
            hidden_dim, n_heads, n_kv_heads, world_size,
            rms_norm_eps=rms_norm_eps, device=device, dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(hidden_dim, eps=rms_norm_eps,
                                                device=device, dtype=dtype)
        self.mlp = Qwen3SeqParallelMLP(
            hidden_dim, intermediate_size, world_size,
            device=device, dtype=dtype,
        )

    @torch.no_grad()
    def forward(self, x, positions, residual=None):
        if residual is None:
            residual = x
            h = self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)

        h = self.self_attn(h, positions, self.timer.ag_mm, self.timer.mm_rs)
        h, residual = self.post_attention_layernorm(h, residual)
        h = self.mlp(h, self.timer.ag_mm, self.timer.mm_rs)
        return h, residual


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------
def report_times(label, times_us):
    if rank != 0:
        return
    times = sorted(times_us)
    n = len(times)
    mean = statistics.mean(times)
    p50 = times[n // 2]
    p99 = times[int(n * 0.99)]
    mn = times[0]
    mx = times[-1]
    print(f"  {label:30s}  mean={mean:8.2f}us  p50={p50:8.2f}us  "
          f"p99={p99:8.2f}us  min={mn:8.2f}us  max={mx:8.2f}us")


# ---------------------------------------------------------------------------
# Raw microbenchmark
# ---------------------------------------------------------------------------
def bench_raw(backends, args):
    if rank == 0:
        print("\n" + "=" * 80)
        print("RAW FUSED GEMM+COMM MICROBENCHMARK")
        print(f"hidden_dim={args.hidden_dim}, world_size={world_size}")
        print("=" * 80)

    K = args.hidden_dim
    head_dim = K // args.n_heads
    local_q_dim = (args.n_heads // world_size) * head_dim
    local_kv_dim = max(1, args.n_kv_heads // world_size) * head_dim
    local_qkv_dim = local_q_dim + 2 * local_kv_dim
    local_intermediate = args.intermediate_size // world_size

    ops = [
        ("AG+MM (qkv_proj)",  "ag_mm", K, local_qkv_dim),
        ("AG+MM (gate_up)",   "ag_mm", K, 2 * local_intermediate),
        ("MM+RS (o_proj)",    "mm_rs", local_q_dim, K),
        ("MM+RS (down_proj)", "mm_rs", local_intermediate, K),
    ]

    token_counts = [int(t) for t in args.tokens.split(",")]
    sp_backends = [b for b in backends if b in BACKEND_FNS]

    for op_name, op_type, op_K, op_N in ops:
        if rank == 0:
            print(f"\n{op_name}: K={op_K}, N={op_N}")

        for M_total in token_counts:
            M_local = M_total // world_size
            if M_local == 0:
                continue
            if rank == 0:
                print(f"  M_total={M_total} ({M_local}/rank):")

            for backend in sp_backends:
                ag_mm_fn, mm_rs_fn = BACKEND_FNS[backend]
                try:
                    # Weight in nn.Linear format (N, K)
                    w = torch.randn(op_N, op_K, dtype=torch.bfloat16, device=device)

                    if op_type == "ag_mm":
                        x = torch.randn(M_local, op_K, dtype=torch.bfloat16, device=device)

                        def fn(x=x, w=w):
                            ag_mm_fn(x, w)
                    else:
                        x = torch.randn(M_total, op_K, dtype=torch.bfloat16, device=device)

                        def fn(x=x, w=w):
                            mm_rs_fn(x, w)

                    # Warmup
                    for _ in range(args.warmup):
                        fn()
                    torch.cuda.synchronize()

                    starts = [torch.cuda.Event(enable_timing=True)
                              for _ in range(args.iters)]
                    ends = [torch.cuda.Event(enable_timing=True)
                            for _ in range(args.iters)]
                    torch.cuda.synchronize()
                    for i in range(args.iters):
                        starts[i].record()
                        fn()
                        ends[i].record()
                    torch.cuda.synchronize()

                    times = [starts[i].elapsed_time(ends[i]) * 1000.0
                             for i in range(args.iters)]
                    report_times(f"    {backend}", times)

                except Exception as e:
                    if rank == 0:
                        import traceback
                        print(f"      {backend:28s}  FAILED: {e}")
                        traceback.print_exc()


# ---------------------------------------------------------------------------
# Transformer layer benchmark
# ---------------------------------------------------------------------------
def bench_transformer(backends, args):
    if rank == 0:
        print("\n" + "=" * 80)
        print("TRANSFORMER LAYER BENCHMARK (Qwen3 Seq-Parallel)")
        print(f"hidden_dim={args.hidden_dim}, n_heads={args.n_heads}, "
              f"n_kv_heads={args.n_kv_heads}, "
              f"intermediate_size={args.intermediate_size}, "
              f"n_layers={args.n_layers}, batch={args.batch_size}, "
              f"seq_len={args.seq_len}")
        print("=" * 80)

    full_seq = args.seq_len
    local_seq = full_seq // world_size
    assert local_seq * world_size == full_seq, \
        f"seq_len={full_seq} must be divisible by world_size={world_size}"

    positions = torch.arange(
        full_seq, dtype=torch.long, device=device
    ).unsqueeze(0).expand(args.batch_size, -1)

    # --- Seq-parallel backends ---
    sp_backends = [b for b in backends if b in BACKEND_FNS]
    for backend in sp_backends:
        ag_mm_fn, mm_rs_fn = BACKEND_FNS[backend]
        try:
            timer = SeqParallelTimer(ag_mm_fn, mm_rs_fn)
            calls_per_iter = 4 * args.n_layers

            call_labels = []
            for l in range(args.n_layers):
                call_labels.extend([
                    f"L{l}.ag_qkv", f"L{l}.rs_attn",
                    f"L{l}.ag_mlp", f"L{l}.rs_mlp",
                ])

            layers = nn.ModuleList([
                Qwen3SeqParallelDecoderLayer(
                    hidden_dim=args.hidden_dim,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    intermediate_size=args.intermediate_size,
                    world_size=world_size,
                    timer=timer,
                    device=device,
                )
                for _ in range(args.n_layers)
            ])

            x_shard = torch.randn(
                args.batch_size, local_seq, args.hidden_dim,
                dtype=torch.bfloat16, device=device,
            )

            def sp_forward(x_shard=x_shard):
                h, residual = x_shard, None
                for layer in layers:
                    h, residual = layer(h, positions, residual)
                return h

            for _ in range(args.warmup):
                sp_forward()
            torch.cuda.synchronize()

            starts = [torch.cuda.Event(enable_timing=True)
                      for _ in range(args.iters)]
            ends = [torch.cuda.Event(enable_timing=True)
                    for _ in range(args.iters)]

            timer.start_recording()
            torch.cuda.synchronize()
            for i in range(args.iters):
                starts[i].record()
                sp_forward()
                ends[i].record()
            torch.cuda.synchronize()
            timer.stop_recording()

            e2e = [starts[i].elapsed_time(ends[i]) * 1000.0
                   for i in range(args.iters)]

            per_call = timer.per_call_times(calls_per_iter)
            per_call_tensor = torch.tensor(per_call, dtype=torch.float64,
                                           device=device)
            all_per_call = [torch.empty_like(per_call_tensor)
                            for _ in range(world_size)]
            dist.all_gather(all_per_call, per_call_tensor)

            if rank == 0:
                stacked = torch.stack(all_per_call)
                min_per_call = stacked.min(dim=0).values.tolist()
                comm_total = sum(min_per_call)
                parts = "  ".join(
                    f"{call_labels[j]}={min_per_call[j]:.1f}us"
                    for j in range(len(min_per_call))
                )
                report_times(backend, e2e)
                print(f"    fused gemm+comm p50: {parts}")
                print(f"    total={comm_total:.1f}us")

        except Exception as e:
            if rank == 0:
                import traceback
                print(f"  {backend:30s}  FAILED: {e}")
                traceback.print_exc()

    # --- Allreduce baseline (Level 1) ---
    if "allreduce_nccl" in backends:
        try:
            def nccl_allreduce(inp, out):
                if inp.data_ptr() != out.data_ptr():
                    out.view(-1).copy_(inp.contiguous().view(-1))
                dist.all_reduce(out)

            ar_timer = AllreduceTimer(nccl_allreduce)
            ar_calls = 2 * args.n_layers
            ar_labels = []
            for l in range(args.n_layers):
                ar_labels.extend([f"L{l}.attn", f"L{l}.mlp"])

            max_tokens = args.batch_size * full_seq
            x_full = torch.randn(
                args.batch_size, full_seq, args.hidden_dim,
                dtype=torch.bfloat16, device=device,
            )

            ar_layers = nn.ModuleList([
                Qwen3DecoderLayer(
                    hidden_dim=args.hidden_dim,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    intermediate_size=args.intermediate_size,
                    world_size=world_size,
                    allreduce_fn=ar_timer,
                    max_tokens=max_tokens,
                    device=device,
                )
                for _ in range(args.n_layers)
            ])

            def ar_forward():
                h, residual = x_full, None
                for layer in ar_layers:
                    h, residual = layer(h, positions, residual)
                return h

            for _ in range(args.warmup):
                ar_forward()
            torch.cuda.synchronize()

            starts = [torch.cuda.Event(enable_timing=True)
                      for _ in range(args.iters)]
            ends = [torch.cuda.Event(enable_timing=True)
                    for _ in range(args.iters)]

            ar_timer.start_recording()
            torch.cuda.synchronize()
            for i in range(args.iters):
                starts[i].record()
                ar_forward()
                ends[i].record()
            torch.cuda.synchronize()
            ar_timer.stop_recording()

            e2e = [starts[i].elapsed_time(ends[i]) * 1000.0
                   for i in range(args.iters)]

            per_call = ar_timer.per_call_times(ar_calls)
            per_call_tensor = torch.tensor(per_call, dtype=torch.float64,
                                           device=device)
            all_per_call = [torch.empty_like(per_call_tensor)
                            for _ in range(world_size)]
            dist.all_gather(all_per_call, per_call_tensor)

            if rank == 0:
                stacked = torch.stack(all_per_call)
                min_per_call = stacked.min(dim=0).values.tolist()
                ar_total = sum(min_per_call)
                parts = "  ".join(
                    f"{ar_labels[j]}={min_per_call[j]:.1f}us"
                    for j in range(len(min_per_call))
                )
                report_times("allreduce_nccl (baseline)", e2e)
                print(f"    allreduce kernel p50: {parts}")
                print(f"    total={ar_total:.1f}us")

        except Exception as e:
            if rank == 0:
                import traceback
                print(f"  {'allreduce_nccl':30s}  FAILED: {e}")
                traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Seq-parallel fused GEMM+comm benchmark")
    p.add_argument("--backend", default="all",
                   help="Comma-separated backends or 'all'")
    p.add_argument("--hidden-dim", type=int, default=4096)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--n-kv-heads", type=int, default=8)
    p.add_argument("--intermediate-size", type=int, default=11008)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128,
                   help="Full sequence length (must be divisible by world_size)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--tokens", default="8,16,32,64,128,256,512,1024",
                   help="Comma-separated total token counts for raw benchmark")
    p.add_argument("--skip-transformer", action="store_true")
    p.add_argument("--skip-raw", action="store_true")
    return p.parse_args()


def main():
    global rank, world_size, device, group_name

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    group_name = dist.group.WORLD.group_name

    args = parse_args()

    if args.backend == "all":
        backends = list(BACKENDS)
    else:
        backends = [b.strip() for b in args.backend.split(",")]

    if rank == 0:
        print(f"Seq-Parallel Fused GEMM+Comm Benchmark")
        print(f"world_size={world_size}, device={device}")
        print(f"backends={backends}")

    dist.barrier()

    if not args.skip_raw:
        bench_raw(backends, args)
    if not args.skip_transformer:
        bench_transformer(backends, args)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
