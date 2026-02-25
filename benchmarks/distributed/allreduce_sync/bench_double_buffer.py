"""Double-buffered nosync allreduce benchmark.

Demonstrates that skipping the end barrier of allreduce is safe when
using double buffering: consecutive allreduces alternate between two
registered buffers, so by the time a buffer is reused, all peers have
finished reading from it (guaranteed by the intervening start barrier).

Safety argument:
  - AR_0 uses buf_A (nosync: no end barrier)
  - AR_1 uses buf_B (nosync): start barrier ensures all ranks finished AR_0
  - AR_2 uses buf_A again: start barrier ensures all ranks finished AR_1,
    which means all ranks finished AR_0's reads long ago → buf_A is safe

Tests:
  1. Numerics: verify nosync + double buffer matches NCCL reference
  2. Standalone chain: N allreduces, compare synced vs nosync+double-buffer
  3. Transformer: Qwen3 decoder with double-buffered allreduce

Usage:
  torchrun --nproc_per_node=N bench_double_buffer.py [options]

Examples:
  torchrun --nproc_per_node=2 bench_double_buffer.py
  torchrun --nproc_per_node=8 bench_double_buffer.py --n-layers 4 --seq-len 1
"""

import argparse
import statistics
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem_mod
import torch.nn as nn
from torch.utils.cpp_extension import load

from qwen3_block import Qwen3Attention, Qwen3MLP, RMSNorm

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
rank = 0
world_size = 1
device = None
ext = None  # JIT-compiled custom AR extension

buf_a = None  # symm_mem buffer A
buf_b = None  # symm_mem buffer B
group_name = None

# GPU clock/power state saved for restore
_orig_power_limit = None   # watts (int)
_orig_gr_clock = None      # MHz (int)
_orig_mem_clock = None     # MHz (int)


# ---------------------------------------------------------------------------
# GPU clock / power management
# ---------------------------------------------------------------------------
def _smi(*args, needs_sudo=False):
    """Run nvidia-smi and return stdout. Silently ignores failures."""
    try:
        cmd = ["sudo", "nvidia-smi", *args] if needs_sudo else ["nvidia-smi", *args]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return ""


def lock_gpu_clocks():
    """Pin SM clock to max and power to max across all GPUs. Only rank 0 runs."""
    global _orig_power_limit, _orig_gr_clock, _orig_mem_clock
    if rank != 0:
        return

    # Query current settings so we can restore later
    out = _smi("-i", "0", "--query-gpu=power.limit,clocks.max.sm,clocks.max.mem",
               "--format=csv,noheader,nounits")
    if not out:
        print("[clock] WARNING: could not query GPU clocks, skipping lock")
        return
    parts = [x.strip() for x in out.split(",")]
    _orig_power_limit = int(float(parts[0]))
    max_sm = int(parts[1])
    max_mem = int(parts[2])

    # Save current application clocks
    out2 = _smi("-i", "0", "--query-gpu=clocks.applications.gr,clocks.applications.mem",
                "--format=csv,noheader,nounits")
    if out2:
        parts2 = [x.strip() for x in out2.split(",")]
        _orig_gr_clock = int(parts2[0])
        _orig_mem_clock = int(parts2[1])

    # Query max power
    out3 = _smi("-i", "0", "--query-gpu=power.max_limit", "--format=csv,noheader,nounits")
    max_power = int(float(out3)) if out3 else _orig_power_limit

    # Lock clocks and power across all GPUs (requires sudo)
    _smi("-pm", "1", needs_sudo=True)
    _smi("--power-limit=" + str(max_power), needs_sudo=True)
    _smi("--lock-gpu-clocks=" + str(max_sm) + "," + str(max_sm), needs_sudo=True)
    _smi("--lock-memory-clocks=" + str(max_mem) + "," + str(max_mem), needs_sudo=True)
    print(f"[clock] Locked: SM={max_sm}MHz, mem={max_mem}MHz, power={max_power}W")


def unlock_gpu_clocks():
    """Restore original GPU clock/power settings. Only rank 0 runs."""
    if rank != 0:
        return
    if _orig_power_limit is None:
        return

    _smi("--reset-gpu-clocks", needs_sudo=True)
    _smi("--reset-memory-clocks", needs_sudo=True)
    _smi("--power-limit=" + str(_orig_power_limit), needs_sudo=True)
    print(f"[clock] Restored: power={_orig_power_limit}W, clocks reset to defaults")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup(max_elems):
    global ext, buf_a, buf_b, group_name

    src_dir = Path(__file__).parent
    ext = load(
        name="custom_all_reduce_ext",
        sources=[str(src_dir / "custom_all_reduce_wrapper.cu")],
        extra_include_paths=[str(src_dir)],
        extra_cuda_cflags=["-O2", "-std=c++17"],
        verbose=(rank == 0),
    )

    # Custom AR IPC buffer (needed for barrier infrastructure)
    elem_bytes = 2
    signal_size = 1024 * 16
    scratch_size = max_elems * elem_bytes
    data_size = max_elems * elem_bytes
    total_size = signal_size + scratch_size + data_size

    ptr_tensor, handle_tensor = ext.allocate_and_get_handle(total_size)
    local_ptr = ptr_tensor.item()

    handle_list = [None] * world_size
    dist.all_gather_object(handle_list, handle_tensor.cpu())

    signal_ptrs = torch.zeros(world_size, dtype=torch.int64)
    data_ptrs = torch.zeros(world_size, dtype=torch.int64)
    for i in range(world_size):
        if i == rank:
            signal_ptrs[i] = local_ptr
            data_ptrs[i] = local_ptr + signal_size + scratch_size
        else:
            peer_ptr = ext.open_ipc_handle(handle_list[i]).item()
            signal_ptrs[i] = peer_ptr
            data_ptrs[i] = peer_ptr + signal_size + scratch_size

    ext.init_custom_ar(signal_ptrs, rank, world_size)
    ext.register_buffer(data_ptrs, rank, max_elems)

    # Two symm_mem buffers for double buffering
    pool = symm_mem_mod.get_mem_pool(device)
    with torch.cuda.use_mem_pool(pool):
        buf_a = torch.empty(max_elems, dtype=torch.bfloat16, device=device)
        buf_b = torch.empty(max_elems, dtype=torch.bfloat16, device=device)
    group_name = dist.group.WORLD.group_name

    # Register both with custom AR barrier infrastructure
    def register_symm_buf(buf, register_fn):
        handle = symm_mem_mod.rendezvous(buf, group=dist.group.WORLD)
        peer_ptrs = torch.zeros(world_size, dtype=torch.int64)
        for i in range(world_size):
            peer_buf = handle.get_buffer(i, (max_elems,), torch.bfloat16)
            peer_ptrs[i] = peer_buf.data_ptr()
        register_fn(peer_ptrs, rank, max_elems)

    register_symm_buf(buf_a, ext.register_symm_buffer)
    register_symm_buf(buf_b, ext.register_symm_buffer_b)

    # Warm up: the IPC memset runs on stream 0 but kernels run on PyTorch's
    # stream. A sync + warm-up allreduce primes the barrier state for all
    # block counts that will be used later.
    torch.cuda.synchronize()
    dist.barrier()
    ext.allreduce_symm_sync_a(max_elems, "bf16")
    ext.allreduce_symm_sync_b(max_elems, "bf16")
    torch.cuda.synchronize()

    dist.barrier()
    if rank == 0:
        print("[setup] Custom AR + double-buffered symm_mem initialized")


# ---------------------------------------------------------------------------
# Allreduce primitives (operate on pre-registered symm_mem buffers)
# ---------------------------------------------------------------------------
def nosync_a(inp, out):
    ext.allreduce_symm_nosync(inp.numel(), "bf16")

def nosync_b(inp, out):
    ext.allreduce_symm_nosync_b(inp.numel(), "bf16")

def sync_a(inp, out):
    ext.allreduce_symm_sync_a(inp.numel(), "bf16")

def sync_b(inp, out):
    ext.allreduce_symm_sync_b(inp.numel(), "bf16")

def nccl_fn(inp, out):
    if inp.data_ptr() != out.data_ptr():
        out.view(-1).copy_(inp.view(-1))
    dist.all_reduce(out)


# ---------------------------------------------------------------------------
# Double-buffer allreduce manager
# ---------------------------------------------------------------------------
class DoubleBufferAllreduce:
    """Alternates allreduce calls between buf_a (slot 0) and buf_b (slot 1).

    If nosync=True, ALL calls use the nosync kernel variant. This is safe
    because double buffering guarantees that by the time a buffer is reused
    (2 ARs later), all peers have passed through an intervening start barrier,
    meaning they finished reading from that buffer long ago. This holds
    across iteration boundaries too: the stream serialization ensures that
    entering AR[0] of iter N+1 implies completion of all iter N work.
    """

    def __init__(self, total_calls_per_iter, nosync=True):
        self.total = total_calls_per_iter
        self.nosync = nosync
        self.idx = 0
        self._nosync_fns = [nosync_a, nosync_b]
        self._sync_fns = [sync_a, sync_b]

    def __call__(self, inp, out):
        slot = self.idx % 2
        if self.nosync:
            self._nosync_fns[slot](inp, out)
        else:
            self._sync_fns[slot](inp, out)
        self.idx = (self.idx + 1) % self.total

    def reset(self):
        self.idx = 0


# ---------------------------------------------------------------------------
# Transformer layer with double-buffered allreduce
# ---------------------------------------------------------------------------
class DoubleBufferDecoderLayer(nn.Module):
    """Qwen3 decoder layer where attn and MLP allreduces use separate buffers.

    attn writes into buf_a, MLP writes into buf_b. This guarantees that
    consecutive allreduces always target different buffers, enabling safe
    nosync (no end barrier) for all but the last allreduce.
    """

    def __init__(self, hidden_dim, n_heads, n_kv_heads, intermediate_size,
                 world_size, allreduce_fn, max_tokens,
                 attn_buf, mlp_buf,
                 rms_norm_eps=1e-6, device=None, dtype=torch.bfloat16):
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

        n = max_tokens * hidden_dim
        self.attn_buf = attn_buf[:n]
        self.mlp_buf = mlp_buf[:n]

    @torch.no_grad()
    def forward(self, x, positions, residual=None):
        if residual is None:
            residual = x
            h = self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)

        n = h.shape[0] * h.shape[1] * self.hidden_dim

        # Attention → buf_a
        attn_flat = self.attn_buf[:n]
        attn_3d = attn_flat.view(x.shape)
        self.self_attn(h, positions, out=attn_3d)
        self.allreduce_fn(attn_flat, attn_flat)

        h, residual = self.post_attention_layernorm(attn_3d, residual)

        # MLP → buf_b
        mlp_flat = self.mlp_buf[:n]
        mlp_3d = mlp_flat.view(x.shape)
        self.mlp(h, out=mlp_3d)
        self.allreduce_fn(mlp_flat, mlp_flat)

        return mlp_3d, residual


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def bench_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    return [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]


def report(label, times_us):
    if rank != 0:
        return
    times = sorted(times_us)
    n = len(times)
    print(f"  {label:30s}  mean={statistics.mean(times):8.2f}µs  "
          f"p50={times[n // 2]:8.2f}µs  "
          f"min={times[0]:8.2f}µs  max={times[-1]:8.2f}µs")


# ---------------------------------------------------------------------------
# Numerics verification
# ---------------------------------------------------------------------------
def verify_numerics(numel):
    if rank == 0:
        print("\n" + "=" * 70)
        print("NUMERICS VERIFICATION")
        print("=" * 70)

    data = torch.arange(numel, dtype=torch.bfloat16, device=device) * (rank + 1)

    # Test 1: single nosync allreduce on buf_a
    buf_a[:numel].copy_(data)
    torch.cuda.synchronize()
    dist.barrier()
    ext.allreduce_symm_nosync(numel, "bf16")
    torch.cuda.synchronize()
    result_nosync = buf_a[:numel].clone()
    dist.barrier()

    # Reference: single sync allreduce on buf_a
    buf_a[:numel].copy_(data)
    torch.cuda.synchronize()
    dist.barrier()
    ext.allreduce_symm_sync_a(numel, "bf16")
    torch.cuda.synchronize()
    result_sync = buf_a[:numel].clone()
    dist.barrier()

    # Test 2: chain of 4 independent nosync allreduces (alternating buffers)
    # This matches the real usage pattern: no mutations between ARs.
    buf_a[:numel].copy_(data)
    buf_b[:numel].copy_(data)
    torch.cuda.synchronize()
    dist.barrier()
    ext.allreduce_symm_nosync(numel, "bf16")     # AR 0: buf_a, nosync
    ext.allreduce_symm_nosync_b(numel, "bf16")    # AR 1: buf_b, nosync
    ext.allreduce_symm_nosync(numel, "bf16")      # AR 2: buf_a, nosync
    ext.allreduce_symm_nosync_b(numel, "bf16")    # AR 3: buf_b, nosync
    torch.cuda.synchronize()
    chain_nosync_a = buf_a[:numel].clone()
    chain_nosync_b = buf_b[:numel].clone()
    dist.barrier()

    # Reference: same chain with all synced
    buf_a[:numel].copy_(data)
    buf_b[:numel].copy_(data)
    torch.cuda.synchronize()
    dist.barrier()
    ext.allreduce_symm_sync_a(numel, "bf16")
    ext.allreduce_symm_sync_b(numel, "bf16")
    ext.allreduce_symm_sync_a(numel, "bf16")
    ext.allreduce_symm_sync_b(numel, "bf16")
    torch.cuda.synchronize()
    chain_sync_a = buf_a[:numel].clone()
    chain_sync_b = buf_b[:numel].clone()

    if rank == 0:
        match1 = torch.equal(result_nosync, result_sync)
        match_a = torch.equal(chain_nosync_a, chain_sync_a)
        match_b = torch.equal(chain_nosync_b, chain_sync_b)
        print(f"  Single AR nosync vs sync:     {'PASS' if match1 else 'FAIL'}")
        if not match1:
            diff = (result_nosync - result_sync).abs().max().item()
            print(f"    max diff: {diff}")
        print(f"  Chain buf_a nosync vs sync:   {'PASS' if match_a else 'FAIL'}")
        if not match_a:
            diff = (chain_nosync_a - chain_sync_a).abs().max().item()
            print(f"    max diff: {diff}")
        print(f"  Chain buf_b nosync vs sync:   {'PASS' if match_b else 'FAIL'}")
        if not match_b:
            diff = (chain_nosync_b - chain_sync_b).abs().max().item()
            print(f"    max diff: {diff}")


# ---------------------------------------------------------------------------
# Standalone chain benchmark
# ---------------------------------------------------------------------------
def bench_chain(numel, n_chain, warmup, iters):
    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"STANDALONE CHAIN BENCHMARK")
        print(f"numel={numel} ({numel * 2 // 1024} KB), chain_length={n_chain}")
        print(f"{'=' * 70}")

    # Fill buffers with data
    buf_a[:numel].fill_(1.0)
    buf_b[:numel].fill_(1.0)
    torch.cuda.synchronize()
    dist.barrier()

    # Build call sequences with identical Python structure.
    # Pre-compute function lists so the hot loop is just fns[i](dummy, dummy).
    dummy = buf_a[:numel]

    sync_fns = [sync_a if i % 2 == 0 else sync_b for i in range(n_chain)]
    nosync_fns = [nosync_a if i % 2 == 0 else nosync_b for i in range(n_chain)]

    def chain_synced():
        for i in range(n_chain):
            sync_fns[i](dummy, dummy)

    def chain_nosync():
        for i in range(n_chain):
            nosync_fns[i](dummy, dummy)

    # Run nosync FIRST to eliminate ordering bias
    times = bench_fn(chain_nosync, warmup, iters)
    report("double_buf nosync", times)

    times = bench_fn(chain_synced, warmup, iters)
    report("double_buf synced", times)

    # NCCL baseline (single buffer)
    nccl_buf = torch.ones(numel, dtype=torch.bfloat16, device=device)

    def chain_nccl():
        for _ in range(n_chain):
            dist.all_reduce(nccl_buf)

    times = bench_fn(chain_nccl, warmup, iters)
    report("nccl", times)


# ---------------------------------------------------------------------------
# Transformer benchmark
# ---------------------------------------------------------------------------
def bench_transformer(args):
    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"TRANSFORMER LAYER BENCHMARK (Qwen3, double-buffered)")
        print(f"hidden={args.hidden_dim}, n_layers={args.n_layers}, "
              f"batch={args.batch_size}, seq_len={args.seq_len}")
        print(f"{'=' * 70}")

    max_tokens = args.batch_size * args.seq_len
    n_elems = max_tokens * args.hidden_dim
    calls_per_iter = 2 * args.n_layers

    x = torch.randn(args.batch_size, args.seq_len, args.hidden_dim,
                     dtype=torch.bfloat16, device=device)
    positions = torch.arange(args.seq_len, dtype=torch.long,
                             device=device).unsqueeze(0).expand(args.batch_size, -1)

    # --- Backend: nosync + double buffer ---
    db_ar = DoubleBufferAllreduce(calls_per_iter, nosync=True)
    db_layers = nn.ModuleList([
        DoubleBufferDecoderLayer(
            hidden_dim=args.hidden_dim, n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            intermediate_size=args.intermediate_size,
            world_size=world_size, allreduce_fn=db_ar,
            max_tokens=max_tokens,
            attn_buf=buf_a[:n_elems], mlp_buf=buf_b[:n_elems],
            device=device,
        )
        for _ in range(args.n_layers)
    ])

    def forward_db(x):
        db_ar.reset()
        h, residual = x, None
        for layer in db_layers:
            h, residual = layer(h, positions, residual)
        return h

    # --- Backend: all-synced + double buffer ---
    sync_ar = DoubleBufferAllreduce(calls_per_iter, nosync=False)
    sync_layers = nn.ModuleList([
        DoubleBufferDecoderLayer(
            hidden_dim=args.hidden_dim, n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            intermediate_size=args.intermediate_size,
            world_size=world_size, allreduce_fn=sync_ar,
            max_tokens=max_tokens,
            attn_buf=buf_a[:n_elems], mlp_buf=buf_b[:n_elems],
            device=device,
        )
        for _ in range(args.n_layers)
    ])

    # Share weights with nosync layers for fair comparison
    for sl, dl in zip(sync_layers, db_layers):
        sl.load_state_dict(dl.state_dict())

    def forward_sync(x):
        sync_ar.reset()
        h, residual = x, None
        for layer in sync_layers:
            h, residual = layer(h, positions, residual)
        return h

    backend = args.backend

    # --- Eager benchmarks ---
    if backend in ("all", "nosync"):
        times = bench_fn(lambda: forward_db(x), args.warmup, args.iters)
        report("nosync (eager)", times)

    if backend in ("all", "synced"):
        times = bench_fn(lambda: forward_sync(x), args.warmup, args.iters)
        report("synced (eager)", times)

    # --- Backend: NCCL baseline ---
    if backend in ("all", "nccl"):
        from qwen3_block import Qwen3DecoderLayer
        nccl_layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_dim=args.hidden_dim, n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                intermediate_size=args.intermediate_size,
                world_size=world_size, allreduce_fn=nccl_fn,
                max_tokens=max_tokens, device=device,
            )
            for _ in range(args.n_layers)
        ])
        for nl, dl in zip(nccl_layers, db_layers):
            nl.load_state_dict(dl.state_dict(), strict=False)

        def forward_nccl(x):
            h, residual = x, None
            for layer in nccl_layers:
                h, residual = layer(h, positions, residual)
            return h

        times = bench_fn(lambda: forward_nccl(x), args.warmup, args.iters)
        report("nccl (eager)", times)

    # --- Full CUDA graph capture ---
    # The custom AR kernels use device-side P2P reads and volatile signal
    # exchange with fixed pointers — no host sync, so the entire forward
    # pass (compute + allreduce) is capturable as a single CUDA graph.
    # This gives zero Python overhead in the measurement loop.
    if args.cudagraph:
        if rank == 0:
            print("\n  --- CUDA Graph (full capture) ---")

        def capture_and_bench(label, ar_fn, ref_layers):
            cg_layers = nn.ModuleList([
                DoubleBufferDecoderLayer(
                    hidden_dim=args.hidden_dim, n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    intermediate_size=args.intermediate_size,
                    world_size=world_size, allreduce_fn=ar_fn,
                    max_tokens=max_tokens,
                    attn_buf=buf_a[:n_elems], mlp_buf=buf_b[:n_elems],
                    device=device,
                )
                for _ in range(args.n_layers)
            ])
            for cl, rl in zip(cg_layers, ref_layers):
                cl.load_state_dict(rl.state_dict())

            def forward_cg():
                ar_fn.reset()
                h, residual = x, None
                for layer in cg_layers:
                    h, residual = layer(h, positions, residual)

            # Warmup (eager)
            for _ in range(args.warmup):
                forward_cg()
            torch.cuda.synchronize()

            # Full graph capture on a side stream (each graph gets its own pool)
            graph = torch.cuda.CUDAGraph()
            graph_pool = torch.cuda.graph_pool_handle()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                ar_fn.reset()
                graph.capture_begin(pool=graph_pool)
                h, residual = x, None
                for layer in cg_layers:
                    h, residual = layer(h, positions, residual)
                graph.capture_end()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            # Warmup replays
            for _ in range(args.warmup):
                graph.replay()
            torch.cuda.synchronize()

            # Timed replays
            times = bench_fn(graph.replay, 0, args.iters)
            report(label, times)

        if backend in ("all", "nosync"):
            db_ar_cg = DoubleBufferAllreduce(calls_per_iter, nosync=True)
            capture_and_bench("nosync (cudagraph)", db_ar_cg, db_layers)

        if backend in ("all", "synced"):
            sync_ar_cg = DoubleBufferAllreduce(calls_per_iter, nosync=False)
            capture_and_bench("synced (cudagraph)", sync_ar_cg, db_layers)

    # --- PyTorch profiler traces ---
    if args.profile:
        from torch.profiler import profile, ProfilerActivity

        profile_iters = min(args.iters, 5)

        def profile_variant(label, forward_fn, graph=None):
            torch.cuda.synchronize()
            dist.barrier()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(profile_iters):
                    if graph is not None:
                        graph.replay()
                    else:
                        forward_fn()
                torch.cuda.synchronize()
            trace_path = f"{label}_rank{rank}.json"
            prof.export_chrome_trace(trace_path)
            if rank == 0:
                print(f"  Trace: {trace_path}")

        if rank == 0:
            print("\n  --- PyTorch Profiler Traces (CUDA) ---")

        if args.cudagraph:
            if backend in ("all", "nosync"):
                # Reuse the CG AR object but re-capture for a clean graph
                cg_ar_ns = DoubleBufferAllreduce(calls_per_iter, nosync=True)
                cg_layers_ns = nn.ModuleList([
                    DoubleBufferDecoderLayer(
                        hidden_dim=args.hidden_dim, n_heads=args.n_heads,
                        n_kv_heads=args.n_kv_heads,
                        intermediate_size=args.intermediate_size,
                        world_size=world_size, allreduce_fn=cg_ar_ns,
                        max_tokens=max_tokens,
                        attn_buf=buf_a[:n_elems], mlp_buf=buf_b[:n_elems],
                        device=device,
                    )
                    for _ in range(args.n_layers)
                ])
                for cl, rl in zip(cg_layers_ns, db_layers):
                    cl.load_state_dict(rl.state_dict())
                for _ in range(3):
                    cg_ar_ns.reset()
                    h, r = x, None
                    for layer in cg_layers_ns:
                        h, r = layer(h, positions, r)
                torch.cuda.synchronize()
                g_ns = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    cg_ar_ns.reset()
                    g_ns.capture_begin(pool=torch.cuda.graph_pool_handle())
                    h, r = x, None
                    for layer in cg_layers_ns:
                        h, r = layer(h, positions, r)
                    g_ns.capture_end()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                for _ in range(3):
                    g_ns.replay()
                torch.cuda.synchronize()
                profile_variant("nosync_cudagraph", None, graph=g_ns)

            if backend in ("all", "synced"):
                cg_ar_sy = DoubleBufferAllreduce(calls_per_iter, nosync=False)
                cg_layers_sy = nn.ModuleList([
                    DoubleBufferDecoderLayer(
                        hidden_dim=args.hidden_dim, n_heads=args.n_heads,
                        n_kv_heads=args.n_kv_heads,
                        intermediate_size=args.intermediate_size,
                        world_size=world_size, allreduce_fn=cg_ar_sy,
                        max_tokens=max_tokens,
                        attn_buf=buf_a[:n_elems], mlp_buf=buf_b[:n_elems],
                        device=device,
                    )
                    for _ in range(args.n_layers)
                ])
                for cl, rl in zip(cg_layers_sy, db_layers):
                    cl.load_state_dict(rl.state_dict())
                for _ in range(3):
                    cg_ar_sy.reset()
                    h, r = x, None
                    for layer in cg_layers_sy:
                        h, r = layer(h, positions, r)
                torch.cuda.synchronize()
                g_sy = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    cg_ar_sy.reset()
                    g_sy.capture_begin(pool=torch.cuda.graph_pool_handle())
                    h, r = x, None
                    for layer in cg_layers_sy:
                        h, r = layer(h, positions, r)
                    g_sy.capture_end()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                for _ in range(3):
                    g_sy.replay()
                torch.cuda.synchronize()
                profile_variant("synced_cudagraph", None, graph=g_sy)
        else:
            if backend in ("all", "nosync"):
                for _ in range(3):
                    forward_db(x)
                torch.cuda.synchronize()
                profile_variant("nosync_eager", lambda: forward_db(x))
            if backend in ("all", "synced"):
                for _ in range(3):
                    forward_sync(x)
                torch.cuda.synchronize()
                profile_variant("synced_eager", lambda: forward_sync(x))

    # --- Numerics comparison (only when running all backends) ---
    if backend == "all":
        if rank == 0:
            print("\n  Transformer numerics check:")
        db_ar.reset()
        out_nosync = forward_db(x)
        sync_ar.reset()
        out_sync = forward_sync(x)
        out_nccl = forward_nccl(x)
        torch.cuda.synchronize()

        if rank == 0:
            diff_ns = (out_nosync - out_sync).abs().max().item()
            diff_nccl = (out_nosync - out_nccl).abs().max().item()
            print(f"    nosync vs synced max diff: {diff_ns}")
            print(f"    nosync vs nccl max diff:   {diff_nccl}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Double-buffered nosync allreduce")
    p.add_argument("--hidden-dim", type=int, default=4096)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--n-kv-heads", type=int, default=8)
    p.add_argument("--intermediate-size", type=int, default=11008)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--chain-length", type=int, default=4,
                   help="Number of allreduces in standalone chain benchmark")
    p.add_argument("--sizes", default="8,32,128,512",
                   help="Comma-separated tensor sizes in KB for chain benchmark")
    p.add_argument("--skip-chain", action="store_true")
    p.add_argument("--skip-transformer", action="store_true")
    p.add_argument("--skip-numerics", action="store_true")
    p.add_argument("--cudagraph", action="store_true",
                   help="Run transformer benchmark with full CUDA graph capture")
    p.add_argument("--backend", default="all",
                   choices=["all", "nosync", "synced", "nccl"],
                   help="Which backend to run (for profiling a single variant)")
    p.add_argument("--profile", action="store_true",
                   help="Export PyTorch profiler Chrome traces (CUDA activities)")
    return p.parse_args()


def main():
    global rank, world_size, device

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    args = parse_args()

    if rank == 0:
        print(f"Double-Buffered Nosync Allreduce Benchmark")
        print(f"world_size={world_size}, device={device}")

    lock_gpu_clocks()
    dist.barrier()

    sizes_kb = [int(s) for s in args.sizes.split(",")]
    max_elems = max(sizes_kb) * 1024 // 2
    transformer_elems = args.batch_size * args.seq_len * args.hidden_dim
    max_elems = max(max_elems, transformer_elems)

    setup(max_elems)

    if not args.skip_numerics:
        verify_numerics(min(4096, max_elems))

    if not args.skip_chain:
        for size_kb in sizes_kb:
            numel = size_kb * 1024 // 2
            bench_chain(numel, args.chain_length, args.warmup, args.iters)

    if not args.skip_transformer:
        bench_transformer(args)

    ext.dispose()
    unlock_gpu_clocks()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
