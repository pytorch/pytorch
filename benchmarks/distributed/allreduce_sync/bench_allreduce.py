"""
Allreduce sync overhead benchmark.

Compares allreduce latency across three implementations:
  1. vLLM custom allreduce (1-stage and 2-stage kernels)
  2. PyTorch symmetric memory (one_shot_all_reduce and two_shot_all_reduce_)
  3. NCCL (baseline via torch.distributed.all_reduce)

Each backend is named as: <kernel>[.nosync]
  - kernel: nccl, custom, symm_one_shot, symm_two_shot
  - .nosync suffix: elide end barrier (only for custom/symm with direct buffers)

In the raw microbenchmark, backends use pre-registered buffers (no copy).
In the transformer benchmark, backends that have registered buffers get
zero-copy (matmul writes directly into the buffer); NCCL uses in-place
all_reduce on a regular tensor.

Usage:
  torchrun --nproc_per_node=N bench_allreduce.py [options]

Examples:
  torchrun --nproc_per_node=2 bench_allreduce.py --backend all
  torchrun --nproc_per_node=8 bench_allreduce.py --backend nccl,custom,symm_one_shot,symm_one_shot.nosync --skip-raw
"""

import argparse
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn as nn
from torch.utils.cpp_extension import load

from qwen3_block import Qwen3DecoderLayer


# ---------------------------------------------------------------------------
# Globals set up during init
# ---------------------------------------------------------------------------
rank = 0
world_size = 1
device = None
custom_ar_ext = None  # JIT-compiled extension module
custom_ar_initialized = False
custom_ar_buf = None  # Tensor wrapping IPC data region

symm_mem_buf = None
symm_mem_group_name = None
symm_nosync_initialized = False

BACKENDS = [
    "nccl",
    "custom",
    "custom.nosync",
    "symm_one_shot",
    "symm_two_shot",
    "symm_one_shot.nosync",
    "symm_two_shot.nosync",
]


# ---------------------------------------------------------------------------
# Custom allreduce setup
# ---------------------------------------------------------------------------
def setup_custom_allreduce(max_elems: int):
    global custom_ar_ext, custom_ar_initialized, custom_ar_buf

    src_dir = Path(__file__).parent
    custom_ar_ext = load(
        name="custom_all_reduce_ext",
        sources=[str(src_dir / "custom_all_reduce_wrapper.cu")],
        extra_include_paths=[str(src_dir)],
        extra_cuda_cflags=["-O2", "-std=c++17"],
        verbose=(rank == 0),
    )

    elem_bytes = 2  # bf16
    signal_size = 1024 * 16
    scratch_size = max_elems * elem_bytes
    data_size = max_elems * elem_bytes
    total_size = signal_size + scratch_size + data_size

    ptr_tensor, handle_tensor = custom_ar_ext.allocate_and_get_handle(total_size)
    local_ptr = ptr_tensor.item()

    handle_list = [None] * world_size
    dist.all_gather_object(handle_list, handle_tensor.cpu())

    signal_ptrs = torch.zeros(world_size, dtype=torch.int64)
    data_ptrs_list = torch.zeros(world_size, dtype=torch.int64)

    for i in range(world_size):
        if i == rank:
            signal_ptrs[i] = local_ptr
            data_ptrs_list[i] = local_ptr + signal_size + scratch_size
        else:
            peer_ptr = custom_ar_ext.open_ipc_handle(handle_list[i]).item()
            signal_ptrs[i] = peer_ptr
            data_ptrs_list[i] = peer_ptr + signal_size + scratch_size

    custom_ar_ext.init_custom_ar(signal_ptrs, rank, world_size)
    custom_ar_ext.register_buffer(data_ptrs_list, rank, max_elems)
    custom_ar_initialized = True

    custom_ar_buf = custom_ar_ext.get_registered_buffer_tensor(max_elems)

    dist.barrier()
    if rank == 0:
        print("[custom allreduce] initialized")


# ---------------------------------------------------------------------------
# Symmetric memory setup
# ---------------------------------------------------------------------------
def setup_symm_mem(max_elems: int):
    global symm_mem_buf, symm_mem_group_name

    pool = symm_mem.get_mem_pool(device)
    with torch.cuda.use_mem_pool(pool):
        symm_mem_buf = torch.empty(max_elems, dtype=torch.bfloat16, device=device)

    symm_mem_group_name = dist.group.WORLD.group_name
    dist.barrier()
    if rank == 0:
        print("[symmetric memory] initialized")


def setup_symm_nosync(max_elems: int):
    global symm_nosync_initialized

    handle = symm_mem.rendezvous(symm_mem_buf, group=dist.group.WORLD)
    peer_ptrs = torch.zeros(world_size, dtype=torch.int64)
    for i in range(world_size):
        peer_buf = handle.get_buffer(i, (max_elems,), torch.bfloat16)
        peer_ptrs[i] = peer_buf.data_ptr()
    custom_ar_ext.register_symm_buffer(peer_ptrs, rank, max_elems)

    symm_nosync_initialized = True
    dist.barrier()
    if rank == 0:
        print("[symm_mem nosync] registered with custom AR barriers")


# ---------------------------------------------------------------------------
# Allreduce functions
#
# All functions have signature (inp: Tensor, out: Tensor) -> None.
# In the transformer benchmark, inp and out may be views of the same
# pre-registered buffer (zero-copy path).
# ---------------------------------------------------------------------------

def nccl_fn(inp: torch.Tensor, out: torch.Tensor):
    if inp.data_ptr() != out.data_ptr():
        out.view(-1).copy_(inp.contiguous().view(-1))
    dist.all_reduce(out)


def custom_fn(inp: torch.Tensor, out: torch.Tensor):
    custom_ar_ext.allreduce_inplace(inp.numel(), "bf16")


def custom_nosync_fn(inp: torch.Tensor, out: torch.Tensor):
    custom_ar_ext.allreduce_inplace_nosync(inp.numel(), "bf16")


def symm_one_shot_fn(inp: torch.Tensor, out: torch.Tensor):
    torch.ops.symm_mem.one_shot_all_reduce(
        inp.reshape(-1), "sum", symm_mem_group_name
    )


def symm_two_shot_fn(inp: torch.Tensor, out: torch.Tensor):
    torch.ops.symm_mem.two_shot_all_reduce_(
        inp.reshape(-1), "sum", symm_mem_group_name
    )


def symm_one_shot_nosync_fn(inp: torch.Tensor, out: torch.Tensor):
    custom_ar_ext.allreduce_symm_nosync(inp.numel(), "bf16")


def symm_two_shot_nosync_fn(inp: torch.Tensor, out: torch.Tensor):
    custom_ar_ext.allreduce_symm_2stage(inp.numel(), "bf16")


BACKEND_FNS = {
    "nccl": nccl_fn,
    "custom": custom_fn,
    "custom.nosync": custom_nosync_fn,
    "symm_one_shot": symm_one_shot_fn,
    "symm_two_shot": symm_two_shot_fn,
    "symm_one_shot.nosync": symm_one_shot_nosync_fn,
    "symm_two_shot.nosync": symm_two_shot_nosync_fn,
}


def get_allreduce_buf(backend: str, n_elems: int):
    """Return the pre-registered buffer for a backend, or None for NCCL."""
    if backend.startswith("custom"):
        return custom_ar_buf[:n_elems] if custom_ar_buf is not None else None
    if backend.startswith("symm_"):
        return symm_mem_buf[:n_elems] if symm_mem_buf is not None else None
    # NCCL: no special buffer needed, but we still want zero-copy
    # (in-place all_reduce works on any CUDA tensor)
    return None


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------
class AllreduceTimer:
    """Wrapper that records CUDA events around each allreduce call.

    When pre_sync=True, inserts a cuda.synchronize() + dist.barrier() before
    each allreduce so all ranks arrive together — isolating kernel time from
    cross-rank wait time.
    """

    def __init__(self, fn, pre_sync=False):
        self.fn = fn
        self.pre_sync = pre_sync
        self.events = []  # list of (start_event, end_event)
        self.recording = False

    def __call__(self, inp, out):
        if self.recording:
            if self.pre_sync:
                torch.cuda.synchronize()
                dist.barrier()
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

    def allreduce_times_per_iter(self, calls_per_iter):
        """Return total allreduce µs per iteration."""
        times = []
        for i in range(0, len(self.events), calls_per_iter):
            batch = self.events[i : i + calls_per_iter]
            total = sum(s.elapsed_time(e) * 1000.0 for s, e in batch)
            times.append(total)
        return times

    def per_call_times(self, calls_per_iter):
        """Return list of per-call-slot median times (µs)."""
        n_iters = len(self.events) // calls_per_iter
        slot_times = [[] for _ in range(calls_per_iter)]
        for i in range(n_iters):
            for j in range(calls_per_iter):
                idx = i * calls_per_iter + j
                s, e = self.events[idx]
                slot_times[j].append(s.elapsed_time(e) * 1000.0)
        return [sorted(t)[len(t) // 2] for t in slot_times]

    def inter_allreduce_gaps(self, calls_per_iter):
        """Return median gap (µs) between end of call j and start of call j+1."""
        n_iters = len(self.events) // calls_per_iter
        gap_times = [[] for _ in range(calls_per_iter - 1)]
        for i in range(n_iters):
            for j in range(calls_per_iter - 1):
                _, prev_end = self.events[i * calls_per_iter + j]
                next_start, _ = self.events[i * calls_per_iter + j + 1]
                gap_times[j].append(prev_end.elapsed_time(next_start) * 1000.0)
        return [sorted(t)[len(t) // 2] for t in gap_times]


class PiecewiseGraphCapture:
    """Piecewise CUDA graph capture around allreduce calls.

    During capture, each allreduce call acts as a graph-break point:
    the preceding compute segment is captured as a CUDA graph, the
    allreduce runs eagerly, then capture resumes for the next segment.

    During replay, graph segments are replayed with eager allreduce
    calls interleaved between them.
    """

    def __init__(self, fn):
        self.fn = fn
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = []
        self.ar_args = []
        self._capturing = False
        self._pending = None

    def start_capture(self):
        self._capturing = True
        self.graphs = []
        self.ar_args = []
        g = torch.cuda.CUDAGraph()
        g.capture_begin(pool=self.pool)
        self._pending = g

    def end_capture(self):
        if self._pending is not None:
            self._pending.capture_end()
            self.graphs.append(self._pending)
            self._pending = None
        self._capturing = False

    def __call__(self, inp, out):
        if self._capturing:
            # End current compute segment capture
            self._pending.capture_end()
            self.graphs.append(self._pending)
            self.ar_args.append((inp, out))
            # Run allreduce eagerly (not captured)
            self.fn(inp, out)
            # Start next compute segment capture
            g = torch.cuda.CUDAGraph()
            g.capture_begin(pool=self.pool)
            self._pending = g
        else:
            self.fn(inp, out)

    def replay(self):
        """Replay: graph → allreduce → graph → allreduce → ... → graph."""
        for i, g in enumerate(self.graphs):
            g.replay()
            if i < len(self.ar_args):
                inp, out = self.ar_args[i]
                self.fn(inp, out)


def bench_fn(fn, inp, out, warmup, iters):
    for _ in range(warmup):
        fn(inp, out)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    torch.cuda.synchronize()
    for i in range(iters):
        start_events[i].record()
        fn(inp, out)
        end_events[i].record()
    torch.cuda.synchronize()

    return [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)]


def report_times(label, times_us, ar_times_us=None):
    if rank != 0:
        return
    times = sorted(times_us)
    n = len(times)
    mean = statistics.mean(times)
    p50 = times[n // 2]
    p99 = times[int(n * 0.99)]
    mn = times[0]
    mx = times[-1]
    line = (f"  {label:25s}  mean={mean:8.2f}µs  p50={p50:8.2f}µs  "
            f"p99={p99:8.2f}µs  min={mn:8.2f}µs  max={mx:8.2f}µs")
    if ar_times_us is not None:
        ar = sorted(ar_times_us)
        ar_mean = statistics.mean(ar)
        ar_p50 = ar[len(ar) // 2]
        line += f"  |  allreduce: mean={ar_mean:8.2f}µs  p50={ar_p50:8.2f}µs"
    print(line)


# ---------------------------------------------------------------------------
# Raw allreduce microbenchmark
# ---------------------------------------------------------------------------
def bench_raw_allreduce(backends, sizes_kb, warmup, iters):
    if rank == 0:
        print("\n" + "=" * 80)
        print("RAW ALLREDUCE MICROBENCHMARK")
        print("=" * 80)

    for size_kb in sizes_kb:
        numel = size_kb * 1024 // 2  # bf16 = 2 bytes

        if rank == 0:
            print(f"\nSize: {size_kb} KB ({numel} bf16 elements)")

        for backend in backends:
            fn = BACKEND_FNS[backend]

            # For raw benchmark, use the registered buffer directly
            buf = get_allreduce_buf(backend, numel)
            if buf is not None:
                inp = buf
                out = buf
            else:
                inp = torch.randn(numel, dtype=torch.bfloat16, device=device)
                out = torch.empty_like(inp)

            try:
                times = bench_fn(fn, inp, out, warmup, iters)
                report_times(backend, times)
            except Exception as e:
                if rank == 0:
                    print(f"  {backend:25s}  FAILED: {e}")

    os.environ.pop("VLLM_CUSTOM_ALLREDUCE_ALGO", None)


# ---------------------------------------------------------------------------
# Transformer layer benchmark
# ---------------------------------------------------------------------------
def bench_transformer_layer(backends, args):
    if rank == 0:
        print("\n" + "=" * 80)
        print("TRANSFORMER LAYER BENCHMARK (Qwen3)")
        print(f"hidden_dim={args.hidden_dim}, n_heads={args.n_heads}, "
              f"n_kv_heads={args.n_kv_heads}, "
              f"intermediate_size={args.intermediate_size}, "
              f"n_layers={args.n_layers}, batch={args.batch_size}, "
              f"seq_len={args.seq_len}"
              + (", cudagraph=True" if args.cudagraph else ""))
        print("=" * 80)

    x = torch.randn(
        args.batch_size, args.seq_len, args.hidden_dim,
        dtype=torch.bfloat16, device=device,
    )
    positions = torch.arange(
        args.seq_len, dtype=torch.long, device=device
    ).unsqueeze(0).expand(args.batch_size, -1)

    for backend in backends:
        fn = BACKEND_FNS[backend]

        try:
            max_tokens = args.batch_size * args.seq_len
            n_elems = max_tokens * args.hidden_dim
            ar_buf = get_allreduce_buf(backend, n_elems)
            calls_per_iter = 2 * args.n_layers

            call_labels = []
            for l in range(args.n_layers):
                call_labels.append(f"L{l}.attn")
                call_labels.append(f"L{l}.mlp")

            timer = AllreduceTimer(fn)

            def make_layers(ar_fn):
                return nn.ModuleList([
                    Qwen3DecoderLayer(
                        hidden_dim=args.hidden_dim,
                        n_heads=args.n_heads,
                        n_kv_heads=args.n_kv_heads,
                        intermediate_size=args.intermediate_size,
                        world_size=world_size,
                        allreduce_fn=ar_fn,
                        max_tokens=max_tokens,
                        allreduce_buf=ar_buf,
                        device=device,
                    )
                    for _ in range(args.n_layers)
                ])

            layers = make_layers(timer)

            def forward_pass(x):
                h, residual = x, None
                for layer in layers:
                    h, residual = layer(h, positions, residual)
                return h

            for _ in range(args.warmup):
                forward_pass(x)
            torch.cuda.synchronize()

            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]

            timer.start_recording()
            torch.cuda.synchronize()
            for i in range(args.iters):
                start_events[i].record()
                forward_pass(x)
                end_events[i].record()
            torch.cuda.synchronize()
            timer.stop_recording()

            e2e_times = [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)]

            # True allreduce kernel time = min across ranks per call slot
            # (the straggler arrives last, sees no barrier wait, measures only kernel)
            per_call = timer.per_call_times(calls_per_iter)
            per_call_tensor = torch.tensor(per_call, dtype=torch.float64, device=device)
            all_per_call = [torch.empty_like(per_call_tensor) for _ in range(world_size)]
            dist.all_gather(all_per_call, per_call_tensor)
            if rank == 0:
                stacked = torch.stack(all_per_call)  # (world_size, calls_per_iter)
                min_per_call = stacked.min(dim=0).values.tolist()
                ar_kernel_total = sum(min_per_call)
                parts = "  ".join(f"{call_labels[j]}={min_per_call[j]:.1f}µs"
                                  for j in range(len(min_per_call)))
                report_times(backend, e2e_times)
                print(f"    allreduce kernel p50: {parts}  total={ar_kernel_total:.1f}µs")

            # --- Piecewise CUDA graph capture ---
            if args.cudagraph:
              try:
                cg_timer = AllreduceTimer(fn)
                pw = PiecewiseGraphCapture(cg_timer)
                cg_layers = make_layers(pw)

                def cg_forward(x):
                    h, residual = x, None
                    for layer in cg_layers:
                        h, residual = layer(h, positions, residual)
                    return h

                # Warmup (eager — pw is not capturing)
                for _ in range(args.warmup):
                    cg_forward(x)
                torch.cuda.synchronize()

                # Capture on a side stream (CUDA graphs require
                # non-default stream for capture_begin).
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream):
                    pw.start_capture()
                    cg_forward(x)
                    pw.end_capture()
                torch.cuda.current_stream().wait_stream(capture_stream)
                torch.cuda.synchronize()

                # Warmup replays
                for _ in range(args.warmup):
                    pw.replay()
                torch.cuda.synchronize()

                # Timed replays
                cg_starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
                cg_ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]

                cg_timer.start_recording()
                torch.cuda.synchronize()
                for i in range(args.iters):
                    cg_starts[i].record()
                    pw.replay()
                    cg_ends[i].record()
                torch.cuda.synchronize()
                cg_timer.stop_recording()

                cg_e2e = [cg_starts[i].elapsed_time(cg_ends[i]) * 1000.0
                          for i in range(args.iters)]

                # Per-call kernel times (min across ranks)
                cg_per_call = cg_timer.per_call_times(calls_per_iter)
                cg_tensor = torch.tensor(cg_per_call, dtype=torch.float64, device=device)
                cg_all = [torch.empty_like(cg_tensor) for _ in range(world_size)]
                dist.all_gather(cg_all, cg_tensor)

                if rank == 0:
                    cg_stacked = torch.stack(cg_all)
                    cg_min = cg_stacked.min(dim=0).values.tolist()
                    cg_ar_total = sum(cg_min)
                    cg_parts = "  ".join(f"{call_labels[j]}={cg_min[j]:.1f}µs"
                                         for j in range(len(cg_min)))
                    report_times(f"{backend} (cudagraph)", cg_e2e)
                    print(f"    allreduce kernel p50: {cg_parts}  total={cg_ar_total:.1f}µs")
              except Exception as cg_err:
                if rank == 0:
                    import traceback
                    print(f"  {backend + ' (cudagraph)':25s}  FAILED: {cg_err}")
                    traceback.print_exc()

        except Exception as e:
            if rank == 0:
                print(f"  {backend:25s}  FAILED: {e}")

    os.environ.pop("VLLM_CUSTOM_ALLREDUCE_ALGO", None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Allreduce sync benchmark")
    parser.add_argument("--backend", default="all",
                        help="Comma-separated backends or 'all'")
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=8,
                        help="Number of KV heads for GQA (Qwen3-8B default: 8)")
    parser.add_argument("--intermediate-size", type=int, default=11008,
                        help="MLP intermediate size (Qwen3-8B default: 11008)")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1,
                        help="Sequence length (1 for decode, >1 for prefill)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--sizes", default="4,8,16,32,64,128,256,512,1024",
                        help="Comma-separated tensor sizes in KB for raw benchmark")
    parser.add_argument("--skip-transformer", action="store_true",
                        help="Skip transformer layer benchmark")
    parser.add_argument("--skip-raw", action="store_true",
                        help="Skip raw allreduce microbenchmark")
    parser.add_argument("--cudagraph", action="store_true",
                        help="Also run transformer benchmark with CUDA graph capture+replay")
    return parser.parse_args()


def main():
    global rank, world_size, device

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    args = parse_args()

    if args.backend == "all":
        backends = list(BACKENDS)
    else:
        backends = [b.strip() for b in args.backend.split(",")]

    sizes_kb = [int(s) for s in args.sizes.split(",")]
    max_elems = max(sizes_kb) * 1024 // 2  # bf16

    transformer_elems = args.batch_size * args.seq_len * args.hidden_dim
    max_elems = max(max_elems, transformer_elems)

    if rank == 0:
        print(f"Allreduce Sync Benchmark")
        print(f"world_size={world_size}, max_elems={max_elems}, device={device}")
        print(f"backends={backends}")

    has_custom = any("custom" in b for b in backends)
    has_symm = any(b.startswith("symm_") for b in backends)
    has_nosync = any("nosync" in b for b in backends)
    has_symm_nosync = any(b.startswith("symm_") and "nosync" in b for b in backends)

    if has_custom or has_nosync:
        setup_custom_allreduce(max_elems)
    if has_symm or has_symm_nosync:
        setup_symm_mem(max_elems)
    if has_symm_nosync:
        setup_symm_nosync(max_elems)

    dist.barrier()

    if not args.skip_raw:
        bench_raw_allreduce(backends, sizes_kb, args.warmup, args.iters)
    if not args.skip_transformer:
        bench_transformer_layer(backends, args)

    if (has_custom or has_nosync) and custom_ar_initialized:
        custom_ar_ext.dispose()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
