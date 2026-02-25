# Allreduce Sync Overhead Benchmarks

Measures allreduce latency and synchronization overhead across different
implementations used in tensor-parallel LLM inference. The key question:
**how much latency does the end barrier of allreduce cost, and can we
safely elide it with double buffering?**

## Benchmarks

### `bench_allreduce.py` — Raw allreduce comparison

Compares latency across five backends at various tensor sizes, plus a
Qwen3 transformer layer benchmark:

| Backend | Description |
|---------|-------------|
| `nccl` | `torch.distributed.all_reduce` |
| `custom_1stage` | vLLM 1-stage (read-reduce-write, all ranks read all data) |
| `custom_2stage` | vLLM 2-stage (reduce-scatter then allgather) |
| `symm_one_shot` | `torch.ops.symm_mem.one_shot_all_reduce` |
| `symm_two_shot` | `torch.ops.symm_mem.two_shot_all_reduce_` |

```bash
torchrun --nproc_per_node=8 bench_allreduce.py --backend all
torchrun --nproc_per_node=8 bench_allreduce.py --backend nccl,custom_1stage --skip-transformer
```

### `bench_double_buffer.py` — Nosync + double buffering

The main benchmark. Tests whether eliding the end barrier ("nosync") is
safe and beneficial when consecutive allreduces alternate between two
pre-registered symmetric memory buffers.

**Safety argument (double buffering):**
- AR_0 uses buf_A (nosync: no end barrier)
- AR_1 uses buf_B (nosync): its start barrier ensures all ranks finished AR_0
- AR_2 uses buf_A again: its start barrier ensures all ranks finished AR_1,
  which means AR_0's reads completed long ago → buf_A is safe to reuse

The start barrier **cannot** be elided — it ensures all ranks have written
their data before any rank reads via P2P. The end barrier **can** be elided
with double buffering because the intervening start barrier on the other
buffer provides the needed synchronization.

**Tests included:**
1. **Numerics verification** — nosync results match synced reference
2. **Standalone chain** — N back-to-back allreduces at various sizes
3. **Transformer layer** — Qwen3 decoder with double-buffered AR (eager + CUDA graph)

```bash
# Full benchmark (chain + transformer + CUDA graph)
torchrun --nproc_per_node=8 bench_double_buffer.py --cudagraph

# Chain only (fast)
torchrun --nproc_per_node=8 bench_double_buffer.py --skip-transformer

# Transformer CUDA graph only
torchrun --nproc_per_node=8 bench_double_buffer.py --cudagraph --skip-chain --skip-numerics

# Profile a single variant for nsys/Perfetto
torchrun --nproc_per_node=8 bench_double_buffer.py --cudagraph --backend nosync --profile
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--cudagraph` | off | Capture full forward pass as CUDA graph |
| `--backend` | `all` | `all`, `nosync`, `synced`, or `nccl` |
| `--profile` | off | Export PyTorch profiler Chrome traces (CUDA activities) |
| `--hidden-dim` | 4096 | Hidden dimension |
| `--n-layers` | 2 | Decoder layers |
| `--batch-size` | 1 | Batch size (1 = single-token decode) |
| `--seq-len` | 1 | Sequence length |
| `--sizes` | 8,32,128,512 | Tensor sizes in KB for chain benchmark |
| `--chain-length` | 4 | Number of allreduces per chain |
| `--skip-chain` | — | Skip standalone chain benchmark |
| `--skip-transformer` | — | Skip transformer benchmark |
| `--skip-numerics` | — | Skip numerics verification |

GPU clocks and power are automatically locked to max for the duration of
the benchmark and restored on exit.

### `bench_seq_parallel.py` — Fused GEMM+comm (sequence parallelism)

Compares fused vs unfused sequence parallelism where allreduce is decomposed
into reduce_scatter + all_gather and fused with adjacent GEMMs.

```bash
torchrun --nproc_per_node=8 bench_seq_parallel.py --backend all
```

### `straggler_repro.py` — Straggler effect reproduction

Demonstrates that when ranks run identical compute, one rank consistently
becomes a straggler, and the allreduce end barrier forces all other ranks
to wait.

```bash
torchrun --nproc_per_node=8 straggler_repro.py
```

## Custom allreduce kernels

The custom allreduce is adapted from vLLM. Source files:

- **`custom_all_reduce.cuh`** — CUDA kernel implementations
  - `cross_device_reduce_1stage` — all ranks read all data, reduce locally (with end barrier)
  - `cross_device_reduce_1stage_nosync` — same but no end barrier
  - `cross_device_reduce_2stage` — reduce-scatter then allgather (for larger payloads)
  - `barrier_at_start` / `barrier_at_end` — P2P flag exchange via NVLink

- **`custom_all_reduce_wrapper.cu`** — PyTorch C++ extension wrapper
  - Exposes `allreduce_symm_nosync`, `allreduce_symm_nosync_b`, `allreduce_symm_sync_a`,
    `allreduce_symm_sync_b` for buf_a and buf_b variants
  - JIT compiled via `torch.utils.cpp_extension.load`

- **`qwen3_block.py`** — Standalone Qwen3 transformer block (GQA, QK-norm, RoPE, fused QKV)

## Key findings

**Measured on 8× H100 (NVLink, SM=1980MHz):**

### Chain benchmark (4 allreduces, no compute between them)

| Size | Nosync p50 | Synced p50 | Savings/AR |
|------|-----------|-----------|------------|
| 8 KB | 24.4µs | 27.7µs | ~0.8µs |
| 128 KB | 35.7µs | 39.6µs | ~1.0µs |
| 512 KB | 73.0µs | 77.9µs | ~1.2µs |

### Transformer CUDA graph (2 layers, 4 ARs, batch=1 seq=1)

| Variant | p50 |
|---------|-----|
| Nosync | 353.1µs |
| Synced | 357.0µs |
| NCCL | ~1941µs (eager only) |

**Per-barrier savings: ~1µs** — dominated by NVLink P2P flag exchange latency.

### Why savings appear small in benchmarks

The end barrier cost has two components:
1. **Fixed overhead** (~0.5-1µs): P2P flag write + `__syncthreads()` — always paid
2. **Spin-wait** (variable): time waiting for the slowest rank

In synthetic benchmarks, ranks stay synchronized (especially with CUDA graphs),
so the spin-wait is near-zero. The real win comes from **straggler jitter** in
production: when one rank is delayed by OS scheduling, memory bandwidth
contention, or NVLink fabric noise, the end barrier forces all ranks to idle.
Without it, fast ranks proceed to compute and the delay is absorbed at the
next start barrier — overlapping useful work with the straggler's catch-up.

For a model like Llama 405B (80 layers, 160 ARs/token) at batch=1 decode:
- Fixed savings: 160 × 1µs = **160µs/token**
- With 3µs avg straggler jitter: 160 × 3µs = **480µs/token** (~5% of decode)

This matters most for **latency-sensitive batch=1 workloads** like agentic
inference, where each token's latency compounds across thousands of decode
steps per task.

## Architecture context for future sessions

The barrier mechanism in `custom_all_reduce.cuh`:
- Each GPU has a `Signal` struct with `start[MAX_BLOCKS][MAX_RANKS]` and
  `end[MAX_BLOCKS][MAX_RANKS]` flag arrays, plus a `_flag[MAX_BLOCKS]`
  monotonic counter
- `barrier_at_start`: thread `t` writes `_flag+1` to peer `t`'s start array,
  then spins until peer `t` writes back. Increments `_flag` by 1.
- `barrier_at_end`: same pattern on end arrays. Increments `_flag` by 1.
- Synced kernel: start barrier (+1) + end barrier (+1) = `_flag` increments by 2
- Nosync kernel: start barrier only (+1) = `_flag` increments by 1
- Both buf_a and buf_b share the same `_flag` counter, so nosync and synced
  calls can be interleaved as long as the counter sequence is consistent
  across all ranks (which CUDA graph replay guarantees)
