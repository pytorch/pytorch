# 0211 – symm_mem Out-Variant Lowering (ExternKernelOut)

## What this change does

In Inductor's `comm_lowering.py`, functional symmetric-memory ops
(`one_shot_all_reduce`, `one_shot_all_reduce_copy`,
`multimem_one_shot_all_reduce`) are currently lowered via
**`FallbackKernel.create()`**.

`FallbackKernel` inherits `should_allocate() = False` from
`ExternKernelAlloc`, so its output buffer is **opaque** to Inductor's
memory planner and can never participate in `AllocateLine.plan()` buffer
reuse.

This change switches the three functional ops to use
**`ir.ExternKernelOut`** (via the corresponding `_out` op).
`ExternKernelOut` has `should_allocate() = True` — the output buffer is
pre-allocated by codegen through `AllocateLine` and can be reused by
later ops with matching `(size, dtype, device)`.

## How to reproduce

```bash
# Clear Inductor cache
rm -rf /tmp/torchinductor_$USER

# Run the benchmark (needs ≥ 2 GPUs with NVLink)
torchrun --nproc_per_node=2 docs/0211_symm_mem_out_variant/benchmark.py [label]
```

To compare baseline vs. modified:
1. Revert `comm_lowering.py` to trunk → run with label `baseline`
2. Apply the change → run with label `modified`

## Results

**Config**: hidden=4096, seq=128, 8 layers of `matmul → one_shot_all_reduce`,
2× NVIDIA H100, bf16, `torch.compile(fullgraph=True)`

| Metric | Baseline (`FallbackKernel`) | Modified (`ExternKernelOut`) |
|---|---|---|
| Peak memory | 290.00 MB | 290.00 MB |
| **Time per iter** | **357.6 μs** | **334.7 μs  (−6.4 %)** |
| Total buffer names | 24 (`buf0`…`buf23`) | 16 (`buf0`…`buf15`) |
| **Buffer reuses** | **7** | **14  (2×)** |
| `empty_strided_cuda` allocs | 0 | 1 |
| P2P allocs | 1 | 1 |
| Out-variant calls (`out=`) | 8 (matmul only) | 16 (matmul + allreduce) |

Peak memory is identical because the 8 weight matrices dominate
(4096 × 4096 × bf16 = 32 MB each ≈ 256 MB). The scratch buffers are
only 1 MB each, well within CUDA allocator granularity.

## Codegen comparison

### Baseline – `FallbackKernel` path

```python
buf0 = empty_strided_p2p(...)                                       # 1 P2P buffer
extern_kernels.mm(arg1_1, arg0_1, out=buf0)

buf1 = torch.ops.symm_mem.one_shot_all_reduce.default(buf0, ...)   # opaque internal alloc
buf2 = buf1
del buf1

buf3 = buf0; del buf0  # reuse                                     # only P2P buffer reuses
extern_kernels.mm(buf2, arg2_1, out=buf3)
del buf2                                                            # ← output FREED, never reused
```

Every `one_shot_all_reduce` call allocates internally and that output
is immediately freed by the next layer — **no output reuse**.

### Modified – `ExternKernelOut` path

```python
buf0 = empty_strided_p2p(...)                                                     # 1 P2P buffer
extern_kernels.mm(arg1_1, arg0_1, out=buf0)

buf1 = empty_strided_cuda(...)                                                    # 1 regular buffer
torch.ops.symm_mem.one_shot_all_reduce_out.default(buf0, 'sum', '0', out=buf1)   # out-variant!

buf2 = buf0; del buf0  # reuse                                                   # P2P buffer reused
extern_kernels.mm(buf1, arg2_1, out=buf2)

buf3 = buf1; del buf1  # reuse                                                   # regular buffer ALSO reused!
torch.ops.symm_mem.one_shot_all_reduce_out.default(buf2, 'sum', '0', out=buf3)
```

Two buffers ping-pong across all 8 layers — **0 extra allocations**.

## Files

| File | Description |
|---|---|
| `benchmark.py` | Benchmark script |
| `generated_code_baseline.py` | Inductor-generated code (trunk) |
| `generated_code_modified.py` | Inductor-generated code (this change) |
| `README.md` | This file |
