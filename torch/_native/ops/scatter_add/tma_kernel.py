"""TMA-based scatter_add for the ``index.unsqueeze(-1).expand(-1, N)`` pattern.

Port of the CUDA C++ kernel from https://github.com/pytorch/pytorch/pull/182675
to CuTeDSL. Each CTA (one warp) handles one source row (within an assigned
chunk-index range): a tile-mode TMA bulk load (``cute.copy`` with
``CopyBulkTensorTileG2SOp`` via ``make_tiled_tma_atom``) stages
``src[i, d_start:d_end]`` into smem, then
``cp.reduce.async.bulk.global.shared::cta.bulk_group.add`` deposits the
reduction into ``out[index[i], d_start:d_end]``.

One module serves both routes: the JIT wrapper (instrumented compile
cache + torch host function) and the AOT ``build(spec)`` entry point
compile the same ``_make_kernel`` body, so same-kernel is by
construction. The AOT export tool imports this module as a package
import with the built torch available (two-stage build).

Restrictions (enforced by the host cond in ``cutedsl_impl.py``):
  - sm_90+ (cp.reduce.async.bulk availability)
  - dim == 0, rank >= 2, self/src inner-contiguous
  - index is the expanded-1D pattern (same shape as src, stride 0 on every
    axis except 0)
  - dtype in {fp32, fp16, bf16}
  - ``row_bytes % 16 == 0`` (for both the TMA load operand and the
    cp.reduce.async.bulk gmem operand)
"""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline
from cutlass import BFloat16, Float16, Float32, Int32, Int64

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache

from ._ptx import cvta_smem, make_bulk_reduce_add, trap_if_oob


_MAX_CHUNK_BYTES = 512
_THREADS_PER_CTA = 32  # one warp per CTA
# cp.async.bulk requires 128B-aligned smem destinations.
_SMEM_ALIGN_BYTES = 128


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


_bulk_reduce_add_f32 = make_bulk_reduce_add("f32")
_bulk_reduce_add_f16 = make_bulk_reduce_add("noftz.f16")
_bulk_reduce_add_bf16 = make_bulk_reduce_add("noftz.bf16")


def _reduce_op_for(dtype):
    if dtype is Float32:
        return _bulk_reduce_add_f32
    if dtype is Float16:
        return _bulk_reduce_add_f16
    if dtype is BFloat16:
        return _bulk_reduce_add_bf16
    raise ValueError(f"unsupported dtype: {dtype}")


def _make_kernel(dtype, elem_bytes: int, chunk_elems: int, reduce_op):
    """Build a dtype-specialized kernel closure. dtype/elem_bytes
    /chunk_elems are Python-time constants that the preprocessor folds
    at cute.compile time; ``N`` and ``num_chunks`` are runtime args so
    one compile serves every row length.

    ``chunk_elems`` is baked in at compile time (it is the static TMA
    box dim). The TMA descriptor built in ``_launch`` (via
    ``make_tiled_tma_atom`` on the source tensor) enables
    OOB-clamp-to-zero for reads past column ``N``; the reduce side then
    writes only the actual valid byte count, so partial final chunks --
    and rows shorter than a full box -- are handled natively.

    One CTA = one warp. The single driver thread (tidx == 0) serves as
    both producer (issues the TMA load) and consumer (issues the
    bulk-reduce). Both CooperativeGroups are ``Agent.Thread, size=1``
    so the mbarrier arrive counts match the single-threaded flow.

    The reduce side always uses the runtime ``out_row_stride`` arg so
    outer-strided outputs (e.g. slices) work; the contiguous case just
    passes ``out.stride(0) == N`` in as that value.
    """

    chunk_bytes = chunk_elems * elem_bytes

    # 2-stage pipeline buffer is laid out column-major; stage i starts
    # at offset i * stage_stride_elems * elem_bytes, so the stride must
    # round chunk_bytes up to a multiple of _SMEM_ALIGN_BYTES. Otherwise
    # stage 1 is misaligned and the kernel faults the first time a CTA
    # writes it. Bites both chunk_bytes < 128 (small D) and chunk_bytes
    # not a multiple of 128 (e.g. fp32 N=36 -> 144 B).
    stage_stride_elems = _round_up(chunk_elems, _SMEM_ALIGN_BYTES // elem_bytes)

    @cute.kernel
    def _kernel(
        tma_atom: cute.CopyAtom,
        tma_tensor_src: cute.Tensor,  # TMA-view of mSrc
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        N: Int32,
        num_chunks: Int32,
        chunks_per_cta: Int32,
        out_row_stride: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdim_x, _, _ = cute.arch.grid_dim()

        # num_entries (M_src) comes from mIndex's shape; mIndex is 1D
        # of length M_src after host-side flattening.
        num_entries = mIndex.shape[0]

        # Chunk-index range assigned to this CTA. bidy partitions the
        # chunk axis into disjoint slices of chunks_per_cta; when
        # chunks_per_cta == num_chunks, grid_y == 1 and every CTA sees
        # the whole D.
        chunk_start = bidy * chunks_per_cta
        chunk_end = chunk_start + chunks_per_cta
        if chunk_end > num_chunks:
            chunk_end = num_chunks

        out_rs = out_row_stride

        smem = cutlass.utils.SmemAllocator()
        sBuf = smem.allocate_tensor(
            dtype,
            cute.make_layout((chunk_elems, 2), stride=(1, stage_stride_elems)),
            _SMEM_ALIGN_BYTES,
        )
        mbar_storage = smem.allocate_array(cutlass.Uint64, num_elems=2 * 2)

        pipe = pipeline.PipelineTmaAsync.create(
            num_stages=2,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1),
            tx_count=chunk_bytes,
            barrier_storage=mbar_storage,
            tidx=tidx,
        )

        # Tile the source (M, N) by (1, chunk_elems) to get
        # (1, chunk_elems, M, num_chunks); tma_partition collapses and
        # returns tma_gmem[None, row_idx, chunk_idx] and tma_smem[None,
        # stage].
        tiled_gmem = cute.local_tile(tma_tensor_src, (1, chunk_elems), (None, None))
        tma_smem, tma_gmem = cpasync.tma_partition(
            tma_atom,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sBuf, 0, 1),
            cute.group_modes(tiled_gmem, 0, 2),
        )

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 2
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 2
        )

        # Software-pipelined schedule: at each iteration, issue the TMA
        # load for the current (entry, chunk) pair, then consume the
        # previous pair (bulk-reduce). With num_stages=2 this keeps one
        # TMA load in flight while a bulk-reduce is running. Final
        # iteration's load is drained in an epilogue after the main loop.
        # pair_count is a runtime Int32 so the ``pair_count > 0`` branch
        # stays in the compiled IR (a Python bool would be baked in at
        # trace time).
        pair_count = Int32(0)
        prev_chunk_idx = Int32(0)
        prev_r = Int64(0)

        base = bidx
        while base < num_entries:
            entry_id = base

            chunk_idx = chunk_start
            while chunk_idx < chunk_end:
                if tidx == Int32(0):
                    r = Int64(mIndex[entry_id])
                    # Bounds check: an OOB index would corrupt unrelated
                    # gmem via cp.reduce.async.bulk. Predicated PTX trap
                    # (same mechanism as aten's CUDA_KERNEL_ASSERT in
                    # TmaScatterAddKernel.cu) -- free on the happy path,
                    # unlike --enable-assertions (~10% device time).
                    # Driver thread only -- no need to replicate the
                    # check across all 32 lanes.
                    trap_if_oob(r, Int64(mOut.shape[0]))

                    pipe.producer_acquire(producer_state)
                    cute.copy(
                        tma_atom,
                        tma_gmem[None, entry_id, chunk_idx],
                        tma_smem[None, producer_state.index],
                        tma_bar_ptr=pipe.producer_get_barrier(producer_state),
                    )
                    pipe.producer_commit(producer_state)
                    producer_state.advance()

                    if pair_count > Int32(0):
                        pipe.consumer_wait(consumer_state)
                        cbuf_ptr = sBuf[None, consumer_state.index].iterator

                        # Partial-chunk handling: actual valid element
                        # count is min(chunk_elems, N - off). TMA
                        # OOB-clamped the tail to 0 in smem, we reduce
                        # only the valid bytes.
                        off = prev_chunk_idx * Int32(chunk_elems)
                        cur_elems = N - off
                        if cur_elems > Int32(chunk_elems):
                            cur_elems = Int32(chunk_elems)
                        cur_bytes = cur_elems * Int32(elem_bytes)

                        dst_off = prev_r * out_rs + Int64(off)
                        gmem_dst_u64 = Int64((mOut.iterator + dst_off).toint())
                        reduce_op(gmem_dst_u64, cvta_smem(cbuf_ptr), cur_bytes)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
                        pipe.consumer_release(consumer_state)
                        consumer_state.advance()

                    prev_chunk_idx = chunk_idx
                    prev_r = r
                    pair_count = pair_count + Int32(1)

                chunk_idx = chunk_idx + Int32(1)

            base = base + gdim_x

        # Epilogue: drain the last outstanding TMA load.
        if tidx == Int32(0):
            if pair_count > Int32(0):
                pipe.consumer_wait(consumer_state)
                cbuf_ptr = sBuf[None, consumer_state.index].iterator

                off = prev_chunk_idx * Int32(chunk_elems)
                cur_elems = N - off
                if cur_elems > Int32(chunk_elems):
                    cur_elems = Int32(chunk_elems)
                cur_bytes = cur_elems * Int32(elem_bytes)

                dst_off = prev_r * out_rs + Int64(off)
                gmem_dst_u64 = Int64((mOut.iterator + dst_off).toint())
                reduce_op(gmem_dst_u64, cvta_smem(cbuf_ptr), cur_bytes)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=False)
                pipe.consumer_release(consumer_state)

    @cute.jit
    def _launch(
        mSrc: cute.Tensor,
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        N: Int32,
        num_chunks: Int32,
        chunks_per_cta: Int32,
        grid_x: Int32,
        grid_y: Int32,
        out_row_stride: Int64,
        stream: cuda.CUstream,
    ):
        # Build the tile-mode TMA descriptor over the (M_src, N) source.
        # Both global dims are dynamic; the ``(1, chunk_elems)`` box is
        # static. TMA clamps OOB column reads to 0, so rows shorter than
        # a full box and rows with ``N % chunk_elems != 0`` are handled
        # natively on the load.
        tma_atom, tma_tensor_src = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mSrc,
            cute.make_layout((1, chunk_elems)),
            (1, chunk_elems),
        )
        _kernel(
            tma_atom,
            tma_tensor_src,
            mIndex,
            mOut,
            N,
            num_chunks,
            chunks_per_cta,
            out_row_stride,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    return _launch


_DTYPES = {"float32": Float32, "float16": Float16, "bfloat16": BFloat16}
_DTYPE_SHORT = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}


def chunk_elems_for(dtype_name: str) -> int:
    """Compile-time ``chunk_elems`` (the static TMA box dim): always
    ``_MAX_CHUNK_BYTES // elem_bytes``. Since ``N`` is a runtime arg
    the box can't shrink to fit small rows -- instead a short row loads
    one box and the TMA descriptor OOB-clamps the tail to 0, and the
    reduce writes only ``min(chunk_elems, N)`` valid elements. 512 B is
    a multiple of 16 for every supported dtype, so the box always meets
    the reduce's 16-byte alignment."""
    dtype = _DTYPES[dtype_name]
    return _MAX_CHUNK_BYTES // (dtype.width // 8)


def build(spec: dict) -> dict:
    """One manifest spec point -> compile inputs + marshalling sidecar.

    ``spec`` carries {"dtype": "float32"|"float16"|"bfloat16", ...};
    everything else about the call (shapes, strides, grid) is a runtime
    argument, so the grid is one kernel per dtype. Index bounds checks
    are always-on predicated PTX traps (``trap_if_oob``); no
    ``--enable-assertions`` needed.
    """
    dtype = _DTYPES[spec["dtype"]]
    elem_bytes = dtype.width // 8
    chunk_elems = chunk_elems_for(spec["dtype"])
    launcher = _make_kernel(dtype, elem_bytes, chunk_elems, _reduce_op_for(dtype))

    src_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    # Index is contiguous int64 (the C++ prelude enforces both); fix
    # stride=1 so `mIndex[i]` doesn't emit a runtime stride multiply.
    idx_fake = cute.runtime.make_fake_tensor(Int64, (cute.sym_int(),), stride=(1,))
    out_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    prefix = f"scatter_add_tma_{_DTYPE_SHORT[spec['dtype']]}"
    return {
        "prefix": prefix,
        "fn": launcher,
        "fake_args": [
            src_fake,
            idx_fake,
            out_fake,
            Int32(0),  # N
            Int32(0),  # num_chunks
            Int32(0),  # chunks_per_cta
            Int32(0),  # grid_x
            Int32(0),  # grid_y
            Int64(0),  # out_row_stride
            cute.runtime.make_fake_stream(),
        ],
        "tensor_args": [
            {
                "name": "mSrc",
                "dynamic_sizes": [0, 1],
                "dynamic_strides": [0],
                "read_only": True,
            },
            {"name": "mIndex", "dynamic_sizes": [0], "read_only": True},
            {"name": "mOut", "dynamic_sizes": [0, 1], "dynamic_strides": [0]},
        ],
        "scalar_args": [
            {"name": "N", "ctype": "int32_t"},
            {"name": "num_chunks", "ctype": "int32_t"},
            {"name": "chunks_per_cta", "ctype": "int32_t"},
            {"name": "grid_x", "ctype": "int32_t"},
            {"name": "grid_y", "ctype": "int32_t"},
            {"name": "out_row_stride", "ctype": "int64_t"},
        ],
    }


_TORCH_TO_NAME = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}


def _chunk_elems_for(torch_dtype: torch.dtype) -> int:
    return chunk_elems_for(_TORCH_TO_NAME[torch_dtype])


@instrumented_cutedsl_cache(
    "aten::scatter_add",
    key_fn=lambda torch_dtype: f"tma {torch_dtype}",
)
def _compile_tma_scatter(torch_dtype: torch.dtype):
    # N and num_chunks are runtime args, so one compile per dtype serves
    # every row length. chunk_elems (the static TMA box) depends only on
    # the dtype's element size.
    name = _TORCH_TO_NAME[torch_dtype]
    dtype = _DTYPES[name]
    elem_bytes = dtype.width // 8
    chunk_elems = chunk_elems_for(name)
    launcher = _make_kernel(dtype, elem_bytes, chunk_elems, _reduce_op_for(dtype))

    mSrc_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    # Index is guaranteed contiguous by _flatten_for_expanded_1d; fix
    # stride=1 so `mIndex[i]` doesn't emit a runtime stride multiply.
    mIndex_fake = cute.runtime.make_fake_tensor(Int64, (cute.sym_int(),), stride=(1,))
    mOut_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    return cute.compile(
        launcher,
        mSrc_fake,
        mIndex_fake,
        mOut_fake,
        Int32(0),  # N
        Int32(0),  # num_chunks
        Int32(0),  # chunks_per_cta
        Int32(0),  # grid_x
        Int32(0),  # grid_y
        Int64(0),  # out_row_stride
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        # Bounds checks on ``r`` are always-on predicated PTX traps in
        # the kernel body (``trap_if_oob``) -- effectively free,
        # unlike ``--enable-assertions`` (~10% device time).
        options="--enable-tvm-ffi",
    )


def min_d_divisor_for(dtype: torch.dtype) -> int:
    """Smallest value D must be divisible by so ``cp.reduce.async.bulk``
    gmem operands stay 16-byte aligned: ``D * sizeof(dtype) % 16 == 0``,
    i.e. ``D % (16 / sizeof(dtype)) == 0``. fp32: %4, bf16/fp16: %8.
    Supported dtypes' itemsize divides 16, so plain // suffices.
    """
    return 16 // dtype.itemsize


def row_shape_supported(dtype: torch.dtype, N: int) -> bool:
    """Host-side check: can the TMA kernel handle an N-element row?

    Only requirement is that ``row_bytes = N * elem_size`` is a multiple
    of 16, which is the PTX operand alignment for both the tile-mode
    TMA load (per-row stride must be 16-aligned) and
    ``cp.reduce.async.bulk`` (gmem address + byte count must be
    16-aligned). Rows that aren't multiples of ``chunk_elems`` are
    handled via the TMA descriptor's OOB-clamp-to-zero behavior.
    """
    return (N * dtype.itemsize) % 16 == 0


def _plan_grid(M: int, D: int, chunk_elems: int, sm: int) -> tuple[int, int, int]:
    """Pick ``(grid_x, grid_y, chunks_per_cta)``.

    Strategy: keep the classic 1D schedule (grid_y=1, whole chunk range
    per CTA with internal double-buffering) whenever the row-axis alone
    saturates the GPU. When M is too small for that, split the
    chunk-axis across grid_y so every SM gets work.
    """
    n_chunks = (D + chunk_elems - 1) // chunk_elems
    # 1 warp per CTA: need many more CTAs than an 8-warp layout to keep
    # occupancy up. sm*32 target with a sm*64 clamp works well across
    # uniform / high_cont / few_idx on B200.
    row_ctas = M
    target_ctas = sm * 32
    if row_ctas >= target_ctas:
        grid_x = min(row_ctas, sm * 64)
        return grid_x, 1, n_chunks
    # Split the chunk axis until we hit the target.
    want_y = max(1, target_ctas // max(row_ctas, 1))
    grid_y = min(n_chunks, want_y)
    chunks_per_cta = (n_chunks + grid_y - 1) // grid_y
    # Recompute grid_y now that each y-slot holds chunks_per_cta chunks.
    grid_y = (n_chunks + chunks_per_cta - 1) // chunks_per_cta
    grid_x = row_ctas
    return grid_x, grid_y, chunks_per_cta


def tma_scatter_add_into(
    out: torch.Tensor,
    index_1d: torch.Tensor,
    src: torch.Tensor,
) -> None:
    """In-place: ``out[index_1d[i], :] += src[i, :]`` for every i.

    ``out`` / ``src`` are 2D with inner-dim stride 1 (outer row stride
    can differ from N, e.g. a slice of a wider buffer). ``index_1d`` is
    1D int64 of length M_src. ``row_bytes = N * elem_size`` must be a
    multiple of 16; the host cond enforces this.
    """
    M, N = src.shape
    chunk_elems = _chunk_elems_for(src.dtype)
    compiled = _compile_tma_scatter(src.dtype)
    sm = torch.cuda.get_device_properties(out.device).multi_processor_count

    grid_x, grid_y, chunks_per_cta = _plan_grid(M, N, chunk_elems, sm)
    num_chunks = (N + chunk_elems - 1) // chunk_elems
    compiled(
        src,
        index_1d,
        out,
        N,
        num_chunks,
        chunks_per_cta,
        grid_x,
        grid_y,
        out.stride(0),
    )
