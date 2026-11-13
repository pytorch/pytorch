import cutlass
import cutlass.cute as cute
import nccl.core.device.cute as nccl_cute  # pyrefly: ignore [missing-import]
from cuda.bindings import driver
from cutlass.cute.runtime import from_dlpack

import torch
from torch.distributed._symmetric_memory._nccl_cute import _context


@cute.kernel
def _reduce_scatter_kernel(
    comm: nccl_cute.DevComm,
    window: nccl_cute.Window,
    outputs: tuple,
    offsets: tuple,
    src_stride: cutlass.Int64,
    rows: cutlass.Constexpr,
    widths: cutlass.Constexpr,
    ctas: cutlass.Constexpr,
    multimem: cutlass.Constexpr,
):
    tile, _, _ = cute.arch.block_idx()
    coop = nccl_cute.cta()
    bar = nccl_cute.lsa_default(coop, comm, index=tile)
    bar.sync(coop, nccl_cute.MemoryOrder.ACQUIRE)
    for slot in cutlass.range_constexpr(len(outputs)):
        start = sum(ctas[:slot])
        if tile >= start and tile < start + ctas[slot]:
            out = outputs[slot]
            width = widths[slot]
            for row in cutlass.range(tile - start, rows[slot], ctas[slot]):
                offset = offsets[slot] + row * src_stride
                dst = cute.make_tensor(
                    out.iterator + row * width, cute.make_layout(width)
                )
                if cutlass.const_expr(multimem):
                    ptr = cute.make_ptr(
                        out.element_type, window.lsa_multimem_pointer(offset, comm)
                    )
                    src = cute.make_tensor(ptr, cute.make_layout(width))
                    nccl_cute.multimem_reduce_sum(coop, src, dst, width)
                else:
                    src = window.tensor(
                        out.element_type, cute.make_layout(width), offset
                    )
                    nccl_cute.lsa_reduce_sum(
                        coop, window, src, dst, width, team=comm.team_lsa
                    )
    bar.sync(coop, nccl_cute.MemoryOrder.RELEASE)
    bar.destroy()


@cute.jit
def _reduce_scatter(
    comm,
    window,
    outputs,
    offsets,
    src_stride,
    rows: cutlass.Constexpr,
    widths: cutlass.Constexpr,
    ctas: cutlass.Constexpr,
    multimem: cutlass.Constexpr,
    stream,
):
    _reduce_scatter_kernel(
        comm,
        window,
        outputs,
        offsets,
        src_stride,
        rows,
        widths,
        ctas,
        multimem,
    ).launch(grid=(sum(ctas), 1, 1), block=(128, 1, 1), stream=stream)


def reduce_scatter_offset(
    input,
    out,
    group,
    *,
    dim,
    offsets=None,
    dst_ranks=None,
    red_op="sum",
    use_multimem=None,
):
    if red_op != "sum" or input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError(
            "reduce_scatter_offset supports sum on float16, bfloat16, and float32"
        )
    if input.ndim != 2 or dim not in (0, 1) or input.stride(-1) != 1:
        raise ValueError(
            "expected a 2-D input with contiguous inner dimension and dim=0 or dim=1"
        )
    with torch.cuda.device(input.device):
        handle, comm, window, base, multicast = _context(input, group)
        peers, rank = handle.world_size, handle.rank
        if offsets is None:
            if input.shape[dim] % peers:
                raise ValueError("input size must be divisible by group size")
            size = input.shape[dim] // peers
            offsets = [size * (i + 1) for i in range(peers)]
        if not 0 < len(offsets) <= 64:
            raise ValueError("expected between 1 and 64 blocks")
        starts = [0, *offsets[:-1]]
        sizes = [end - start for start, end in zip(starts, offsets)]
        if (
            any(size < 0 or size > 65535 for size in sizes)
            or offsets[-1] > input.shape[dim]
        ):
            raise ValueError("invalid block offsets")
        if dst_ranks is None:
            dst_ranks = [i % peers for i in range(len(offsets))]
        if len(dst_ranks) != len(offsets) or any(
            r < 0 or r >= peers for r in dst_ranks
        ):
            raise ValueError("dst_ranks must contain a valid rank for each block")
        owned = [
            [i for i, r in enumerate(dst_ranks) if r == peer] for peer in range(peers)
        ]
        counts = [len(indices) for indices in owned]
        if len(set(counts)) != 1 or not 0 < counts[rank] <= 16:
            raise ValueError(
                "each rank must own the same number of blocks, between 1 and 16"
            )
        slot_sizes = [sizes[i] for i in owned[rank]]
        if any([sizes[i] for i in indices] != slot_sizes for indices in owned):
            raise ValueError("each output slot must have equal size across ranks")
        rows = tuple(input.shape[0] if dim == 1 else size for size in slot_sizes)
        widths = tuple(size if dim == 1 else input.shape[1] for size in slot_sizes)
        if len(out) != len(rows):
            raise ValueError("out must contain one tensor for each owned block")
        for output, nrows, width in zip(out, rows, widths):
            if output.shape != (nrows, width) or not output.is_contiguous():
                raise ValueError(
                    f"output must be contiguous with shape {(nrows, width)}"
                )
            if output.device != input.device or output.dtype != input.dtype:
                raise ValueError("output must have the input's device and dtype")
        if use_multimem is True and not multicast:
            raise ValueError("multicast is unavailable for this allocation")
        multimem = multicast if use_multimem is None else use_multimem
        itemsize = input.element_size()
        stride = input.stride(0) * itemsize
        block_stride = itemsize if dim == 1 else stride
        byte_offsets = tuple(
            cutlass.Int64(base + starts[i] * block_stride) for i in owned[rank]
        )
        ctas = tuple(
            max(1, min((r * w * itemsize + 8191) // 8192, 16))
            for r, w in zip(rows, widths)
        )
        outputs = tuple(from_dlpack(output.view(-1)) for output in out)
        stream = driver.CUstream(torch.cuda.current_stream(input.device).cuda_stream)
        _reduce_scatter(
            comm,
            window,
            outputs,
            byte_offsets,
            cutlass.Int64(stride),
            rows,
            widths,
            ctas,
            multimem,
            stream,
        )
