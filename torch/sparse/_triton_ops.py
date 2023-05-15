import torch
from torch._inductor.cuda_properties import get_device_capability


def _has_triton():
    if not torch.cuda.is_available():
        return False
    try:
        import triton

        return triton is not None and get_device_capability() >= (7, 0)
    except ImportError:
        return False


def make_triton_contiguous(t):
    # Triton does not distinguish between row- and col-majorness
    # and will be fast as long as there is a contiguous dimension.
    if not (t.is_contiguous() or t.transpose(-2, -1).is_contiguous()):
        return t.contiguous()
    else:
        return t


def broadcast_batch_dims(*tensors):
    return torch.broadcast_shapes(*(t.shape[:-2] for t in tensors))


def slicer(dim, slice_range, *tensors):
    for t in tensors:
        slices = [slice(None)] * t.dim()
        slices[dim] = slice_range
        yield t[slices]

def multidim_slicer(dims, slices, *tensors):
    for t in tensors:
        s = [slice(None)] * t.dim()
        for d, d_slice in zip(dims, slices):
            if d is not None:
                s[d] = d_slice
        yield t[s]

def ptr_stride_extractor(*tensors):
    for t in tensors:
        yield t
        yield from t.stride()

def grid_partitioner(full_grid, grid_blocks, tensor_dims_map):
    assert 0 <= len(full_grid) <= 3
    assert 0 <= len(grid_blocks) <= 3

    import itertools

    def generate_grid_points():
        for fg, mg in zip(full_grid, grid_blocks):
            yield range(0, fg, mg)

    def generate_sliced_tensors(slices):
        for t, t_dims in tensor_dims_map.items():
            yield next(multidim_slicer(t_dims, slices, t))

    for grid_point in itertools.product(*generate_grid_points()):
        grid = [min(fg - gp, mg) for fg, gp, mg in zip(full_grid, grid_point, grid_blocks)]
        slices = [slice(gp, gp + g) for gp, g in zip(grid_point, grid)]
        # grid_points are iterated in a "contiguous" order, i.e.
        # left dimensions traversed slower than right dimensions.
        # This order is reversed for CUDA grids.
        yield grid[::-1], *generate_sliced_tensors(slices)

def launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks=None):
    # cuda_max_grid = (2 ** 31 - 1, 2 ** 16 - 1, 2 ** 16 - 1)
    cuda_max_grid = (2147483647, 65535, 65535)[::-1]
    if grid_blocks is None:
        grid_blocks = cuda_max_grid
    else:

        def valid_grid_dim(g, mg):
            if g is None:
                return mg
            else:
                # grid must be at least 1 and no greater than mg
                return max(1, min(g, mg))

        grid_blocks = tuple(
            valid_grid_dim(g, mg) for g, mg in zip(grid_blocks, cuda_max_grid)
        )  # type: ignore[assignment]

    for grid, *sliced_tensors in grid_partitioner(full_grid, grid_blocks, tensor_dims_map):
        kernel(grid, *sliced_tensors)

if _has_triton():
    import triton
    import triton.language as tl
    from typing import Optional, Tuple

    @triton.jit
    def _bsr_strided_dense_rowspace_kernel(
        BLOCKSIZE_ROW: tl.constexpr,
        BLOCKSIZE_COL: tl.constexpr,
        # values prologue
        values_ptr,
        values_batch_stride,
        values_nnz_stride,
        values_row_block_stride,
        values_col_block_stride,
        # values epilogue
        # crow_indices prologue
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        # crow_indices epilogue
        # col_indices prologue
        col_indices_ptr,
        col_indices_batch_stride,
        col_indices_stride,
        # col_indices epilogue
        # dense prologue
        dense_ptr,
        dense_batch_stride,
        dense_tiled_row_stride,
        dense_tiled_col_stride,
        dense_row_block_stride,
        dense_col_block_stride,
        # dense epilogue
        # output prologue
        output_ptr,
        output_batch_stride,
        output_tiled_row_stride,
        output_tiled_col_stride,
        output_row_block_stride,
        output_col_block_stride,
        # output epilogue
        acc_dtype: tl.constexpr,
        allow_tf32: tl.constexpr,
        GROUP_SIZE_ROW: tl.constexpr,
    ):
        batch_pid = tl.program_id(axis=2)
        row_block_pid = tl.program_id(axis=0)
        col_block_pid = tl.program_id(axis=1)
        n_block_rows = tl.num_programs(axis=0)
        n_block_cols = tl.num_programs(axis=1)

        row_block_pid, col_block_pid = tl.swizzle2d(
            row_block_pid, col_block_pid, n_block_rows, n_block_cols, GROUP_SIZE_ROW
        )

        crow_indices_offset_ptr = (
            crow_indices_ptr
            + crow_indices_batch_stride * batch_pid
            + crow_indices_stride * row_block_pid
        )
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

        # Compute nnz for the row with number row_block_pid.
        # If it is zero, skip the row.
        row_nnz = nnz_offset_next - nnz_offset
        if row_nnz == 0:
            return

        row_block_arange = tl.arange(0, BLOCKSIZE_ROW)
        col_block_arange = tl.arange(0, BLOCKSIZE_COL)

        # Pointers are set to the first block of the current row.
        values_block_ptrs = (
            values_ptr
            + values_batch_stride * batch_pid
            + values_nnz_stride * nnz_offset
            + values_row_block_stride * row_block_arange[:, None]
            + values_col_block_stride * col_block_arange[None, :]
        )

        # NOTE: dense is advanced into all dimensions but the tiled row one.
        # That will be advanced in the loop according to values in col_indices.
        dense_block_ptrs = (
            dense_ptr
            + dense_batch_stride * batch_pid
            + dense_tiled_col_stride * col_block_pid
            + dense_row_block_stride * col_block_arange[:, None]
            + dense_col_block_stride * row_block_arange[None, :]
        )

        # Pointers are set to exact write-to locations
        output_ptrs = (
            output_ptr
            + output_batch_stride * batch_pid
            + output_tiled_row_stride * row_block_pid
            + output_tiled_col_stride * col_block_pid
            + output_row_block_stride * row_block_arange[:, None]
            + output_col_block_stride * row_block_arange[None, :]
        )

        # Set pointer to the first nonzero element in the current row
        col_index_nnz_ptr = (
            col_indices_ptr
            + col_indices_batch_stride * batch_pid
            + col_indices_stride * nnz_offset
        )

        output_acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_ROW), dtype=acc_dtype)
        for _ in range(row_nnz):
            values_block = tl.load(values_block_ptrs)

            # find which row of dense needs to get loaded
            # for multiplication with values_block.
            dense_row_idx = tl.load(col_index_nnz_ptr)
            dense_block = tl.load(dense_block_ptrs + dense_tiled_row_stride * dense_row_idx)

            # do block mm
            output_acc_block += tl.dot(values_block, dense_block, allow_tf32=allow_tf32)

            # move val/col_index ptrs to the next block in the row
            values_block_ptrs += values_nnz_stride
            col_index_nnz_ptr += col_indices_stride

        # write back the result
        tl.store(output_ptrs, output_acc_block.to(output_ptr.dtype.element_ty))


    def _run_dense_rowspace_kernel(
        blocksize, values, crow_indices, col_indices, dense, output, max_grid
    ):
        n_batches = dense.size(0)
        n_block_rows = crow_indices.size(-1) - 1
        n_block_cols = dense.size(-3)

        full_grid = (n_batches, n_block_cols, n_block_rows)
        if max_grid is not None:
            grid_blocks = tuple(max_grid[:3][::-1]) + (None,) * (3 - len(max_grid[:3]))
        else:
            grid_blocks = None
        tensor_dims_map = {
            values: (0, None, None),
            crow_indices: (0, None, -1),
            col_indices: (0, None, None),
            dense: (0, -3, None),
            output: (0, -3, -4)
        }
        if values.dtype in (torch.half, torch.bfloat16):
            acc_dtype = tl.float32
            allow_tf32 = True
        else:
            acc_dtype = tl.float64
            allow_tf32 = False

        def kernel(grid, *sliced_tensors):
            _bsr_strided_dense_rowspace_kernel[grid](
                *blocksize,
                *ptr_stride_extractor(*sliced_tensors),
                acc_dtype=acc_dtype,
                allow_tf32=allow_tf32,
                GROUP_SIZE_ROW=4,
                num_stages=1,
                num_warps=4
            )

        launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)


    def bsr_dense_mm(
        bsr: torch.Tensor,
        dense: torch.Tensor,
        *,
        skip_checks: bool = False,
        max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
        out: Optional[torch.Tensor] = None,
    ):
        def check(cond, msg):
            if not cond:
                raise ValueError(msg)

        if not skip_checks:
            check(
                bsr.layout == torch.sparse_bsr,
                "bsr_dense_mm(): only BSR sparse format is supported for the sparse argument.",
            )

            check(
                bsr.device == dense.device and bsr.device.type == "cuda",
                "bsr_dense_mm(): all inputs are expected to be on the same GPU device.",
            )

            check(
                bsr.dtype == dense.dtype
                and bsr.dtype in (torch.half, torch.bfloat16, torch.float),
                "bsr_dense_mm(): all inputs are expected to be of the same dtype "
                "and one of (half, bfloat16, float32), "
                f"but got bsr.dtype == {bsr.dtype} and dense.dtype == {dense.dtype}.",
            )

            check(
                bsr.dim() >= 2 and dense.dim() >= 2,
                "bsr_dense_mm(): all inputs are expected to be at least 2D, "
                f"but got bsr.dim() == {bsr.dim()} and dense.dim() == {dense.dim()}.",
            )

            m, kl = bsr.shape[-2:]
            kr, n = dense.shape[-2:]

            check(
                kl == kr,
                "bsr_dense_mm(): argument sizes are not compatible for matrix multiplication, "
                f"got bsr.shape[-1] == {kl} which is not equal to dense.shape[-2] == {kr}.",
            )

            row_block, col_block = bsr.values().shape[-2:]
            check(
                not n % row_block,
                f"bsr_dense_mm(): dense.size(-1) == {n} should be divisible by "
                f"blocksize[0] == {row_block}.",
            )

            def is_power_of_two(v):
                return not (v & (v - 1))

            def is_compatible_blocksize(b):
                assert len(b) == 2
                res = True
                for blocksize in b:
                    # Triton loads only blocks which are at least 16 and powers of 2.
                    res = (blocksize >= 16 and is_power_of_two(blocksize)) and res
                return res

            check(
                is_compatible_blocksize((row_block, col_block)),
                f"bsr_dense_mm(): sparse inputs' blocksize ({row_block}, {col_block}) "
                "should be at least 16 and a power of 2 in each dimension.",
            )
        else:
            m, kl = bsr.shape[-2:]
            kr, n = dense.shape[-2:]

        original_batch_dims_broadcasted = broadcast_batch_dims(bsr, dense)

        if out is not None and not skip_checks:
            expected_out_shape = original_batch_dims_broadcasted + (m, n)
            check(
                out.shape == expected_out_shape,
                "bsr_dense_mm(): `out` argument has wrong shape, "
                f"expected {expected_out_shape}, but got {out.shape}.",
            )
            check(
                out.is_contiguous() or out.transpose(-2, -1).is_contiguous(),
                "bsr_dense_mm(): only row-major/col-major `out` arguments are supported, "
                "i.e. (out.is_contiguous() or out.transpose(-2, -1).is_contiguous()) "
                "should be True.",
            )

        # Allocate out
        if out is None:
            out = dense.new_zeros(original_batch_dims_broadcasted + (m, n))
        else:
            out.zero_()

        # Short circuit if lhs is zero
        if bsr._nnz() == 0:
            return out

        # Introduce fake batch dimension if not present for convenience.
        crow_indices = bsr.crow_indices().unsqueeze(0)
        col_indices = bsr.col_indices().unsqueeze(0)
        values = make_triton_contiguous(bsr.values().unsqueeze(0))
        dense = make_triton_contiguous(dense.unsqueeze(0))
        nnz = values.shape[-3]
        blocksize = values.shape[-2:]

        # Compute broadcasted batch dimension
        bsr_batch_dims = values.shape[:-3]
        dense_batch_dims = dense.shape[:-2]
        batch_dims_broadcasted = torch.broadcast_shapes(bsr_batch_dims, dense_batch_dims)

        # Broadcast batch dimensions and squash
        def batch_broadcast_and_squash(t, batch_dims, invariant_dims):
            return t.broadcast_to(batch_dims + invariant_dims).flatten(
                0, len(batch_dims) - 1
            )

        crow_indices = batch_broadcast_and_squash(
            crow_indices, batch_dims_broadcasted, (-1,)
        )

        col_indices = batch_broadcast_and_squash(
            col_indices, batch_dims_broadcasted, (-1,)
        )
        values = batch_broadcast_and_squash(
            values, batch_dims_broadcasted, values.shape[-3:]
        )

        dense = batch_broadcast_and_squash(dense, batch_dims_broadcasted, dense.shape[-2:])

        # NOTE: out is contiguous, so batch_broadcast_and_squash will create a view
        # out gets modified in-place, so we store a backup copy.
        out_backup = out
        out = batch_broadcast_and_squash(out, batch_dims_broadcasted, out.shape[-2:])

        # NOTE: this function will ALWAYS create a view
        def tile_to_blocksize(t, blocksize):
            *rest, m, n = t.shape
            new_shape = rest + [
                m // blocksize[0],
                blocksize[0],
                n // blocksize[1],
                blocksize[1],
            ]
            return t.reshape(new_shape).transpose(-3, -2)

        # "Blockify" the row dimension of dense with blocksize[1]
        # since dense is on the rhs of matmul
        dense = tile_to_blocksize(dense, blocksize[::-1])
        # "Blockify" the row dimension of out with blocksize[0]
        # which is inherited from the bsr input.
        # NOTE: tile_to_blocksize will create a view.
        # NOTE: out.blocksize[-1] == dense.blocksize[-1],
        # so it could be any value in [1, dense.shape[-1]).
        # We need to probably use the largest possible blocksize
        # so that it fits into SRAM.
        out = tile_to_blocksize(out, (blocksize[0], blocksize[0]))

        # Launch kernel
        kernel = _run_dense_rowspace_kernel
        kernel(blocksize, values, crow_indices, col_indices, dense, out, max_grid)

        return out_backup
else:
    bsr_dense_mm = None  # type: ignore[assignment]
