import torch
import triton
import triton.language as tl
import itertools
from typing import Optional, Tuple


def compressed_indices_to_plain_indices(cidx, pidx):
    nnz = pidx.shape[-1]
    cdim = cidx.shape[-1] - 1
    batch_numel = cidx.shape[0]
    batch_offset = torch.arange(batch_numel, dtype=cidx.dtype, device=cidx.device)[
        :, None
    ]

    cidx_batch_offsetted = cidx[:, :-1] + nnz * batch_offset
    cidx_linear = torch.empty(
        (batch_numel * cdim + 1,), dtype=cidx.dtype, device=cidx.device
    )
    cidx_linear[:-1] = cidx_batch_offsetted.reshape(-1)
    cidx_linear[-1] = nnz * batch_numel

    idx_linear = torch._convert_indices_from_csr_to_coo(
        cidx_linear, pidx.reshape(-1), out_int32=(cidx.dtype == torch.int32)
    ).select(0, 0)

    return idx_linear.reshape(batch_numel, -1).sub_(cdim * batch_offset)


def slicer(dim, slice_range, *tensors):
    for t in tensors:
        slices = [slice(None)] * t.dim()
        slices[dim] = slice_range
        yield t[slices]


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

    output_acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_ROW), tl.float32)
    for _ in range(row_nnz):
        values_block = tl.load(values_block_ptrs)

        # find which row of dense needs to get loaded
        # for multiplication with values_block.
        dense_row_idx = tl.load(col_index_nnz_ptr)
        dense_block = tl.load(dense_block_ptrs + dense_tiled_row_stride * dense_row_idx)

        # do block mm
        output_acc_block += tl.dot(values_block, dense_block)

        # move val/col_index ptrs to the next block in the row
        values_block_ptrs += values_nnz_stride
        col_index_nnz_ptr += col_indices_stride

    # write back the result
    tl.store(output_ptrs, output_acc_block.to(output_ptr.dtype.element_ty))


@triton.jit
def _bsr_strided_sparse_rowspace_kernel(
    BLOCKSIZE_ROW: tl.constexpr,
    BLOCKSIZE_COL: tl.constexpr,
    batch_idx_ptr,
    row_idx_ptr,
    nnz_per_row_ptr,
    nnz_per_row_cumsum_ptr,
    col_indices_ptr,
    col_indices_stride,
    # values prologue
    values_ptr,
    values_nnz_stride,
    values_row_block_stride,
    values_col_block_stride,
    # values epilogue
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
    GROUP_SIZE_ROW: tl.constexpr,
):
    row_block_pid = tl.program_id(axis=0)
    col_block_pid = tl.program_id(axis=1)
    n_block_rows = tl.num_programs(axis=0)
    n_block_cols = tl.num_programs(axis=1)

    row_block_pid, col_block_pid = tl.swizzle2d(
        row_block_pid, col_block_pid, n_block_rows, n_block_cols, GROUP_SIZE_ROW
    )

    batch_idx = tl.load(batch_idx_ptr + row_block_pid)
    row_idx = tl.load(row_idx_ptr + row_block_pid)
    row_idx_nnz = tl.load(nnz_per_row_ptr + row_block_pid)
    row_idx_nnz_cumsum = tl.load(nnz_per_row_cumsum_ptr + row_block_pid)
    row_idx_nnz_offset = row_idx_nnz_cumsum - row_idx_nnz

    row_block_arange = tl.arange(0, BLOCKSIZE_ROW)
    col_block_arange = tl.arange(0, BLOCKSIZE_COL)

    # Pointers are set to the first block of the current row.
    values_block_ptrs = (
        values_ptr
        + values_nnz_stride * row_idx_nnz_offset
        + values_row_block_stride * row_block_arange[:, None]
        + values_col_block_stride * col_block_arange[None, :]
    )

    # NOTE: dense is advanced into all dimensions but the tiled row one.
    # That will be advanced in the loop according to values in col_indices.
    dense_block_ptrs = (
        dense_ptr
        + dense_batch_stride * batch_idx
        + dense_tiled_col_stride * col_block_pid
        + dense_row_block_stride * col_block_arange[:, None]
        + dense_col_block_stride * row_block_arange[None, :]
    )

    # Pointers are set to exact write-to locations
    output_ptrs = (
        output_ptr
        + output_batch_stride * batch_idx
        + output_tiled_row_stride * row_idx
        + output_tiled_col_stride * col_block_pid
        + output_row_block_stride * row_block_arange[:, None]
        + output_col_block_stride * row_block_arange[None, :]
    )

    output_acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_ROW), tl.float32)
    col_index_nnz_ptr = col_indices_ptr + row_idx_nnz_offset * col_indices_stride
    for _ in range(row_idx_nnz):
        values_block = tl.load(values_block_ptrs)

        # find which row of dense needs to get loaded
        # for multiplication with values_block.
        dense_row_idx = tl.load(col_index_nnz_ptr)
        dense_block = tl.load(dense_block_ptrs + dense_tiled_row_stride * dense_row_idx)

        # do block mm
        output_acc_block += tl.dot(values_block, dense_block)

        # move val/col_index ptrs to the next block in the row
        values_block_ptrs += values_nnz_stride
        col_index_nnz_ptr += col_indices_stride

    # write back the result
    tl.store(output_ptrs, output_acc_block.to(output_ptr.dtype.element_ty))


def _run_sparse_rowspace_kernel(
    blocksize, values, crow_indices, col_indices, dense, output, max_grid
):
    # Compute a vector of non-zero elements numbers per each row.
    # We want to ultimately iterate over non-zero rows.
    nnz_per_row = crow_indices[:, 1:] - crow_indices[:, :-1]

    # Compute indices of non-zero counts.
    # batch_idx maps to a broadcasted batch index, while
    # row_idx tracks non-zero rows of the sparse argument
    # and rows of the output that get modified.
    batch_idx, row_idx = nnz_per_row.nonzero(as_tuple=True)

    # Compress the vector of counts to hold only non-zero values.
    nnz_per_row = nnz_per_row[batch_idx, row_idx]
    # Compute cumulative counts which along with nnz_per_row
    # are used to compute offsets into nnz values.
    nnz_per_row_cumsum = nnz_per_row.cumsum(-1)

    n_nnz_block_rows = row_idx.size(-1)
    n_block_cols = dense.size(-3)
    max_n_nnz_block_rows, max_n_block_cols = max_grid[:2]

    for c_start in range(0, n_block_cols, max_n_block_cols):
        c_dense, c_output = slicer(
            -3, slice(c_start, c_start + max_n_block_cols), dense, output
        )
        c_grid = min(n_block_cols - c_start, max_n_block_cols)

        for r_start in range(0, n_nnz_block_rows, max_n_nnz_block_rows):
            r_batch_idx, r_row_idx, r_nnz_per_row, r_nnz_per_row_cumsum = slicer(
                0,
                slice(r_start, r_start + max_n_nnz_block_rows),
                batch_idx,
                row_idx,
                nnz_per_row,
                nnz_per_row_cumsum,
            )
            r_grid = min(n_nnz_block_rows - r_start, max_n_nnz_block_rows)

            _bsr_strided_sparse_rowspace_kernel[(r_grid, c_grid)](
                *blocksize,
                r_batch_idx,
                r_row_idx,
                r_nnz_per_row,
                r_nnz_per_row_cumsum,
                col_indices,
                *col_indices.stride(),
                values,
                *values.stride(),
                c_dense,
                *c_dense.stride(),
                c_output,
                *c_output.stride(),
                GROUP_SIZE_ROW=4,
                num_stages=4,
                num_warps=4,
            )


def _run_dense_rowspace_kernel(
    blocksize, values, crow_indices, col_indices, dense, output, max_grid
):
    # Launch kernel
    n_batches = dense.size(0)
    n_block_rows = crow_indices.size(-1) - 1
    n_block_cols = dense.size(-3)
    max_n_block_rows, max_n_block_cols, max_n_batches = max_grid

    for b_start in range(0, n_batches, max_n_batches):
        b_v, b_crow, b_col, b_d, b_o = slicer(
            0,
            slice(b_start, b_start + max_n_batches),
            values,
            crow_indices,
            col_indices,
            dense,
            output,
        )
        b_grid = min(n_batches - b_start, max_n_batches)

        for c_start in range(0, n_block_cols, max_n_block_cols):
            bc_d, bc_o = slicer(
                -3, slice(c_start, c_start + max_n_block_cols), b_d, b_o
            )
            c_grid = min(n_block_cols - c_start, max_n_block_cols)

            for r_start in range(0, n_block_rows, max_n_block_rows):
                r_slice = slice(r_start, r_start + max_n_block_rows)
                br_crow = next(slicer(-1, r_slice, b_crow))
                brc_o = next(slicer(-4, r_slice, bc_o))
                r_grid = min(n_block_rows - r_start, max_n_block_rows)

                _bsr_strided_dense_rowspace_kernel[(r_grid, c_grid, b_grid)](
                    *blocksize,
                    b_v,
                    *b_v.stride(),
                    br_crow,
                    *br_crow.stride(),
                    b_col,
                    *b_col.stride(),
                    bc_d,
                    *bc_d.stride(),
                    brc_o,
                    *brc_o.stride(),
                    GROUP_SIZE_ROW=4,
                    num_stages=4,
                    num_warps=4,
                )


def bsr_dense_mm(
    bsr: torch.Tensor,
    dense: torch.Tensor,
    *,
    is_sparse_rowspace_mode: Optional[bool] = None,
    out: torch.Tensor = None,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
):
    def check(cond, msg):
        if not cond:
            raise ValueError(msg)

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

    # Required to undo the fake batch dimension insertion.
    original_batch_dims_broadcasted = torch.broadcast_shapes(
        bsr.shape[:-2], dense.shape[:-2]
    )

    if out is not None:
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

    # Short circuit if lhs is zero
    if bsr._nnz() == 0:
        return dense.new_zeros(original_batch_dims_broadcasted + (m, n))

    # TODO: insert switch
    if is_sparse_rowspace_mode is None:
        is_sparse_rowspace_mode = False

    # Introduce fake batch dimension if not present for convenience.
    def unsqueeze_batch_dim(t, n_non_batch_dims):
        if t.dim() > n_non_batch_dims:
            return t
        else:
            return t.unsqueeze(0)

    def make_triton_contiguous(t):
        # Triton does not distinguish between row- and col-majorness
        # and will be fast as long as there is a contiguous dimension.
        if not (t.is_contiguous() or t.transpose(-2, -1).is_contiguous()):
            return t.contiguous()
        else:
            return t

    crow_indices = unsqueeze_batch_dim(bsr.crow_indices(), 1)
    col_indices = unsqueeze_batch_dim(bsr.col_indices(), 1)
    values = make_triton_contiguous(unsqueeze_batch_dim(bsr.values(), 3))
    dense = make_triton_contiguous(unsqueeze_batch_dim(dense, 2))
    nnz = values.shape[-3]
    blocksize = values.shape[-2:]

    # Compute broadcasted batch dimension
    bsr_batch_dims = values.shape[:-3]
    dense_batch_dims = dense.shape[:-2]
    batch_dims_broadcasted = torch.broadcast_shapes(bsr_batch_dims, dense_batch_dims)

    # Allocate out
    if out is None:
        out = dense.new_zeros(batch_dims_broadcasted + (m, n))

    # Broadcast batch dimensions and squash
    def batch_broadcast_and_squash(t, batch_dims, invariant_dims):
        return t.broadcast_to(batch_dims + invariant_dims).flatten(
            0, len(batch_dims) - 1
        )

    crow_indices = batch_broadcast_and_squash(
        crow_indices, batch_dims_broadcasted, (-1,)
    )

    if is_sparse_rowspace_mode:
        # Flatten batch dimension with nnz dimension
        # as required by the sparse rowspace kernel.
        col_indices = batch_broadcast_and_squash(
            col_indices, batch_dims_broadcasted + (-1,), ()
        )
        values = batch_broadcast_and_squash(
            values, batch_dims_broadcasted + (values.shape[-3],), values.shape[-2:]
        )
    else:
        col_indices = batch_broadcast_and_squash(
            col_indices, batch_dims_broadcasted, (-1,)
        )
        values = batch_broadcast_and_squash(
            values, batch_dims_broadcasted, values.shape[-3:]
        )

    dense = batch_broadcast_and_squash(dense, batch_dims_broadcasted, dense.shape[-2:])

    # NOTE: out is contiguous, so batch_broadcast_and_squash will create a view
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
    if is_sparse_rowspace_mode:
        kernel = _run_sparse_rowspace_kernel
    else:
        kernel = _run_dense_rowspace_kernel

    # cuda_max_grid = (2 ** 31 - 1, 2 ** 16 - 1, 2 ** 16 - 1)
    cuda_max_grid = (2147483647, 65535, 65535)
    if max_grid is None:
        max_grid = cuda_max_grid
    else:

        def valid_grid_dim(g, mg):
            if g is None:
                return mg
            else:
                # grid must be at least 1 and no greater than mg
                return max(1, min(g, mg))

        max_grid = tuple(
            valid_grid_dim(g, mg) for g, mg in zip(max_grid, cuda_max_grid)
        )

    kernel(blocksize, values, crow_indices, col_indices, dense, out, max_grid)

    # Block dims need to rejoin with the corresponding block dimensions
    # prior to reshape so that blocks do not end up being transposed.
    return out.transpose(-3, -2).reshape(original_batch_dims_broadcasted + (m, n))


if __name__ == "__main__":
    torch.set_printoptions(threshold=2000, linewidth=150)
    torch.manual_seed(13)
    dtype = torch.float32
    p = 0.5
    batch_size_bsr = (10,)
    mask_size = (8, 8)
    block_size = (64, 64)
    size = (mask_size[0] * block_size[0], mask_size[1] * block_size[1])

    n_exp = 512
    diff = torch.ones(n_exp, device="cuda", dtype=torch.float32)
    for i in range(n_exp):
        mask = torch.rand(*mask_size, device="cuda") < p
        x = torch.rand(*mask_size, *block_size, dtype=dtype, device="cuda") / 10
        x = (
            (mask[:, :, None, None] * x)
            .transpose(-3, -2)
            .reshape(*size)
            .to_sparse_bsr(*block_size)
        )
        y = torch.rand(5, *size, dtype=dtype, device="cuda") / 10
        res_dense = x.to_dense() @ y
        res = bsr_dense_mm(x, y)
        diff[i] = (res - res_dense).abs().max()
    print(f"mean: {diff.mean()}, std: {diff.std()}")
    print(f"max diff: {diff.max()}")
