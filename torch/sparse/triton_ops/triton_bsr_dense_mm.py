import torch
import triton
import triton.language as tl


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


@triton.jit
def _bsr_strided_mm_kernel(
    batch_idx_ptr,
    row_idx_ptr,
    nnz_per_row_ptr,
    nnz_per_row_cumsum_ptr,
    BLOCKSIZE_ROW: tl.constexpr,
    BLOCKSIZE_COL: tl.constexpr,
    col_indices_ptr,
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
    for nnz_idx in range(row_idx_nnz):
        values_block = tl.load(values_block_ptrs)

        # find which row of dense needs to get loaded
        # for multiplication with values_block.
        # TODO: vectorize the load.
        dense_row_idx = tl.load(col_indices_ptr + row_idx_nnz_offset + nnz_idx)
        dense_block = tl.load(dense_block_ptrs + dense_tiled_row_stride * dense_row_idx)

        # do block mm
        output_acc_block += tl.dot(values_block, dense_block)

        # move val ptrs to the next block in the row
        values_block_ptrs += values_nnz_stride

    # write back the result
    tl.store(output_ptrs, output_acc_block.to(output_ptr.dtype.element_ty))


def bsr_dense_mm(bsr, dense):
    # TODO: insert checks
    m, k = bsr.shape[-2:]
    k1, n = dense.shape[-2:]

    # TODO: probably make sure that inputs are properly contiguous

    # Required to undo the fake batch dimension insertion.
    original_batch_dims_broadcasted = torch.broadcast_shapes(
        bsr.shape[:-2], dense.shape[:-2]
    )

    # Short circuit if lhs is zero
    if bsr._nnz() == 0:
        return torch.zeros(
            original_batch_dims_broadcasted + (m, n),
            dtype=dense.dtype,
            device=dense.device,
        )

    # Introduce fake batch dimension if not present for convenience.
    def unsqueeze_batch_dim(t, n_non_batch_dims):
        if t.dim() > n_non_batch_dims:
            return t
        else:
            return t.unsqueeze(0)

    crow_indices = unsqueeze_batch_dim(bsr.crow_indices(), 1)
    col_indices = unsqueeze_batch_dim(bsr.col_indices(), 1)
    values = unsqueeze_batch_dim(bsr.values(), 3)
    dense = unsqueeze_batch_dim(dense, 2)
    nnz = values.shape[-3]
    blocksize = values.shape[-2:]

    # Compute broadcasted batch dimension
    bsr_batch_dims = values.shape[:-3]
    dense_batch_dims = dense.shape[:-2]
    batch_dims_broadcasted = torch.broadcast_shapes(bsr_batch_dims, dense_batch_dims)

    # Allocate output
    output = torch.zeros(
        batch_dims_broadcasted + (m, n), dtype=dense.dtype, device=dense.device
    )

    # Broadcast batch dimensions and squash
    def batch_broadcast_and_squash(t, batch_dims, invariant_dims):
        return t.broadcast_to(batch_dims + invariant_dims).flatten(
            0, len(batch_dims) - 1
        )

    crow_indices = batch_broadcast_and_squash(
        crow_indices, batch_dims_broadcasted, (-1,)
    )
    col_indices = batch_broadcast_and_squash(col_indices, batch_dims_broadcasted, (-1,))
    values = batch_broadcast_and_squash(
        values, batch_dims_broadcasted, values.shape[-3:]
    )
    dense = batch_broadcast_and_squash(dense, batch_dims_broadcasted, dense.shape[-2:])
    # NOTE: output is contiguous, so batch_broadcast_and_squash will create a view
    output = batch_broadcast_and_squash(
        output, batch_dims_broadcasted, output.shape[-2:]
    )

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
    # "Blockify" the row dimension of output with blocksize[0]
    # which is inherited from the bsr input.
    # NOTE: tile_to_blocksize will create a view.
    # NOTE: output.blocksize[-1] == dense.blocksize[-1],
    # so it could be any value in [1, dense.shape[-1]).
    # We need to probably use the largest possible blocksize
    # so that it fits into SRAM.
    output = tile_to_blocksize(output, (blocksize[0], blocksize[0]))

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

    # The meta-data for search into values assumes
    # the batch and the nnz dimensions squashed together.
    # TODO: batch dims for values and col_indices are
    # already flattened. Fuse batch flattening with
    # batch-info flattening together.
    values = values.flatten(0, 1)
    col_indices = col_indices.flatten(0, 1)

    # Launch kernel
    n_nnz_block_rows = row_idx.size(-1)
    n_block_cols = dense.size(-3)
    grid = (n_nnz_block_rows, n_block_cols)
    _bsr_strided_mm_kernel[grid](
        batch_idx,
        row_idx,
        nnz_per_row,
        nnz_per_row_cumsum,
        *blocksize,
        col_indices,
        values,
        *values.stride(),
        dense,
        *dense.stride(),
        output,
        *output.stride(),
        GROUP_SIZE_ROW=4,
        num_stages=4,
        num_warps=4,
    )

    # Block dims need to rejoin with the corresponding block dimensions
    # prior to reshape so that blocks do not end up being transposed.
    return output.transpose(-3, -2).reshape(original_batch_dims_broadcasted + (m, n))


if __name__ == "__main__":
    torch.set_printoptions(threshold=2000, linewidth=150)
    torch.manual_seed(13)
    dtype = torch.float32
    p = 0.01
    batch_size_bsr = (10,)
    mask_size = (8, 8)
    block_size = (64, 64)
    size = (mask_size[0] * block_size[0], mask_size[1] * block_size[1])

    n_exp = 512
    diff = torch.ones(n_exp, device="cuda", dtype=torch.double)
    for i in range(n_exp):
        # mask = torch.rand(*mask_size, device='cuda') < p
        mask = torch.zeros(*mask_size).to(torch.bool)
        mask[0, 0] = True
        mask[1, 0] = True
        mask = mask.cuda()
        x = torch.rand(*mask_size, *block_size, dtype=dtype, device="cuda") / 10
        x = (
            (mask[:, :, None, None] * x)
            .transpose(-3, -2)
            .reshape(*size)
            .to_sparse_bsr(*block_size)
        )
        print(x)
        y = torch.rand(5, *size, dtype=dtype, device="cuda") / 10
        res_dense = x.to_dense() @ y
        res = bsr_dense_mm(x, y)
        diff[i] = (res - res_dense).abs().max()
    print(f"mean: {diff.mean()}, std: {diff.std()}")
    print(f"max diff: {diff.max()}")
