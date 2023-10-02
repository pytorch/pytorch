from typing import Dict, List
import math
import torch
from torch.utils._triton import has_triton


def check(cond, msg):
    if not cond:
        raise ValueError(msg)


def check_bsr_layout(f_name, t):
    check(
        t.layout == torch.sparse_bsr,
        f"{f_name}(): only BSR sparse format is supported for the sparse argument.",
    )


def check_device(f_name, t, device):
    check(
        t.device == device and t.device.type == "cuda",
        f"{f_name}(): all inputs are expected to be on the same GPU device.",
    )


def check_mm_compatible_shapes(f_name, lhs, rhs):
    check(
        lhs.dim() >= 2 and rhs.dim() >= 2,
        f"{f_name}(): all inputs involved in the matrix product are expected to be at least 2D, "
        f"but got lhs.dim() == {lhs.dim()} and rhs.dim() == {rhs.dim()}."
    )

    m, kl = lhs.shape[-2:]
    kr, n = rhs.shape[-2:]

    check(
        kl == kr,
        f"{f_name}(): arguments' sizes involved in the matrix product are not compatible for matrix multiplication, "
        f"got lhs.shape[-1] == {kl} which is not equal to rhs.shape[-2] == {kr}.",
    )


def check_dtype(f_name, t, dtype, *additional_dtypes):
    check(
        t.dtype == dtype
        and t.dtype in ((torch.half, torch.bfloat16, torch.float) + tuple(*additional_dtypes)),
        f"{f_name}(): all inputs are expected to be of the same dtype "
        f"and one of (half, bfloat16, float32) or {additional_dtypes}, "
        f"but got dtype == {t.dtype}.",
    )


def check_blocksize(f_name, blocksize):
    assert len(blocksize) == 2

    def is_power_of_two(v):
        return not (v & (v - 1))

    def is_compatible_blocksize(b):
        res = True
        for blocksize in b:
            # Triton loads only blocks which are at least 16 and powers of 2.
            res = (blocksize >= 16 and is_power_of_two(blocksize)) and res
        return res

    check(
        is_compatible_blocksize(blocksize),
        f"{f_name}(): sparse inputs' blocksize ({blocksize[0]}, {blocksize[1]}) "
        "should be at least 16 and a power of 2 in each dimension.",
    )


def make_triton_contiguous(t):
    if (t.stride(-2) > 1 or t.dtype is torch.float32) and t.stride(-1) > 1:
        return t.contiguous()
    else:
        return t


def broadcast_batch_dims(f_name, *tensors):
    try:
        return torch.broadcast_shapes(*(t.shape[:-2] for t in tensors))
    except Exception:
        check(False, f"{f_name}(): inputs' batch dimensions are not broadcastable!")


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


def prepare_inputs(bsr, *dense_tensors):
    # Introduce fake batch dimension if not present for convenience.
    crow_indices = bsr.crow_indices().unsqueeze(0)
    col_indices = bsr.col_indices().unsqueeze(0)
    values = make_triton_contiguous(bsr.values().unsqueeze(0))
    tensors = [make_triton_contiguous(t.unsqueeze(0)) for t in dense_tensors]

    # Compute broadcasted batch dimension
    batch_dims_broadcasted = torch.broadcast_shapes(values.shape[:-3], *(t.shape[:-2] for t in tensors))

    # Broadcast batch dimensions and squash.
    # The result can be either a view or a copy.
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
    tensors = [
        batch_broadcast_and_squash(t, batch_dims_broadcasted, t.shape[-2:]) for t in tensors
    ]

    return crow_indices, col_indices, values, *tensors


def broadcast_batch_dims_bsr(f_name, bsr, *tensors):
    batch_shape = broadcast_batch_dims(f_name, bsr, *tensors)

    crow_indices = bsr.crow_indices().broadcast_to(batch_shape + (-1,))
    col_indices = bsr.col_indices().broadcast_to(batch_shape + (-1,))
    values = bsr.values().broadcast_to(batch_shape + bsr.values().shape[-3:])
    size = batch_shape + bsr.shape[-2:]
    return torch.sparse_compressed_tensor(crow_indices, col_indices, values, size=size, layout=bsr.layout)


# NOTE: this function will ALWAYS create a view
def tile_to_blocksize(t, blocksize):
    *rest, m, n = t.shape
    new_shape = rest + [
        m // blocksize[0],
        blocksize[0],
        n // blocksize[1],
        blocksize[1],
    ]
    # using .view instead of .reshape to ensure that the result is
    # indeed a view:
    return t.view(new_shape).transpose(-3, -2)


def scatter_mm(blocks, others, indices_data, *, accumulators=None):
    """Performs matrix multiplication of matrices scattered in batches of tensors.

    If :attr:`blocks` is a :math:`(* \times M \times K) tensor,
    :attr:`others` is a :math:`(* \times K \times N)` tensor,
    :attr:`accumulators` is a :math:`(* \times M \times N)` tensor,
    and :attr:`indices = indices_data['indices']` is a :math:`(*
    \times 3)` tensor, then the operation is equivalent to the
    following code::

      for r, p, q in indices:
          accumulators[r] += blocks[p] @ others[q]

    If :attr:`blocks` is a :math:`(* \times M \times K) tensor,
    :attr:`others` is a :math:`(BK \times BN)` tensor,
    :attr:`accumulators` is a :math:`(BM \times BN)` tensor, then the
    operation is equivalent to the following code::

      for r, p, q in indices_data['tasks4'][0]:
          r0, r1 = divmod(r, BN)
          q0, q1 = divmod(q, BN)
          accumulators[r0:r0 + M, r1:r1 + N] += blocks[p] @ others[q0:q0 + K, q1:q1 + N]

    where ``N = BN // indices_data['tasks4'][1]``.

    Parameters
    ----------
    blocks (Tensor): a 3-D tensor of first matrices to be multiplied
    others (Tensor): a 3-D tensor of second matrices to be multiplied
    indices_data (dict): a mapping to define which matrices in blocks
      participate in matrix multiplications and which matrices
      accumulate the results.

    Keyword arguments
    -----------------
    accumulators (Tensor, optional): a 3-D tensor of matrix product
      accumulators. When specified, the caller is responsible for
      initializing the accumulators, say, with zeros.

    """
    tasks2 = indices_data.get('tasks2')
    tasks5 = indices_data.get('tasks5')
    assert tasks2 is not None or tasks5 is not None

    assert blocks.ndim == 3
    P, M, K = blocks.shape

    if tasks5 is not None:
        assert others.ndim == 2
        BK, BN = others.shape
        assert BK % K == 0

        compressed_indices, r_offsets, pq_offsets, SPLIT_N = tasks5

        if accumulators is None:
            BM = M + (r_offsets.max().item() + 1) // BN
            accumulators = torch.zeros((BM, BN), dtype=blocks.dtype, device=blocks.device)
        else:
            BM, BN_ = accumulators.shape
            assert BN_ == BN

        N = BN // SPLIT_N

        if M % 16 or K % 16 or N % 16 or _scatter_mm5 is None:
            accumulators.zero_()
            for r in range(len(compressed_indices) - 1):
                g0 = compressed_indices[r]
                g1 = compressed_indices[r + 1]
                r0, r1 = divmod(r_offsets[r].item(), BN)
                acc = accumulators[r0:r0 + M, r1:r1 + N]
                for g in range(g0, g1):
                    p, q = pq_offsets[g]
                    q0, q1 = divmod(q.item(), BN)
                    acc += blocks[p] @ others[q0:q0 + K, q1:q1 + N]
        else:
            _scatter_mm5(blocks, others, compressed_indices, r_offsets, pq_offsets, SPLIT_N, accumulators)
        return accumulators

    assert others.ndim == 3
    Q, K_, N = others.shape
    assert K == K_

    if tasks2 is not None:
        if accumulators is None:
            R = tasks2[0].shape[0] - 1
            accumulators = torch.zeros((R, M, N), dtype=blocks.dtype, device=blocks.device)
        else:
            R, M_, N_ = accumulators.shape
            assert M_ == M
            assert N_ == N

        if M % 16 or K % 16 or N % 16 or _scatter_mm2 is None:
            pq_offsets, pq = tasks2
            for r in range(pq_offsets.shape[0] - 1):
                g0 = pq_offsets[r]
                g1 = pq_offsets[r + 1]
                for g in range(g0, g1):
                    p, q = pq[g]
                    accumulators[r] += blocks[p] @ others[q]
        else:
            pq_offsets, pq = tasks2
            _scatter_mm2(blocks, others, pq_offsets, pq, accumulators)
        return accumulators

    assert 0


def bsr_scatter_mm_indices_data(bsr, other, SPLIT_N=1):

    assert bsr.dense_dim() == 0
    assert bsr.ndim == 2  # no batch dims
    crow_indices = bsr.crow_indices()
    col_indices = bsr.col_indices()
    blocksize = bsr.values().shape[-2:]
    if 1:
        BM, BK = bsr.shape
        BK_, BN = other.shape
        assert BK_ == BK
        M, K = blocksize
        assert BN % SPLIT_N == 0, (BN, SPLIT_N)
        N = BN // SPLIT_N
        tasks: Dict[int, list] = {}
        for m in range(BM // M):
            r0 = crow_indices[m].item()
            r1 = crow_indices[m + 1].item()
            for n in range(SPLIT_N):
                for t in range(r1 - r0):
                    r = n * N + m * M * BN
                    p = r0 + t
                    q = n * N + col_indices[p].item() * K * BN

                    lst = tasks.get(r)
                    if lst is None:
                        lst = tasks[r] = []
                    lst.append([p, q])
        rpq_full = []

        for r in sorted(tasks):
            pq = tasks[r]
            for p, q in pq:
                rpq_full.append([r, p, q])

        compressed_indices = [0]
        r_offsets = []
        pq_offsets = []
        last_r = None
        for r, p, q in sorted(rpq_full, key=lambda item: (item[0], item[1], item[2])):
            # assume that rpq_full is sorted with respect to r
            if r == last_r:
                compressed_indices[-1] += 1
            else:
                compressed_indices.append(compressed_indices[-1] + 1)
                r_offsets.append(r)
            pq_offsets.append([p, q])
            last_r = r

        tasks5 = (torch.tensor(compressed_indices, dtype=torch.int32, device=bsr.device),
                  torch.tensor(r_offsets, dtype=torch.int32, device=bsr.device),
                  torch.tensor(pq_offsets, dtype=torch.int32, device=bsr.device),
                  SPLIT_N)

    if 1:
        M, K, N = bsr.shape[-2], bsr.shape[-1], other.shape[-1]

        BM = M // blocksize[0]
        BN = N // blocksize[0]
        BK = K // blocksize[1]
        assert (len(crow_indices) - 1) == BM

        tasks2: Dict[int, list] = {}
        for m in range(BM):
            r0 = crow_indices[m].item()
            r1 = crow_indices[m + 1].item()
            for n in range(BN):
                for t in range(r1 - r0):
                    p = r0 + t
                    q = col_indices[p].item() * BN + n
                    r = m * BN + n
                    lst = tasks2.get(r)
                    if lst is None:
                        lst = tasks2[r] = []
                    lst.append([p, q])

        pq_offsets2: List[int] = [0]
        pq_full = []
        for r in sorted(tasks2):
            pq = tasks2[r]
            pq_offsets2.append(pq_offsets2[-1] + len(pq))
            pq_full.extend(pq)

        tasks2 = (torch.tensor(pq_offsets2, dtype=torch.int32, device=crow_indices.device),
                  torch.tensor(pq_full, dtype=torch.int32, device=crow_indices.device).view((len(pq_full), 2)))

    return dict(tasks2=tasks2, tasks5=tasks5)


def bsr_scatter_mm(bsr, other, indices_data=None):
    """BSR @ strided -> strided
    """

    assert bsr.ndim == 2
    assert other.ndim == 2

    M, K, N = bsr.shape[-2], bsr.shape[-1], other.shape[1]
    blocksize = bsr.values().shape[-2:]

    if indices_data is None:
        indices_data = bsr_scatter_mm_indices_data(bsr, other)

    if bsr._nnz() == 0:
        # TFLOPS is 770
        result = torch.zeros((M, N), dtype=bsr.dtype, device=bsr.device)

    elif 'tasks5' in indices_data:
        result = torch.zeros((M, N), dtype=bsr.dtype, device=bsr.device)
        scatter_mm(bsr.values(), other, dict(tasks5=indices_data['tasks5']), accumulators=result)
    elif 'tasks2' in indices_data:
        # Using torch.zeros is equivalent to using torch.empty followed by zero_() call
        accumulators = torch.zeros((M // blocksize[0] * N // blocksize[0], blocksize[0], blocksize[0]),
                                   dtype=bsr.dtype, device=bsr.device)

        others = (other.transpose(0, 1)
                  .view(N // blocksize[0], blocksize[0], K // blocksize[1], blocksize[1])
                  .movedim((2, 0, 3, 1), (0, 1, 2, 3))  # equivalent to .transpose(1, 2).transpose(2, 3).transpose(0, 1)
                  .flatten(0, 1)  # when nnz==0, drops TFLOPS from 770 to 188
                  )  # it's a copy

        scatter_mm(bsr.values(), others, indices_data, accumulators=accumulators)

        result = (accumulators
                  .unflatten(0, (M // blocksize[0], N // blocksize[0]))
                  .movedim((0, 1, 2, 3), (2, 0, 3, 1))  # equivalent to .transpose(0, 1).transpose(2, 3).transpose(1, 2)
                  .reshape(N, M)  # drops TFLOPS from 770 to 105
                  .transpose(0, 1))  # it's a copy
    else:
        raise NotImplementedError(f'{indices_data.keys()}')

    return result


if has_triton():
    import triton
    import triton.language as tl
    from typing import Optional, Tuple

    @triton.jit
    def _sampled_addmm_kernel(
        alpha,
        beta,
        IS_BETA_ZERO: tl.constexpr,
        BLOCKSIZE_ROW: tl.constexpr,
        BLOCKSIZE_COL: tl.constexpr,
        k,
        TILE_K: tl.constexpr,
        values_ptr,
        values_batch_stride,
        values_nnz_stride,
        values_row_block_stride,
        values_col_block_stride,
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        col_indices_ptr,
        col_indices_batch_stride,
        col_indices_stride,
        mat1_ptr,
        mat1_batch_stride,
        mat1_tiled_row_stride,
        mat1_tiled_col_stride,
        mat1_row_block_stride,
        mat1_col_block_stride,
        mat2_ptr,
        mat2_batch_stride,
        mat2_tiled_row_stride,
        mat2_tiled_col_stride,
        mat2_row_block_stride,
        mat2_col_block_stride,
        acc_dtype: tl.constexpr,
        allow_tf32: tl.constexpr,
    ):
        batch_pid = tl.program_id(axis=1)
        row_block_pid = tl.program_id(axis=0)

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

        col_index_nnz_ptr = (
            col_indices_ptr
            + col_indices_batch_stride * batch_pid
            + col_indices_stride * nnz_offset
        )

        # Advance mat1 to the current tiled row, ignore columns.
        mat1_block_ptrs = (
            mat1_ptr
            + mat1_batch_stride * batch_pid
            + mat1_tiled_row_stride * row_block_pid
            + mat1_row_block_stride * row_block_arange[:, None]
        )

        # Advance mat2 in batch and block col dimension.
        mat2_block_ptrs = (
            mat2_ptr
            + mat2_batch_stride * batch_pid
            + mat2_col_block_stride * col_block_arange[None, :]
        )

        k_tile_arange = tl.arange(0, TILE_K)
        for _ in range(row_nnz):
            acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

            # find column block index
            col_block = tl.load(col_index_nnz_ptr)

            for k_tile in range(0, k, TILE_K):
                k_offsets = k_tile + k_tile_arange
                mask_k = k_offsets < k

                mat1_block = tl.load(
                    mat1_block_ptrs
                    + mat1_col_block_stride * k_offsets[None, :],
                    mask=mask_k[None, :], other=0.0
                )

                mat2_block = tl.load(
                    mat2_block_ptrs
                    + mat2_tiled_col_stride * col_block
                    + mat2_row_block_stride * k_offsets[:, None],
                    mask=mask_k[:, None], other=0.0
                )

                acc_block += tl.dot(mat1_block, mat2_block, allow_tf32=allow_tf32, out_dtype=acc_dtype)

            if IS_BETA_ZERO:
                acc_block *= alpha
            else:
                acc_block = alpha * acc_block + beta * tl.load(values_block_ptrs)

            # write result
            tl.store(values_block_ptrs, acc_block.to(values_ptr.dtype.element_ty))

            # advance val/col_index ptrs to the next block in the row.
            values_block_ptrs += values_nnz_stride
            col_index_nnz_ptr += col_indices_stride

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
            output_acc_block += tl.dot(values_block, dense_block, allow_tf32=allow_tf32, out_dtype=acc_dtype)

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


    def _run_sampled_addmm_kernel(
        alpha, beta, is_beta_zero,
        blocksize, k, tile_k,
        values, crow_indices, col_indices,
        mat1, mat2,
        max_grid
    ):
        n_batches = values.size(0)
        n_block_rows = crow_indices.size(-1) - 1

        full_grid = (n_batches, n_block_rows)
        if max_grid is not None:
            grid_blocks = tuple(max_grid[:2][::-1]) + (None,) * (2 - len(max_grid[:2]))
        else:
            grid_blocks = None
        tensor_dims_map = {
            values: (0, None),
            crow_indices: (0, -1),
            col_indices: (0, None),
            mat1: (0, -4),
            mat2: (0, None),
        }
        if values.dtype in (torch.half, torch.bfloat16):
            acc_dtype = tl.float32
            allow_tf32 = True
        else:
            acc_dtype = tl.float64
            allow_tf32 = False

        def kernel(grid, *sliced_tensors):
            _sampled_addmm_kernel[grid](
                alpha, beta, is_beta_zero,
                *blocksize, k, tile_k,
                *ptr_stride_extractor(*sliced_tensors),
                acc_dtype=acc_dtype,
                allow_tf32=allow_tf32,
                num_stages=1,
                num_warps=4
            )

        launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)


    def sampled_addmm(
        input: torch.Tensor,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        *,
        beta=1.0,
        alpha=1.0,
        out: Optional[torch.Tensor] = None,
        skip_checks: bool = False,
        max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
    ):
        f_name = "sampled_addmm"

        check_bsr_layout(f_name, input)
        input_broadcasted = broadcast_batch_dims_bsr(f_name, input, mat1, mat2)

        if not skip_checks:
            check_device(f_name, mat1, input.device)
            check_device(f_name, mat2, input.device)
            if beta != 0.0 and input.dtype is torch.bool:
                check(
                    False,
                    f"{f_name}(): having beta == {beta} not equal to 0.0 with boolean mask is not allowed."
                )
            if input.dtype is not torch.bool:
                check_dtype(f_name, mat1, input.dtype)
                check_dtype(f_name, mat2, input.dtype)
            else:
                check_dtype(f_name, mat1, mat2.dtype)
            check_mm_compatible_shapes(f_name, mat1, mat2)
            if out is not None:
                check_bsr_layout(f_name, out)
                check_device(f_name, out, mat1.device)
                check_dtype(f_name, out, input.dtype)
                check(
                    out.shape == input_broadcasted.shape
                    and out._nnz() == input._nnz(),
                    f"{f_name}(): Expects `out` to be of shape {input_broadcasted.shape} "
                    f"and with nnz equal to {input_broadcasted._nnz()} "
                    f"but got out.shape = {out.shape} and out.nnz = {out._nnz()}"
                )

        if out is None:
            out = input_broadcasted.to(mat1.dtype, copy=True)
        else:
            out.copy_(input_broadcasted)

        if out.numel() == 0 or out._nnz() == 0:
            return out

        blocksize = out.values().shape[-2:]
        m = mat1.size(-2)
        n = mat2.size(-1)
        k = mat1.size(-1)

        # NOTE: (m, 0) @ (0, n) == zeros(m, n)
        if alpha == 0.0 or k == 0:
            out.values().mul_(beta)
            return out

        # prepare inputs by reshaping them to be kernel-compatible
        out_backup = out
        crow_indices, col_indices, values, mat1, mat2 = prepare_inputs(out, mat1, mat2)

        mat1 = tile_to_blocksize(mat1, (blocksize[0], k))
        mat2 = tile_to_blocksize(mat2, (k, blocksize[1]))
        tile_k = max(*blocksize)

        _run_sampled_addmm_kernel(
            alpha, beta, beta == 0.0,
            blocksize, k, tile_k,
            values, crow_indices, col_indices,
            mat1, mat2,
            max_grid
        )

        # If nnz x block strides are not the same in out_backup.values and values,
        # it means that out_backup.values and values are not the views of each other,
        # so we have to copy.
        if out_backup.values().stride()[-3:] != values.stride()[-3:]:
            out_backup.values().copy_(values.reshape(out_backup.values().shape))
        return out_backup


    def bsr_dense_mm(
        bsr: torch.Tensor,
        dense: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        skip_checks: bool = False,
        max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
    ):
        f_name = "bsr_dense_mm"
        if not skip_checks:
            check_bsr_layout(f_name, bsr)
            check_device(f_name, bsr, dense.device)
            check_dtype(f_name, bsr, dense.dtype)
            check_mm_compatible_shapes(f_name, bsr, dense)

            m = bsr.size(-2)
            n = dense.size(-1)
            row_block, col_block = bsr.values().shape[-2:]
            check(
                not n % row_block,
                f"bsr_dense_mm(): dense.size(-1) == {n} should be divisible by "
                f"blocksize[0] == {row_block}.",
            )
            check_blocksize(f_name, (row_block, col_block))
        else:
            m, kl = bsr.shape[-2:]
            kr, n = dense.shape[-2:]

        original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)

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
            out = dense.new_empty(original_batch_dims_broadcasted + (m, n))

        # Short circuit if lhs is zero
        if bsr._nnz() == 0:
            return out.zero_()

        blocksize = bsr.values().shape[-2:]

        # NOTE: out is contiguous, so prepare_inputs will create a view.
        # out gets modified in-place, so we store a backup copy.
        out_backup = out

        # prepare inputs by reshaping them to be kernel-compatible.
        crow_indices, col_indices, values, dense, out = prepare_inputs(bsr, dense, out)

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
        _run_dense_rowspace_kernel(blocksize, values, crow_indices, col_indices, dense, out, max_grid)

        return out_backup


    @triton.jit
    def _bsr_softmax_kernel(
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        values_ptr,
        values_batch_stride,
        values_row_block_stride,
        values_nnz_col_block_stride,
        row_block, col_block,
        MAX_ROW_NNZ: tl.constexpr,
        TILE: tl.constexpr
    ):
        batch_pid = tl.program_id(axis=2)
        row_block_offset_pid = tl.program_id(axis=1)
        row_block_pid = tl.program_id(axis=0)

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

        row_arange = tl.arange(0, TILE)
        mask = row_arange < row_nnz * col_block

        curr_row_values_ptrs = (
            values_ptr
            + values_batch_stride * batch_pid
            + values_row_block_stride * row_block_offset_pid
            + nnz_offset * col_block
        )

        # find max in the row
        row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
        max_row_value = tl.max(row_tile, axis=0)
        for _ in range(TILE, MAX_ROW_NNZ, TILE):
            row_arange += TILE
            mask = row_arange < row_nnz * col_block
            row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
            curr_max_row_value = tl.max(row_tile, axis=0)
            max_row_value = tl.where(max_row_value > curr_max_row_value, max_row_value, curr_max_row_value)

        # find denominator for stable softmax
        num = tl.exp(row_tile - max_row_value)
        denom = tl.sum(num, axis=0)
        for _ in range(TILE, MAX_ROW_NNZ, TILE):
            row_arange -= TILE
            mask = row_arange < row_nnz * col_block
            row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
            num = tl.exp(row_tile - max_row_value)
            denom += tl.sum(num, axis=0)

        # populate output
        tl.store(curr_row_values_ptrs + row_arange, (num / denom).to(values_ptr.dtype.element_ty), mask=mask)
        for _ in range(TILE, MAX_ROW_NNZ, TILE):
            row_arange += TILE
            mask = row_arange < row_nnz * col_block
            row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
            num = tl.exp(row_tile - max_row_value)
            tl.store(curr_row_values_ptrs + row_arange, (num / denom).to(values_ptr.dtype.element_ty), mask=mask)


    def bsr_softmax(input, max_row_nnz=None):
        f_name = "bsr_softmax"

        check_bsr_layout(f_name, input)
        check_dtype(f_name, input, input.dtype)

        if input._nnz() == 0 or input.numel() == 0:
            return input.clone()

        m, n = input.shape[-2:]
        nnz = input._nnz()
        row_block, col_block = input.values().shape[-2:]

        if max_row_nnz is None:
            max_row_nnz = triton.next_power_of_2(n)
        else:
            max_row_nnz = triton.next_power_of_2(max_row_nnz)

        crow_indices = input.crow_indices().unsqueeze(0).flatten(0, -2)
        # reshape values from
        # (b1, ..., bn, nnz, row_block, col_block) to
        # (b1 * ... * bn, row_block, nnz * col_block).
        # This simplifies batch dim manipulation and unlocks
        # the possibility to access all nnzs in any given row.
        if input.values().transpose(-3, -2).is_contiguous():
            # Need to clone to avoid `contiguous` returning a view.
            values = input.values().clone()
        else:
            values = input.values()
        values = values.transpose(-3, -2).contiguous().unsqueeze(0).flatten(0, -4).reshape(-1, row_block, nnz * col_block)
        full_grid = (values.shape[0], row_block, m // row_block)
        grid_blocks = None
        tensor_dims_map = {
            # We span nnz number of blocks, not nnz + 1,
            # hence crow_indices[..., :-1]
            crow_indices[..., :-1]: (0, None, -1),
            values: (0, None, None),
        }

        def kernel(grid, *sliced_tensors):
            _bsr_softmax_kernel[grid](
                *ptr_stride_extractor(*sliced_tensors),
                row_block, col_block,
                max_row_nnz,
                # Triton's max numel is bounded by 2 ** 17.
                min(2 ** 17, max_row_nnz)
            )

        launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)

        values = values.reshape(-1, row_block, nnz, col_block).transpose(-3, -2).reshape(*input.values().shape)

        return torch.sparse_compressed_tensor(
            input.crow_indices().clone(),
            input.col_indices().clone(),
            values,
            size=input.shape,
            layout=input.layout
        )

    def _scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None
    ):
        f_name = "_scaled_dot_product_attention"
        check(
            not is_causal,
            f"{f_name}(): is_causal == True is not supported."
        )
        check(
            attn_mask is not None,
            f"{f_name}(): attn_mask == None is not supported."
        )
        assert attn_mask is not None

        check(
            attn_mask.layout == torch.sparse_bsr,
            f"{f_name}(): "
            f"attn_mask.layout must be {torch.sparse_bsr}, but got "
            f"attn_mask.layout == {attn_mask.layout}."
        )

        check_device(f_name, key, query.device)
        check_device(f_name, value, query.device)
        check_device(f_name, attn_mask, query.device)

        check_dtype(f_name, key, query.dtype)
        check_dtype(f_name, value, query.dtype)
        if attn_mask.dtype is not torch.bool:
            check_dtype(f_name, attn_mask, query.dtype)

        sdpa = sampled_addmm(attn_mask, query, key.transpose(-2, -1), beta=0.0, skip_checks=False)
        if scale is None and query.size(-1) == 0 or scale == 0.0:
            check(
                False,
                f"{f_name}(): current value of scale == {scale} "
                "results in division by zero."
            )
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        sdpa.values().mul_(scale_factor)
        sdpa = bsr_softmax(sdpa)
        torch.nn.functional.dropout(sdpa.values(), p=dropout_p, inplace=True)
        sdpa = bsr_dense_mm(sdpa, value)
        return sdpa

    @triton.jit
    def _scatter_mm2_kernel(
            M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
            blocks_ptr, blocks_stride_P, blocks_stride_M, blocks_stride_K,
            others_ptr, others_stride_Q, others_stride_K, others_stride_N,
            accumulators_ptr, accumulators_stride_R, accumulators_stride_M, accumulators_stride_N,
            pq_offsets_ptr, pq_offsets_stride,
            pq_ptr, pq_stride_T, pq_stride_1,
            dot_out_dtype: tl.constexpr,
            TILE_M: tl.constexpr,
            TILE_N: tl.constexpr):

        BLOCK_M = M // TILE_M
        BLOCK_N = N // TILE_N

        pid_t = tl.program_id(axis=0)

        pid = tl.program_id(axis=1)
        pid_m = pid // BLOCK_M
        pid_n = pid % BLOCK_M

        rm = (pid_m * TILE_M + tl.arange(0, TILE_M))
        rn = (pid_n * TILE_N + tl.arange(0, TILE_N))
        rk = tl.arange(0, K)

        A_ptr = blocks_ptr + (rm[:, None] * blocks_stride_M + rk[None, :] * blocks_stride_K)
        B_ptr = others_ptr + (rk[:, None] * others_stride_K + rn[None, :] * others_stride_N)

        g0 = tl.load(pq_offsets_ptr + pid_t * pq_offsets_stride)
        g1 = tl.load(pq_offsets_ptr + (pid_t + 1) * pq_offsets_stride)

        if g0 == g1:
            return

        acc_block = tl.zeros((TILE_M, TILE_N), dtype=dot_out_dtype)

        for i in range(g0, g1):
            p = tl.load(pq_ptr + i * pq_stride_T)
            q = tl.load(pq_ptr + i * pq_stride_T + pq_stride_1)
            A = tl.load(A_ptr + p * blocks_stride_P)
            B = tl.load(B_ptr + q * others_stride_Q)
            acc_block += tl.dot(A, B, out_dtype=dot_out_dtype)

        C_ptr = accumulators_ptr + pid_t * accumulators_stride_R + (
            rm[:, None] * accumulators_stride_M + rn[None, :] * accumulators_stride_N)
        tl.store(C_ptr, acc_block.to(accumulators_ptr.dtype.element_ty))

    def _scatter_mm2(
            blocks: torch.Tensor,
            others: torch.Tensor,
            pq_offsets: torch.Tensor,
            pq_indices: torch.Tensor,
            accumulators: torch.Tensor
    ):
        P, M, K = blocks.shape
        Q, _, N = others.shape
        R, _, _ = accumulators.shape

        meta = dict(TILE_M=max(16, M // 4), TILE_N=max(16, N // 4), num_stages=1, num_warps=2)  # TODO: optimize

        def grid(META):
            return (pq_offsets.shape[0] - 1, triton.cdiv(M, META['TILE_M']) * triton.cdiv(N, META['TILE_N']), 1)

        dot_out_dtype = {torch.float16: tl.float32,
                         torch.bfloat16: tl.float32,
                         torch.float32: tl.float64,
                         torch.float64: tl.float64}[accumulators.dtype]

        _scatter_mm2_kernel[grid](
            M, K, N,
            blocks, blocks.stride(0), blocks.stride(1), blocks.stride(2),
            others, others.stride(0), others.stride(1), others.stride(2),
            accumulators, accumulators.stride(0), accumulators.stride(1), accumulators.stride(2),
            pq_offsets, pq_offsets.stride(0),
            pq_indices, pq_indices.stride(0), pq_indices.stride(1),
            dot_out_dtype=dot_out_dtype,
            **meta
        )



    @triton.jit
    def _scatter_mm5_kernel(
            M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, BN: tl.constexpr,
            blocks_ptr, blocks_stride_P, blocks_stride_M, blocks_stride_K,
            others_ptr, others_stride_K, others_stride_N,
            accumulators_ptr, accumulators_stride_M, accumulators_stride_N,
            compressed_indices_ptr, compressed_indices_stride,
            r_offsets_ptr, r_offsets_stride,
            pq_offsets_ptr, pq_offsets_stride_T, pq_offsets_stride_1,
            dot_out_dtype: tl.constexpr,
            TILE_M: tl.constexpr,
            TILE_N: tl.constexpr):

        pid_t = tl.program_id(axis=0)

        BLOCK_M = M // TILE_M
        BLOCK_N = N // TILE_N
        pid = tl.program_id(axis=1)
        pid_m = pid // BLOCK_N
        pid_n = pid % BLOCK_N

        rm = (pid_m * TILE_M + tl.arange(0, TILE_M))
        rn = (pid_n * TILE_N + tl.arange(0, TILE_N))
        rk = tl.arange(0, K)

        A_ptr = blocks_ptr + (rm[:, None] * blocks_stride_M + rk[None, :] * blocks_stride_K)
        B_ptr = others_ptr + (rk[:, None] * others_stride_K + rn[None, :] * others_stride_N)

        acc_block = tl.zeros((TILE_M, TILE_N), dtype=dot_out_dtype)

        g0 = tl.load(compressed_indices_ptr + pid_t * compressed_indices_stride)
        g1 = tl.load(compressed_indices_ptr + (pid_t + 1) * compressed_indices_stride)

        if g0 == g1:
            return

        for i in range(g0, g1):
            p = tl.load(pq_offsets_ptr + i * pq_offsets_stride_T)
            q = tl.load(pq_offsets_ptr + i * pq_offsets_stride_T + pq_offsets_stride_1)

            A = tl.load(A_ptr + p * blocks_stride_P)
            q0 = q // BN
            q1 = q % BN
            B = tl.load(B_ptr + others_stride_K * q0 + others_stride_N * q1)
            acc_block += tl.dot(A, B, out_dtype=dot_out_dtype)

        r = tl.load(r_offsets_ptr + pid_t * r_offsets_stride)
        r0 = r // BN
        r1 = r % BN
        C_ptr = accumulators_ptr + (accumulators_stride_M * r0 + accumulators_stride_N * r1) + (
            rm[:, None] * accumulators_stride_M + rn[None, :] * accumulators_stride_N)
        tl.store(C_ptr, acc_block.to(accumulators_ptr.dtype.element_ty))

    def _scatter_mm5(
            blocks: torch.Tensor,
            others: torch.Tensor,
            compressed_indices: torch.Tensor,
            r_offsets: torch.Tensor,
            pq_offsets: torch.Tensor,
            SPLIT_N: int,
            accumulators: torch.Tensor
    ):
        P, M, K = blocks.shape
        BK, BN = others.shape
        BM, BN_ = accumulators.shape
        assert BN_ == BN
        N = BN // SPLIT_N

        meta = dict(TILE_M=max(16, M // 1), TILE_N=max(16, N // 32), num_stages=1, num_warps=8)
        meta = dict(TILE_M=max(16, M // 2), TILE_N=max(16, N // 8), num_stages=1, num_warps=8 // 2)
        meta = dict(TILE_M=max(16, M // 2), TILE_N=max(16, N // 32), num_stages=1, num_warps=8 // 2)
        meta = dict(TILE_M=M, TILE_N=N, num_stages=1, num_warps=8 // 2)
        meta = dict(TILE_M=min(64, M), TILE_N=min(64, N), num_stages=1, num_warps=2)

        def grid(META):
            return (r_offsets.shape[0], triton.cdiv(M, META['TILE_M']) * triton.cdiv(N, META['TILE_N']), 1)

        dot_out_dtype = {torch.float16: tl.float32,
                         torch.bfloat16: tl.float32,
                         torch.float32: tl.float64,
                         torch.float64: tl.float64}[accumulators.dtype]
        _scatter_mm5_kernel[grid](
            M, K, N, BN,
            blocks, blocks.stride(0), blocks.stride(1), blocks.stride(2),
            others, others.stride(0), others.stride(1),
            accumulators, accumulators.stride(0), accumulators.stride(1),
            compressed_indices, compressed_indices.stride(0),
            r_offsets, r_offsets.stride(0),
            pq_offsets, pq_offsets.stride(0), pq_offsets.stride(1),
            dot_out_dtype=dot_out_dtype,
            **meta
        )
else:
    bsr_softmax = None  # type: ignore[assignment]
    bsr_dense_mm = None  # type: ignore[assignment]
    sampled_addmm = None  # type: ignore[assignment]
    _scaled_dot_product_attention = None  # type: ignore[assignment]
    _scatter_mm2 = None  # type: ignore[assignment]
    _scatter_mm5 = None  # type: ignore[assignment]
