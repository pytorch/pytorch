import torch


def sparse_semi_structured_from_dense(dense):
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"
        )

    m, k = dense.shape
    device = dense.device

    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError("Invalid number of elements per meta element calculated")

    if m % 32 != 0:
        raise RuntimeError(
            f"Number rows columns of dense matrix {m} must be divisible by 32"
        )
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}"
        )
    meta_ncols = k // (4 * quadbits_per_meta_elem)

    dense_4 = dense.view(-1, k // 4, 4)
    m0, m1, m2, m3 = (dense_4 != 0).unbind(-1)

    # Encoding quadruples of True/False values as follows:
    #     [True,  True,  False, False] -> 0b0100
    #     [True,  False, True,  False] -> 0b1000
    #     [False, True,  True,  False] -> 0b1001
    #     [True,  False, False, True ] -> 0b1100
    #     [False, True,  False, True ] -> 0b1101
    #     [False, False, True,  True ] -> 0b1110
    # Thus, lower two bits in the encoding are index of the True value
    # at the lowest index in the quadruple, and the higher two bits in
    # the encoding are index of the other True value in the quadruple.
    # In case there are less than two True values, than False value or
    # values at some index or indices are considered True for the
    # encoding.  In case there are more than two True values, then the
    # excess True value(s) at some indices are considered False for
    # the encoding.  The exact encodings used for these cases are as
    # follows:
    #     [False, False, False, False] -> 0b1110
    #     [False, False, False, True ] -> 0b1110
    #     [False, False, True,  False] -> 0b1110
    #     [False, True,  False, False] -> 0b1101
    #     [False, True,  True,  True ] -> 0b1001
    #     [True,  False, False, False] -> 0b1100
    #     [True,  False, True,  True ] -> 0b1000
    #     [True,  True,  False, True ] -> 0b0100
    #     [True,  True,  True,  False] -> 0b1000
    #     [True,  True,  True,  True ] -> 0b1000
    # These particular encodings are chosen, with the help of Espresso
    # logic minimizer software, for the purpose of minimization of
    # corresponding Boolean functions, that translate non-zero flags
    # into encoding bits.

    bit0 = ~m0 & m1
    bit1 = ~m0 & ~m1
    bit2 = bit1 | ~m2
    bit3 = bit0 | ~m1 | m2
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)

    sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))
    sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
    sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)

    meta_4 = idxs0 | (idxs1 << 2)
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)

    if quadbits_per_meta_elem == 4:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
        )
    elif quadbits_per_meta_elem == 8:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
            | (meta_n[:, :, 4] << 16)
            | (meta_n[:, :, 5] << 20)
            | (meta_n[:, :, 6] << 24)
            | (meta_n[:, :, 7] << 28)
        )

    # Metadata values are now to be reshuffled in a way given in
    # reorder_meta() function, in
    # tools/util/include/cutlass/util/host_reorder.h file of CUTLASS
    # source tree.  Furthermore, CUTLASS template for sparse GEMM
    # decides upon layout of this matrix, and at the moment for the
    # sparse GEMM executed on tensor cores, this is layout described
    # by ColumnMajorInterleaved<2> data structure, in
    # include/cutlass/layout/matrix.h of CUTLASS source tree.  The
    # reordering of meta matrix into meta_reordered matrix calculated
    # according to these segments of CUTLASS code is given below.
    # However, this calculation produces offsets for scatter access
    # from metadata matrix to redordered metadata matrix, and gather
    # pattern is more efficient.  For this reason, the scatter offsets
    # are reverted and printed, through enabling commented block at
    # the end of following code.  Resulting gather offsets are then
    # analyzed, on several (m, k) value pairs (in particular: (32,
    # 128), (32, 256), (64, 128) and (64, 256)), and the code that
    # follows this comment is written to reproduce these gather offsets.
    #
    #    dst_rows = torch.arange(0, m, device=device)[:, None].repeat(1, meta_ncols)
    #    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(m, 1)
    #
    #    # Reorder the rows, then swizzle the 2x2 blocks.
    #    group = 32 if meta_dtype.itemsize == 2 else 16
    #    interweave = 4 if meta_dtype.itemsize == 2 else 2
    #    dst_rows = (
    #        dst_rows // group * group
    #        + (dst_rows % 8) * interweave
    #        + (dst_rows % group) // 8
    #    )
    #
    #    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    #    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    #    dst_rows += topright - bottomleft
    #    dst_cols -= topright - bottomleft
    #
    #    # Assumed that meta tensor is to be stored in CUTLASS
    #    # InterleavedColumnMajor layout, and reverse engineered
    #    # corresponding code to store values into this tensor.
    #    interleave = 2
    #    cols_maj = dst_cols // interleave
    #    cols_min = dst_cols % interleave
    #    meta_reordered_offsets = (
    #        cols_maj * m * interleave + dst_rows * interleave + cols_min
    #    )
    #
    #    meta_reordered = torch.empty((m, meta_ncols), dtype=meta_dtype, device=device)
    #    meta_reordered.view(-1)[meta_reordered_offsets.view(-1)] = meta.view(-1)
    #
    #    # Uncomment to have gather pattern for meta_reordered printed
    #    #
    #    #offsets = torch.empty(
    #    #    (m, meta_ncols), dtype=meta_reordered_offsets.dtype, device=device
    #    #)
    #    #offsets.view(-1)[meta_reordered_offsets.view(-1)] = torch.arange(
    #    #    0, m * meta_ncols, dtype=meta_reordered_offsets.dtype, device=device
    #    #)
    #    #torch.set_printoptions(threshold=1000000)
    #    #print("------------------------------------------------------------")
    #    #print("dtype =", dtype, ", m =", m, ", k =", k, ", meta_ncols =", meta_ncols)
    #    #print(offsets.view(-1))
    #

    # No point to try to understand this code: as mentioned in the
    # comment above it is written to reproduce gather offsets, as
    # these would be calculated by CUTLASS, and to be efficient, but
    # it contains several magic values and magic calculations that
    # make it rather hard to read, let alone understand.
    if meta_dtype == torch.int32:
        magic0 = 4
        magic1 = 32
        magic2 = 16
        magic3 = k // 2
        magic4 = [0, k // 4, 1, k // 4 + 1]
    elif meta_dtype == torch.int16:
        magic0 = 8
        magic1 = 64
        magic2 = 32
        magic3 = 2 * k
        magic4 = [0, k // 2, 1, k // 2 + 1, k, 3 * k // 2, k + 1, 3 * k // 2 + 1]
    tmp0 = torch.zeros(m * meta_ncols, dtype=torch.int64, device=device)
    tmp1 = (
        tmp0.view(meta_ncols // 2, -1)
        + torch.arange(0, meta_ncols, 2, device=device).view(meta_ncols // 2, 1)
    ).view(-1, magic1)
    tmp2 = (
        (
            torch.arange(0, 8, device=device).view(-1, 1)
            * torch.ones((magic0,), dtype=torch.int64, device=device)
            * meta_ncols
        )
        .view(-1)
        .repeat(m * meta_ncols // magic1)
        .view(-1, magic1)
    )
    tmp3 = (torch.arange(0, m // magic2, device=device).view(-1, 1) * magic3).repeat(
        meta_ncols // 2, magic1
    )
    tmp4 = torch.tensor(magic4, device=device).repeat(tmp3.shape[0], 8)
    meta_offsets = tmp1 + tmp2 + tmp3 + tmp4

    meta_reordered = torch.gather(meta.view(-1), 0, meta_offsets.view(-1)).view(
        m, meta_ncols
    )
    return (sparse, meta_reordered)


def sparse_semi_structured_to_dense(sparse, meta_reordered):
    if sparse.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor"
        )

    m, k = sparse.shape
    device = sparse.device

    if meta_reordered.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor"
        )
    if meta_reordered.device != device:
        raise RuntimeError(
            f"Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device"
        )

    meta_dtype = meta_reordered.dtype
    if meta_dtype not in (torch.int16, torch.int32):
        raise RuntimeError(f"Invalid datatype {meta_dtype} of meta matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4

    meta_nrows, meta_ncols = meta_reordered.shape
    if meta_nrows != m:
        raise RuntimeError(
            f"Number of rows of meta matrix {meta_nrows} must be equal to number of columns of spase matrix {m}"
        )
    if meta_ncols * 4 * quadbits_per_meta_elem != 2 * k:
        raise RuntimeError(
            f"Number of columns of sparse matrix {k} different from the {meta_ncols * 4 * quadbits_per_meta_elem // 2}, "
            "expected according to the number of columns of meta matrix"
        )

    if meta_dtype == torch.int32:
        magic0 = 4
        magic1 = [0, 1, 32, 33]
    elif meta_dtype == torch.int16:
        magic0 = 8
        magic1 = [0, 1, 4, 5]
    tmp1 = torch.tensor([0, 2], dtype=torch.int64, device=device).repeat(
        meta_nrows, meta_ncols // 2
    )
    tmp2 = (
        (torch.arange(0, meta_ncols // 2, device=device) * 2 * meta_nrows)
        .view(-1, 1)
        .repeat(1, 2)
        .view(-1)
        .repeat(m, 1)
    )
    tmp3 = (
        (torch.arange(0, 8, device=device) * magic0)
        .view(-1, 1)
        .repeat(m // 8, meta_ncols)
    )
    tmp4 = (
        torch.tensor(magic1, device=device)
        .view(-1, 1)
        .repeat(1, 8 * meta_ncols)
        .repeat(meta_nrows // 32, 1)
        .view(meta_nrows, meta_ncols)
    )
    tmp5 = (
        (torch.arange(0, meta_nrows // 32, device=device) * 64)
        .view(-1, 1)
        .repeat(1, 32 * meta_ncols)
        .view(meta_nrows, meta_ncols)
    )
    meta_offsets = tmp1 + tmp2 + tmp3 + tmp4 + tmp5

    meta = torch.gather(meta_reordered.view(-1), 0, meta_offsets.view(-1)).view(
        m, meta_ncols
    )

    meta_2 = torch.empty(
        (m, meta_ncols, 2 * quadbits_per_meta_elem), dtype=meta_dtype, device=device
    )
    if quadbits_per_meta_elem == 4:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
    elif quadbits_per_meta_elem == 8:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
        meta_2[:, :, 8] = (meta >> 16) & 0b11
        meta_2[:, :, 9] = (meta >> 18) & 0b11
        meta_2[:, :, 10] = (meta >> 20) & 0b11
        meta_2[:, :, 11] = (meta >> 22) & 0b11
        meta_2[:, :, 12] = (meta >> 24) & 0b11
        meta_2[:, :, 13] = (meta >> 26) & 0b11
        meta_2[:, :, 14] = (meta >> 28) & 0b11
        meta_2[:, :, 15] = (meta >> 30) & 0b11

    dense_offsets = meta_2.view(-1) + (
        torch.arange(0, m * k // 2, device=device) * 4
    ).view(-1, 1).repeat(1, 2).view(-1)

    dense = torch.zeros((m * 2 * k,), dtype=sparse.dtype, device=device)
    dense.scatter_(0, dense_offsets, sparse.view(-1))

    return dense.view(m, 2 * k)
