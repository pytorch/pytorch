import torch


# This is PyTorch implementation of main part of reorder_meta()
# function, from tools/util/include/cutlass/util/host_reorder.h file
# of CUTLASS source tree.  Furthermore, CUTLASS template for sparse
# GEMM decides upon layout of this matrix, and at the moment for the
# sparse GEMM executed on tensor cores, this is layout described by
# ColumnMajorInterleaved<2> data structure, in
# include/cutlass/layout/matrix.h of CUTLASS source tree.  The
# reordering of meta matrix into meta_reordered matrix calculated
# according to these segments of CUTLASS code is re-implemented here.
# Note that this calculation produces offsets for scattering metadata
# matrix elements into reordered metadata matrix elements (or,
# equivalently, for gathering reordered metadata matrix element back
# into metadata matrix elements).
def _calculate_meta_reordering_scatter_offsets(m, meta_ncols, meta_dtype, device):
    dst_rows = torch.arange(0, m, device=device)[:, None].repeat(1, meta_ncols)
    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(m, 1)

    # Reorder the rows, then swizzle the 2x2 blocks.
    group = 32 if meta_dtype.itemsize == 2 else 16
    interweave = 4 if meta_dtype.itemsize == 2 else 2
    dst_rows = (
        dst_rows // group * group
        + (dst_rows % 8) * interweave
        + (dst_rows % group) // 8
    )

    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    dst_rows += topright - bottomleft
    dst_cols -= topright - bottomleft

    # Assumed that meta tensor is to be stored in CUTLASS
    # InterleavedColumnMajor layout, and reverse engineered
    # corresponding code to store values into this tensor.
    interleave = 2
    cols_maj = dst_cols // interleave
    cols_min = dst_cols % interleave
    return (cols_maj * m * interleave + dst_rows * interleave + cols_min).view(-1)


# This function converts dense matrix into sparse semi-structured
# representation, producing "compressed" matrix, in the layout used by
# CUTLASS backend, and corresponding metadata matrix.
def sparse_semi_structured_from_dense_cutlass(dense):
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"
        )

    m, k = dense.shape
    device = dense.device

    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16, torch.float]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError("Invalid number of elements per meta element calculated")

    if meta_dtype == torch.int32:
        if m % 16 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 16"
            )
    else:
        if m % 32 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 32"
            )
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}"
        )

    if dense.dtype != torch.float:
        ksparse = 4
        dense_4 = dense.view(-1, k // ksparse, ksparse)
        m0, m1, m2, m3 = (dense_4 != 0).unbind(-1)
    else:
        ksparse = 2
        dense_2 = dense.view(-1, k // ksparse, ksparse)
        m0, m2 = m1, m3 = (dense_2 != 0).unbind(-1)
    meta_ncols = k // (ksparse * quadbits_per_meta_elem)

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
    #     [False, True,  False, False] -> 0b1001
    #     [False, True,  True,  True ] -> 0b1101
    #     [True,  False, False, False] -> 0b1000
    #     [True,  False, True,  True ] -> 0b1100
    #     [True,  True,  False, True ] -> 0b0100
    #     [True,  True,  True,  False] -> 0b0100
    #     [True,  True,  True,  True ] -> 0b0100
    # These particular encodings are chosen, with the help of Espresso
    # logic minimizer software, for the purpose of minimization of
    # corresponding Boolean functions, that translate non-zero flags
    # into encoding bits.  Note also possible choices for the first
    # and last of these encodings were limited only to (0b0100,
    # 0b1110), in order to produce valid encodings for 1:2 sparsity
    # case.

    expr0 = m0 & m1
    expr1 = ~m0 & m1
    expr2 = ~m0 & ~m1
    bit0 = expr1
    bit1 = expr2
    bit2 = expr0 | expr2 | m3
    bit3 = expr1 | ~m1
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)

    if dense.dtype != torch.float:
        sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))  # type: ignore[possibly-undefined]
        sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
        sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    else:
        sparse = dense_2.gather(-1, idxs0.unsqueeze(-1) // 2).view(m, k // 2)  # type: ignore[possibly-undefined]

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

    # Reorder meta tensor elements.
    meta_reordered = meta.new_empty((m * meta_ncols,))  # type: ignore[possibly-undefined]
    meta_offsets = _calculate_meta_reordering_scatter_offsets(
        m, meta_ncols, meta_dtype, device
    )
    meta_reordered.scatter_(0, meta_offsets, meta.view(-1))

    return (sparse, meta_reordered.view(m, meta_ncols))


# This function performs reverse of the function above - it
# reconstructs dense matrix from a pair of "compressed" matrix, given
# in the layout used by CUTLASS backend, and accompanying metadata
# matrix.
def sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered):
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

    if sparse.dtype != torch.float:
        ksparse = 4
    else:
        ksparse = 2

    meta_nrows, meta_ncols = meta_reordered.shape
    if meta_nrows != m:
        raise RuntimeError(
            f"Number of rows of meta matrix {meta_nrows} must be equal to number of columns of spase matrix {m}"
        )
    if meta_ncols * ksparse * quadbits_per_meta_elem != 2 * k:
        raise RuntimeError(
            f"Number of columns of sparse matrix {k} different from the {meta_ncols * ksparse * quadbits_per_meta_elem // 2}, "
            "expected according to the number of columns of meta matrix"
        )

    # Undo meta tensor elements reordering.
    meta_offsets = _calculate_meta_reordering_scatter_offsets(
        m, meta_ncols, meta_dtype, device
    )
    meta = torch.gather(meta_reordered.view(-1), 0, meta_offsets).view(m, meta_ncols)

    # Unpack sparse tensor back to original dense tensor, using
    # information provided by meta tensor.  Note that torch.float
    # datatype is handled pretty much the same as
    # torch.half/torch.bfloat16, as metadata for a pair of torch.float
    # value is encoded as if underlying 8 bytes contain four
    # torch.half/torch.bfloat16 values, where either first two or last
    # two are zeros.
    meta_2 = torch.empty(
        (m, meta_ncols, 2 * quadbits_per_meta_elem),
        dtype=meta_dtype,
        device=device,
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
        torch.arange(0, 2 * m * k // ksparse, device=device) * 4
    ).view(-1, 1).repeat(1, 2).view(-1)

    dense = torch.zeros((m * 2 * k,), dtype=sparse.dtype, device=device)
    if sparse.dtype != torch.float:
        dense.scatter_(0, dense_offsets, sparse.view(-1))
    else:
        dense.view(torch.half).scatter_(
            0, dense_offsets, sparse.view(torch.half).view(-1)
        )

    return dense.view(m, 2 * k)
