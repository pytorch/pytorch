import torch
from .semi_structured_sparse_tensor import SemiStructuredSparseTensor
import random


def gen_semi_structured_sparse_mask(r, c, dtype=torch.float16, device="cuda"):
    """
    This function returns a 1:2 sparse matrix of size (r, c).
    Note that this means this matrix will also be 2:4 and 4:8 sparse as well.
    """

    choices = [[0, 1], [1, 0]]

    mask_entries = [random.choice(choices) for i in range(r * c // 2)]
    # print(mask_entries)

    return (
        torch.tensor(mask_entries, dtype=dtype, device=device)
        .reshape(r, c)
        .contiguous()
    )


def is_semi_structured_sparse(tensor: torch.Tensor, zeros_per_block=2):
    """
    return wether a tensor is semi_structured sparse
    """

    if not tensor.is_contiguous():
        raise Exception("Tensor is not contiguous")

    block_size = 2 * zeros_per_block
    contiguous_flattened = tensor.view(-1)
    # okay if not the same tensor since values will be the same
    block_tensor = contiguous_flattened.reshape(-1, block_size)
    assert ((block_tensor == 0).sum(dim=1) == zeros_per_block).all()


def to_semi_structured_sparse(
    original_tensor: torch.Tensor,
    transposed=False,
    backend="cusparselt",
):
    # This code calculates the size of the compressed tensor.

    num_bytes = original_tensor.nelement() * original_tensor.element_size()

    # compression factor is different based on dtype
    if original_tensor.dtype in {torch.float16, torch.bfloat16, torch.float32}:
        compression_factor = 9
    elif original_tensor.dtype is torch.int8:
        compression_factor = 10

    compressed_size_bytes = num_bytes * compression_factor // 16
    compressed_size = compressed_size_bytes // original_tensor.element_size()

    compressed_tensor = torch.empty(
        (compressed_size,),
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )

    if backend == "cusparselt":
        cslt = torch.classes.cusparselt.CusparseLt(compressed_tensor)
        cslt.compress(original_tensor, False)

        return SemiStructuredSparseTensor(
            original_tensor.shape, compressed_tensor, cslt, transposed
        )
    else:
        return SemiStructuredSparseTensor(
            original_tensor.shape, compressed_tensor, None, transposed
        )


def convert_2by4_dense_to_sparse_meta(dense):
    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype == torch.half:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    nelems_per_meta_elem = meta_dtype.itemsize * 8 // 4

    m, k = dense.shape
    if k % (4 * nelems_per_meta_elem) != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by 16"
        )
    meta_ncols = k // 4 // nelems_per_meta_elem

    mask = dense != 0

    sparse = dense.masked_select(mask).view(m, k // 2)

    mask_4 = mask.reshape((-1, k // 4, 4))
    if torch.any(mask_4.sum(-1) != 2):
        raise RuntimeError("Dense matrix does not have 2:4 sparsity pattern")

    pattern4 = torch.tensor([True, True, False, False], device=mask.device)
    pattern8 = torch.tensor([True, False, True, False], device=mask.device)
    pattern9 = torch.tensor([False, True, True, False], device=mask.device)
    pattern12 = torch.tensor([True, False, False, True], device=mask.device)
    pattern13 = torch.tensor([False, True, False, True], device=mask.device)
    pattern14 = torch.tensor([False, False, True, True], device=mask.device)
    meta_4 = (
        (mask_4 == pattern4).prod(-1) * 4
        + (mask_4 == pattern8).prod(-1) * 8
        + (mask_4 == pattern9).prod(-1) * 9
        + (mask_4 == pattern12).prod(-1) * 12
        + (mask_4 == pattern13).prod(-1) * 13
        + (mask_4 == pattern14).prod(-1) * 14
    )
    meta_n = meta_4.reshape((-1, meta_ncols, nelems_per_meta_elem)).to(meta_dtype)

    if nelems_per_meta_elem == 4:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
        )
    elif nelems_per_meta_elem == 8:
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
    else:
        raise RuntimeError(f"Invalid number of elements per meta element calculated")
    meta = torch.squeeze(meta, -1)

    dst_rows = torch.arange(0, m)[:, None].repeat(1, meta_ncols)
    dst_cols = torch.arange(0, meta_ncols).repeat(m, 1)

    # Reorder the rows, then swizzle the 2x2 blocks.
    group = 32 if meta_dtype.itemsize == 2 else 16
    interweave = 4 if meta_dtype.itemsize == 2 else 2
    dst_rows = (
        dst_rows // group * group
        + (dst_rows % 8) * interweave
        + (dst_rows % group) // 8
    )
    topright = (dst_rows % 2 == 0) & (dst_cols % 2 == 1)
    bottomleft = (dst_rows % 2 == 1) & (dst_cols % 2 == 0)
    dst_rows[topright] += 1
    dst_cols[topright] -= 1
    dst_rows[bottomleft] -= 1
    dst_cols[bottomleft] += 1

    # Assumed that meta tensor is to be stored in CUTLASS
    # InterleavedColumnMajor layout, and reverse engineered
    # corresponding code to store values into this tensor.
    interleave = 2
    cols_maj = dst_cols // interleave
    cols_min = dst_cols % interleave
    offsets = cols_maj * meta.shape[0] * interleave + dst_rows * interleave + cols_min

    meta_reordered = torch.empty(
        (m, meta_ncols), dtype=meta_dtype, layout=torch.strided, device=dense.device
    )
    meta_reordered.view(-1)[offsets.view(-1)] = meta.view(-1)

    return (sparse, meta_reordered)


def from_semi_structured_sparse(sparse_tensor):
    raise NotImplementedError("Currently not supported")


__all__ = [
    "to_semi_structured_sparse",
    "is_semi_structured_sparse",
    "SemiStructuredSparseTensor",
    "gen_semi_structured_sparse_mask",
]
