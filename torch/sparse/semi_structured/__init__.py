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


def from_semi_structured_sparse(sparse_tensor):
    raise NotImplementedError("Currently not supported")


__all__ = [
    "to_semi_structured_sparse",
    "is_semi_structured_sparse",
    "gen_semi_structured_sparse_mask",
    "SemiStructuredSparseTensor",
]
