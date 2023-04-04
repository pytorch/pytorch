import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
    generate_elementwise_binary_tensors,
    SampleInput,
)


def sample_inputs_elementwise_binary_operation_sparse(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    """Sample inputs for elementwise binary operations on sparse tensors.

    The samples include regular, zero-sized, batched, and hybrid
    sparse tensors as well as rhs scalars. All tensors are full tensors.
    """

    def _to_sparse(tensor, **kwargs):
        return tensor.detach().to_sparse(**kwargs).requires_grad_(requires_grad)

    for sample_input in generate_elementwise_binary_tensors(
        op_info,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=True,
        **kwargs,
    ):
        lhs, rhs = sample_input.input, sample_input.args[0]
        min_dense_dim = 0
        max_dense_dim = lhs.ndim - 1
        if layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            if lhs.ndim < 2:
                # sparse compressed tensors sparse_dim must be 2
                continue
            max_dense_dim = lhs.ndim - 2

        for dense_dim in range(min_dense_dim, max_dense_dim + 1):
            if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                blocksizes = [(1, 1)]
                if lhs.numel() > 0:
                    blocksizes.append(
                        (
                            lhs.shape[lhs.ndim - 2 - dense_dim],
                            lhs.shape[lhs.ndim - 1 - dense_dim],
                        )
                    )
            else:
                blocksizes = [None]
            for blocksize in blocksizes:
                to_sparse_kwargs = dict(
                    layout=layout, dense_dim=dense_dim, blocksize=blocksize
                )
                lhs_sparse = _to_sparse(lhs, **to_sparse_kwargs)
                rhs_sparse = _to_sparse(rhs, **to_sparse_kwargs)
                # op(sparse, sparse)
                yield SampleInput(
                    lhs_sparse,
                    args=(rhs_sparse, *sample_input.args[1:]),
                    kwargs=sample_input.kwargs,
                )
                # op(sparse, scalar)
                yield SampleInput(
                    lhs_sparse,
                    args=(
                        make_tensor(
                            (), dtype=dtype, device=device, requires_grad=requires_grad
                        ),
                        *sample_input.args[1:],
                    ),
                    kwargs=sample_input.kwargs,
                )
