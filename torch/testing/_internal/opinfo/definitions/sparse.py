import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
    ErrorInput,
    generate_elementwise_binary_tensors,
    sample_inputs_reduction,
    SampleInput,
)


def _filter_sample_inputs(
    sample_and_error_inputs_func,
    op_info,
    device,
    dtype,
    requires_grad,
    layout,
    **kwargs,
):
    for sample in sample_and_error_inputs_func(
        op_info, device, dtype, requires_grad, layout, **kwargs
    ):
        if isinstance(sample, SampleInput):
            yield sample


def _filter_error_inputs(sample_and_error_inputs_func, op_info, device, **kwargs):
    for layout in (
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
        torch.sparse_coo,
    ):
        for dtype in op_info.supported_dtypes(device):
            for requires_grad in (
                [False, True]
                if op_info.supports_autograd
                and (dtype.is_complex or dtype.is_floating_point)
                else [False]
            ):
                for sample in sample_and_error_inputs_func(
                    op_info, device, dtype, requires_grad, layout, **kwargs
                ):
                    if isinstance(sample, ErrorInput):
                        yield sample


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


def _sample_and_error_inputs_mul_sparse(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    for sample in sample_inputs_elementwise_binary_operation_sparse(
        op_info, device, dtype, requires_grad, layout, **kwargs
    ):
        t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
        batch_dim = t_inp.dim() - t_inp.dense_dim() - t_inp.sparse_dim()
        if layout is torch.sparse_csr and batch_dim > 0 and t_args[0].ndim > 0:
            yield ErrorInput(
                sample,
                error_regex="crow_indices is supposed to be a vector, but got 2 dimensional tensor"
            )
        elif layout is torch.sparse_csc and t_args[0].ndim > 0:
            yield ErrorInput(
                sample, error_regex="Expected result Tensor to be of format CSR"
            )
        elif layout is torch.sparse_bsr and t_args[0].ndim > 0:
            yield ErrorInput(
                sample,
                error_regex="empty_sparse_compressed expected sparse compressed [(]non-block[)] tensor layout but got SparseBsr",
            )
        elif layout is torch.sparse_bsc and t_args[0].ndim > 0:
            yield ErrorInput(
                sample,
                error_regex="empty_sparse_compressed expected sparse compressed [(]non-block[)] tensor layout but got SparseBsc",
            )
        elif (
            dtype == torch.bool
            and t_args[0].ndim > 0
            and t_inp.is_cpu
            and t_inp.numel() > 0
            and t_inp.dense_dim() > 0
        ):
            yield ErrorInput(
                sample, error_regex="\"addcmul_cpu_out\" not implemented for 'Bool'"
            )
        elif (
            dtype == torch.bool
            and t_args[0].ndim > 0
            and t_inp.is_cpu
            and t_inp.numel() > 0
        ):
            yield ErrorInput(
                sample, error_regex="\"mul_out_sparse\" not implemented for 'Bool'"
            )
        else:
            yield sample


def sample_inputs_mul_sparse(op_info, device, dtype, requires_grad, layout, **kwargs):
    """Sample inputs for mul operation on sparse tensors."""
    yield from _filter_sample_inputs(
        _sample_and_error_inputs_mul_sparse,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def error_inputs_mul_sparse(op_info, device, **kwargs):
    """Error inputs for mul operation on sparse tensors."""
    yield from _filter_error_inputs(
        _sample_and_error_inputs_mul_sparse, op_info, device, **kwargs
    )


def sample_inputs_reduction_sparse(
    op_info, device, dtype, requires_grad, layout, blocksize=None, **kwargs
):
    layout_name = str(layout).split(".", 1)[-1].rsplit("_coo", 1)[0]
    op_supports_layout = getattr(op_info, "supports_" + layout_name)
    if not op_supports_layout:
        return

    for sample_input in sample_inputs_reduction(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        if sample_input.input.ndim == 0:
            # scalar sparse tensors are not supported
            continue
        if layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            if sample_input.input.ndim < 2:
                # conversion to sparse compressed tensors requires at
                # least 2 dimensional tensors
                continue
            if sample_input.input.ndim > 2 and (sample_input.input == 0).any():
                # Skip batched sparse compressed samples that contain
                # explicit values because to_sparse(layout=..) will
                # fail, see gh-98495.
                # TODO: remove this if-block after gh-98495 is fixed.
                continue

        if layout in {torch.sparse_bsr, torch.sparse_bsc} and blocksize is None:
            blocksize = (1, 1)

        yield SampleInput(
            sample_input.input.detach()
            .to_sparse(layout=layout, blocksize=blocksize)
            .requires_grad_(requires_grad),
            args=sample_input.args,
            kwargs=sample_input.kwargs,
        )

        if layout is torch.sparse_coo and (dtype.is_floating_point or dtype.is_complex):
            # uncoalesced samples
            inp = sample_input.input.detach().to_sparse(layout=layout)
            inp = torch.sparse_coo_tensor(
                inp.indices().repeat(1, 2),
                inp.values().repeat(2),
                inp.shape,
                dtype=inp.dtype,
                device=inp.device,
            )
            assert not inp.is_coalesced()
            yield SampleInput(
                inp.requires_grad_(requires_grad),
                args=sample_input.args,
                kwargs=sample_input.kwargs,
            )

        if sample_input.input.ndim > 2:
            # hybrid samples
            yield SampleInput(
                sample_input.input.detach()
                .to_sparse(
                    layout=layout,
                    blocksize=blocksize,
                    dense_dim=sample_input.input.ndim - 2,
                )
                .requires_grad_(requires_grad),
                args=sample_input.args,
                kwargs=sample_input.kwargs,
            )


def _sample_and_error_inputs_reduction_sparse_sum(
    op_info, device, dtype, requires_grad, layout, blocksize=None, **kwargs
):
    for sample in sample_inputs_reduction_sparse(
        op_info, device, dtype, requires_grad, layout, blocksize=blocksize, **kwargs
    ):
        t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
        if layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            if (
                isinstance(t_kwargs.get("dim"), int)
                and (t_inp.dim() != 2 or t_kwargs.get("keepdim"))
            ) or (
                isinstance(t_kwargs.get("dim"), (list, tuple))
                and (
                    (
                        (t_inp.dim() != 2 and len(t_kwargs.get("dim")) != t_inp.dim())
                        or t_kwargs.get("keepdim")
                    )
                )
            ):
                if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                    yield ErrorInput(
                        sample,
                        error_regex=(
                            "empty_sparse_compressed expected sparse compressed [(]non-block[)] tensor"
                            " layout but got Sparse(Bsr|Bsc)"
                        ),
                    )
                else:
                    yield ErrorInput(
                        sample,
                        error_regex="Could not run 'aten::sum.IntList_out' with arguments from the 'SparseCsr(CPU|CUDA)' backend",
                    )
                continue
            elif t_kwargs and not t_kwargs.get("keepdim"):
                # reductions on sparse compressed tensors require
                # keepdim==True when reduction is over sparse dimensions
                yield ErrorInput(
                    sample,
                    # FIXME: raise a better exception message
                    error_regex="torch.empty: Only batched sparse compressed [(]non-block[)] tensors are supported",
                )
                continue
        yield sample


def sample_inputs_reduction_sparse_sum(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    yield from _filter_sample_inputs(
        _sample_and_error_inputs_reduction_sparse_sum,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def error_inputs_reduction_sparse_sum(op_info, device, **kwargs):
    yield from _filter_error_inputs(
        _sample_and_error_inputs_reduction_sparse_sum, op_info, device, **kwargs
    )
