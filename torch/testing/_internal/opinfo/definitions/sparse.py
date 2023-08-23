import os

import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
    BinaryUfuncInfo,
    ErrorInput,
    generate_elementwise_binary_tensors,
    ReductionOpInfo,
    sample_inputs_reduction,
    SampleInput,
)


def _check_validate(op_info, sample):
    def _check_fail(sample):
        try:
            op_info(
                sample.sample_input.input,
                *sample.sample_input.args,
                **sample.sample_input.kwargs,
            )
        except sample.error_type:
            pass
        except Exception as msg:
            raise AssertionError(
                f"{op_info.name} on {sample.sample_input=} expected exception "
                f"{sample.error_type}: {sample.error_regex}, got {type(msg).__name__}: {msg}"
            )
        else:
            raise AssertionError(
                f"{op_info.name} on {sample.sample_input=} expected exception "
                f"{sample.error_type}: {sample.error_regex}, got none."
            )

    def _check_success(sample):
        try:
            op_info(sample.input, *sample.args, **sample.kwargs)
        except Exception as msg:
            raise AssertionError(
                f"{op_info.name} on {sample=} expected to succeed "
                f", got {type(msg).__name__}: {msg}"
            )

    if isinstance(sample, ErrorInput):
        _check_fail(sample)
    else:
        _check_success(sample)


def _sample_inputs_sparse(
    sample_inputs,
    maybe_failing_sample_inputs,
    validate_sample_input,
    op_info,
    *args,
    **kwargs,
):
    check_validate = (
        os.environ.get("PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES", "0") == "1"
    )
    for sample in sample_inputs(op_info, *args, **kwargs):
        sample = validate_sample_input(op_info, sample, check_validate=check_validate)
        if isinstance(sample, SampleInput):
            yield sample
        # Error inputs are handled in error_inputs_sparse

    for sample in maybe_failing_sample_inputs(op_info, *args, **kwargs):
        sample = validate_sample_input(op_info, sample, check_validate=check_validate)
        if isinstance(sample, SampleInput):
            yield sample


def _error_inputs_sparse(
    maybe_failing_sample_inputs, validate_sample_input, op_info, *args, **kwargs
):
    check_validate = (
        os.environ.get("PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES", "0") == "1"
    )
    for sample in maybe_failing_sample_inputs(op_info, *args, **kwargs):
        sample = validate_sample_input(op_info, sample, check_validate=check_validate)
        if isinstance(sample, ErrorInput):
            yield sample
        # Sample inputs are handled in sample_inputs_sparse


def _apply_requires_grad_to_samples(sample_inputs):
    """Decorator to _maybe_failing_sample_inputs_... generator functions
    that clones and sets requires_grad argument to tensors in sample
    input arguments. This is needed when the generated samples share
    tensor instances.
    """

    def wrapper(op_info, device, dtype, requires_grad, layout, **kwargs):
        def apply_requires_grad(x):
            if (
                not isinstance(x, torch.Tensor)
                or x.requires_grad
                or not requires_grad
                or not (x.is_floating_point() or x.is_complex())
            ):
                return x
            return x.detach().clone().requires_grad_(requires_grad)

        if requires_grad:
            for sample_input in sample_inputs(
                op_info, device, dtype, requires_grad, layout, **kwargs
            ):
                yield sample_input.transform(apply_requires_grad)
        else:
            yield from sample_inputs(
                op_info, device, dtype, requires_grad, layout, **kwargs
            )

    return wrapper


def sample_inputs_sparse_reduction(
    op_info, device, dtype, requires_grad, layout, blocksize=None, **kwargs
):
    """Sample inputs for reduction operations on sparse tensors."""
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
                # explicit zeros because to_sparse(layout=..) will
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


def _validate_sample_input_sparse_reduction(op_info, sample, check_validate=False):
    """Return the specified sample when it is valid and supported by the
    operation. Otherwise, return the sample as ErrorInput instance.

    When check_validate is True, the result is validated against
    calling the op on the sample.
    """
    UNSPECIFIED = object()
    if op_info.name == "sum":
        sample = _validate_sample_input_sparse_reduction_sum(sample)

    if op_info.name in {"masked.sum"}:
        mask = sample.kwargs.get("mask", UNSPECIFIED)
        if (
            mask not in {None, UNSPECIFIED}
            and mask.ndim > 2
            and mask.layout is torch.strided
            and (mask == 0).any()
        ):
            # TODO: remove this if-block after gh-98495 is fixed.
            sample = ErrorInput(
                sample,
                error_regex="Expect the same number of specified elements per batch.",
            )
        elif not sample.kwargs.get("keepdim"):
            sample = ErrorInput(
                sample,
                error_type=(AssertionError, RuntimeError),
                error_regex="reduction operations on (CSR|CSC) tensors with keepdim=False is unsupported",
            )
        elif mask is UNSPECIFIED:
            sample = ErrorInput(
                sample,
                error_type=ValueError,
                error_regex="masked (.*) expects explicit mask for sparse_csr tensor input",
            )
        elif sample.input.ndim > 2:
            sample = ErrorInput(
                sample,
                error_regex="crow_indices is supposed to be a vector, but got 3 dimensional tensor.",
            )

    if op_info.name in {"masked.amax", "masked.amin", "masked.mean", "masked.prod"}:
        t_inp = sample.input
        batch_dim = t_inp.dim() - t_inp.dense_dim() - t_inp.sparse_dim()
        mask = sample.kwargs.get("mask")
        if (
            mask is not None
            and mask.ndim > 2
            and mask.layout is torch.strided
            and (mask == 0).any()
        ):
            # TODO: remove this if-block after gh-98495 is fixed.
            sample = ErrorInput(
                sample,
                error_regex="Expect the same number of specified elements per batch.",
            )
        elif mask is None:
            sample = ErrorInput(
                sample,
                error_type=ValueError,
                error_regex="masked (.*) expects explicit mask for sparse_csr tensor input",
            )
        elif (
            mask.layout is sample.input.layout
            and mask.ndim > 2
            and op_info.name == "masked.mean"
        ):
            sample = ErrorInput(
                sample,
                error_type=TypeError,
                error_regex=(
                    "where[(][)] received an invalid combination of arguments"
                    " - got [(]Tensor, Tensor, NoneType[)]"
                ),
            )
        elif not sample.kwargs.get("keepdim"):
            sample = ErrorInput(
                sample,
                error_type=(AssertionError, RuntimeError),
                error_regex="reduction operations on (CSR|CSC) tensors with keepdim=False is unsupported",
            )
        elif (
            sample.input.ndim > 2
            and (sample.kwargs.get("dim") not in {0, 1})
            and mask.ndim > 2
            and mask.layout is not torch.strided
        ):
            if sample.kwargs.get("dim") == (0, -1):
                sample = ErrorInput(
                    sample,
                    error_regex="tensor dimensionality must be sum of batch, base, and dense dimensionalities",
                )
            elif op_info.name == "masked.prod":
                sample = ErrorInput(
                    sample,
                    error_regex="input_dim == 2 INTERNAL ASSERT FAILED at",
                )
            else:
                sample = ErrorInput(
                    sample,
                    error_type=AssertionError,
                    error_regex="Sparse CSR tensors are 2D and only support reduction along dim 0 or 1.",
                )
        elif sample.input.ndim > 2:
            sample = ErrorInput(
                sample,
                error_regex="crow_indices is supposed to be a vector, but got 3 dimensional tensor.",
            )
        elif (
            mask.layout is t_inp.layout
            and mask._nnz() != t_inp._nnz()
            and t_inp.dense_dim() > 0
        ):
            sample = ErrorInput(
                sample,
                error_regex="Index tensor must have the same number of dimensions as src tensor",
            )

    if check_validate:
        _check_validate(op_info, sample)

    return sample


def _validate_sample_input_sparse_reduction_sum(sample, check_validate=False):
    # NOTE: When fixing a failing sample case, remove the
    #       corresponding if-block
    t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
    dim = t_kwargs.get("dim")
    keepdim = t_kwargs.get("keepdim")
    layout = t_inp.layout
    if layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }:
        if (isinstance(dim, int) and (t_inp.dim() != 2 or keepdim)) or (
            isinstance(dim, (list, tuple))
            and (((t_inp.dim() != 2 and len(dim) != t_inp.dim()) or keepdim))
        ):
            if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                return ErrorInput(
                    sample,
                    error_regex=(
                        "empty_sparse_compressed expected sparse compressed [(]non-block[)] tensor"
                        " layout but got Sparse(Bsr|Bsc)"
                    ),
                )
            else:
                return ErrorInput(
                    sample,
                    error_type=NotImplementedError,
                    error_regex="Could not run 'aten::sum.IntList_out' with arguments from the 'SparseCsr(CPU|CUDA)' backend",
                )
        elif t_kwargs and not keepdim:
            # reductions on sparse compressed tensors require
            # keepdim==True when reduction is over sparse dimensions
            return ErrorInput(
                sample,
                # FIXME: raise a better exception message
                error_regex="torch.empty: Only batched sparse compressed [(]non-block[)] tensors are supported",
            )
    return sample


def _maybe_failing_sample_inputs_sparse_reduction_sum(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    """Generator of samples that are known to fail or that were failing in past."""
    # NOTE: When fixing a failing case, remove the Exception comment
    #       but keep the `yield sample` statement.
    if layout in [
        torch.sparse_csr,
        torch.sparse_csc,
    ]:
        # NotImplementedError: Could not run 'aten::sum.IntList_out' with arguments from the 'SparseCsrCPU' backend.
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=0, keepdim=True),
        )
        yield SampleInput(
            torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype)
            .to_sparse(layout=layout, dense_dim=1)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=0),
        )
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=(0,)),
        )
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=(0,), keepdim=True),
        )
        yield SampleInput(
            torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype)
            .to_sparse(layout=layout, dense_dim=1)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=(0,)),
        )

        # RuntimeError: torch.empty: Only batched sparse compressed (non-block) tensors are supported, but got size [2]
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=0),
        )

    if layout in [
        torch.sparse_bsr,
        torch.sparse_bsc,
    ]:
        # RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout, blocksize=(2, 2))
            .requires_grad_(requires_grad),
            kwargs=dict(dim=0, keepdim=True),
        )
        yield SampleInput(
            torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype)
            .to_sparse(layout=layout, dense_dim=1, blocksize=(1, 1))
            .requires_grad_(requires_grad),
            kwargs=dict(dim=0),
        )
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout, blocksize=(1, 1))
            .requires_grad_(requires_grad),
            kwargs=dict(dim=(0,)),
        )
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout, blocksize=(1, 1))
            .requires_grad_(requires_grad),
            kwargs=dict(dim=(0,), keepdim=True),
        )
        yield SampleInput(
            torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype)
            .to_sparse(layout=layout, blocksize=(1, 1), dense_dim=1)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=(0,)),
        )

        # RuntimeError: torch.empty: Only batched sparse compressed (non-block) tensors are supported, but got size [2]
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout, blocksize=(1, 1))
            .requires_grad_(requires_grad),
            kwargs=dict(dim=0),
        )


def sample_inputs_sparse_reduction_sum(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    """Sample inputs for sum on sparse tensors."""
    yield from _sample_inputs_sparse(
        sample_inputs_sparse_reduction,
        _maybe_failing_sample_inputs_sparse_reduction_sum,
        _validate_sample_input_sparse_reduction,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def error_inputs_sparse_reduction_sum(op_info, device, layout, **kwargs):
    """Error inputs for sum on sparse tensors."""
    dtype = torch.float64
    requires_grad = False
    yield from _error_inputs_sparse(
        _maybe_failing_sample_inputs_sparse_reduction_sum,
        _validate_sample_input_sparse_reduction,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def sample_inputs_sparse_elementwise_binary_operation(
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


def _validate_sample_input_elementwise_binary_sparse_mul(sample):
    # NOTE: When fixing a failing sample case, remove the
    #       corresponding if-block
    t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
    batch_dim = t_inp.dim() - t_inp.dense_dim() - t_inp.sparse_dim()
    layout = t_inp.layout
    dtype = t_inp.dtype
    if layout is torch.sparse_csr and batch_dim > 0 and t_args[0].ndim > 0:
        return ErrorInput(
            sample,
            error_regex="crow_indices is supposed to be a vector, but got 2 dimensional tensor",
        )
    elif layout is torch.sparse_csc and t_args[0].ndim > 0:
        return ErrorInput(
            sample, error_regex="Expected result Tensor to be of format CSR"
        )
    elif layout is torch.sparse_bsr and t_args[0].ndim > 0:
        return ErrorInput(
            sample,
            error_regex="empty_sparse_compressed expected sparse compressed [(]non-block[)] tensor layout but got SparseBsr",
        )
    elif layout is torch.sparse_bsc and t_args[0].ndim > 0:
        return ErrorInput(
            sample,
            error_regex="empty_sparse_compressed expected sparse compressed [(]non-block[)] tensor layout but got SparseBsc",
        )
    elif (
        layout is torch.sparse_coo
        and dtype is torch.bool
        and t_args[0].ndim > 0
        and t_inp.is_cpu
        and t_inp.numel() > 0
        and t_inp.dense_dim() > 0
    ):
        return ErrorInput(
            sample, error_regex="\"addcmul_cpu_out\" not implemented for 'Bool'"
        )
    elif (
        layout in {torch.sparse_coo, torch.sparse_csr}
        and dtype is torch.bool
        and t_inp._nnz() > 0
        and t_args[0].ndim > 0
        and t_inp.is_cpu
        and t_inp.numel() > 0
    ):
        return ErrorInput(
            sample, error_regex="\"mul_out_sparse\" not implemented for 'Bool'"
        )
    elif (
        layout is torch.sparse_csr
        and t_args[0].layout is torch.strided
        and 0 < t_args[0].ndim
        and t_args[0].ndim < t_inp.ndim
    ):
        return ErrorInput(
            sample, error_regex="sparse_mask_sparse_csr expects self to be 2D"
        )
    elif layout is torch.sparse_csr and (
        (t_args[0].layout is torch.strided and 0 < t_args[0].ndim)
        or (t_args[0].layout is layout and t_inp.shape != t_args[0].shape)
    ):
        return ErrorInput(
            sample,
            error_regex=(
                "expects sparse inputs with equal dimensionality, number of sparse dimensions,"
                " and shape of sparse dimensions"
            ),
        )
    elif (
        layout is torch.sparse_csr
        and t_inp.dense_dim() > 0
        and t_inp._nnz() > 0
        and t_inp.is_cpu
        and dtype is torch.float16
        and t_args[0].ndim > 0
    ):
        return ErrorInput(
            sample, error_regex="\"addcmul_cpu_out\" not implemented for 'Half'"
        )
    return sample


@_apply_requires_grad_to_samples
def _maybe_failing_sample_inputs_sparse_elementwise_binary_mul(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    """Generator of samples that are known to fail or that were failing in past."""
    # NOTE: When fixing a failing case, remove the Exception comment
    #       but keep the `yield sample` statement.

    blocksize = (1, 1) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None
    regular = torch.tensor([[1, 2], [3, 4]], device=device, dtype=dtype).to_sparse(
        layout=layout, dense_dim=0, blocksize=blocksize
    )
    batch = torch.tensor(
        [[[1, 2], [3, 4]], [[4, 5], [6, 7]]], device=device, dtype=dtype
    ).to_sparse(layout=layout, dense_dim=0, blocksize=blocksize)
    hybrid = torch.tensor(
        [[[1], [2]], [[3], [4]]], device=device, dtype=dtype
    ).to_sparse(layout=layout, dense_dim=1, blocksize=blocksize)

    if layout is torch.sparse_csr:
        # RuntimeError: crow_indices is supposed to be a vector, but got 2 dimensional tensor
        yield SampleInput(batch, args=(batch,))
        # RuntimeError: Only tensors with two sparse dimensions can be
        # converted to the SparseCsr layout, got self with 3 sparse
        # dimensions.
        yield SampleInput(
            torch.zeros_like(hybrid).requires_grad_(requires_grad),
            args=(torch.zeros_like(hybrid).requires_grad_(requires_grad),),
        )
        if dtype is torch.complex32:
            # RuntimeError: "mul_out_sparse" not implemented for 'ComplexHalf'
            yield SampleInput(regular, args=(regular,))
        if dtype is torch.bool and regular.is_cpu:
            # RuntimeError: "mul_out_sparse" not implemented for 'Bool'
            yield SampleInput(regular, args=(regular,))
    if layout is torch.sparse_csc:
        # RuntimeError: Expected result Tensor to be of format CSR
        yield SampleInput(regular, args=(regular,))
    if layout is torch.sparse_bsr:
        # RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr
        yield SampleInput(regular, args=(regular,))
    if layout is torch.sparse_bsc:
        # RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsc
        yield SampleInput(regular, args=(regular,))
    if layout is torch.sparse_coo:
        if dtype is torch.complex32:
            # RuntimeError: "mul_out_sparse" not implemented for 'ComplexHalf'
            yield SampleInput(regular, args=(regular,))
        if dtype is torch.bool and regular.is_cpu:
            # RuntimeError: "mul_out_sparse" not implemented for 'Bool'
            yield SampleInput(regular, args=(regular,))
        if dtype in {torch.bool, torch.float16} and regular.is_cpu:
            # RuntimeError: "addcmul_cpu_out" not implemented for '(Bool|Half)'
            yield SampleInput(hybrid, args=(hybrid,))


def _validate_sample_input_sparse_elementwise_binary_operation(
    op_info, sample, check_validate=False
):
    if op_info.name == "mul":
        sample = _validate_sample_input_elementwise_binary_sparse_mul(sample)

    if check_validate:
        _check_validate(op_info, sample)
    return sample


def sample_inputs_sparse_mul(op_info, device, dtype, requires_grad, layout, **kwargs):
    """Sample inputs for mul operation on sparse tensors."""
    yield from _sample_inputs_sparse(
        sample_inputs_sparse_elementwise_binary_operation,
        _maybe_failing_sample_inputs_sparse_elementwise_binary_mul,
        _validate_sample_input_sparse_elementwise_binary_operation,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def error_inputs_sparse_mul(op_info, device, layout, **kwargs):
    """Error inputs for mul operation on sparse tensors."""
    dtype = torch.float64
    requires_grad = False
    yield from _error_inputs_sparse(
        _maybe_failing_sample_inputs_sparse_elementwise_binary_mul,
        _validate_sample_input_sparse_elementwise_binary_operation,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def _sample_inputs_sparse_like_fns(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    from torch.testing._internal.common_utils import TestCase

    for tensor in TestCase().generate_simple_inputs(
        layout,
        device=device,
        dtype=dtype,
        enable_batch=True,
        enable_hybrid=True,
        enable_zero_sized=True,
        enable_non_contiguous_indices=False,
        enable_non_contiguous_values=False,
    ):
        yield SampleInput(tensor, args=(), kwargs={})
        yield SampleInput(
            tensor, args=(), kwargs=dict(device=device, dtype=dtype, layout=layout)
        )

        if dtype is not torch.float64:
            yield SampleInput(tensor, args=(), kwargs=dict(dtype=torch.float64))

        if torch.cuda.is_available():
            other_device = "cuda" if tensor.device.type == "cpu" else "cpu"
            yield SampleInput(tensor, args=(), kwargs=dict(device=other_device))

        if layout is torch.sparse_csr:
            other_layout = torch.sparse_csc
        elif layout is torch.sparse_csc:
            other_layout = torch.sparse_csr
        elif layout is torch.sparse_bsr:
            other_layout = torch.sparse_bsc
        elif layout is torch.sparse_bsc:
            other_layout = torch.sparse_bsr
        else:
            other_layout = torch.strided
        yield SampleInput(tensor, args=(), kwargs=dict(layout=other_layout))

        if layout is not torch.sparse_coo:
            yield SampleInput(tensor, args=(), kwargs=dict(layout=torch.sparse_coo))


def _validate_sample_input_sparse_like_fns(op_info, sample, check_validate=False):
    if sample.input.layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }:
        if sample.kwargs.get("device", sample.input.device) != sample.input.device:
            return ErrorInput(
                sample,
                error_regex=(
                    "device of (ccol|crow)_indices \\(=(cpu|cuda.*)\\) must"
                    " match device of values \\(=(cuda.*|cpu)\\)"
                ),
            )
        if sample.kwargs.get("layout", sample.input.layout) != sample.input.layout:
            return ErrorInput(
                sample,
                error_regex=(
                    "empty_like with different sparse layout is not supported"
                    " \\(self is Sparse(Csc|Csr|Bsc|Bsr) but you requested Sparse(Csr|Csc|Bsr|Bsc)\\)"
                ),
            )
    if sample.input.layout is torch.sparse_coo:
        return ErrorInput(
            sample,
            error_regex=(
                "Could not run 'aten::normal_' with arguments from the 'Sparse(CPU|CUDA)' backend."
            ),
        )
    if check_validate:
        _check_validate(op_info, sample)
    return sample


def _maybe_failing_sample_inputs_sparse_like_fns(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    if torch.cuda.is_available() and layout is not torch.sparse_coo:
        other_device = "cuda" if torch.device(device).type == "cpu" else "cpu"
        if layout is torch.sparse_csr:
            other_layout = torch.sparse_csc
        elif layout is torch.sparse_csc:
            other_layout = torch.sparse_csr
        elif layout is torch.sparse_bsr:
            other_layout = torch.sparse_bsc
        elif layout is torch.sparse_bsc:
            other_layout = torch.sparse_bsr
        else:
            other_layout = torch.strided

        blocksize = (1, 1) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None

        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype, device=device).to_sparse(
                layout=layout, blocksize=blocksize
            ),
            kwargs=dict(device=other_device),
        )

        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype, device=device).to_sparse(
                layout=layout, blocksize=blocksize
            ),
            kwargs=dict(layout=other_layout),
        )


def sample_inputs_sparse_like_fns(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    """Sample inputs for like-functions on sparse tensors."""
    yield from _sample_inputs_sparse(
        _sample_inputs_sparse_like_fns,
        _maybe_failing_sample_inputs_sparse_like_fns,
        _validate_sample_input_sparse_like_fns,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def error_inputs_sparse_like_fns(op_info, device, layout, **kwargs):
    """Error inputs for like-functions on sparse tensors."""
    dtype = torch.float64
    requires_grad = False
    yield from _error_inputs_sparse(
        _maybe_failing_sample_inputs_sparse_like_fns,
        _validate_sample_input_sparse_like_fns,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def _validate_sample_input_sparse_default(op_info, sample, check_validate=False):
    if op_info.name == "to_sparse":
        if (
            sample.input.layout
            in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}
            and len(sample.args) == 1
            and isinstance(sample.args[0], int)
            and sample.args[0] != 2
        ):
            sample = ErrorInput(
                sample,
                error_regex="sparse dim argument must be 2 for sparse_compressed_to_sparse",
            )

    if check_validate:
        _check_validate(op_info, sample)
    return sample


def validate_sample_input_sparse(op_info, sample, check_validate=False):
    """Return the specified sample when it is valid and supported by the
    operation. Otherwise, return the sample as ErrorInput instance.

    When check_validate is True, the result is validated against
    calling the op on the sample.
    """
    if isinstance(op_info, ReductionOpInfo):
        return _validate_sample_input_sparse_reduction(
            op_info, sample, check_validate=check_validate
        )
    elif isinstance(op_info, BinaryUfuncInfo):
        return _validate_sample_input_sparse_elementwise_binary_operation(
            op_info, sample, check_validate=check_validate
        )
    else:
        return _validate_sample_input_sparse_default(
            op_info, sample, check_validate=check_validate
        )
