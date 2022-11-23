import itertools
import unittest
from functools import partial
from itertools import product
from typing import Iterable, List

import numpy as np
from numpy import inf

import torch

from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
    _get_magma_version,
    _get_torch_cuda_version,
    CUDA11OrLater,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import (
    has_cusolver,
    skipCPUIfNoLapack,
    skipCUDAIf,
    skipCUDAIfNoCusolver,
    skipCUDAIfNoMagma,
    skipCUDAIfNoMagmaAndNoCusolver,
    skipCUDAIfRocm,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_dtype import (
    all_types_and_complex,
    all_types_and_complex_and,
    floating_and_complex_types,
    floating_and_complex_types_and,
)
from torch.testing._internal.common_utils import (
    GRADCHECK_NONDET_TOL,
    IS_MACOS,
    make_fullrank_matrices_with_distinct_singular_values,
    skipIfSlowGradcheckEnv,
    slowTest,
    TEST_WITH_ROCM,
)
from torch.testing._internal.opinfo.core import (
    clone_sample,
    DecorateInfo,
    ErrorInput,
    gradcheck_wrapper_hermitian_input,
    OpInfo,
    ReductionOpInfo,
    S,
    SampleInput,
)
from torch.testing._internal.opinfo.refs import PythonRefInfo, ReductionPythonRefInfo


def sample_kwargs_vector_norm(t, **kwargs):
    # orders with / without identity
    def ords():
        has_id = (6, 4, 2, 1, 0, 0.9)
        no_id = (inf, -2.1, -inf)
        if t.numel() == 0:
            dim = kwargs.get("dim")
            if dim is None:
                return has_id
            if not isinstance(dim, Iterable):
                dim = (dim,)
            for d in dim:
                if t.size(d) == 0:
                    return has_id
        return has_id + no_id

    return (((), dict(ord=o)) for o in ords())


def sample_inputs_svd(op_info, device, dtype, requires_grad=False, **kwargs):
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(
        make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad
    )

    is_linalg_svd = "linalg.svd" in op_info.name
    batches = [(), (0,), (3,)]
    ns = [0, 3, 5]

    def uniformize(usv):
        S = usv[1]
        k = S.shape[-1]
        U = usv[0][..., :k]
        Vh = usv[2] if is_linalg_svd else usv[2].mH
        Vh = Vh[..., :k, :]
        return U, S, Vh

    def fn_U(usv):
        U, _, _ = uniformize(usv)
        return U.abs()

    def fn_S(usv):
        return uniformize(usv)[1]

    def fn_Vh(usv):
        # We also return S to test
        _, S, Vh = uniformize(usv)
        return S, Vh.abs()

    def fn_UVh(usv):
        U, S, Vh = uniformize(usv)
        return U @ Vh, S

    fns = (fn_U, fn_S, fn_Vh, fn_UVh)

    fullmat = "full_matrices" if is_linalg_svd else "some"

    for batch, n, k, fullmat_val, fn in product(batches, ns, ns, (True, False), fns):
        shape = batch + (n, k)
        yield SampleInput(
            make_arg(*shape), kwargs={fullmat: fullmat_val}, output_process_fn_grad=fn
        )


def sample_inputs_cross(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    yield SampleInput(make_arg((S, 3)), args=(make_arg((S, 3)),))
    yield SampleInput(
        make_arg((S, 3, S)), args=(make_arg((S, 3, S)),), kwargs=dict(dim=1)
    )
    yield SampleInput(make_arg((1, 3)), args=(make_arg((S, 3)),), kwargs=dict(dim=-1))


def error_inputs_cross(op_info, device, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)

    sample = SampleInput(input=make_arg((S, 3)), args=(make_arg((S, 1)),))
    err = "inputs dimension -1 must have length 3"
    yield ErrorInput(sample, error_regex=err, error_type=RuntimeError)

    sample = SampleInput(input=make_arg((5, S, 3)), args=(make_arg((S, 3)),))
    err = "inputs must have the same number of dimensions"
    yield ErrorInput(sample, error_regex=err, error_type=RuntimeError)

    sample = SampleInput(input=make_arg((S, 2)), args=(make_arg((S, 2)),))
    err = "must have length 3"
    yield ErrorInput(sample, error_regex=err, error_type=RuntimeError)

    sample = SampleInput(
        input=make_arg((S, 2)), args=(make_arg((S, 2)),), kwargs=dict(dim=2)
    )
    err = "Dimension out of range"
    yield ErrorInput(sample, error_regex=err, error_type=IndexError)


def sample_inputs_householder_product(op_info, device, dtype, requires_grad, **kwargs):
    """
    This function generates input for torch.linalg.householder_product (torch.orgqr).
    The first argument should be a square matrix or batch of square matrices, the second argument is a vector or batch of vectors.
    Empty, square, rectangular, batched square and batched rectangular input is generated.
    """
    make_arg = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-2,
        high=2,
    )
    # Each column of the matrix is getting multiplied many times leading to very large values for
    # the Jacobian matrix entries and making the finite-difference result of grad check less accurate.
    # That's why gradcheck with the default range [-9, 9] fails and [-2, 2] is used here.
    yield SampleInput(make_arg((S, S)), make_arg((S,)))
    yield SampleInput(make_arg((S + 1, S)), make_arg((S,)))
    yield SampleInput(make_arg((2, 1, S, S)), make_arg((2, 1, S)))
    yield SampleInput(make_arg((2, 1, S + 1, S)), make_arg((2, 1, S)))
    yield SampleInput(
        make_arg((0, 0), low=None, high=None),
        make_arg((0,), low=None, high=None),
    )
    yield SampleInput(make_arg((S, S)), make_arg((0,), low=None, high=None))
    # m = n = S, k = S - 2
    yield SampleInput(make_arg((S, S)), make_arg((S - 2,), low=None, high=None))
    # m = S, n = S -1, k = S - 2
    yield SampleInput(make_arg((S, S - 1)), make_arg((S - 2,), low=None, high=None))


def sample_inputs_linalg_det_singular(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype)

    def make_singular_matrix_batch_base(size, rank):
        assert size[-1] == size[-2]
        assert rank > 0 and rank < size[-1]

        n = size[-1]
        a = make_arg(size[:-2] + (n, rank)) / 10
        b = make_arg(size[:-2] + (rank, n)) / 10
        x = a @ b
        lu, pivs, _ = torch.linalg.lu_factor_ex(x)
        p, l, u = torch.lu_unpack(lu, pivs)
        u_diag_abs = u.diagonal(0, -2, -1).abs()
        u_diag_abs_largest = u_diag_abs.max(dim=-1, keepdim=True).values
        u_diag_abs_smallest_idxs = torch.topk(
            u_diag_abs, k=(n - rank), largest=False
        ).indices
        u.diagonal(0, -2, -1).div_(u_diag_abs_largest)
        u.diagonal(0, -2, -1)[..., u_diag_abs_smallest_idxs] = torch.finfo(dtype).eps
        matrix = p @ l @ u

        matrix.requires_grad_(requires_grad)
        return matrix

    for batch, size in product(((), (2,), (2, 2)), range(6)):
        shape = batch + (size, size)
        for rank in range(1, size):
            yield SampleInput(make_singular_matrix_batch_base(shape, rank))


def sample_inputs_linalg_matrix_power(op_info, device, dtype, requires_grad, **kwargs):
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    make_arg_fullrank = partial(
        make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad
    )
    # (<matrix_size>, (<batch_sizes, ...>))
    test_sizes = [
        (1, ()),
        (2, (0,)),
        (2, (2,)),
    ]

    for matrix_size, batch_sizes in test_sizes:
        size = batch_sizes + (matrix_size, matrix_size)
        for n in (0, 3, 5):
            yield SampleInput(make_arg(size), args=(n,))
        for n in [-4, -2, -1]:
            yield SampleInput(make_arg_fullrank(*size), args=(n,))


def sample_inputs_linalg_det_logdet_slogdet(
    op_info, device, dtype, requires_grad, **kwargs
):
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(
        make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad
    )
    batches = [(), (0,), (3,)]
    ns = [0, 1, 5]

    is_logdet = op_info.name == "logdet"

    for (
        batch,
        n,
    ) in product(batches, ns):
        shape = batch + (n, n)
        A = make_arg(*shape)
        # Need to make the matrices in A have positive determinant for autograd
        # To do so, we multiply A by its determinant to flip the sign of its determinant
        if is_logdet and not A.is_complex() and A.numel() > 0:
            s = torch.linalg.slogdet(A).sign
            A = A * s.unsqueeze(-1).unsqueeze(-1)
            A.requires_grad_(requires_grad)
        yield SampleInput(A)


def sample_inputs_lu_solve(op_info, device, dtype, requires_grad=False, **kwargs):
    """Samples the inputs for both linalg.lu_solve and lu_solve"""
    make_fn = make_fullrank_matrices_with_distinct_singular_values
    make_a = partial(make_fn, dtype=dtype, device=device)
    make_b = partial(make_tensor, dtype=dtype, device=device)

    def clone(X, requires_grad):
        Y = X.clone()
        Y.requires_grad_(requires_grad)
        return Y

    is_linalg_lu_solve = op_info.name == "linalg.lu_solve"

    batches = ((), (0,), (2,))
    ns = (3, 1, 0)
    nrhs = (4, 1, 0)

    for n, batch, rhs in product(ns, batches, nrhs):
        A = make_a(*(batch + (n, n)))
        LU, pivots = torch.linalg.lu_factor(A)

        B = make_b(batch + (n, rhs))

        grads = (False,) if not requires_grad else (True, False)
        # we try all possible combinations of requires_grad for each input
        for LU_grad, B_grad in product(grads, grads):
            # when requires_grad == True, at least one input has to have requires_grad enabled
            if requires_grad and not LU_grad and not B_grad:
                continue

            if is_linalg_lu_solve:
                for adjoint, left in product((True, False), repeat=2):
                    yield SampleInput(
                        clone(LU, LU_grad),
                        args=(pivots, clone(B if left else B.mT, B_grad)),
                        kwargs=dict(adjoint=adjoint, left=left),
                    )
            else:
                yield SampleInput(clone(B, B_grad), args=(clone(LU, LU_grad), pivots))


def sample_inputs_linalg_multi_dot(op_info, device, dtype, requires_grad, **kwargs):
    # Each test case consists of the sizes in the chain of multiplications
    # e.g. [2, 3, 4, 5] generates matrices (2, 3) @ (3, 4) @ (4, 5)
    test_cases = [
        [1, 2, 1],
        [2, 0, 2],
        [0, 2, 2],
        [2, 2, 2, 2],
        [2, 3, 4, 5],
        [5, 4, 0, 2],
        [2, 4, 3, 5, 3, 2],
    ]

    for sizes in test_cases:
        tensors = []
        for size in zip(sizes[:-1], sizes[1:]):
            t = make_tensor(
                size, dtype=dtype, device=device, requires_grad=requires_grad
            )
            tensors.append(t)
        yield SampleInput(tensors)


def sample_inputs_linalg_matrix_norm(op_info, device, dtype, requires_grad, **kwargs):
    low_precision_dtypes = (torch.float16, torch.bfloat16, torch.complex32)
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    sizes = ((2, 2), (2, 3, 2))
    if dtype in low_precision_dtypes:
        # svdvals not supported for low precision dtypes
        ords = ("fro", inf, -inf, 1, -1)
    else:
        ords = ("fro", "nuc", inf, -inf, 1, -1, 2, -2)
    dims = ((-2, -1), (-1, 0))

    for size, ord, dim, keepdim in product(sizes, ords, dims, [True, False]):
        yield SampleInput(make_arg(size), args=(ord, dim, keepdim))


def sample_inputs_linalg_norm(
    op_info, device, dtype, requires_grad, *, variant=None, **kwargs
):
    if variant is not None and variant not in ("subgradient_at_zero",):
        raise ValueError(
            f"Unsupported variant, expected variant to be 'subgradient_at_zero' but got: {variant}"
        )

    test_sizes = [
        (S,),
        (0,),
        (S, S),
        (0, 0),
        (S, 0),
        (0, S),
        (S, S, S),
        (0, S, S),
        (S, 0, S),
        (0, 0, 0),
    ]

    vector_ords = (None, 0, 0.5, 1, 2, 3.5, inf, -0.5, -1, -2, -3.5, -inf)
    if dtype in {torch.float16, torch.bfloat16, torch.complex32}:
        # svdvals not supported for low precision dtypes
        matrix_ords = ("fro", inf, -inf, 1, -1)
    else:
        matrix_ords = (None, "fro", "nuc", inf, -inf, 1, -1, 2, -2)

    make_arg = partial(
        make_tensor,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        low=None,
        high=None,
    )

    for test_size in test_sizes:
        is_vector_norm = len(test_size) == 1
        is_matrix_norm = len(test_size) == 2

        # IndexError: amax(): Expected reduction dim 0 to have non-zero size.
        is_valid_for_p2 = is_vector_norm or (test_size[-1] != 0 and test_size[-2] != 0)

        for keepdim in [False, True]:
            if variant != "subgradient_at_zero" and is_valid_for_p2:
                yield SampleInput(make_arg(test_size), keepdim=keepdim)

            if not (is_vector_norm or is_matrix_norm):
                continue

            ords = vector_ords if is_vector_norm else matrix_ords

            for ord in ords:
                if is_vector_norm and test_size[-1] == 0:
                    if ord == np.inf or (ord is not None and ord < 0):
                        # RuntimeError: linalg.vector_norm cannot compute the
                        # {ord} norm on an empty tensor because the operation
                        # does not have an identity
                        continue
                elif is_matrix_norm:
                    dims_to_check = {
                        None: (0,),
                        np.inf: (0,),
                        2: (0, 1),
                        1: (1,),
                        -1: (1,),
                        -2: (0, 1),
                        -np.inf: (0,),
                    }.get(ord, ())

                    if any(test_size[d] == 0 for d in dims_to_check):
                        # IndexError: amax(): Expected reduction dim {dim} to
                        # have non-zero size.
                        continue

                if variant == "subgradient_at_zero":
                    yield SampleInput(
                        torch.zeros(
                            test_size,
                            dtype=dtype,
                            device=device,
                            requires_grad=requires_grad,
                        ),
                        ord,
                        keepdim=keepdim,
                    )
                else:
                    yield SampleInput(make_arg(test_size), ord, keepdim=keepdim)

                    if ord in ["nuc", "fro"]:
                        yield SampleInput(
                            make_arg(test_size), ord=ord, keepdim=keepdim, dim=(0, 1)
                        )


def sample_inputs_linalg_vecdot(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    batches = ((), (0,), (1,), (5,))
    ns = (0, 1, 3, 5)
    for b, n in product(batches, ns):
        shape = b + (n,)
        yield SampleInput(make_arg(shape), args=(make_arg(shape),))
        for i in range(len(shape)):
            yield SampleInput(
                make_arg(shape), args=(make_arg(shape),), kwargs=dict(dim=i)
            )


def sample_inputs_linalg_invertible(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    """
    This function generates invertible inputs for linear algebra ops
    The input is generated as the itertools.product of 'batches' and 'ns'.
    In total this function generates 8 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices,
        (1, 1) - 1x1 batch of matrices
    'ns' gives 0x0 and 5x5 matrices.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    """
    make_fn = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(make_fn, dtype=dtype, device=device, requires_grad=requires_grad)

    batches = [(), (0,), (2,), (1, 1)]
    ns = [5, 0]

    for batch, n in product(batches, ns):
        yield SampleInput(make_arg(*batch, n, n))


def sample_inputs_matrix_rank(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function produces inputs for matrix rank that test
    all possible combinations for atol and rtol
    """

    def make_tol_arg(kwarg_type, inp):
        if kwarg_type == "none":
            return None
        if kwarg_type == "float":
            return 1.0
        assert kwarg_type == "tensor"
        return torch.ones(inp.shape[:-2], device=device)

    for tol_type in ["float", "tensor"]:
        for atol_type, rtol_type in product(["none", tol_type], repeat=2):
            if (
                not atol_type and not rtol_type
            ):  # default behavior, so skipped here so it's not tested 2 extra times
                continue
            for sample in sample_inputs_linalg_invertible(
                op_info, device, dtype, requires_grad
            ):
                assert sample.kwargs == {}
                sample.kwargs = {
                    "atol": make_tol_arg(atol_type, sample.input),
                    "rtol": make_tol_arg(rtol_type, sample.input),
                }
                yield sample

    for sample in sample_inputs_linalg_invertible(
        op_info, device, dtype, requires_grad
    ):
        yield sample  # default kwargs


def sample_inputs_linalg_pinv_singular(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    """
    This function produces factors `a` and `b` to generate inputs of the form `a @ b.t()` to
    test the backward method of `linalg_pinv`. That way we always preserve the rank of the
    input no matter the perturbations applied to it by the gradcheck.
    Note that `pinv` is Frechet-differentiable in a rank-preserving neighborhood.
    """
    batches = [(), (0,), (2,), (1, 1)]
    # the size of at least 30 is required to cause failures for the previous implicit implementation
    # of the pinv's backward method, albeit it is slow.
    size = [0, 3, 50]

    for batch, m, n in product(batches, size, size):
        for k in range(min(3, min(m, n))):
            # Note that by making the columns of `a` and `b` orthonormal we make sure that
            # the product matrix `a @ b.t()` has condition number 1 when restricted to its image
            a = (
                torch.rand(*batch, m, k, device=device, dtype=dtype)
                .qr()
                .Q.requires_grad_(requires_grad)
            )
            b = (
                torch.rand(*batch, n, k, device=device, dtype=dtype)
                .qr()
                .Q.requires_grad_(requires_grad)
            )
            yield SampleInput(a, args=(b,))


def sample_inputs_linalg_cond(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )

    # autograd is not supported for inputs with zero number of elements
    shapes = (
        (S, S),
        (2, S, S),
        (2, 1, S, S),
    )

    for shape in shapes:
        yield SampleInput(make_arg(shape))


def sample_inputs_linalg_vander(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )

    shapes = (
        (),
        (1,),
        (S,),
        (2, S),
    )

    for shape in shapes:
        if len(shape) > 0 and shape[-1] > 1:
            yield SampleInput(make_arg(shape))
        n = shape[-1] if len(shape) > 0 else 1
        for i in range(3):
            # n-1, n, n+1
            N = n + i - 1
            if N < 2:
                continue
            yield SampleInput(make_arg(shape), kwargs=dict(N=N))


def np_vander_batched(x, N=None):
    # Wrapper around np.vander that supports batches of 1 dimension (enough for the tests)
    if x.ndim == 0:
        x = x[np.newaxis]
    if x.ndim == 1:
        y = np.vander(x, N=N, increasing=True)
        return y
    else:
        if N is None:
            N = x.shape[-1]
        y = np.vander(x.ravel(), N=N, increasing=True).reshape((*x.shape, N))
        return y


def sample_inputs_linalg_cholesky_inverse(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    from torch.testing._internal.common_utils import random_well_conditioned_matrix

    # Cholesky factorization is for positive-definite matrices
    single_well_conditioned_matrix = random_well_conditioned_matrix(
        S, S, dtype=dtype, device=device
    )
    batch_well_conditioned_matrices = random_well_conditioned_matrix(
        2, S, S, dtype=dtype, device=device
    )
    single_pd = single_well_conditioned_matrix @ single_well_conditioned_matrix.mH
    batch_pd = batch_well_conditioned_matrices @ batch_well_conditioned_matrices.mH

    inputs = (
        torch.zeros(0, 0, dtype=dtype, device=device),  # 0x0 matrix
        torch.zeros(0, 2, 2, dtype=dtype, device=device),  # zero batch of matrices
        single_pd,
        batch_pd,
    )
    test_cases = (torch.linalg.cholesky(a, upper=False) for a in inputs)
    for l in test_cases:
        # generated lower-triangular samples
        l.requires_grad = requires_grad
        yield SampleInput(l)  # upper=False by default
        yield SampleInput(
            l.detach().clone().requires_grad_(requires_grad), kwargs=dict(upper=False)
        )

        # generate upper-triangular inputs
        u = l.detach().clone().mT.contiguous().requires_grad_(requires_grad)
        yield SampleInput(u, kwargs=dict(upper=True))


def sample_inputs_linalg_ldl_factor(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    from torch.testing._internal.common_utils import (
        random_hermitian_pd_matrix,
        random_symmetric_pd_matrix,
    )

    device = torch.device(device)

    # Symmetric inputs
    yield SampleInput(
        random_symmetric_pd_matrix(S, dtype=dtype, device=device),
        kwargs=dict(hermitian=False),
    )  # single matrix
    yield SampleInput(
        random_symmetric_pd_matrix(S, 2, dtype=dtype, device=device),
        kwargs=dict(hermitian=False),
    )  # batch of matrices
    yield SampleInput(
        torch.zeros(0, 0, dtype=dtype, device=device), kwargs=dict(hermitian=False)
    )  # 0x0 matrix
    yield SampleInput(
        torch.zeros(0, 2, 2, dtype=dtype, device=device), kwargs=dict(hermitian=False)
    )  # zero batch of matrices

    # Hermitian inputs
    # hermitian=True for complex inputs on CUDA is supported only with MAGMA 2.5.4+
    magma_254_available = device.type == "cuda" and _get_magma_version() >= (2, 5, 4)
    if dtype.is_complex and (device.type == "cpu" or magma_254_available):
        yield SampleInput(
            random_hermitian_pd_matrix(S, dtype=dtype, device=device),
            kwargs=dict(hermitian=True),
        )  # single matrix
        yield SampleInput(
            random_hermitian_pd_matrix(S, 2, dtype=dtype, device=device),
            kwargs=dict(hermitian=True),
        )  # batch of matrices


def sample_inputs_linalg_ldl_solve(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    # Generate LDL factors of symmetric (and Hermitian on CPU) matrices
    from torch.testing._internal.common_utils import (
        random_hermitian_pd_matrix,
        random_symmetric_pd_matrix,
    )

    device = torch.device(device)
    symmetric_inputs = (
        random_symmetric_pd_matrix(S, dtype=dtype, device=device),  # single matrix
        random_symmetric_pd_matrix(
            S, 2, dtype=dtype, device=device
        ),  # batch of matrices
        torch.zeros(0, 0, dtype=dtype, device=device),  # 0x0 matrix
        torch.zeros(0, 2, 2, dtype=dtype, device=device),  # zero batch of matrices
    )
    hermitian_inputs = (
        (
            random_hermitian_pd_matrix(S, dtype=dtype, device=device),
            random_hermitian_pd_matrix(S, 2, dtype=dtype, device=device),
        )
        if device.type == "cpu" and dtype.is_complex
        else ()
    )
    test_cases1 = (
        torch.linalg.ldl_factor_ex(a, hermitian=False) for a in symmetric_inputs
    )
    test_cases2 = (
        torch.linalg.ldl_factor_ex(a, hermitian=True) for a in hermitian_inputs
    )

    # Symmetric case
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    for test_case in test_cases1:
        factors, pivots, _ = test_case
        factors.requires_grad = requires_grad
        for B_batch_shape in ((), factors.shape[:-2]):
            B = make_arg((*B_batch_shape, factors.shape[-1], S))
            yield SampleInput(factors, args=(pivots, B), kwargs=dict(hermitian=False))
            clone_factors = factors.detach().clone().requires_grad_(requires_grad)
            yield SampleInput(
                clone_factors, args=(pivots, B), kwargs=dict(hermitian=False)
            )

    # Hermitian case
    for test_case in test_cases2:
        factors, pivots, _ = test_case
        factors.requires_grad = requires_grad
        for B_batch_shape in ((), factors.shape[:-2]):
            B = make_arg((*B_batch_shape, factors.shape[-1], S))
            yield SampleInput(factors, args=(pivots, B), kwargs=dict(hermitian=True))
            clone_factors = factors.detach().clone().requires_grad_(requires_grad)
            yield SampleInput(
                clone_factors, args=(pivots, B), kwargs=dict(hermitian=True)
            )


def sample_inputs_linalg_lstsq(op_info, device, dtype, requires_grad=False, **kwargs):
    from torch.testing._internal.common_utils import random_well_conditioned_matrix

    device = torch.device(device)

    drivers: Tuple[str, ...]
    if device.type == "cuda":
        drivers = ("gels",)
    else:
        drivers = ("gels", "gelsy", "gelss", "gelsd")

    # we generate matrices of shape (..., n + delta, n)
    deltas: Tuple[int, ...]
    if device.type == "cpu" or has_cusolver():
        deltas = (-1, 0, +1)
    # only square systems if Cusolver is not available
    # becase we solve a lstsq problem with a transposed matrix in the backward
    else:
        deltas = (0,)

    for batch, driver, delta in product(((), (3,), (3, 3)), drivers, deltas):
        shape = batch + (3 + delta, 3)
        a = random_well_conditioned_matrix(*shape, dtype=dtype, device=device)
        a.requires_grad_(requires_grad)
        b = make_tensor(
            shape,
            dtype=dtype,
            device=device,
            low=None,
            high=None,
            requires_grad=requires_grad,
        )
        yield SampleInput(a, b, driver=driver)


def error_inputs_lstsq(op_info, device, **kwargs):
    zero_d = torch.randn((), device=device)
    yield ErrorInput(
        SampleInput(zero_d, args=(zero_d,)),
        error_type=RuntimeError,
        error_regex="at least 2 dimensions",
    )


def error_inputs_lstsq_grad_oriented(op_info, device, **kwargs):
    zero_d = torch.randn((), device=device)
    yield ErrorInput(
        SampleInput(zero_d, args=(zero_d, None)),
        error_type=RuntimeError,
        error_regex="at least 2 dimensions",
    )


def sample_inputs_linalg_cholesky(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    """
    This function generates always positive-definite input for torch.linalg.cholesky using
    random_hermitian_pd_matrix.
    The input is generated as the itertools.product of 'batches' and 'ns'.
    In total this function generates 8 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices,
        (1, 1) - 1x1 batch of matrices
    'ns' gives 0x0 and 5x5 matrices.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    """
    from torch.testing._internal.common_utils import random_hermitian_pd_matrix

    batches = [(), (0,), (2,), (1, 1)]
    ns = [5, 0]
    for batch, n, upper in product(batches, ns, [True, False]):
        a = random_hermitian_pd_matrix(n, *batch, dtype=dtype, device=device)
        a.requires_grad = requires_grad
        yield SampleInput(a, upper=upper)


def sample_inputs_linalg_eig(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.eig
    """

    def out_fn(output):
        return output[0], abs(output[1])

    samples = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)
    for sample in samples:
        sample.output_process_fn_grad = out_fn
        yield sample


def sample_inputs_linalg_eigh(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.eigh/eigvalsh with UPLO="U" or "L" keyword argument.
    """

    def out_fn(output):
        if isinstance(output, tuple):
            # eigh function
            return output[0], abs(output[1])
        else:
            # eigvalsh function
            return output

    # Samples do not need to be Hermitian, as we're using gradcheck_wrapper_hermitian_input
    samples = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)
    for sample in samples:
        sample.kwargs = {"UPLO": np.random.choice(["L", "U"])}
        sample.output_process_fn_grad = out_fn
        yield sample


def sample_inputs_linalg_pinv(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.pinv with hermitian=False keyword argument.
    """
    for o in sample_inputs_linalg_invertible(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        real_dtype = o.input.real.dtype if dtype.is_complex else dtype
        # requires_grad path for rtol tensor is not implemented
        for rtol in (None, 1.0, torch.tensor(1.0, dtype=real_dtype, device=device)):
            o = clone_sample(o)
            o.kwargs = {"rtol": rtol}
            yield o


def sample_inputs_linalg_pinv_hermitian(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    """
    This function generates input for torch.linalg.pinv with hermitian=True keyword argument.
    """
    for o in sample_inputs_linalg_invertible(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        o.kwargs = {"hermitian": True}
        yield o


def sample_inputs_linalg_solve(
    op_info, device, dtype, requires_grad=False, vector_rhs_allowed=True, **kwargs
):
    """
    This function generates always solvable input for torch.linalg.solve
    We sample a fullrank square matrix (i.e. invertible) A
    The first input to torch.linalg.solve is generated as the itertools.product of 'batches' and 'ns'.
    The second input is generated as the product of 'batches', 'ns' and 'nrhs'.
    In total this function generates 18 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices.
    'ns' gives 0x0 and 5x5 matrices.
    and 'nrhs' controls the number of vectors to solve for:
        () - using 1 as the number of vectors implicitly
        (1,) - same as () but explicit
        (3,) - solve for 3 vectors.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    'vector_rhs_allowed' controls whether to include nrhs = () to the list of SampleInputs.
    torch.solve / triangular_solve / cholesky_solve (opposed to torch.linalg.solve) do not allow
    1D tensors (vectors) as the right-hand-side.
    Once torch.solve / triangular_solve / cholesky_solve and its testing are removed,
    'vector_rhs_allowed' may be removed here as well.
    """
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_a = partial(
        make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad
    )
    make_b = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )

    batches = [(), (0,), (2,)]
    ns = [5, 0]
    if vector_rhs_allowed:
        nrhs = [(), (1,), (3,)]
    else:
        nrhs = [(1,), (3,)]

    for n, batch, rhs in product(ns, batches, nrhs):
        yield SampleInput(make_a(*batch, n, n), args=(make_b((batch + (n,) + rhs)),))


def sample_inputs_linalg_solve_triangular(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    make_arg = partial(make_tensor, dtype=dtype, device=device)
    bs = (1, 2, 0)
    ns = (3, 0)
    ks = (1, 3, 0)

    for b, n, k, (left, upper, uni) in product(
        bs, ns, ks, product((True, False), repeat=3)
    ):
        if b == 1:
            A = make_arg((n, n)) if left else make_arg((k, k))
            B = make_arg((n, k))
        else:
            A = make_arg((b, n, n)) if left else make_arg((b, k, k))
            B = make_arg((b, n, k))
        if uni:
            # Not really necessary, but writing it for consistency
            A.diagonal(0, -2, -1).fill_(1.0)
        else:
            d = A.diagonal(0, -2, -1)
            d[d.abs() < 1e-6] = 1.0
        if upper:
            A.triu_()
        else:
            A.tril_()
        kwargs = {"upper": upper, "left": left, "unitriangular": uni}
        if requires_grad:
            for grad_A, grad_B in product((True, False), repeat=2):
                # Either A or B needs to have a gradient
                if not grad_A and not grad_B:
                    continue
                yield SampleInput(
                    A.clone().requires_grad_(grad_A),
                    args=(B.clone().requires_grad_(grad_B),),
                    kwargs=kwargs,
                )
        else:
            yield SampleInput(A, args=(B,), kwargs=kwargs)


def sample_inputs_legacy_solve(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates always solvable input for legacy solve functions
    (the ones that are not in torch.linalg module).
    The difference from sample_inputs_linalg_solve is that here the right-hand-side of A x = b equation
    should have b.ndim >= 2, vectors are not allowed.
    Also the arguments order is swapped.
    """
    out = sample_inputs_linalg_solve(
        op_info, device, dtype, requires_grad=requires_grad, vector_rhs_allowed=False
    )

    def out_fn(output):
        return output[0]

    # Reverses tensor order
    for sample in out:
        sample.input, sample.args = sample.args[0], (sample.input,)
        if op_info.name == "solve":
            sample.output_process_fn_grad = out_fn
        yield sample


def sample_inputs_linalg_lu(op_info, device, dtype, requires_grad=False, **kwargs):
    full_rank = op_info.name == "linalg.lu_factor"
    make_fn = (
        make_tensor
        if not full_rank
        else make_fullrank_matrices_with_distinct_singular_values
    )
    make_arg = partial(make_fn, dtype=dtype, device=device, requires_grad=requires_grad)

    def out_fn(output):
        if op_info.name == "linalg.lu":
            return output[1], output[2]
        else:
            return output

    batch_shapes = ((), (3,), (3, 3))
    # pivot=False only supported in CUDA
    pivots = (True, False) if torch.device(device).type == "cuda" else (True,)
    deltas = (-2, -1, 0, +1, +2)
    for batch_shape, pivot, delta in product(batch_shapes, pivots, deltas):
        shape = batch_shape + (S + delta, S)
        # Insanely annoying that make_fullrank_blablabla accepts a *shape and not a tuple!
        A = make_arg(shape) if not full_rank else make_arg(*shape)
        yield SampleInput(A, kwargs={"pivot": pivot}, output_process_fn_grad=out_fn)


def sample_inputs_linalg_svdvals(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )

    batches = [(), (0,), (2,), (1, 1)]
    ns = [5, 2, 0]

    for batch, m, n in product(batches, ns, ns):
        yield SampleInput(make_arg(batch + (m, n)))


def sample_inputs_linalg_qr_geqrf(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    # QR is just well defined when the matrix is full rank
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(
        make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad
    )

    batches = [(), (0,), (2,), (1, 1)]
    ns = [5, 2, 0]

    for batch, (m, n) in product(batches, product(ns, ns)):
        shape = batch + (m, n)
        yield SampleInput(make_arg(*shape))


def sample_inputs_tensorsolve(op_info, device, dtype, requires_grad, **kwargs):
    a_shapes = [(2, 3, 6), (3, 4, 4, 3)]
    # Zero-dim tensors are not supported in NumPy, so we skip them for now.
    # NumPy is used in reference check tests.
    # See https://github.com/numpy/numpy/pull/20482 for tracking NumPy bugfix.
    # a_shapes += [(0, 0, 1, 2, 3, 0)]
    dimss = [None, (0, 2)]

    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    for a_shape, dims in itertools.product(a_shapes, dimss):
        a = make_arg(a_shape)
        b = make_arg(a_shape[:2])
        yield SampleInput(a, b, dims=dims)


def sample_inputs_tensorinv(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = make_fullrank_matrices_with_distinct_singular_values

    def make_input():
        return make_arg(12, 12, device=device, dtype=dtype, requires_grad=requires_grad)

    # lhs / rhs shape can have any number of dimensions as long as their product equals 12
    shapes = [
        ((2, 2, 3), (12, 1)),
        ((4, 3), (6, 1, 2)),
    ]

    for shape_lhs, shape_rhs in shapes:
        inp = make_input().reshape(*shape_lhs, *shape_rhs).detach()
        inp.requires_grad_(requires_grad)
        yield SampleInput(inp, ind=len(shape_lhs))


op_db: List[OpInfo] = [
    OpInfo(
        "linalg.cross",
        ref=lambda x, y, dim=-1: np.cross(x, y, axis=dim),
        op=torch.linalg.cross,
        dtypes=all_types_and_complex_and(torch.bfloat16),
        dtypesIfCUDA=all_types_and_complex_and(torch.half),
        aten_name="linalg_cross",
        sample_inputs_func=sample_inputs_cross,
        error_inputs_func=error_inputs_cross,
        supports_out=True,
        supports_fwgrad_bwgrad=True,
        supports_forward_ad=True,
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
    OpInfo(
        "linalg.det",
        aten_name="linalg_det",
        op=torch.linalg.det,
        aliases=("det",),
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_det_logdet_slogdet,
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
        check_batched_gradgrad=False,
    ),
    OpInfo(
        "linalg.det",
        aten_name="linalg_det",
        op=torch.linalg.det,
        variant_test_name="singular",
        aliases=("det",),
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_gradgrad=False,
        sample_inputs_func=sample_inputs_linalg_det_singular,
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
        skips=(
            DecorateInfo(
                unittest.skip("The backward may give different results"),
                "TestCommon",
                "test_noncontiguous_samples",
            ),
            DecorateInfo(
                unittest.skip("Gradients are incorrect on macos"),
                "TestBwdGradients",
                "test_fn_grad",
                device_type="cpu",
                dtypes=(torch.float64,),
                active_if=IS_MACOS,
            ),
            DecorateInfo(
                unittest.skip("Gradients are incorrect on macos"),
                "TestFwdGradients",
                "test_forward_mode_AD",
                device_type="cpu",
                dtypes=(torch.float64,),
                active_if=IS_MACOS,
            ),
            # Both Hessians are incorrect on complex inputs??
            DecorateInfo(
                unittest.expectedFailure,
                "TestBwdGradients",
                "test_fn_gradgrad",
                dtypes=(torch.complex128,),
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                dtypes=(torch.complex128,),
            ),
            DecorateInfo(
                unittest.skip("Skipped, see https://github.com//issues/84192"),
                "TestBwdGradients",
                "test_fn_gradgrad",
                device_type="cuda",
            ),
            DecorateInfo(
                unittest.skip("Skipped, see https://github.com//issues/84192"),
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cuda",
            ),
        ),
    ),
    OpInfo(
        "linalg.cholesky",
        aten_name="linalg_cholesky",
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_cholesky,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.cholesky_ex",
        aten_name="linalg_cholesky_ex",
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_cholesky,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.vecdot",
        aten_name="linalg_vecdot",
        ref=lambda x, y, *, dim=-1: (x.conj() * y).sum(dim),
        dtypes=floating_and_complex_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_and_complex_types_and(
            torch.half, *[torch.bfloat16] if (CUDA11OrLater or TEST_WITH_ROCM) else []
        ),
        sample_inputs_func=sample_inputs_linalg_vecdot,
        check_batched_forward_grad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            # Issue with conj and torch dispatch, see https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestSchemaCheckModeOpInfo",
                "test_schema_correctness",
                dtypes=(torch.complex64, torch.complex128),
            ),
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
    OpInfo(
        "linalg.cond",
        aten_name="linalg_cond",
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_cond,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
    ),
    OpInfo(
        "linalg.eig",
        aten_name="linalg_eig",
        op=torch.linalg.eig,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_eig,
        check_batched_forward_grad=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            # AssertionError: Scalars are not equal!
            DecorateInfo(
                unittest.expectedFailure, "TestCommon", "test_out", device_type="cpu"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack, with_tf32_off],
    ),
    OpInfo(
        "linalg.eigvals",
        aten_name="linalg_eigvals",
        op=torch.linalg.eigvals,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_invertible,
        check_batched_forward_grad=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            # exits early on eager extremal value test
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCudaFuserOpInfo",
                "test_nvfuser_extremal_values",
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.eigh",
        aten_name="linalg_eigh",
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_eigh,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        check_batched_forward_grad=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack, with_tf32_off],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.eigvalsh",
        aten_name="linalg_eigvalsh",
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_eigh,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        check_batched_forward_grad=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            # Pre-existing condition; Needs to be fixed
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.householder_product",
        aten_name="linalg_householder_product",
        op=torch.linalg.householder_product,
        aliases=("orgqr",),
        dtypes=floating_and_complex_types(),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        # TODO: backward uses in-place operations that vmap doesn't like
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_householder_product,
        decorators=[
            skipCUDAIfNoCusolver,
            skipCPUIfNoLapack,
            DecorateInfo(
                toleranceOverride({torch.complex64: tol(atol=1e-3, rtol=1e-3)})
            ),
            DecorateInfo(
                unittest.skip("Skipped! Flaky"),
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cpu",
                dtypes=(torch.complex128,),
            ),
        ],
    ),
    OpInfo(
        "linalg.ldl_factor",
        aten_name="linalg_ldl_factor",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_linalg_ldl_factor,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, skipCUDAIfRocm],
    ),
    OpInfo(
        "linalg.ldl_factor_ex",
        aten_name="linalg_ldl_factor_ex",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_linalg_ldl_factor,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, skipCUDAIfRocm],
    ),
    OpInfo(
        "linalg.ldl_solve",
        aten_name="linalg_ldl_solve",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_linalg_ldl_solve,
        decorators=[
            skipCUDAIf(
                _get_torch_cuda_version() < (11, 4), "not available before CUDA 11.3.1"
            ),
            skipCUDAIfNoCusolver,
            skipCUDAIfRocm,
            skipCPUIfNoLapack,
        ],
    ),
    OpInfo(
        "linalg.lstsq",
        aten_name="linalg_lstsq",
        dtypes=floating_and_complex_types(),
        supports_out=True,
        sample_inputs_func=sample_inputs_linalg_lstsq,
        error_inputs_func=error_inputs_lstsq,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            # we skip gradient checks for this suite as they are tested in
            # variant_test_name='grad_oriented'
            DecorateInfo(unittest.skip("Skipped!"), "TestFwdGradients"),
            DecorateInfo(unittest.skip("Skipped!"), "TestBwdGradients"),
            # The values for attribute 'shape' do not match
            DecorateInfo(unittest.skip("Skipped!"), "TestCommon", "test_out"),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.lstsq",
        aten_name="linalg_lstsq",
        variant_test_name="grad_oriented",
        # gradchecks for forward AD fails with multi-Tensor outputs
        op=lambda a, b, driver: torch.linalg.lstsq(a, b, driver=driver)[0],
        supports_out=False,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_lstsq,
        error_inputs_func=error_inputs_lstsq_grad_oriented,
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_autograd=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            # tests do not work with passing lambda for op
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestOperatorSignatures",
                "test_get_torch_func_signature_exhaustive",
            ),
        ),
    ),
    OpInfo(
        "linalg.matrix_power",
        aliases=("matrix_power",),
        aten_name="linalg_matrix_power",
        dtypes=floating_and_complex_types(),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_inplace_autograd=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_grad=False,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        sample_inputs_func=sample_inputs_linalg_matrix_power,
    ),
    OpInfo(
        "linalg.multi_dot",
        # Need this lambda because gradcheck does not work with TensorList inputs
        aten_name="linalg_multi_dot",
        dtypes=all_types_and_complex_and(torch.bfloat16),
        dtypesIfCUDA=floating_and_complex_types_and(
            torch.half, *[torch.bfloat16] if (CUDA11OrLater or TEST_WITH_ROCM) else []
        ),
        supports_inplace_autograd=False,
        # Batched grad checks fail for empty input tensors (see https://github.com/pytorch/pytorch/issues/53407)
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # https://github.com/pytorch/pytorch/issues/66357
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_multi_dot,
        gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
        skips=(
            # https://github.com/pytorch/pytorch/issues/67470
            DecorateInfo(
                unittest.skip("67470!"), "TestCommon", "test_noncontiguous_samples"
            ),
            # Fails on XLA.
            # AssertionError: False is not true : Tensors failed to compare as equal!
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestOpInfo",
                device_type="xla",
                dtypes=(torch.long,),
            ),
            # https://github.com/pytorch/pytorch/issues/71774
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestNNCOpInfo",
                "test_nnc_correctness",
                device_type="cpu",
                dtypes=(torch.long,),
            ),
        ),
    ),
    # NB: linalg.norm has two variants so that different skips can be used for different sample inputs
    OpInfo(
        "linalg.norm",
        aten_name="linalg_norm",
        op=torch.linalg.norm,
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        sample_inputs_func=sample_inputs_linalg_norm,
        supports_forward_ad=True,
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        skips=(
            DecorateInfo(
                unittest.expectedFailure, "TestBwdGradients", "test_fn_gradgrad"
            ),
        ),
    ),
    OpInfo(
        "linalg.norm",
        op=torch.linalg.norm,
        variant_test_name="subgradients_at_zero",
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        sample_inputs_func=partial(
            sample_inputs_linalg_norm, variant="subgradient_at_zero"
        ),
        aten_name="linalg_norm",
        supports_forward_ad=True,
        # torch.autograd.gradcheck.GradcheckError: While computing batched gradients, got:
        # Could not allocate memory to change Tensor SizesAndStrides!
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        skips=(
            # [NEW] Skips specifically for sample inputs at zero
            # norm's vjp/jvp are not well-conditioned near zero
            DecorateInfo(
                unittest.expectedFailure, "TestBwdGradients", "test_fn_gradgrad"
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestFwdGradients", "test_fn_fwgrad_bwgrad"
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestFwdGradients", "test_forward_mode_AD"
            ),
            DecorateInfo(unittest.expectedFailure, "TestBwdGradients", "test_fn_grad"),
        ),
    ),
    OpInfo(
        "linalg.matrix_norm",
        aten_name="linalg_matrix_norm",
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        supports_forward_ad=True,
        check_batched_forward_grad=False,
        check_batched_gradgrad=False,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        sample_inputs_func=sample_inputs_linalg_matrix_norm,
    ),
    OpInfo(
        "linalg.qr",
        aten_name="linalg_qr",
        op=torch.linalg.qr,
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # In-place ops
        check_batched_gradgrad=False,
        sample_inputs_func=sample_inputs_linalg_qr_geqrf,
        decorators=[skipCUDAIfNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.slogdet",
        aten_name="linalg_slogdet",
        op=torch.linalg.slogdet,
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_det_logdet_slogdet,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.vander",
        aten_name="linalg_vander",
        ref=np_vander_batched,
        op=torch.linalg.vander,
        dtypes=all_types_and_complex(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
        sample_inputs_func=sample_inputs_linalg_vander,
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
    ReductionOpInfo(
        "linalg.vector_norm",
        op=torch.linalg.vector_norm,
        identity=0,
        nan_policy="propagate",
        supports_multiple_dims=True,
        complex_to_real=True,
        supports_forward_ad=True,
        # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
        # got: Could not allocate memory to change Tensor SizesAndStrides!
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        generate_args_kwargs=sample_kwargs_vector_norm,
        aten_name="linalg_vector_norm",
        skips=(
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
        ),
    ),
    OpInfo(
        "linalg.lu_factor",
        aten_name="linalg_lu_factor",
        op=torch.linalg.lu_factor,
        dtypes=floating_and_complex_types(),
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_lu,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # linalg.lu_factor: LU without pivoting is not implemented on the CPU
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
    OpInfo(
        "linalg.lu_factor_ex",
        aten_name="linalg_lu_factor_ex",
        op=torch.linalg.lu_factor_ex,
        dtypes=floating_and_complex_types(),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_lu,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # linalg.lu_factor: LU without pivoting is not implemented on the CPU
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
    OpInfo(
        "linalg.lu",
        aten_name="linalg_lu",
        op=torch.linalg.lu,
        dtypes=floating_and_complex_types(),
        # https://github.com/pytorch/pytorch/issues/80411
        # Runs very slowly on slow-gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_lu,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # linalg.lu_factor: LU without pivoting is not implemented on the CPU
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
    OpInfo(
        "linalg.lu_solve",
        op=torch.linalg.lu_solve,
        aten_name="linalg_lu_solve",
        dtypes=floating_and_complex_types(),
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_lu_solve,
        skips=(
            DecorateInfo(
                unittest.skip("Tests different backward paths"),
                "TestCommon",
                "test_floating_inputs_are_differentiable",
            ),
        ),
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
    ),
    OpInfo(
        "linalg.inv",
        aten_name="linalg_inv",
        op=torch.linalg.inv,
        aliases=("inverse",),
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_invertible,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.inv_ex",
        aten_name="linalg_inv_ex",
        op=torch.linalg.inv_ex,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_invertible,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.solve",
        aten_name="linalg_solve",
        op=torch.linalg.solve,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_solve,
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.solve_ex",
        aten_name="linalg_solve_ex",
        op=torch.linalg.solve_ex,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_solve,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.solve_triangular",
        aten_name="linalg_solve_triangular",
        op=torch.linalg.solve_triangular,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_solve_triangular,
        supports_fwgrad_bwgrad=True,
        skips=(skipCPUIfNoLapack,),
        # linalg.solve_triangular cannot be batched over because of a call to out.copy_(result);
        supports_forward_ad=True,
    ),
    OpInfo(
        "linalg.matrix_rank",
        aten_name="linalg_matrix_rank",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_matrix_rank,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            # jit doesn't accept tensor inputs for matrix rank
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                dtypes=[torch.complex64, torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.matrix_rank",
        aten_name="linalg_matrix_rank",
        variant_test_name="hermitian",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.pinv",
        aten_name="linalg_pinv",
        op=torch.linalg.pinv,
        dtypes=floating_and_complex_types(),
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_pinv,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # errors with "leaked XXXX bytes CUDA memory on device 0"
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="cuda",
            ),
        ),
    ),
    OpInfo(
        "linalg.pinv",
        aten_name="linalg_pinv",
        variant_test_name="singular",
        # pinv is Frechet-differentiable in a rank-preserving neighborhood,
        # so we feed inputs that are the products of two full-rank factors,
        # to avoid any rank changes caused by the perturbations in the gradcheck
        op=lambda a, b: torch.linalg.pinv(a @ b.mT),
        dtypes=floating_and_complex_types(),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_pinv_singular,
        # Only large tensors show issues with implicit backward used prior to
        # explicit backward implementation.
        decorators=[slowTest, skipCUDAIfNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # CUDA runs out of memory
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cuda",
                dtypes=[torch.cdouble],
            ),
            # This test takes almost 2 hours to run!
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestBwdGradients",
                "test_fn_gradgrad",
                device_type="cuda",
                dtypes=[torch.cdouble],
            ),
        ),
    ),
    OpInfo(
        "linalg.pinv",
        aten_name="linalg_pinv",
        variant_test_name="hermitian",
        dtypes=floating_and_complex_types(),
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1e-5, rtol=1e-5)}),
                "TestCommon",
                "test_noncontiguous_samples",
                device_type="cuda",
            ),
            # This test is flaky under slow gradcheck, likely due to rounding issues
            DecorateInfo(
                skipIfSlowGradcheckEnv,
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cuda",
            ),
        ),
    ),
    OpInfo(
        "linalg.svd",
        op=torch.linalg.svd,
        aten_name="linalg_svd",
        decomp_aten_name="_linalg_svd",
        dtypes=floating_and_complex_types(),
        # Runs very slowly on slow-gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_fwgrad_bwgrad=True,
        supports_forward_ad=True,
        check_batched_forward_grad=False,
        # We're using at::allclose, which does not have a batching rule
        check_batched_grad=False,
        check_batched_gradgrad=False,
        sample_inputs_func=sample_inputs_svd,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.svdvals",
        op=torch.linalg.svdvals,
        aten_name="linalg_svdvals",
        decomp_aten_name="_linalg_svd",
        dtypes=floating_and_complex_types(),
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        supports_forward_ad=True,
        # We're using at::allclose, which does not have a batching rule
        check_batched_gradgrad=False,
        sample_inputs_func=sample_inputs_linalg_svdvals,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
    ),
    OpInfo(
        "linalg.tensorinv",
        ref=np.linalg.tensorinv,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_tensorinv,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
    OpInfo(
        "linalg.tensorsolve",
        ref=lambda a, b, dims=None: np.linalg.tensorsolve(a, b, axes=dims),
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_tensorsolve,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[
            skipCUDAIfNoMagmaAndNoCusolver,
            skipCPUIfNoLapack,
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1e-03, rtol=1e-03)}),
                "TestCommon",
                "test_noncontiguous_samples",
                device_type="cuda",
            ),
        ],
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
]

python_ref_db: List[OpInfo] = [
    #
    # torch.linalg
    #
    ReductionPythonRefInfo(
        "_refs.linalg.vector_norm",
        torch_opinfo_name="linalg.vector_norm",
        supports_out=True,
        supports_nvfuser=False,  # clone_default
        op_db=op_db,
    ),
    PythonRefInfo(
        "_refs.linalg.matrix_norm",
        torch_opinfo_name="linalg.matrix_norm",
        supports_out=True,
        # Uses svdvals which does not support nvfuser
        supports_nvfuser=False,
        # Uses vector_norm inside and vector_norm is affected by
        # https://github.com/pytorch/pytorch/issues/77216
        validate_view_consistency=False,
        op_db=op_db,
    ),
    PythonRefInfo(
        "_refs.linalg.norm",
        torch_opinfo_name="linalg.norm",
        supports_out=True,
        # Uses svdvals which does not support nvfuser
        supports_nvfuser=False,
        # Uses vector_norm inside and vector_norm is affected by
        # https://github.com/pytorch/pytorch/issues/77216
        validate_view_consistency=False,
        op_db=op_db,
    ),
    PythonRefInfo(
        "_refs.linalg.svd",
        torch_opinfo_name="linalg.svd",
        supports_out=True,
        supports_nvfuser=False,
        op_db=op_db,
    ),
    PythonRefInfo(
        "_refs.linalg.svdvals",
        torch_opinfo_name="linalg.svdvals",
        supports_out=True,
        supports_nvfuser=False,
        op_db=op_db,
    ),
]
