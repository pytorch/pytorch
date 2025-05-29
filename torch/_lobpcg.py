# mypy: allow-untyped-defs
"""Locally Optimal Block Preconditioned Conjugate Gradient methods."""
# Author: Pearu Peterson
# Created: February 2020

from typing import Optional

import torch
from torch import _linalg_utils as _utils, Tensor
from torch.overrides import handle_torch_function, has_torch_function


__all__ = ["lobpcg"]


def _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U):
    # compute F, such that F_ij = (d_j - d_i)^{-1} for i != j, F_ii = 0
    F = D.unsqueeze(-2) - D.unsqueeze(-1)
    F.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
    F.pow_(-1)

    # A.grad = U (D.grad + (U^T U.grad * F)) U^T
    Ut = U.mT.contiguous()
    res = torch.matmul(
        U, torch.matmul(torch.diag_embed(D_grad) + torch.matmul(Ut, U_grad) * F, Ut)
    )

    return res


def _polynomial_coefficients_given_roots(roots):
    """
    Given the `roots` of a polynomial, find the polynomial's coefficients.

    If roots = (r_1, ..., r_n), then the method returns
    coefficients (a_0, a_1, ..., a_n (== 1)) so that
    p(x) = (x - r_1) * ... * (x - r_n)
         = x^n + a_{n-1} * x^{n-1} + ... a_1 * x_1 + a_0

    Note: for better performance requires writing a low-level kernel
    """
    poly_order = roots.shape[-1]
    poly_coeffs_shape = list(roots.shape)
    # we assume p(x) = x^n + a_{n-1} * x^{n-1} + ... + a_1 * x + a_0,
    # so poly_coeffs = {a_0, ..., a_n, a_{n+1}(== 1)},
    # but we insert one extra coefficient to enable better vectorization below
    poly_coeffs_shape[-1] += 2
    poly_coeffs = roots.new_zeros(poly_coeffs_shape)
    poly_coeffs[..., 0] = 1
    poly_coeffs[..., -1] = 1

    # perform the Horner's rule
    for i in range(1, poly_order + 1):
        # note that it is computationally hard to compute backward for this method,
        # because then given the coefficients it would require finding the roots and/or
        # calculating the sensitivity based on the Vieta's theorem.
        # So the code below tries to circumvent the explicit root finding by series
        # of operations on memory copies imitating the Horner's method.
        # The memory copies are required to construct nodes in the computational graph
        # by exploting the explicit (not in-place, separate node for each step)
        # recursion of the Horner's method.
        # Needs more memory, O(... * k^2), but with only O(... * k^2) complexity.
        poly_coeffs_new = poly_coeffs.clone() if roots.requires_grad else poly_coeffs
        out = poly_coeffs_new.narrow(-1, poly_order - i, i + 1)
        out -= roots.narrow(-1, i - 1, 1) * poly_coeffs.narrow(
            -1, poly_order - i + 1, i + 1
        )
        poly_coeffs = poly_coeffs_new

    return poly_coeffs.narrow(-1, 1, poly_order + 1)


def _polynomial_value(poly, x, zero_power, transition):
    """
    A generic method for computing poly(x) using the Horner's rule.

    Args:
      poly (Tensor): the (possibly batched) 1D Tensor representing
                     polynomial coefficients such that
                     poly[..., i] = (a_{i_0}, ..., a{i_n} (==1)), and
                     poly(x) = poly[..., 0] * zero_power + ... + poly[..., n] * x^n

      x (Tensor): the value (possible batched) to evalate the polynomial `poly` at.

      zero_power (Tensor): the representation of `x^0`. It is application-specific.

      transition (Callable): the function that accepts some intermediate result `int_val`,
                             the `x` and a specific polynomial coefficient
                             `poly[..., k]` for some iteration `k`.
                             It basically performs one iteration of the Horner's rule
                             defined as `x * int_val + poly[..., k] * zero_power`.
                             Note that `zero_power` is not a parameter,
                             because the step `+ poly[..., k] * zero_power` depends on `x`,
                             whether it is a vector, a matrix, or something else, so this
                             functionality is delegated to the user.
    """

    res = zero_power.clone()
    for k in range(poly.size(-1) - 2, -1, -1):
        res = transition(res, x, poly[..., k])
    return res


def _matrix_polynomial_value(poly, x, zero_power=None):
    """
    Evaluates `poly(x)` for the (batched) matrix input `x`.
    Check out `_polynomial_value` function for more details.
    """

    # matrix-aware Horner's rule iteration
    def transition(curr_poly_val, x, poly_coeff):
        res = x.matmul(curr_poly_val)
        res.diagonal(dim1=-2, dim2=-1).add_(poly_coeff.unsqueeze(-1))
        return res

    if zero_power is None:
        zero_power = torch.eye(
            x.size(-1), x.size(-1), dtype=x.dtype, device=x.device
        ).view(*([1] * len(list(x.shape[:-2]))), x.size(-1), x.size(-1))

    return _polynomial_value(poly, x, zero_power, transition)


def _vector_polynomial_value(poly, x, zero_power=None):
    """
    Evaluates `poly(x)` for the (batched) vector input `x`.
    Check out `_polynomial_value` function for more details.
    """

    # vector-aware Horner's rule iteration
    def transition(curr_poly_val, x, poly_coeff):
        res = torch.addcmul(poly_coeff.unsqueeze(-1), x, curr_poly_val)
        return res

    if zero_power is None:
        zero_power = x.new_ones(1).expand(x.shape)

    return _polynomial_value(poly, x, zero_power, transition)


def _symeig_backward_partial_eigenspace(D_grad, U_grad, A, D, U, largest):
    # compute a projection operator onto an orthogonal subspace spanned by the
    # columns of U defined as (I - UU^T)
    Ut = U.mT.contiguous()
    proj_U_ortho = -U.matmul(Ut)
    proj_U_ortho.diagonal(dim1=-2, dim2=-1).add_(1)

    # compute U_ortho, a basis for the orthogonal complement to the span(U),
    # by projecting a random [..., m, m - k] matrix onto the subspace spanned
    # by the columns of U.
    #
    # fix generator for determinism
    gen = torch.Generator(A.device)

    # orthogonal complement to the span(U)
    U_ortho = proj_U_ortho.matmul(
        torch.randn(
            (*A.shape[:-1], A.size(-1) - D.size(-1)),
            dtype=A.dtype,
            device=A.device,
            generator=gen,
        )
    )
    U_ortho_t = U_ortho.mT.contiguous()

    # compute the coefficients of the characteristic polynomial of the tensor D.
    # Note that D is diagonal, so the diagonal elements are exactly the roots
    # of the characteristic polynomial.
    chr_poly_D = _polynomial_coefficients_given_roots(D)

    # the code belows finds the explicit solution to the Sylvester equation
    # U_ortho^T A U_ortho dX - dX D = -U_ortho^T A U
    # and incorporates it into the whole gradient stored in the `res` variable.
    #
    # Equivalent to the following naive implementation:
    # res = A.new_zeros(A.shape)
    # p_res = A.new_zeros(*A.shape[:-1], D.size(-1))
    # for k in range(1, chr_poly_D.size(-1)):
    #     p_res.zero_()
    #     for i in range(0, k):
    #         p_res += (A.matrix_power(k - 1 - i) @ U_grad) * D.pow(i).unsqueeze(-2)
    #     res -= chr_poly_D[k] * (U_ortho @ poly_D_at_A.inverse() @ U_ortho_t @  p_res @ U.t())
    #
    # Note that dX is a differential, so the gradient contribution comes from the backward sensitivity
    # Tr(f(U_grad, D_grad, A, U, D)^T dX) = Tr(g(U_grad, A, U, D)^T dA) for some functions f and g,
    # and we need to compute g(U_grad, A, U, D)
    #
    # The naive implementation is based on the paper
    # Hu, Qingxi, and Daizhan Cheng.
    # "The polynomial solution to the Sylvester matrix equation."
    # Applied mathematics letters 19.9 (2006): 859-864.
    #
    # We can modify the computation of `p_res` from above in a more efficient way
    # p_res =   U_grad * (chr_poly_D[1] * D.pow(0) + ... + chr_poly_D[k] * D.pow(k)).unsqueeze(-2)
    #       + A U_grad * (chr_poly_D[2] * D.pow(0) + ... + chr_poly_D[k] * D.pow(k - 1)).unsqueeze(-2)
    #       + ...
    #       + A.matrix_power(k - 1) U_grad * chr_poly_D[k]
    # Note that this saves us from redundant matrix products with A (elimination of matrix_power)
    U_grad_projected = U_grad
    series_acc = U_grad_projected.new_zeros(U_grad_projected.shape)
    for k in range(1, chr_poly_D.size(-1)):
        poly_D = _vector_polynomial_value(chr_poly_D[..., k:], D)
        series_acc += U_grad_projected * poly_D.unsqueeze(-2)
        U_grad_projected = A.matmul(U_grad_projected)

    # compute chr_poly_D(A) which essentially is:
    #
    # chr_poly_D_at_A = A.new_zeros(A.shape)
    # for k in range(chr_poly_D.size(-1)):
    #     chr_poly_D_at_A += chr_poly_D[k] * A.matrix_power(k)
    #
    # Note, however, for better performance we use the Horner's rule
    chr_poly_D_at_A = _matrix_polynomial_value(chr_poly_D, A)

    # compute the action of `chr_poly_D_at_A` restricted to U_ortho_t
    chr_poly_D_at_A_to_U_ortho = torch.matmul(
        U_ortho_t, torch.matmul(chr_poly_D_at_A, U_ortho)
    )
    # we need to invert 'chr_poly_D_at_A_to_U_ortho`, for that we compute its
    # Cholesky decomposition and then use `torch.cholesky_solve` for better stability.
    # Cholesky decomposition requires the input to be positive-definite.
    # Note that `chr_poly_D_at_A_to_U_ortho` is positive-definite if
    # 1. `largest` == False, or
    # 2. `largest` == True and `k` is even
    # under the assumption that `A` has distinct eigenvalues.
    #
    # check if `chr_poly_D_at_A_to_U_ortho` is positive-definite or negative-definite
    chr_poly_D_at_A_to_U_ortho_sign = -1 if (largest and (k % 2 == 1)) else +1
    chr_poly_D_at_A_to_U_ortho_L = torch.linalg.cholesky(
        chr_poly_D_at_A_to_U_ortho_sign * chr_poly_D_at_A_to_U_ortho
    )

    # compute the gradient part in span(U)
    res = _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U)

    # incorporate the Sylvester equation solution into the full gradient
    # it resides in span(U_ortho)
    res -= U_ortho.matmul(
        chr_poly_D_at_A_to_U_ortho_sign
        * torch.cholesky_solve(
            U_ortho_t.matmul(series_acc), chr_poly_D_at_A_to_U_ortho_L
        )
    ).matmul(Ut)

    return res


def _symeig_backward(D_grad, U_grad, A, D, U, largest):
    # if `U` is square, then the columns of `U` is a complete eigenspace
    if U.size(-1) == U.size(-2):
        return _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U)
    else:
        return _symeig_backward_partial_eigenspace(D_grad, U_grad, A, D, U, largest)


class LOBPCGAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        A: Tensor,
        k: Optional[int] = None,
        B: Optional[Tensor] = None,
        X: Optional[Tensor] = None,
        n: Optional[int] = None,
        iK: Optional[Tensor] = None,
        niter: Optional[int] = None,
        tol: Optional[float] = None,
        largest: Optional[bool] = None,
        method: Optional[str] = None,
        tracker: None = None,
        ortho_iparams: Optional[dict[str, int]] = None,
        ortho_fparams: Optional[dict[str, float]] = None,
        ortho_bparams: Optional[dict[str, bool]] = None,
    ) -> tuple[Tensor, Tensor]:
        # makes sure that input is contiguous for efficiency.
        # Note: autograd does not support dense gradients for sparse input yet.
        A = A.contiguous() if (not A.is_sparse) else A
        if B is not None:
            B = B.contiguous() if (not B.is_sparse) else B

        D, U = _lobpcg(
            A,
            k,
            B,
            X,
            n,
            iK,
            niter,
            tol,
            largest,
            method,
            tracker,
            ortho_iparams,
            ortho_fparams,
            ortho_bparams,
        )

        ctx.save_for_backward(A, B, D, U)
        ctx.largest = largest

        return D, U

    @staticmethod
    def backward(ctx, D_grad, U_grad):
        A_grad = B_grad = None
        grads = [None] * 14

        A, B, D, U = ctx.saved_tensors
        largest = ctx.largest

        # lobpcg.backward has some limitations. Checks for unsupported input
        if A.is_sparse or (B is not None and B.is_sparse and ctx.needs_input_grad[2]):
            raise ValueError(
                "lobpcg.backward does not support sparse input yet."
                "Note that lobpcg.forward does though."
            )
        if (
            A.dtype in (torch.complex64, torch.complex128)
            or B is not None
            and B.dtype in (torch.complex64, torch.complex128)
        ):
            raise ValueError(
                "lobpcg.backward does not support complex input yet."
                "Note that lobpcg.forward does though."
            )
        if B is not None:
            raise ValueError(
                "lobpcg.backward does not support backward with B != I yet."
            )

        if largest is None:
            largest = True

        # symeig backward
        if B is None:
            A_grad = _symeig_backward(D_grad, U_grad, A, D, U, largest)

        # A has index 0
        grads[0] = A_grad
        # B has index 2
        grads[2] = B_grad
        return tuple(grads)


def lobpcg(
    A: Tensor,
    k: Optional[int] = None,
    B: Optional[Tensor] = None,
    X: Optional[Tensor] = None,
    n: Optional[int] = None,
    iK: Optional[Tensor] = None,
    niter: Optional[int] = None,
    tol: Optional[float] = None,
    largest: Optional[bool] = None,
    method: Optional[str] = None,
    tracker: None = None,
    ortho_iparams: Optional[dict[str, int]] = None,
    ortho_fparams: Optional[dict[str, float]] = None,
    ortho_bparams: Optional[dict[str, bool]] = None,
) -> tuple[Tensor, Tensor]:
    """Find the k largest (or smallest) eigenvalues and the corresponding
    eigenvectors of a symmetric positive definite generalized
    eigenvalue problem using matrix-free LOBPCG methods.

    This function is a front-end to the following LOBPCG algorithms
    selectable via `method` argument:

      `method="basic"` - the LOBPCG method introduced by Andrew
      Knyazev, see [Knyazev2001]. A less robust method, may fail when
      Cholesky is applied to singular input.

      `method="ortho"` - the LOBPCG method with orthogonal basis
      selection [StathopoulosEtal2002]. A robust method.

    Supported inputs are dense, sparse, and batches of dense matrices.

    .. note:: In general, the basic method spends least time per
      iteration. However, the robust methods converge much faster and
      are more stable. So, the usage of the basic method is generally
      not recommended but there exist cases where the usage of the
      basic method may be preferred.

    .. warning:: The backward method does not support sparse and complex inputs.
      It works only when `B` is not provided (i.e. `B == None`).
      We are actively working on extensions, and the details of
      the algorithms are going to be published promptly.

    .. warning:: While it is assumed that `A` is symmetric, `A.grad` is not.
      To make sure that `A.grad` is symmetric, so that `A - t * A.grad` is symmetric
      in first-order optimization routines, prior to running `lobpcg`
      we do the following symmetrization map: `A -> (A + A.t()) / 2`.
      The map is performed only when the `A` requires gradients.

    Args:

      A (Tensor): the input tensor of size :math:`(*, m, m)`

      B (Tensor, optional): the input tensor of size :math:`(*, m,
                  m)`. When not specified, `B` is interpreted as
                  identity matrix.

      X (tensor, optional): the input tensor of size :math:`(*, m, n)`
                  where `k <= n <= m`. When specified, it is used as
                  initial approximation of eigenvectors. X must be a
                  dense tensor.

      iK (tensor, optional): the input tensor of size :math:`(*, m,
                  m)`. When specified, it will be used as preconditioner.

      k (integer, optional): the number of requested
                  eigenpairs. Default is the number of :math:`X`
                  columns (when specified) or `1`.

      n (integer, optional): if :math:`X` is not specified then `n`
                  specifies the size of the generated random
                  approximation of eigenvectors. Default value for `n`
                  is `k`. If :math:`X` is specified, the value of `n`
                  (when specified) must be the number of :math:`X`
                  columns.

      tol (float, optional): residual tolerance for stopping
                 criterion. Default is `feps ** 0.5` where `feps` is
                 smallest non-zero floating-point number of the given
                 input tensor `A` data type.

      largest (bool, optional): when True, solve the eigenproblem for
                 the largest eigenvalues. Otherwise, solve the
                 eigenproblem for smallest eigenvalues. Default is
                 `True`.

      method (str, optional): select LOBPCG method. See the
                 description of the function above. Default is
                 "ortho".

      niter (int, optional): maximum number of iterations. When
                 reached, the iteration process is hard-stopped and
                 the current approximation of eigenpairs is returned.
                 For infinite iteration but until convergence criteria
                 is met, use `-1`.

      tracker (callable, optional) : a function for tracing the
                 iteration process. When specified, it is called at
                 each iteration step with LOBPCG instance as an
                 argument. The LOBPCG instance holds the full state of
                 the iteration process in the following attributes:

                   `iparams`, `fparams`, `bparams` - dictionaries of
                   integer, float, and boolean valued input
                   parameters, respectively

                   `ivars`, `fvars`, `bvars`, `tvars` - dictionaries
                   of integer, float, boolean, and Tensor valued
                   iteration variables, respectively.

                   `A`, `B`, `iK` - input Tensor arguments.

                   `E`, `X`, `S`, `R` - iteration Tensor variables.

                 For instance:

                   `ivars["istep"]` - the current iteration step
                   `X` - the current approximation of eigenvectors
                   `E` - the current approximation of eigenvalues
                   `R` - the current residual
                   `ivars["converged_count"]` - the current number of converged eigenpairs
                   `tvars["rerr"]` - the current state of convergence criteria

                 Note that when `tracker` stores Tensor objects from
                 the LOBPCG instance, it must make copies of these.

                 If `tracker` sets `bvars["force_stop"] = True`, the
                 iteration process will be hard-stopped.

      ortho_iparams, ortho_fparams, ortho_bparams (dict, optional):
                 various parameters to LOBPCG algorithm when using
                 `method="ortho"`.

    Returns:

      E (Tensor): tensor of eigenvalues of size :math:`(*, k)`

      X (Tensor): tensor of eigenvectors of size :math:`(*, m, k)`

    References:

      [Knyazev2001] Andrew V. Knyazev. (2001) Toward the Optimal
      Preconditioned Eigensolver: Locally Optimal Block Preconditioned
      Conjugate Gradient Method. SIAM J. Sci. Comput., 23(2),
      517-541. (25 pages)
      https://epubs.siam.org/doi/abs/10.1137/S1064827500366124

      [StathopoulosEtal2002] Andreas Stathopoulos and Kesheng
      Wu. (2002) A Block Orthogonalization Procedure with Constant
      Synchronization Requirements. SIAM J. Sci. Comput., 23(6),
      2165-2182. (18 pages)
      https://epubs.siam.org/doi/10.1137/S1064827500370883

      [DuerschEtal2018] Jed A. Duersch, Meiyue Shao, Chao Yang, Ming
      Gu. (2018) A Robust and Efficient Implementation of LOBPCG.
      SIAM J. Sci. Comput., 40(5), C655-C676. (22 pages)
      https://arxiv.org/abs/1704.07458

    """

    if not torch.jit.is_scripting():
        tensor_ops = (A, B, X, iK)
        if not set(map(type, tensor_ops)).issubset(
            (torch.Tensor, type(None))
        ) and has_torch_function(tensor_ops):
            return handle_torch_function(
                lobpcg,
                tensor_ops,
                A,
                k=k,
                B=B,
                X=X,
                n=n,
                iK=iK,
                niter=niter,
                tol=tol,
                largest=largest,
                method=method,
                tracker=tracker,
                ortho_iparams=ortho_iparams,
                ortho_fparams=ortho_fparams,
                ortho_bparams=ortho_bparams,
            )

    if not torch._jit_internal.is_scripting():
        if A.requires_grad or (B is not None and B.requires_grad):
            # While it is expected that `A` is symmetric,
            # the `A_grad` might be not. Therefore we perform the trick below,
            # so that `A_grad` becomes symmetric.
            # The symmetrization is important for first-order optimization methods,
            # so that (A - alpha * A_grad) is still a symmetric matrix.
            # Same holds for `B`.
            A_sym = (A + A.mT) / 2
            B_sym = (B + B.mT) / 2 if (B is not None) else None

            return LOBPCGAutogradFunction.apply(
                A_sym,
                k,
                B_sym,
                X,
                n,
                iK,
                niter,
                tol,
                largest,
                method,
                tracker,
                ortho_iparams,
                ortho_fparams,
                ortho_bparams,
            )
    else:
        if A.requires_grad or (B is not None and B.requires_grad):
            raise RuntimeError(
                "Script and require grads is not supported atm."
                "If you just want to do the forward, use .detach()"
                "on A and B before calling into lobpcg"
            )

    return _lobpcg(
        A,
        k,
        B,
        X,
        n,
        iK,
        niter,
        tol,
        largest,
        method,
        tracker,
        ortho_iparams,
        ortho_fparams,
        ortho_bparams,
    )


def _lobpcg(
    A: Tensor,
    k: Optional[int] = None,
    B: Optional[Tensor] = None,
    X: Optional[Tensor] = None,
    n: Optional[int] = None,
    iK: Optional[Tensor] = None,
    niter: Optional[int] = None,
    tol: Optional[float] = None,
    largest: Optional[bool] = None,
    method: Optional[str] = None,
    tracker: None = None,
    ortho_iparams: Optional[dict[str, int]] = None,
    ortho_fparams: Optional[dict[str, float]] = None,
    ortho_bparams: Optional[dict[str, bool]] = None,
) -> tuple[Tensor, Tensor]:
    # A must be square:
    assert A.shape[-2] == A.shape[-1], A.shape
    if B is not None:
        # A and B must have the same shapes:
        assert A.shape == B.shape, (A.shape, B.shape)

    dtype = _utils.get_floating_dtype(A)
    device = A.device
    if tol is None:
        feps = {torch.float32: 1.2e-07, torch.float64: 2.23e-16}[dtype]
        tol = feps**0.5

    m = A.shape[-1]
    k = (1 if X is None else X.shape[-1]) if k is None else k
    n = (k if n is None else n) if X is None else X.shape[-1]

    if m < 3 * n:
        raise ValueError(
            f"LPBPCG algorithm is not applicable when the number of A rows (={m})"
            f" is smaller than 3 x the number of requested eigenpairs (={n})"
        )

    method = "ortho" if method is None else method

    iparams = {
        "m": m,
        "n": n,
        "k": k,
        "niter": 1000 if niter is None else niter,
    }

    fparams = {
        "tol": tol,
    }

    bparams = {"largest": True if largest is None else largest}

    if method == "ortho":
        if ortho_iparams is not None:
            iparams.update(ortho_iparams)
        if ortho_fparams is not None:
            fparams.update(ortho_fparams)
        if ortho_bparams is not None:
            bparams.update(ortho_bparams)
        iparams["ortho_i_max"] = iparams.get("ortho_i_max", 3)
        iparams["ortho_j_max"] = iparams.get("ortho_j_max", 3)
        fparams["ortho_tol"] = fparams.get("ortho_tol", tol)
        fparams["ortho_tol_drop"] = fparams.get("ortho_tol_drop", tol)
        fparams["ortho_tol_replace"] = fparams.get("ortho_tol_replace", tol)
        bparams["ortho_use_drop"] = bparams.get("ortho_use_drop", False)

    if not torch.jit.is_scripting():
        LOBPCG.call_tracker = LOBPCG_call_tracker  # type: ignore[method-assign]

    if len(A.shape) > 2:
        N = int(torch.prod(torch.tensor(A.shape[:-2])))
        bA = A.reshape((N,) + A.shape[-2:])
        bB = B.reshape((N,) + A.shape[-2:]) if B is not None else None
        bX = X.reshape((N,) + X.shape[-2:]) if X is not None else None
        bE = torch.empty((N, k), dtype=dtype, device=device)
        bXret = torch.empty((N, m, k), dtype=dtype, device=device)

        for i in range(N):
            A_ = bA[i]
            B_ = bB[i] if bB is not None else None
            X_ = (
                torch.randn((m, n), dtype=dtype, device=device) if bX is None else bX[i]
            )
            assert len(X_.shape) == 2 and X_.shape == (m, n), (X_.shape, (m, n))
            iparams["batch_index"] = i
            worker = LOBPCG(A_, B_, X_, iK, iparams, fparams, bparams, method, tracker)
            worker.run()
            bE[i] = worker.E[:k]
            bXret[i] = worker.X[:, :k]

        if not torch.jit.is_scripting():
            LOBPCG.call_tracker = LOBPCG_call_tracker_orig  # type: ignore[method-assign]

        return bE.reshape(A.shape[:-2] + (k,)), bXret.reshape(A.shape[:-2] + (m, k))

    X = torch.randn((m, n), dtype=dtype, device=device) if X is None else X
    assert len(X.shape) == 2 and X.shape == (m, n), (X.shape, (m, n))

    worker = LOBPCG(A, B, X, iK, iparams, fparams, bparams, method, tracker)

    worker.run()

    if not torch.jit.is_scripting():
        LOBPCG.call_tracker = LOBPCG_call_tracker_orig  # type: ignore[method-assign]

    return worker.E[:k], worker.X[:, :k]


class LOBPCG:
    """Worker class of LOBPCG methods."""

    def __init__(
        self,
        A: Optional[Tensor],
        B: Optional[Tensor],
        X: Tensor,
        iK: Optional[Tensor],
        iparams: dict[str, int],
        fparams: dict[str, float],
        bparams: dict[str, bool],
        method: str,
        tracker: None,
    ) -> None:
        # constant parameters
        self.A = A
        self.B = B
        self.iK = iK
        self.iparams = iparams
        self.fparams = fparams
        self.bparams = bparams
        self.method = method
        self.tracker = tracker
        m = iparams["m"]
        n = iparams["n"]

        # variable parameters
        self.X = X
        self.E = torch.zeros((n,), dtype=X.dtype, device=X.device)
        self.R = torch.zeros((m, n), dtype=X.dtype, device=X.device)
        self.S = torch.zeros((m, 3 * n), dtype=X.dtype, device=X.device)
        self.tvars: dict[str, Tensor] = {}
        self.ivars: dict[str, int] = {"istep": 0}
        self.fvars: dict[str, float] = {"_": 0.0}
        self.bvars: dict[str, bool] = {"_": False}

    def __str__(self):
        lines = ["LOPBCG:"]
        lines += [f"  iparams={self.iparams}"]
        lines += [f"  fparams={self.fparams}"]
        lines += [f"  bparams={self.bparams}"]
        lines += [f"  ivars={self.ivars}"]
        lines += [f"  fvars={self.fvars}"]
        lines += [f"  bvars={self.bvars}"]
        lines += [f"  tvars={self.tvars}"]
        lines += [f"  A={self.A}"]
        lines += [f"  B={self.B}"]
        lines += [f"  iK={self.iK}"]
        lines += [f"  X={self.X}"]
        lines += [f"  E={self.E}"]
        r = ""
        for line in lines:
            r += line + "\n"
        return r

    def update(self):
        """Set and update iteration variables."""
        if self.ivars["istep"] == 0:
            X_norm = float(torch.norm(self.X))
            iX_norm = X_norm**-1
            A_norm = float(torch.norm(_utils.matmul(self.A, self.X))) * iX_norm
            B_norm = float(torch.norm(_utils.matmul(self.B, self.X))) * iX_norm
            self.fvars["X_norm"] = X_norm
            self.fvars["A_norm"] = A_norm
            self.fvars["B_norm"] = B_norm
            self.ivars["iterations_left"] = self.iparams["niter"]
            self.ivars["converged_count"] = 0
            self.ivars["converged_end"] = 0

        if self.method == "ortho":
            self._update_ortho()
        else:
            self._update_basic()

        self.ivars["iterations_left"] = self.ivars["iterations_left"] - 1
        self.ivars["istep"] = self.ivars["istep"] + 1

    def update_residual(self):
        """Update residual R from A, B, X, E."""
        mm = _utils.matmul
        self.R = mm(self.A, self.X) - mm(self.B, self.X) * self.E

    def update_converged_count(self):
        """Determine the number of converged eigenpairs using backward stable
        convergence criterion, see discussion in Sec 4.3 of [DuerschEtal2018].

        Users may redefine this method for custom convergence criteria.
        """
        # (...) -> int
        prev_count = self.ivars["converged_count"]
        tol = self.fparams["tol"]
        A_norm = self.fvars["A_norm"]
        B_norm = self.fvars["B_norm"]
        E, X, R = self.E, self.X, self.R
        rerr = torch.norm(R, 2, (0,)) / (
            torch.norm(X, 2, (0,)) * (A_norm + torch.abs(E[: X.shape[-1]]) * B_norm)
        )
        converged = rerr < tol
        count = 0
        for b in converged:
            if not b:
                # ignore convergence of following pairs to ensure
                # strict ordering of eigenpairs
                break
            count += 1
        assert count >= prev_count, (
            f"the number of converged eigenpairs (was {prev_count}, got {count}) cannot decrease"
        )
        self.ivars["converged_count"] = count
        self.tvars["rerr"] = rerr
        return count

    def stop_iteration(self):
        """Return True to stop iterations.

        Note that tracker (if defined) can force-stop iterations by
        setting ``worker.bvars['force_stop'] = True``.
        """
        return (
            self.bvars.get("force_stop", False)
            or self.ivars["iterations_left"] == 0
            or self.ivars["converged_count"] >= self.iparams["k"]
        )

    def run(self):
        """Run LOBPCG iterations.

        Use this method as a template for implementing LOBPCG
        iteration scheme with custom tracker that is compatible with
        TorchScript.
        """
        self.update()

        if not torch.jit.is_scripting() and self.tracker is not None:
            self.call_tracker()

        while not self.stop_iteration():
            self.update()

            if not torch.jit.is_scripting() and self.tracker is not None:
                self.call_tracker()

    @torch.jit.unused
    def call_tracker(self):
        """Interface for tracking iteration process in Python mode.

        Tracking the iteration process is disabled in TorchScript
        mode. In fact, one should specify tracker=None when JIT
        compiling functions using lobpcg.
        """
        # do nothing when in TorchScript mode

    # Internal methods

    def _update_basic(self):
        """
        Update or initialize iteration variables when `method == "basic"`.
        """
        mm = torch.matmul
        ns = self.ivars["converged_end"]
        nc = self.ivars["converged_count"]
        n = self.iparams["n"]
        largest = self.bparams["largest"]

        if self.ivars["istep"] == 0:
            Ri = self._get_rayleigh_ritz_transform(self.X)
            M = _utils.qform(_utils.qform(self.A, self.X), Ri)
            E, Z = _utils.symeig(M, largest)
            self.X[:] = mm(self.X, mm(Ri, Z))
            self.E[:] = E
            np = 0
            self.update_residual()
            nc = self.update_converged_count()
            self.S[..., :n] = self.X

            W = _utils.matmul(self.iK, self.R)
            self.ivars["converged_end"] = ns = n + np + W.shape[-1]
            self.S[:, n + np : ns] = W
        else:
            S_ = self.S[:, nc:ns]
            Ri = self._get_rayleigh_ritz_transform(S_)
            M = _utils.qform(_utils.qform(self.A, S_), Ri)
            E_, Z = _utils.symeig(M, largest)
            self.X[:, nc:] = mm(S_, mm(Ri, Z[:, : n - nc]))
            self.E[nc:] = E_[: n - nc]
            P = mm(S_, mm(Ri, Z[:, n : 2 * n - nc]))
            np = P.shape[-1]

            self.update_residual()
            nc = self.update_converged_count()
            self.S[..., :n] = self.X
            self.S[:, n : n + np] = P
            W = _utils.matmul(self.iK, self.R[:, nc:])

            self.ivars["converged_end"] = ns = n + np + W.shape[-1]
            self.S[:, n + np : ns] = W

    def _update_ortho(self):
        """
        Update or initialize iteration variables when `method == "ortho"`.
        """
        mm = torch.matmul
        ns = self.ivars["converged_end"]
        nc = self.ivars["converged_count"]
        n = self.iparams["n"]
        largest = self.bparams["largest"]

        if self.ivars["istep"] == 0:
            Ri = self._get_rayleigh_ritz_transform(self.X)
            M = _utils.qform(_utils.qform(self.A, self.X), Ri)
            _E, Z = _utils.symeig(M, largest)
            self.X = mm(self.X, mm(Ri, Z))
            self.update_residual()
            np = 0
            nc = self.update_converged_count()
            self.S[:, :n] = self.X
            W = self._get_ortho(self.R, self.X)
            ns = self.ivars["converged_end"] = n + np + W.shape[-1]
            self.S[:, n + np : ns] = W

        else:
            S_ = self.S[:, nc:ns]
            # Rayleigh-Ritz procedure
            E_, Z = _utils.symeig(_utils.qform(self.A, S_), largest)

            # Update E, X, P
            self.X[:, nc:] = mm(S_, Z[:, : n - nc])
            self.E[nc:] = E_[: n - nc]
            P = mm(S_, mm(Z[:, n - nc :], _utils.basis(Z[: n - nc, n - nc :].mT)))
            np = P.shape[-1]

            # check convergence
            self.update_residual()
            nc = self.update_converged_count()

            # update S
            self.S[:, :n] = self.X
            self.S[:, n : n + np] = P
            W = self._get_ortho(self.R[:, nc:], self.S[:, : n + np])
            ns = self.ivars["converged_end"] = n + np + W.shape[-1]
            self.S[:, n + np : ns] = W

    def _get_rayleigh_ritz_transform(self, S):
        """Return a transformation matrix that is used in Rayleigh-Ritz
        procedure for reducing a general eigenvalue problem :math:`(S^TAS)
        C = (S^TBS) C E` to a standard eigenvalue problem :math: `(Ri^T
        S^TAS Ri) Z = Z E` where `C = Ri Z`.

        .. note:: In the original Rayleight-Ritz procedure in
          [DuerschEtal2018], the problem is formulated as follows::

            SAS = S^T A S
            SBS = S^T B S
            D = (<diagonal matrix of SBS>) ** -1/2
            R^T R = Cholesky(D SBS D)
            Ri = D R^-1
            solve symeig problem Ri^T SAS Ri Z = Theta Z
            C = Ri Z

          To reduce the number of matrix products (denoted by empty
          space between matrices), here we introduce element-wise
          products (denoted by symbol `*`) so that the Rayleight-Ritz
          procedure becomes::

            SAS = S^T A S
            SBS = S^T B S
            d = (<diagonal of SBS>) ** -1/2    # this is 1-d column vector
            dd = d d^T                         # this is 2-d matrix
            R^T R = Cholesky(dd * SBS)
            Ri = R^-1 * d                      # broadcasting
            solve symeig problem Ri^T SAS Ri Z = Theta Z
            C = Ri Z

          where `dd` is 2-d matrix that replaces matrix products `D M
          D` with one element-wise product `M * dd`; and `d` replaces
          matrix product `D M` with element-wise product `M *
          d`. Also, creating the diagonal matrix `D` is avoided.

        Args:
        S (Tensor): the matrix basis for the search subspace, size is
                    :math:`(m, n)`.

        Returns:
        Ri (tensor): upper-triangular transformation matrix of size
                     :math:`(n, n)`.

        """
        B = self.B
        SBS = _utils.qform(B, S)
        d_row = SBS.diagonal(0, -2, -1) ** -0.5
        d_col = d_row.reshape(d_row.shape[0], 1)
        # TODO use torch.linalg.cholesky_solve once it is implemented
        R = torch.linalg.cholesky((SBS * d_row) * d_col, upper=True)
        return torch.linalg.solve_triangular(
            R, d_row.diag_embed(), upper=True, left=False
        )

    def _get_svqb(self, U: Tensor, drop: bool, tau: float) -> Tensor:
        """Return B-orthonormal U.

        .. note:: When `drop` is `False` then `svqb` is based on the
                  Algorithm 4 from [DuerschPhD2015] that is a slight
                  modification of the corresponding algorithm
                  introduced in [StathopolousWu2002].

        Args:

          U (Tensor) : initial approximation, size is (m, n)
          drop (bool) : when True, drop columns that
                     contribution to the `span([U])` is small.
          tau (float) : positive tolerance

        Returns:

          U (Tensor) : B-orthonormal columns (:math:`U^T B U = I`), size
                       is (m, n1), where `n1 = n` if `drop` is `False,
                       otherwise `n1 <= n`.

        """
        if torch.numel(U) == 0:
            return U
        UBU = _utils.qform(self.B, U)
        d = UBU.diagonal(0, -2, -1)

        # Detect and drop exact zero columns from U. While the test
        # `abs(d) == 0` is unlikely to be True for random data, it is
        # possible to construct input data to lobpcg where it will be
        # True leading to a failure (notice the `d ** -0.5` operation
        # in the original algorithm). To prevent the failure, we drop
        # the exact zero columns here and then continue with the
        # original algorithm below.
        nz = torch.where(abs(d) != 0.0)
        assert len(nz) == 1, nz
        if len(nz[0]) < len(d):
            U = U[:, nz[0]]
            if torch.numel(U) == 0:
                return U
            UBU = _utils.qform(self.B, U)
            d = UBU.diagonal(0, -2, -1)
            nz = torch.where(abs(d) != 0.0)
            assert len(nz[0]) == len(d)

        # The original algorithm 4 from [DuerschPhD2015].
        d_col = (d**-0.5).reshape(d.shape[0], 1)
        DUBUD = (UBU * d_col) * d_col.mT
        E, Z = _utils.symeig(DUBUD)
        t = tau * abs(E).max()
        if drop:
            keep = torch.where(E > t)
            assert len(keep) == 1, keep
            E = E[keep[0]]
            Z = Z[:, keep[0]]
            d_col = d_col[keep[0]]
        else:
            E[(torch.where(E < t))[0]] = t

        return torch.matmul(U * d_col.mT, Z * E**-0.5)

    def _get_ortho(self, U, V):
        """Return B-orthonormal U with columns are B-orthogonal to V.

        .. note:: When `bparams["ortho_use_drop"] == False` then
                  `_get_ortho` is based on the Algorithm 3 from
                  [DuerschPhD2015] that is a slight modification of
                  the corresponding algorithm introduced in
                  [StathopolousWu2002]. Otherwise, the method
                  implements Algorithm 6 from [DuerschPhD2015]

        .. note:: If all U columns are B-collinear to V then the
                  returned tensor U will be empty.

        Args:

          U (Tensor) : initial approximation, size is (m, n)
          V (Tensor) : B-orthogonal external basis, size is (m, k)

        Returns:

          U (Tensor) : B-orthonormal columns (:math:`U^T B U = I`)
                       such that :math:`V^T B U=0`, size is (m, n1),
                       where `n1 = n` if `drop` is `False, otherwise
                       `n1 <= n`.
        """
        mm = torch.matmul
        mm_B = _utils.matmul
        m = self.iparams["m"]
        tau_ortho = self.fparams["ortho_tol"]
        tau_drop = self.fparams["ortho_tol_drop"]
        tau_replace = self.fparams["ortho_tol_replace"]
        i_max = self.iparams["ortho_i_max"]
        j_max = self.iparams["ortho_j_max"]
        # when use_drop==True, enable dropping U columns that have
        # small contribution to the `span([U, V])`.
        use_drop = self.bparams["ortho_use_drop"]

        # clean up variables from the previous call
        for vkey in list(self.fvars.keys()):
            if vkey.startswith("ortho_") and vkey.endswith("_rerr"):
                self.fvars.pop(vkey)
        self.ivars.pop("ortho_i", 0)
        self.ivars.pop("ortho_j", 0)

        BV_norm = torch.norm(mm_B(self.B, V))
        BU = mm_B(self.B, U)
        VBU = mm(V.mT, BU)
        i = j = 0
        for i in range(i_max):
            U = U - mm(V, VBU)
            drop = False
            tau_svqb = tau_drop
            for j in range(j_max):
                if use_drop:
                    U = self._get_svqb(U, drop, tau_svqb)
                    drop = True
                    tau_svqb = tau_replace
                else:
                    U = self._get_svqb(U, False, tau_replace)
                if torch.numel(U) == 0:
                    # all initial U columns are B-collinear to V
                    self.ivars["ortho_i"] = i
                    self.ivars["ortho_j"] = j
                    return U
                BU = mm_B(self.B, U)
                UBU = mm(U.mT, BU)
                U_norm = torch.norm(U)
                BU_norm = torch.norm(BU)
                R = UBU - torch.eye(UBU.shape[-1], device=UBU.device, dtype=UBU.dtype)
                R_norm = torch.norm(R)
                # https://github.com/pytorch/pytorch/issues/33810 workaround:
                rerr = float(R_norm) * float(BU_norm * U_norm) ** -1
                vkey = f"ortho_UBUmI_rerr[{i}, {j}]"
                self.fvars[vkey] = rerr
                if rerr < tau_ortho:
                    break
            VBU = mm(V.mT, BU)
            VBU_norm = torch.norm(VBU)
            U_norm = torch.norm(U)
            rerr = float(VBU_norm) * float(BV_norm * U_norm) ** -1
            vkey = f"ortho_VBU_rerr[{i}]"
            self.fvars[vkey] = rerr
            if rerr < tau_ortho:
                break
            if m < U.shape[-1] + V.shape[-1]:
                # TorchScript needs the class var to be assigned to a local to
                # do optional type refinement
                B = self.B
                assert B is not None
                raise ValueError(
                    "Overdetermined shape of U:"
                    f" #B-cols(={B.shape[-1]}) >= #U-cols(={U.shape[-1]}) + #V-cols(={V.shape[-1]}) must hold"
                )
        self.ivars["ortho_i"] = i
        self.ivars["ortho_j"] = j
        return U


# Calling tracker is separated from LOBPCG definitions because
# TorchScript does not support user-defined callback arguments:
LOBPCG_call_tracker_orig = LOBPCG.call_tracker


def LOBPCG_call_tracker(self):
    self.tracker(self)
