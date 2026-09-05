# flake8: noqa
"""Pass tests for torch.linalg, torch.fft, and torch.special typing."""

from collections.abc import Sequence

import torch


# torch.linalg operations
def test_linalg(axes: Sequence[int], tensors: Sequence[torch.Tensor]) -> None:
    t = torch.randn(3, 3)
    v = torch.randn(3)

    # Decompositions
    svd_result = torch.linalg.svd(t)
    qr_result = torch.linalg.qr(t)
    eig_result = torch.linalg.eig(t)
    eigh_result = torch.linalg.eigh(t)
    lu_result = torch.linalg.lu_factor(t)
    chol = torch.linalg.cholesky(t @ t.T)  # Ensure positive definite

    # Norms
    norm_val = torch.linalg.norm(t)
    vec_norm = torch.linalg.vector_norm(v)
    mat_norm = torch.linalg.matrix_norm(t)
    # Test with list dim (this was the issue in reverted PR #160750)
    vec_norm_list = torch.linalg.vector_norm(t, dim=[0, 1])

    # Matrix properties
    det_val = torch.linalg.det(t)
    slogdet_result = torch.linalg.slogdet(t)

    # Inverses and solvers
    inv_t = torch.linalg.inv(t)
    pinv_t = torch.linalg.pinv(t)
    solve_result = torch.linalg.solve(t, v)

    # Products
    multi_dot = torch.linalg.multi_dot([t, t])
    torch.linalg.multi_dot((t, t))
    torch.linalg.multi_dot(tensors)
    torch.linalg.tensorsolve(t, v, dims=axes)
    torch.linalg.vector_norm(t, dim=axes)
    torch.linalg.det(A=t)
    torch.linalg.svd(A=t)
    torch.linalg.svd(t, out=(t, v, t))
    torch.linalg.qr(A=t)
    torch.linalg.solve(A=t, B=v)
    torch.linalg.pinv(t, rcond=1e-5)
    torch.linalg.matrix_rank(t, tol=1e-5)
    torch.linalg.ldl_factor(t)
    torch.linalg.ldl_factor_ex(t)
    torch.linalg.ldl_solve(t, torch.ones(3, dtype=torch.int32), t)
    torch.linalg.lu_factor_ex(t)
    torch.linalg.matrix_sqrth(t)
    torch.linalg.polar(t)
    error: type[RuntimeError] = torch.linalg.LinAlgError
    cross_result = torch.linalg.cross(v, v)


# torch.fft operations
def test_fft(axes: Sequence[int]) -> None:
    t = torch.randn(8)
    t2d = torch.randn(8, 8)

    # 1D FFT
    fft_result = torch.fft.fft(t)
    ifft_result = torch.fft.ifft(fft_result)
    rfft_result = torch.fft.rfft(t)
    irfft_result = torch.fft.irfft(rfft_result)

    # 2D FFT
    fft2_result = torch.fft.fft2(t2d)
    ifft2_result = torch.fft.ifft2(fft2_result)

    # N-D FFT
    fftn_result = torch.fft.fftn(t2d, dim=axes)
    ifftn_result = torch.fft.ifftn(fftn_result)
    torch.fft.ifftn(t2d, dim=axes)
    torch.fft.rfftn(t2d, dim=axes)
    torch.fft.irfftn(t2d, dim=axes)
    torch.fft.hfftn(t2d, dim=axes)
    torch.fft.ihfftn(t2d, dim=axes)

    # Helper functions
    freqs = torch.fft.fftfreq(8)
    rfreqs = torch.fft.rfftfreq(8)
    torch.fft.fftfreq(8, out=torch.empty(8))
    torch.fft.rfftfreq(8, out=torch.empty(5))
    shifted = torch.fft.fftshift(fft_result)
    unshifted = torch.fft.ifftshift(shifted)


# torch.special operations
def test_special() -> None:
    t = torch.randn(5)
    t_pos = torch.abs(t) + 0.1  # positive values for gamma functions

    # Error functions
    erf_val = torch.special.erf(t)
    erfc_val = torch.special.erfc(t)
    erfinv_val = torch.special.erfinv(t.clamp(-1, 1))

    # Exponential and logarithmic
    exp2_val = torch.special.exp2(t)
    expm1_val = torch.special.expm1(t)
    log1p_val = torch.special.log1p(t_pos)
    logit_val = torch.special.logit(torch.sigmoid(t))

    # Gamma functions
    gammaln_val = torch.special.gammaln(t_pos)
    digamma_val = torch.special.digamma(t_pos)

    # Bessel functions
    i0_val = torch.special.i0(t)
    i1_val = torch.special.i1(t)

    # Other special functions
    ndtr_val = torch.special.ndtr(t)
    sinc_val = torch.special.sinc(t)
    torch.special.psi(t_pos)
    torch.special.airy_ai(x=t)
    torch.special.xlog1py(input=1.0, other=t_pos)
    torch.special.xlog1py(input=t_pos, other=1.0)
    torch.special.zeta(t_pos, 2.0)
    torch.special.chebyshev_polynomial_t(t, 2)
