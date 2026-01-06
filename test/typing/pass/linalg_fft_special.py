# flake8: noqa
"""Pass tests for torch.linalg, torch.fft, and torch.special typing."""

import torch


# torch.linalg operations
def test_linalg() -> None:
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
    cross_result = torch.linalg.cross(v, v)


# torch.fft operations
def test_fft() -> None:
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
    fftn_result = torch.fft.fftn(t2d)
    ifftn_result = torch.fft.ifftn(fftn_result)

    # Helper functions
    freqs = torch.fft.fftfreq(8)
    rfreqs = torch.fft.rfftfreq(8)
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
