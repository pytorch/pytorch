# flake8: noqa
"""Reveal tests for torch.linalg, torch.fft, and torch.special return types."""

import torch


t = torch.randn(3, 3)
v = torch.randn(3)

# torch.linalg decompositions with named tuple returns
svd_result = torch.linalg.svd(t)
reveal_type(svd_result)  # E: fallback=torch.linalg.SVDResult

qr_result = torch.linalg.qr(t)
reveal_type(qr_result)  # E: fallback=torch.linalg.QRResult

eig_result = torch.linalg.eig(t)
reveal_type(eig_result)  # E: fallback=torch.linalg.EigResult

eigh_result = torch.linalg.eigh(t)
reveal_type(eigh_result)  # E: fallback=torch.linalg.EighResult

lu_result = torch.linalg.lu_factor(t)
reveal_type(lu_result)  # E: fallback=torch.linalg.LUResult

slogdet_result = torch.linalg.slogdet(t)
reveal_type(slogdet_result)  # E: fallback=torch.linalg.SlogdetResult

# torch.linalg functions returning Tensor
norm_val = torch.linalg.norm(t)
reveal_type(norm_val)  # E: torch._tensor.Tensor

det_val = torch.linalg.det(t)
reveal_type(det_val)  # E: torch._tensor.Tensor

inv_t = torch.linalg.inv(t)
reveal_type(inv_t)  # E: torch._tensor.Tensor

# torch.fft functions
fft_result = torch.fft.fft(v)
reveal_type(fft_result)  # E: torch._tensor.Tensor

fft2_result = torch.fft.fft2(t)
reveal_type(fft2_result)  # E: torch._tensor.Tensor

freqs = torch.fft.fftfreq(8)
reveal_type(freqs)  # E: torch._tensor.Tensor

# torch.special functions
erf_val = torch.special.erf(v)
reveal_type(erf_val)  # E: torch._tensor.Tensor

gammaln_val = torch.special.gammaln(torch.abs(v) + 0.1)
reveal_type(gammaln_val)  # E: torch._tensor.Tensor

i0_val = torch.special.i0(v)
reveal_type(i0_val)  # E: torch._tensor.Tensor
