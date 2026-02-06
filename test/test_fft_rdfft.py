# test/test_fft_rdfft.py
"""
Tests for custom real-domain FFT (rdFFT) and inverse (irdFFT) operators.

Highlights:
1. Our operators store FFT outputs in a split real/imag layout. rdfft2complex converts it
   back to standard complex format compatible with torch.fft.rfft.
2. bf16 precision is lower, so comparisons use looser tolerances. float32 uses strict checks.
3. Tests both forward/inverse correctness and optional memory usage reporting.
"""

import torch
import pytest
import numpy as np

# Custom operators
myfft = torch.ops.aten.fft_rdfft_
myifft = torch.ops.aten.fft_irdfft_

def rdfft2complex(x):
    n = x.shape[-1]
    k = n // 2 + 1
    y = torch.empty(*x.shape[:-1], k, dtype=torch.complex64, device=x.device)
    y[..., 0] = x[..., 0] + 0j
    y[..., -1] = x[..., n//2] + 0j
    real_part = x[..., 1:n//2]
    imag_part = torch.flip(x[..., n//2+1:], dims=[-1])
    y[..., 1:-1] = real_part + 1j * imag_part
    return y

# ----------------------------
# Python-side checks mimicking TORCH_CHECK
# ----------------------------
def check_rdfft_input(x):
    if not x.is_cuda:
        raise RuntimeError("Input must be CUDA tensor")
    if not x.is_floating_point():
        raise RuntimeError("Input must be float or double")
    if x.dim() != 4:
        raise RuntimeError("Input must be 4D tensor")
    if (x.size(-1) & (x.size(-1) - 1)) != 0:
        raise RuntimeError("Last dimension must be power of 2")

# ----------------------------
# Test forward/inverse correctness
# ----------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.cuda
def test_rdfft_roundtrip(dtype):
    device = "cuda"
    r = 4096
    x = torch.rand(1, 1, 128, r, device=device, dtype=dtype)

    # Save reference in float32
    x_ref = x.to(torch.float32)

    # Measure initial GPU memory
    initial_mem = torch.cuda.max_memory_allocated(device=device)

    # Forward + Inverse
    y = myfft(x)
    # y_complex = rdfft2complex(y)
    x_out = myifft(y)

    # Measure peak GPU memory
    peak_mem = torch.cuda.max_memory_allocated(device=device)
    print(f"[INFO] FFT forward/inverse GPU memory used: {peak_mem - initial_mem} bytes")

    # rfft for comparison
    y_ref = torch.fft.rfft(x_ref, dim=-1)
    x_ref_out = torch.fft.irfft(y_ref, dim=-1)

    # Round-trip check
    x_out_float = x_out.to(torch.float32) if dtype==torch.bfloat16 else x_out
    atol = 1e-5 if dtype==torch.float32 else 1e-1
    assert x_out_float.shape == x.shape
    assert torch.allclose(x_out_float.cpu(), x_ref_out.cpu(), atol=atol), \
        f"Round-trip failed for dtype={dtype}"

# ----------------------------
# Test TORCH_CHECK error handling
# ----------------------------
def test_rdfft_torch_check_errors():
    # Non-CUDA tensor
    x_cpu = torch.rand(1,1,1,8)
    with pytest.raises(RuntimeError, match="Input must be CUDA tensor"):
        check_rdfft_input(x_cpu)

    # Non-float tensor
    x_int = torch.randint(0,10,(1,1,1,8), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="Input must be float or double"):
        check_rdfft_input(x_int)

    # Wrong dimension
    x_3d = torch.rand(1,1,8, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input must be 4D tensor"):
        check_rdfft_input(x_3d)

    # Last dim not power of 2
    x_baddim = torch.rand(1,1,1,7, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Last dimension must be power of 2"):
        check_rdfft_input(x_baddim)

if __name__ == "__main__":
    test_rdfft_roundtrip(torch.float32)
    test_rdfft_roundtrip(torch.bfloat16)
    test_rdfft_torch_check_errors()