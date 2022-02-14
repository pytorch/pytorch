import torch

def _test_large_cum_fn_helper(x, fn):
    x_cpu = x.cpu().float()
    expected = fn(x_cpu)
    actual = fn(x).cpu().float()
    torch.allclose(expected, actual, rtol=1.3e-06, atol=1e-05, equal_nan=True)
    print(" Pass")

def test_large_cumprod(val):
    device = "cuda:0"
    dtype = torch.float16
    print("----------------------")
    print("test_large")
    print(val)
    print("----------------------")

    p = int(val)

    x = torch.empty(2**p + 200, device=device, dtype=dtype)
    print(x.shape[0])
    x[::3] = 8
    x[1::3] = .25
    x[2::3] = .5
    _test_large_cum_fn_helper(x, lambda x: torch.cumprod(x, 0))

import sys

test_large_cumprod(sys.argv[1])