import torch

def _compare_large_tensors(expected, actual, num_splits):
    length = expected.shape[0]
    for i in range(num_splits):
        start = (i * length) // num_splits
        end = ((i + 1) * length) // num_splits
        print("  " + str(i))
        torch.allclose(expected[start:end], actual[start:end], rtol=1.3e-06, atol=1e-05, equal_nan=True)

def _test_large_cum_fn_helper(x, fn, num_splits):
    x_cpu = x.cpu().float()
    print(" Point 2", flush=True)
    expected = fn(x_cpu)
    print(" Point 3", flush=True)
    actual = fn(x).cpu().float()
    print(" Point 4", flush=True)
    _compare_large_tensors(expected, actual, num_splits)
    print(" Pass", flush=True)

def test_large_cumprod(p, num_splits):
    device = "cuda:0"
    dtype = torch.float16
    print("----------------------", flush=True)
    print("test_large", flush=True)
    print(p, flush=True)
    print("----------------------", flush=True)

    x = torch.empty(2**p + 200, device=device, dtype=dtype)
    x[::3] = 8
    x[1::3] = .25
    x[2::3] = .5
    print(" Point 1", flush=True)
    _test_large_cum_fn_helper(x, lambda x: torch.cumprod(x, 0), num_splits)

import sys

test_large_cumprod(int(sys.argv[1]), int(sys.argv[2]))