import torch
import sys

def _compare_large_tensors(expected, actual, num_splits):
    length = expected.shape[0]
    for i in range(num_splits):
        start = (i * length) // num_splits
        end = ((i + 1) * length) // num_splits
        sys.stdout.write("  " + str(i))
        torch.allclose(expected[start:end], actual[start:end], rtol=1.3e-06, atol=1e-05, equal_nan=True)

def _test_large_cum_fn_helper(x, fn, num_splits):
    x_cpu = x.cpu().float()
    sys.stdout.write(" Point 2" + "\n")
    expected = fn(x_cpu)
    sys.stdout.write(" Point 3" + "\n")
    actual = fn(x).cpu().float()
    sys.stdout.write(" Point 4" + "\n")
    _compare_large_tensors(expected, actual, num_splits)
    sys.stdout.write(" Pass" + "\n")

def test_large_cumprod(p, num_splits):
    device = "cuda:0"
    dtype = torch.float16
    sys.stdout.write("----------------------" + "\n")
    sys.stdout.write("test_large" + "\n")
    sys.stdout.write(str(p) + "\n")
    sys.stdout.write("----------------------" + "\n")

    x = torch.empty(2**p + 200, device=device, dtype=dtype)
    x[::3] = 8
    x[1::3] = .25
    x[2::3] = .5
    sys.stdout.write(" Point 1" + "\n")
    _test_large_cum_fn_helper(x, lambda x: torch.cumprod(x, 0), num_splits)

test_large_cumprod(int(sys.argv[1]), int(sys.argv[2]))