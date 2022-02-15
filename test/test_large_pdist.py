import torch

def _compare_large_tensors(expected, actual, num_splits):
    length = expected.shape[0]
    for i in range(num_splits):
        start = (i * length) // num_splits
        end = ((i + 1) * length) // num_splits
        sys.stdout.write("  " + str(i) + "\n")
        torch.allclose(expected[start:end], actual[start:end], rtol=1.3e-06, atol=1e-05, equal_nan=True)

def test_pdist_norm_large(size, num_splits):
    device = "cuda:0"

    sys.stdout.write("----------------------" + "\n")
    sys.stdout.write("test_pdist" + "\n")
    sys.stdout.write(str(size) + "\n")
    sys.stdout.write("----------------------" + "\n")

    x = torch.randn(size, 1, dtype=torch.float32)
    sys.stdout.write(" Point 1" + "\n")
    expected_cpu = torch.pdist(x, p=2)
    sys.stdout.write(" Point 2" + "\n")
    actual_gpu = torch.pdist(x.to(device), p=2)
    sys.stdout.write(" Point 3" + "\n")
    _compare_large_tensors(expected_cpu, actual_gpu.cpu(), num_splits)
    sys.stdout.write(" Pass" + "\n")
            

import sys

test_pdist_norm_large(int(sys.argv[1]), int(sys.argv[2]))