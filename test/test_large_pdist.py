import torch

def _compare_large_tensors(expected, actual, num_splits):
    length = expected.shape[0]
    for i in range(num_splits):
        start = (i * length) // num_splits
        end = ((i + 1) * length) // num_splits
        print("  " + str(i), flush=True)
        torch.allclose(expected[start:end], actual[start:end], rtol=1.3e-06, atol=1e-05, equal_nan=True)

def test_pdist_norm_large(size, num_splits):
    device = "cuda:0"

    print("----------------------", flush=True)
    print("test_pdist", flush=True)
    print(size, flush=True)
    print("----------------------", flush=True)

    x = torch.randn(size, 1, dtype=torch.float32)
    print(" Point 1", flush=True)
    expected_cpu = torch.pdist(x, p=2)
    print(" Point 2", flush=True)
    actual_gpu = torch.pdist(x.to(device), p=2)
    print(" Point 3", flush=True)
    _compare_large_tensors(expected_cpu, actual_gpu.cpu(), num_splits)
    print(" Pass", flush=True)
            

import sys

test_pdist_norm_large(int(sys.argv[1]), int(sys.argv[2]))