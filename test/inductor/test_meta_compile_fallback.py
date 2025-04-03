"""
Test for meta tensor fallback in torch.compile.

This test verifies that when a function compiled with torch.compile
receives an input tensor on the "meta" device, it falls back to eager execution,
thus avoiding the lowering exception described in issue #144607.
"""

import torch
import pytest

# Define a simple function that doubles its input.
@torch.compile
def foobar(x):
    return x * 2

def run_test(device: str):
    # Call the compiled function twice with different input shapes.
    out1 = foobar(torch.empty((1, 16, 128, 128), device=device))
    out2 = foobar(torch.empty((1, 32, 64, 64), device=device))
    return out1, out2

def test_cuda():
    # Skip this test if CUDA is not available.
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    out1, out2 = run_test("cuda")
    # Check that the outputs are on CUDA and have the expected shapes.
    assert out1.device.type == "cuda"
    assert out1.shape == (1, 16, 128, 128)
    assert out2.device.type == "cuda"
    assert out2.shape == (1, 32, 64, 64)

def test_meta():
    # When a meta tensor is passed, the fallback should trigger.
    out1, out2 = run_test("meta")
    # We expect that the returned outputs are still meta tensors with the correct shapes.
    assert out1.device.type == "meta"
    assert out1.shape == (1, 16, 128, 128)
    assert out2.device.type == "meta"
    assert out2.shape == (1, 32, 64, 64)