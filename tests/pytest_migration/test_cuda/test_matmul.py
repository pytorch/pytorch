"""
CUDA matrix multiplication tests using pytest.
These tests verify the correctness of torch.matmul operations on CUDA devices.
"""
import pytest
import torch
import itertools
from typing import Tuple

@pytest.mark.cuda
def test_matmul_2d(cuda_device):
    """Test basic 2D matrix multiplication."""
    a = torch.randn(2, 3, device=cuda_device)
    b = torch.randn(3, 2, device=cuda_device)
    
    c = torch.matmul(a, b)
    
    # Verify shape
    assert c.shape == (2, 2)
    
    # Verify result against CPU computation
    c_expected = torch.matmul(a.cpu(), b.cpu())
    assert torch.allclose(c.cpu(), c_expected, rtol=1e-5, atol=1e-5)

@pytest.mark.cuda
@pytest.mark.parametrize(
    "shape_a,shape_b,expected_shape",
    [
        ((2, 3), (3, 4), (2, 4)),
        ((4, 2, 3), (3, 4), (4, 2, 4)),
        ((2, 3, 4), (4, 5), (2, 3, 5)),
        ((1, 2, 3, 4), (4, 5), (1, 2, 3, 5)),
    ],
    ids=lambda x: str(x)
)
def test_matmul_shapes(
    cuda_device: torch.device,
    shape_a: Tuple[int, ...],
    shape_b: Tuple[int, ...],
    expected_shape: Tuple[int, ...]
):
    """Test matrix multiplication with various tensor shapes."""
    a = torch.randn(*shape_a, device=cuda_device)
    b = torch.randn(*shape_b, device=cuda_device)
    
    c = torch.matmul(a, b)
    assert c.shape == expected_shape
    
    # Verify against CPU computation
    c_expected = torch.matmul(a.cpu(), b.cpu())
    assert torch.allclose(c.cpu(), c_expected, rtol=1e-5, atol=1e-5)

@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype1,dtype2",
    list(itertools.product([torch.float32, torch.float64], repeat=2)),
    ids=lambda x: str(x)
)
def test_matmul_dtypes(
    cuda_device: torch.device,
    dtype1: torch.dtype,
    dtype2: torch.dtype
):
    """Test matrix multiplication with different dtype combinations."""
    shapes = (2, 3)
    a = torch.randn(*shapes, device=cuda_device, dtype=dtype1)
    b = torch.randn(shapes[1], shapes[0], device=cuda_device, dtype=dtype2)
    
    c = torch.matmul(a, b)
    
    # Result should be in the higher precision dtype
    expected_dtype = torch.float64 if torch.float64 in (dtype1, dtype2) else torch.float32
    assert c.dtype == expected_dtype

    # Verify computation
    c_expected = torch.matmul(a.cpu(), b.cpu())
    assert torch.allclose(c.cpu(), c_expected, rtol=1e-5, atol=1e-5)

@pytest.mark.cuda
def test_matmul_errors(cuda_device: torch.device):
    """Test that matrix multiplication fails appropriately with invalid inputs."""
    # Test incompatible shapes
    a = torch.randn(2, 3, device=cuda_device)
    b = torch.randn(2, 3, device=cuda_device)  # Incorrect inner dimension
    
    with pytest.raises(RuntimeError):
        torch.matmul(a, b)
    
    # Test invalid dimensions
    a = torch.randn(2, device=cuda_device)  # 1D tensor
    b = torch.randn(2, 2, device=cuda_device)  # 2D tensor
    
    with pytest.raises(RuntimeError):
        torch.matmul(a, b)