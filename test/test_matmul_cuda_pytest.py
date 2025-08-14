"""
CUDA matrix multiplication tests using pytest.
These tests verify the correctness of torch.matmul operations on CUDA devices.
"""
import pytest
import torch
import itertools
from typing import Tuple
from torch.testing._internal.common_utils import TEST_CUDA

@pytest.mark.skipif(not TEST_CUDA, reason="CUDA not available")
class TestMatmulCuda:
    def test_matmul_2d(self):
        """Test basic 2D matrix multiplication."""
        a = torch.randn(2, 3, device="cuda")
        b = torch.randn(3, 2, device="cuda")
        
        c = torch.matmul(a, b)
        
        # Verify shape
        assert c.shape == (2, 2)
        
        # Verify result against CPU computation
        c_expected = torch.matmul(a.cpu(), b.cpu())
        assert torch.allclose(c.cpu(), c_expected, rtol=1e-5, atol=1e-5)

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
        self,
        shape_a: Tuple[int, ...],
        shape_b: Tuple[int, ...],
        expected_shape: Tuple[int, ...]
    ):
        """Test matrix multiplication with various tensor shapes."""
        a = torch.randn(*shape_a, device="cuda")
        b = torch.randn(*shape_b, device="cuda")
        
        c = torch.matmul(a, b)
        assert c.shape == expected_shape
        
        # Verify against CPU computation
        c_expected = torch.matmul(a.cpu(), b.cpu())
        assert torch.allclose(c.cpu(), c_expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "dtype1,dtype2",
        list(itertools.product([torch.float32, torch.float64], repeat=2)),
        ids=lambda x: str(x)
    )
    def test_matmul_dtypes(
        self,
        dtype1: torch.dtype,
        dtype2: torch.dtype
    ):
        """Test matrix multiplication with different dtype combinations."""
        shapes = (2, 3)
        a = torch.randn(*shapes, device="cuda", dtype=dtype1)
        b = torch.randn(shapes[1], shapes[0], device="cuda", dtype=dtype2)
        
        c = torch.matmul(a, b)
        
        # Result should be in the higher precision dtype
        expected_dtype = torch.float64 if torch.float64 in (dtype1, dtype2) else torch.float32
        assert c.dtype == expected_dtype

        # Verify computation
        c_expected = torch.matmul(a.cpu(), b.cpu())
        assert torch.allclose(c.cpu(), c_expected, rtol=1e-5, atol=1e-5)

    def test_matmul_errors(self):
        """Test that matrix multiplication fails appropriately with invalid inputs."""
        # Test incompatible shapes
        a = torch.randn(2, 3, device="cuda")
        b = torch.randn(2, 3, device="cuda")  # Incorrect inner dimension
        
        with pytest.raises(RuntimeError):
            torch.matmul(a, b)
        
        # Test invalid dimensions
        a = torch.randn(2, device="cuda")  # 1D tensor
        b = torch.randn(2, 2, device="cuda")  # 2D tensor
        
        with pytest.raises(RuntimeError):
            torch.matmul(a, b)
