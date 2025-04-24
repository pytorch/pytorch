import math
import random
import unittest
import warnings
from functools import partial
from itertools import product
from typing import List

import torch
import numpy as np

from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
)
from torch.testing._internal.common_dtype import all_types_and_complex
from torch.testing._internal.common_utils import (
    TestCase,
    make_tensor,
    run_tests,
    TEST_WITH_ROCM,
    suppress_warnings,
)

def divup_reference(a, b):
    """Reference implementation of ceiling division."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.ceil(a / b)
    return math.ceil(a / b)

class TestDivUp(TestCase):
    """Tests for torch.divup and torch.ceiling_divide operations."""
    
    @dtypes(*all_types_and_complex())
    def test_divup_basic(self, device, dtype):
        """Test basic divup operation."""
        a = make_tensor((10, 10), dtype=dtype, device=device, low=None, high=None)
        b = make_tensor((10, 10), dtype=dtype, device=device, low=None, high=None) + 1  # Avoid zeros
        
        # Test ceiling_divide
        result = torch.ceiling_divide(a, b)
        expected = divup_reference(a, b)
        self.assertEqual(result, expected)
        
        # Test divup alias
        result = torch.divup(a, b)
        self.assertEqual(result, expected)
        
        # Test in-place
        c = a.clone()
        c.ceiling_divide_(b)
        self.assertEqual(c, expected)
        
        # Test scalar
        scalar = random.random() + 0.5  # Positive random number
        result = torch.ceiling_divide(a, scalar)
        expected = divup_reference(a, scalar)
        self.assertEqual(result, expected)
    
    @dtypes(torch.int32, torch.int64)
    def test_divup_integer(self, device, dtype):
        """Test divup with integer inputs."""
        a = torch.tensor([5, 10, 11, -5, -10, -11], dtype=dtype, device=device)
        b = torch.tensor([2, 3, 3, 2, 3, 3], dtype=dtype, device=device)
        
        # For integers, ceiling division should be (a + b - 1) // b when a and b are positive
        result = torch.ceiling_divide(a, b)
        expected = torch.tensor([3, 4, 4, -2, -3, -3], dtype=dtype, device=device)
        self.assertEqual(result, expected)
        
        # Test divup alias
        result = torch.divup(a, b)
        self.assertEqual(result, expected)
        
        # Test scalar
        scalar = 2
        result = torch.ceiling_divide(a, scalar)
        expected = torch.tensor([3, 5, 6, -2, -5, -5], dtype=dtype, device=device)
        self.assertEqual(result, expected)
    
    @dtypes(torch.float32, torch.float64)
    def test_divup_edge_cases(self, device, dtype):
        """Test divup with edge cases."""
        # Test with zero
        a = torch.tensor([0.0, 1.0, -1.0], dtype=dtype, device=device)
        b = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
        result = torch.ceiling_divide(a, b)
        expected = torch.tensor([0.0, 1.0, -1.0], dtype=dtype, device=device)
        self.assertEqual(result, expected)
        
        # Test with infinity
        a = torch.tensor([float('inf'), -float('inf'), 1.0], dtype=dtype, device=device)
        result = torch.ceiling_divide(a, b)
        expected = torch.tensor([float('inf'), -float('inf'), 1.0], dtype=dtype, device=device)
        self.assertEqual(result, expected)

instantiate_device_type_tests(TestDivUp, globals())

if __name__ == "__main__":
    run_tests()