#!/usr/bin/env python3

"""
Tests for the MultiDimKernelDispatcher functionality.
"""

import unittest
from typing import Callable, Sequence
import math

import sys
import os

# Add the torch path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch", "_inductor"))

from torch._inductor.kernel_dispatcher import (
    MultiDimKernelDispatcher,
    create_symint_specialization_points,
    get_symint_range_bounds,
)


# Mock kernel classes for testing
class MockKernel:
    def __init__(self, name: str):
        self.name = name
        self.run_calls = []
    
    def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        return f"executed_{self.name}"


class TestMultiDimKernelDispatcher(unittest.TestCase):
    
    def setUp(self):
        self.kernel_128 = MockKernel("kernel_128")
        self.kernel_1024 = MockKernel("kernel_1024") 
        self.kernel_4096 = MockKernel("kernel_4096")
        
        # Create dispatcher with different specialization points
        self.dispatcher = MultiDimKernelDispatcher([
            (self.kernel_128, (128, 0, 0)),
            (self.kernel_1024, (1024, 0, 0)),
            (self.kernel_4096, (4096, 0, 0)),
        ])
    
    def test_basic_dispatch_closest_match(self):
        """Test that dispatch selects the closest kernel."""
        # Test point close to 128
        kernel, grid = self.dispatcher.dispatch((100, 0, 0))
        self.assertEqual(kernel, self.kernel_128)
        
        # Test point close to 1024
        kernel, grid = self.dispatcher.dispatch((900, 0, 0))
        self.assertEqual(kernel, self.kernel_1024)
        
        # Test point close to 4096
        kernel, grid = self.dispatcher.dispatch((3500, 0, 0))
        self.assertEqual(kernel, self.kernel_4096)
    
    def test_euclidean_distance_calculation(self):
        """Test that Euclidean distance is calculated correctly."""
        # Point (500, 0, 0) distances:
        # Distance to 128: sqrt((500-128)^2) = 372
        # Distance to 1024: sqrt((500-1024)^2) = 524  
        # Distance to 4096: sqrt((500-4096)^2) = 3596
        # 128 is actually closer to 500 than 1024!
        kernel, grid = self.dispatcher.dispatch((500, 0, 0))
        self.assertEqual(kernel, self.kernel_128)
        
        # Point (200, 0, 0) should be closest to (128, 0, 0)
        kernel, grid = self.dispatcher.dispatch((200, 0, 0))
        self.assertEqual(kernel, self.kernel_128)
    
    def test_grid_computation(self):
        """Test that grid dimensions are calculated correctly."""
        kernel, grid = self.dispatcher.dispatch((100, 0, 0))
        # Grid calculation: 64 * ((127 + 100) // 128) = 64 * (227 // 128) = 64 * 1 = 64
        expected_grid = (64, 1, 1)
        self.assertEqual(grid, expected_grid)
        
        kernel, grid = self.dispatcher.dispatch((300, 0, 0))
        # Grid calculation: 64 * ((127 + 300) // 128) = 64 * (427 // 128) = 64 * 3 = 192
        expected_grid = (192, 1, 1)
        self.assertEqual(grid, expected_grid)
    
    def test_empty_kernels_list(self):
        """Test error handling for empty kernels list."""
        empty_dispatcher = MultiDimKernelDispatcher([])
        with self.assertRaises(RuntimeError):
            empty_dispatcher.dispatch((100, 0, 0))
    
    def test_multidimensional_dispatch(self):
        """Test dispatch with multiple dimensions."""
        multi_dispatcher = MultiDimKernelDispatcher([
            (self.kernel_128, (128, 64, 32)),
            (self.kernel_1024, (1024, 512, 256)),
            (self.kernel_4096, (4096, 2048, 1024)),
        ])
        
        # Test point closest to first kernel
        kernel, grid = multi_dispatcher.dispatch((100, 50, 25))
        self.assertEqual(kernel, self.kernel_128)
        
        # Test point closest to second kernel
        kernel, grid = multi_dispatcher.dispatch((1000, 500, 250))
        self.assertEqual(kernel, self.kernel_1024)


class TestSymintSpecializationPoints(unittest.TestCase):
    
    def test_create_specialization_points_basic(self):
        """Test creating basic specialization points."""
        points = create_symint_specialization_points(0, 1024, 4)
        expected = [128.0, 384.0, 640.0, 896.0]  # Midpoints of 4 ranges
        self.assertEqual(points, expected)
    
    def test_create_specialization_points_single(self):
        """Test creating a single specialization point."""
        points = create_symint_specialization_points(0, 100, 1)
        expected = [50.0]  # Midpoint of single range
        self.assertEqual(points, expected)
    
    def test_create_specialization_points_custom_range(self):
        """Test creating specialization points for custom range."""
        points = create_symint_specialization_points(100, 900, 2)
        expected = [300.0, 700.0]  # Midpoints of 2 ranges
        self.assertEqual(points, expected)
    
    def test_invalid_num_points(self):
        """Test error handling for invalid number of points."""
        with self.assertRaises(ValueError):
            create_symint_specialization_points(0, 100, 0)
        
        with self.assertRaises(ValueError):
            create_symint_specialization_points(0, 100, -1)
    
    def test_get_symint_range_bounds(self):
        """Test getting range bounds for symbolic integers."""
        # For now, this should return the default bounds
        min_val, max_val = get_symint_range_bounds(None, 0, 4096)
        self.assertEqual(min_val, 0)
        self.assertEqual(max_val, 4096)
        
        # Test with different defaults
        min_val, max_val = get_symint_range_bounds(None, 10, 1000)
        self.assertEqual(min_val, 0)  # Should extend to ensure full coverage
        self.assertEqual(max_val, 4096)  # Should extend to ensure full coverage


if __name__ == "__main__":
    unittest.main()