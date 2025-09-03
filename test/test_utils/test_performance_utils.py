"""
Test utilities for PyTorch performance testing.

This module provides unit tests for the performance benchmarking utilities
to ensure they work correctly across different devices and configurations.
"""

import unittest
import torch
import sys
import os

# Add PyTorch root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmarks.performance_tests.gpu_cpu_benchmark import (
    DeviceDetector,
    MatrixBenchmark,
    NeuralNetworkBenchmark
)


class TestDeviceDetector(unittest.TestCase):
    """Test device detection functionality."""
    
    def test_get_best_device(self):
        """Test that device detection returns valid device."""
        device, name = DeviceDetector.get_best_device()
        self.assertIsInstance(device, torch.device)
        self.assertIsInstance(name, str)
        self.assertIn(device.type, ['cpu', 'cuda', 'mps'])


class TestMatrixBenchmark(unittest.TestCase):
    """Test matrix benchmarking functionality."""
    
    def test_matrix_multiplication_cpu(self):
        """Test matrix multiplication benchmark on CPU."""
        cpu_device = torch.device('cpu')
        time_taken = MatrixBenchmark.benchmark_matrix_multiplication(
            size=100, 
            device=cpu_device, 
            warmup=False
        )
        self.assertIsInstance(time_taken, float)
        self.assertGreater(time_taken, 0)
    
    def test_matrix_multiplication_different_sizes(self):
        """Test benchmark with different matrix sizes."""
        cpu_device = torch.device('cpu')
        sizes = [50, 100, 200]
        
        for size in sizes:
            with self.subTest(size=size):
                time_taken = MatrixBenchmark.benchmark_matrix_multiplication(
                    size=size,
                    device=cpu_device,
                    warmup=False
                )
                self.assertIsInstance(time_taken, float)
                self.assertGreater(time_taken, 0)
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_matrix_multiplication_cuda(self):
        """Test matrix multiplication benchmark on CUDA."""
        cuda_device = torch.device('cuda')
        time_taken = MatrixBenchmark.benchmark_matrix_multiplication(
            size=100,
            device=cuda_device,
            warmup=True
        )
        self.assertIsInstance(time_taken, float)
        self.assertGreater(time_taken, 0)
    
    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_matrix_multiplication_mps(self):
        """Test matrix multiplication benchmark on MPS."""
        mps_device = torch.device('mps')
        time_taken = MatrixBenchmark.benchmark_matrix_multiplication(
            size=100,
            device=mps_device,
            warmup=True
        )
        self.assertIsInstance(time_taken, float)
        self.assertGreater(time_taken, 0)


class TestNeuralNetworkBenchmark(unittest.TestCase):
    """Test neural network benchmarking functionality."""
    
    def test_benchmark_net_creation(self):
        """Test that benchmark network can be created."""
        model = NeuralNetworkBenchmark.BenchmarkNet()
        self.assertIsInstance(model, torch.nn.Module)
        
        # Test forward pass
        x = torch.randn(10, 1000)
        output = model(x)
        self.assertEqual(output.shape, (10, 100))
    
    def test_training_benchmark_cpu(self):
        """Test neural network training benchmark on CPU."""
        cpu_device = torch.device('cpu')
        time_taken = NeuralNetworkBenchmark.benchmark_training(
            device=cpu_device,
            batch_size=32,
            epochs=2,
            learning_rate=0.001
        )
        self.assertIsInstance(time_taken, float)
        self.assertGreater(time_taken, 0)
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_training_benchmark_cuda(self):
        """Test neural network training benchmark on CUDA."""
        cuda_device = torch.device('cuda')
        time_taken = NeuralNetworkBenchmark.benchmark_training(
            device=cuda_device,
            batch_size=32,
            epochs=2,
            learning_rate=0.001
        )
        self.assertIsInstance(time_taken, float)
        self.assertGreater(time_taken, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the benchmarking system."""
    
    def test_full_benchmark_run(self):
        """Test that a full benchmark run completes without errors."""
        try:
            # Import the main function
            from benchmarks.performance_tests.gpu_cpu_benchmark import (
                run_matrix_benchmark,
                run_neural_network_benchmark
            )
            
            # Run small benchmarks (should not raise exceptions)
            matrix_results = run_matrix_benchmark()
            self.assertIsInstance(matrix_results, list)
            
            # Note: Neural network benchmark might be skipped on CPU-only systems
            # so we don't assert on its return value
            
        except Exception as e:
            self.fail(f"Full benchmark run failed with error: {e}")


if __name__ == '__main__':
    unittest.main()
