"""
GPU vs CPU Performance Benchmarking for PyTorch

This module provides comprehensive benchmarking tools to compare
GPU and CPU performance for various PyTorch operations.

Usage:
    python benchmarks/performance_tests/gpu_cpu_benchmark.py
"""

import torch
import time
import sys
import os
from typing import Tuple, Dict, List, Optional, Union

# Add PyTorch root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class DeviceDetector:
    """Utility class for detecting and managing PyTorch devices."""
    
    @staticmethod
    def get_best_device() -> Tuple[torch.device, str]:
        """
        Detect the best available device for computation.
        
        Priority order: CUDA > MPS > CPU
        
        Returns:
            Tuple of (device, device_name)
        """
        if torch.cuda.is_available():
            return torch.device('cuda'), 'NVIDIA CUDA'
        elif torch.backends.mps.is_available():
            return torch.device('mps'), 'Apple Metal (MPS)'
        else:
            return torch.device('cpu'), 'CPU Only'


class MatrixBenchmark:
    """Benchmarking utilities for matrix operations."""
    
    @staticmethod
    def benchmark_matrix_multiplication(
        size: int, 
        device: torch.device, 
        warmup: bool = True,
        dtype: torch.dtype = torch.float32
    ) -> float:
        """
        Benchmark matrix multiplication performance.
        
        Args:
            size: Matrix dimension (size x size)
            device: PyTorch device to run on
            warmup: Whether to perform warmup run
            dtype: Data type for tensors
            
        Returns:
            Execution time in seconds
        """
        # Create tensors on specified device
        x = torch.randn(size, size, device=device, dtype=dtype)
        y = torch.randn(size, size, device=device, dtype=dtype)
        
        # Warmup run for GPU
        if warmup and device.type != 'cpu':
            _ = torch.mm(x, y)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
        
        # Synchronize before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        result = torch.mm(x, y)
        
        # Synchronize after computation
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        return end_time - start_time


class NeuralNetworkBenchmark:
    """Benchmarking utilities for neural network operations."""
    
    class BenchmarkNet(torch.nn.Module):
        """Test neural network for benchmarking."""
        
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(1000, 2000),
                torch.nn.ReLU(),
                torch.nn.Linear(2000, 2000),
                torch.nn.ReLU(),
                torch.nn.Linear(2000, 2000),
                torch.nn.ReLU(),
                torch.nn.Linear(2000, 1000),
                torch.nn.ReLU(),
                torch.nn.Linear(1000, 100)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    @staticmethod
    def benchmark_training(
        device: torch.device,
        batch_size: int = 256,
        epochs: int = 10,
        learning_rate: float = 0.001
    ) -> float:
        """
        Benchmark neural network training performance.
        
        Args:
            device: PyTorch device to run on
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Optimizer learning rate
            
        Returns:
            Training time in seconds
        """
        model = NeuralNetworkBenchmark.BenchmarkNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Create training data
        x_train = torch.randn(batch_size, 1000, device=device)
        y_train = torch.randn(batch_size, 100, device=device)
        
        # Warmup run
        if device.type != 'cpu':
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
        
        # Benchmark training
        start_time = time.time()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        return time.time() - start_time


class PerformanceReporter:
    """Utility class for reporting benchmark results."""
    
    @staticmethod
    def print_header():
        """Print benchmark header."""
        print("=" * 70)
        print("🚀 PyTorch GPU vs CPU Performance Benchmark")
        print("=" * 70)
        print(f"PyTorch version: {torch.__version__}")
        print()
    
    @staticmethod
    def print_device_info(cpu_device: torch.device, gpu_device: torch.device, gpu_name: str):
        """Print device information."""
        print(f"🖥️  CPU Device: {cpu_device}")
        print(f"⚡ GPU Device: {gpu_device} ({gpu_name})")
        print()
    
    @staticmethod
    def print_matrix_results(results: List[Dict]):
        """Print matrix benchmark results."""
        print("📊 Matrix Multiplication Results:")
        print("-" * 70)
        print(f"{'Size':<8} {'CPU (s)':<10} {'GPU (s)':<10} {'Speedup':<10} {'Winner':<10}")
        print("-" * 70)
        
        gpu_wins = 0
        cpu_wins = 0
        
        for result in results:
            size = result['size']
            cpu_time = result['cpu_time']
            gpu_time = result['gpu_time']
            speedup = result['speedup']
            winner = result['winner']
            
            if winner == "GPU":
                gpu_wins += 1
            else:
                cpu_wins += 1
            
            winner_emoji = "🚀 GPU" if winner == "GPU" else "🖥️ CPU"
            print(f"{size}x{size:<3} {cpu_time:<10.4f} {gpu_time:<10.4f} {speedup:<10.2f}x {winner_emoji}")
        
        print("-" * 70)
        print(f"🏆 Final Score: GPU wins: {gpu_wins}, CPU wins: {cpu_wins}")
        
        if gpu_wins > cpu_wins:
            print("🎉 GPU is faster for larger matrices!")
        else:
            print("💻 CPU dominates for these matrix sizes.")


def run_matrix_benchmark() -> List[Dict]:
    """Run matrix multiplication benchmark."""
    gpu_device, gpu_name = DeviceDetector.get_best_device()
    cpu_device = torch.device('cpu')
    
    PerformanceReporter.print_device_info(cpu_device, gpu_device, gpu_name)
    
    if gpu_device.type == 'cpu':
        print("❌ No GPU acceleration available. Skipping comparison.")
        return []
    
    test_sizes = [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000]
    results = []
    
    for size in test_sizes:
        try:
            cpu_time = MatrixBenchmark.benchmark_matrix_multiplication(size, cpu_device, warmup=False)
            gpu_time = MatrixBenchmark.benchmark_matrix_multiplication(size, gpu_device, warmup=True)
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            winner = "GPU" if speedup > 1.0 else "CPU"
            if winner == "CPU":
                speedup = gpu_time / cpu_time
            
            results.append({
                'size': size,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'winner': winner
            })
        except Exception as e:
            print(f"Error benchmarking {size}x{size}: {e}")
            continue
    
    return results


def run_neural_network_benchmark():
    """Run neural network training benchmark."""
    gpu_device, gpu_name = DeviceDetector.get_best_device()
    cpu_device = torch.device('cpu')
    
    if gpu_device.type == 'cpu':
        print("❌ No GPU available for neural network benchmark")
        return
    
    print("\n" + "=" * 70)
    print("🧠 Neural Network Training Performance")
    print("=" * 70)
    
    batch_size = 256
    epochs = 10
    
    # Count parameters
    model = NeuralNetworkBenchmark.BenchmarkNet()
    param_count = sum(p.numel() for p in model.parameters())
    
    print(f"Training setup:")
    print(f"  • Network: 5 layers (1000→2000→2000→2000→1000→100)")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Epochs: {epochs}")
    print(f"  • Parameters: {param_count:,}")
    print()
    
    # CPU benchmark
    print("🖥️  Training on CPU...")
    cpu_time = NeuralNetworkBenchmark.benchmark_training(cpu_device, batch_size, epochs)
    print(f"✅ CPU training completed in {cpu_time:.4f}s")
    
    # GPU benchmark
    print(f"⚡ Training on {gpu_name}...")
    gpu_time = NeuralNetworkBenchmark.benchmark_training(gpu_device, batch_size, epochs)
    print(f"✅ GPU training completed in {gpu_time:.4f}s")
    
    # Results
    speedup = cpu_time / gpu_time
    print(f"\n📊 Neural Network Training Results:")
    print(f"  • CPU time: {cpu_time:.4f}s")
    print(f"  • GPU time: {gpu_time:.4f}s")
    if speedup > 1.0:
        print(f"  🚀 GPU is {speedup:.2f}x faster for neural network training!")
    else:
        print(f"  🖥️ CPU is {1/speedup:.2f}x faster for neural network training")


def main():
    """Main benchmark execution."""
    PerformanceReporter.print_header()
    
    # Run matrix benchmarks
    matrix_results = run_matrix_benchmark()
    if matrix_results:
        PerformanceReporter.print_matrix_results(matrix_results)
    
    # Run neural network benchmarks
    run_neural_network_benchmark()
    
    print("\n" + "=" * 70)
    print("✅ Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
