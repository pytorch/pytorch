#!/usr/bin/env python3
"""
Benchmark script for PyTorch DataLoader with different worker methods.

This script measures:
1. Dataloader initialization time
2. Dataloading speed (time per batch)
3. CPU memory utilization

Usage:
    python dataloader_benchmark.py --data_path /path/to/dataset --worker_method thread --batch_size 32 --num_workers 4
"""
import argparse
import copy
import gc
import os
import sys
import threading
import time

import psutil
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torchvision.models import resnet18


def get_memory_usage(include_children=True):
    """
    Get current memory usage in MB.

    Args:
        include_children: If True, include memory of child processes

    Returns:
        Total memory usage in MB
    """
    process = psutil.Process()

    main_memory = process.memory_full_info().pss

    # If requested, add memory of all child processes
    if include_children:
        for child in process.children(recursive=True):
            try:
                child_mem = child.memory_full_info().pss
                main_memory += child_mem
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                # Process might have terminated or doesn't support PSS
                try:
                    # Fall back to USS
                    print(f"Failed to get PSS for {child}, falling back to USS")
                    child_mem = child.memory_info().uss
                    main_memory += child_mem
                except:
                    pass

    return main_memory / (1024 * 1024)


def print_detailed_memory():
    """Print detailed memory information."""
    process = psutil.Process()
    print("\nDetailed memory information:")
    try:
        print(
            f"  USS (Unique Set Size): {process.memory_full_info().uss / (1024 * 1024):.2f} MB"
        )
        print(
            f"  PSS (Proportional Set Size): {process.memory_full_info().pss / (1024 * 1024):.2f} MB"
        )
        print(
            f"  RSS (Resident Set Size): {process.memory_info().rss / (1024 * 1024):.2f} MB"
        )
    except:
        print("  Detailed memory info not available")


def create_model():
    """Create a simple model for benchmarking."""
    model = resnet18()
    return model


def benchmark_dataloader(
    dataset,
    batch_size,
    num_workers,
    worker_method,
    num_epochs=1,
    max_batches=10,
    multiprocessing_context=None,
):
    """Benchmark a dataloader with specific configuration."""
    print(f"\n--- Benchmarking DataLoader with worker_method={worker_method} ---")

    # Clear memory before starting
    gc.collect()
    torch.cuda.empty_cache()

    # Measure memory before dataloader creation
    memory_before = get_memory_usage()
    print(f"Memory before DataLoader creation: {memory_before:.2f} MB")
    print_detailed_memory()

    # Measure dataloader initialization time
    start = time.time()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_method=worker_method,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context=multiprocessing_context,
    )
    it = iter(dataloader)
    dataloader_init_time = time.time() - start

    # Measure memory after dataloader creation
    memory_after = get_memory_usage()
    print(f"Memory after DataLoader creation: {memory_after:.2f} MB")
    print(f"Memory increase: {memory_after - memory_before:.2f} MB")

    # Create model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Benchmark dataloading speed
    model.train()
    total_batches = 0
    total_samples = 0
    total_time = 0
    start_time = time.time()

    # Measure peak memory during training
    peak_memory = memory_after

    print(
        f"\nStarting training loop with {num_epochs} epochs (max {max_batches} batches per epoch)"
    )

    for epoch in range(num_epochs):
        batch_times = []

        while total_batches < max_batches:
            batch_start = time.time()

            inputs, labels = next(it)

            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)

            total_batches += 1
            total_samples += inputs.size(0)
            total_time += batch_time

            # Update peak memory and log memory usage periodically
            if total_batches % 5 == 0:
                # Force garbage collection before measuring memory
                gc.collect()
                current_memory = get_memory_usage()
                elapsed = time.time() - start_time

                if current_memory > peak_memory:
                    peak_memory = current_memory

            if total_batches % 10 == 0:
                print(
                    f"Epoch {epoch+1}, Batch {total_batches}, "
                    f"Time: {batch_time:.4f}s, "
                    f"Memory: {current_memory:.2f} MB"
                )

    # Calculate statistics
    avg_batch_time = total_time / total_batches if total_batches > 0 else 0
    samples_per_second = total_samples / total_time if total_time > 0 else 0

    results = {
        "dataloader_init_time": dataloader_init_time,
        "worker_method": worker_method,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "total_batches": total_batches,
        "avg_batch_time": avg_batch_time,
        "samples_per_second": samples_per_second,
        "peak_memory_mb": peak_memory,
        "memory_increase_mb": peak_memory - memory_before,
    }

    print("\nResults:")
    print(f"  Worker method: {worker_method}")
    print(f"  DataLoader init time: {dataloader_init_time:.4f} seconds")
    print(f"  Average batch time: {avg_batch_time:.4f} seconds")
    print(f"  Samples per second: {samples_per_second:.2f}")
    print(f"  Peak memory usage: {peak_memory:.2f} MB")
    print(f"  Memory increase: {peak_memory - memory_before:.2f} MB")

    # Clean up
    del model, optimizer
    del dataloader

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch DataLoader with different worker methods"
    )
    parser.add_argument("--data_path", required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--max_batches",
        type=int,
        default=100,
        help="Maximum number of batches per epoch",
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--worker_method",
        choices=["thread", "multiprocessing"],
        default="multiprocessing",
        help="Worker method to use (thread or multiprocessing)",
    )
    parser.add_argument(
        "--multiprocessing_context",
        choices=["fork", "spawn", "forkserver"],
        default="forkserver",
        help="Multiprocessing context to use (fork, spawn, forkserver)",
    )
    parser.add_argument(
        "--dataset_copies",
        type=int,
        default=1,
        help="Number of copies of the dataset to concatenate (for testing memory usage)",
    )
    args = parser.parse_args()

    # Print system info
    print("System Information:")
    # The following are handy for debugging if building from source worked correctly
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  PyTorch location: {torch.__file__}")
    print(f"  Torchvision version: {torchvision.__version__}")
    print(f"  Torchvision location: {torchvision.__file__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  CPU count: {psutil.cpu_count(logical=True)}")
    print(f"  Physical CPU cores: {psutil.cpu_count(logical=False)}")
    print(f"  Total system memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset
    print(f"\nLoading dataset from {args.data_path} ({args.dataset_copies} copies)")

    # Try to load as ImageFolder
    datasets = []
    for _ in range(args.dataset_copies):
        base_dataset = torchvision.datasets.ImageFolder(
            args.data_path, transform=transform
        )
        datasets.append(copy.deepcopy(base_dataset))
        del base_dataset
    dataset = ConcatDataset(datasets)

    print(f"Dataset size: {len(dataset)}")

    # Run benchmark with specified worker method
    results = benchmark_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        multiprocessing_context=(
            args.multiprocessing_context
            if args.worker_method == "multiprocessing"
            else None
        ),
        worker_method=args.worker_method,
        num_epochs=args.num_epochs,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
