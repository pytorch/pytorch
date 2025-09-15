#!/usr/bin/env python3
"""
Demo script showing distributed logging behavior before and after the fix.

Usage:
    # Test with multiple processes (shows the fix working)
    torchrun --nproc_per_node=2 demo_distributed_logging.py
    
    # Test with single process (shows normal behavior)
    python demo_distributed_logging.py
"""

import os
import sys
import logging
import warnings
import torch
import torch.distributed as dist

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_log_spew():
    """Demonstrate various types of log spew that would occur without the fix."""
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    print(f"\n{'='*60}")
    print(f"Process Info: Rank {rank}/{world_size}")
    print(f"{'='*60}\n")
    
    # 1. warnings.warn - would normally appear on all ranks
    print(f"[Rank {rank}] Emitting warnings.warn()...")
    warnings.warn("This is a test warning - should only appear once on rank 0")
    warnings.warn("Another warning - should only appear once on rank 0")
    warnings.warn("This is a test warning - should only appear once on rank 0")  # Duplicate
    
    # 2. Logger warnings - would normally appear on all ranks
    print(f"[Rank {rank}] Emitting logger warnings...")
    logger.warning("Logger warning - should only appear on rank 0")
    logger.info("Logger info - should only appear on rank 0")
    
    # 3. Direct logging module - would normally appear on all ranks
    print(f"[Rank {rank}] Emitting logging module warnings...")
    logging.warning("Direct logging warning - should only appear on rank 0")
    logging.info("Direct logging info - should only appear on rank 0")
    
    # 4. The specific cpp_extension case from your PR
    print(f"[Rank {rank}] Testing cpp_extension case...")
    try:
        # Set debug level to see the message
        cpp_logger = logging.getLogger('torch.utils.cpp_extension')
        cpp_logger.setLevel(logging.DEBUG)
        
        from torch.utils.cpp_extension import _get_cuda_arch_flags
        _get_cuda_arch_flags()
        print(f"[Rank {rank}] cpp_extension test completed")
    except Exception as e:
        print(f"[Rank {rank}] cpp_extension test skipped: {e}")
    
    print(f"\n[Rank {rank}] Demo completed!\n")


def test_distributed_utilities():
    """Test the new distributed logging utilities."""
    from torch.distributed._distributed_logging import (
        distributed_print,
        force_log_on_all_ranks,
        get_distributed_logging_stats,
        is_distributed_logging_enabled
    )
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    print(f"\n{'='*60}")
    print(f"Testing Distributed Utilities (Rank {rank})")
    print(f"{'='*60}\n")
    
    # Check if patching is enabled
    enabled = is_distributed_logging_enabled()
    print(f"[Rank {rank}] Distributed logging enabled: {enabled}")
    
    # Test distributed_print
    distributed_print("This message via distributed_print - only on rank 0")
    
    # Test force_log_on_all_ranks
    force_log_on_all_ranks("This CRITICAL message appears on ALL ranks")
    
    # Show stats
    stats = get_distributed_logging_stats()
    print(f"[Rank {rank}] Logging stats: {stats}")


def main():
    # Initialize distributed if running with torchrun
    if 'RANK' in os.environ:
        print("Initializing distributed environment...")
        dist.init_process_group(backend='gloo')  # Use gloo for CPU testing
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        print(f"✓ Distributed initialized: rank={rank}, world_size={world_size}")
        
        # Check if distributed logging was automatically enabled
        from torch.distributed._distributed_logging import is_distributed_logging_enabled
        if is_distributed_logging_enabled():
            print(f"✓ Distributed logging patch is ACTIVE (warnings/logs will only appear on rank 0)")
        else:
            print(f"✗ Distributed logging patch is NOT active")
    else:
        print("Running in single-process mode (no distributed)")
        print("To test distributed behavior, run with:")
        print("  torchrun --nproc_per_node=2 demo_distributed_logging.py")
    
    # Run the main demo
    demonstrate_log_spew()
    
    # Test utilities if in distributed mode
    if dist.is_initialized():
        test_distributed_utilities()
        
        # Clean up
        dist.destroy_process_group()
        print("Distributed environment destroyed")


if __name__ == "__main__":
    main()