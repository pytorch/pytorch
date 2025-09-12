#!/usr/bin/env python3
"""
Distributed test for cpp_extension warning (issue #161629)

Run this with torchrun to test distributed behavior:
    torchrun --nproc_per_node=2 test_distributed_warning.py

This demonstrates that the warning should only appear on rank 0.
"""

import os
import sys
import logging

# Setup logging - force torch.utils.cpp_extension to DEBUG level
logging.basicConfig(
    level=logging.WARNING,  # General level
    format=f'[rank{os.environ.get("RANK", "?")}] [%(levelname)s] %(name)s: %(message)s'
)
# Force DEBUG for cpp_extension to see the message
logging.getLogger('torch.utils.cpp_extension').setLevel(logging.DEBUG)

# Ensure TORCH_CUDA_ARCH_LIST is not set
os.environ.pop('TORCH_CUDA_ARCH_LIST', None)

import torch
import torch.distributed as dist

def main():
    # Initialize distributed if running with torchrun
    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"[Rank {rank}/{world_size}] Initialized distributed")
    else:
        rank = 0
        world_size = 1
        print("Running in single-process mode")
    
    # Import after distributed is initialized
    from torch.utils.cpp_extension import _get_cuda_arch_flags
    
    if torch.cuda.is_available():
        print(f"[Rank {rank}] CUDA available, calling _get_cuda_arch_flags()...")
        
        # This should only log on rank 0 with the fix
        flags = _get_cuda_arch_flags()
        
        print(f"[Rank {rank}] Got {len(flags)} arch flags")
        
        # Synchronize before exiting
        if dist.is_initialized():
            dist.barrier()
            
        print(f"[Rank {rank}] Test complete")
        
        if rank == 0:
            print("\n" + "=" * 60)
            print("EXPECTED BEHAVIOR:")
            print("- BEFORE FIX: Warning appears on ALL ranks")
            print("- AFTER FIX:  Debug message appears ONLY on rank 0")
            print("=" * 60)
    else:
        print(f"[Rank {rank}] CUDA not available")
        
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()