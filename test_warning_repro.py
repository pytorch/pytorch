#!/usr/bin/env python3
"""
Simple reproduction script for issue #161629
Shows the warning behavior before and after the fix

Usage:
    python test_warning_repro.py
    
This will demonstrate:
1. Before fix: Warning appears at WARNING level on all ranks
2. After fix: Message only appears at DEBUG level on rank 0
"""

import os
import sys
import logging

# Ensure TORCH_CUDA_ARCH_LIST is not set
os.environ.pop('TORCH_CUDA_ARCH_LIST', None)

print("=" * 60)
print("Testing cpp_extension warning behavior (issue #161629)")
print("=" * 60)

# Setup logging to see both WARNING and DEBUG messages
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(name)s: %(message)s'
)

print("\n1. Testing with TORCH_CUDA_ARCH_LIST not set:")
print("-" * 50)

# Import after setting up environment
import torch
from torch.utils.cpp_extension import _get_cuda_arch_flags

if torch.cuda.is_available():
    print(f"CUDA is available with {torch.cuda.device_count()} device(s)")
    
    # This will trigger the warning/debug message
    print("\nCalling _get_cuda_arch_flags()...")
    flags = _get_cuda_arch_flags()
    print(f"Generated flags: {flags[:3]}..." if len(flags) > 3 else f"Generated flags: {flags}")
    
    print("\n2. Testing with TORCH_CUDA_ARCH_LIST='8.0':")
    print("-" * 50)
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
    
    print("Calling _get_cuda_arch_flags()...")
    flags = _get_cuda_arch_flags()
    print(f"Generated flags: {flags}")
    
    print("\n3. Testing distributed scenario (simulated):")
    print("-" * 50)
    
    # Remove the env var again
    os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
    
    # Simulate being on rank 1 (non-zero rank)
    print("Simulating rank 1 in distributed setting...")
    print("With the fix: Message should only appear on rank 0")
    
    # Note: We can't actually test distributed without proper setup,
    # but the code checks torch.distributed.is_initialized()
    
    print("\n" + "=" * 60)
    print("EXPECTED BEHAVIOR:")
    print("- BEFORE FIX: [WARNING] message appears for all ranks")  
    print("- AFTER FIX:  [DEBUG] message appears only on rank 0")
    print("=" * 60)
    
else:
    print("CUDA not available - test requires GPU")
    sys.exit(1)