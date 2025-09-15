#!/usr/bin/env python3
"""
Demo: No log spew with distributed logging patch.
Run with: torchrun --nproc_per_node=2 demo_no_spew.py
"""
import os
import warnings
import logging
import torch
import torch.distributed as dist

# Initialize distributed
if 'RANK' in os.environ:
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
else:
    rank = 0
    world_size = 1

print(f"=== Process {rank}/{world_size} ===")

# Test warnings
warnings.warn("This warning should only appear ONCE (from rank 0)")

# Test logging
logging.warning("This logging should only appear ONCE (from rank 0)")

# Test the original cpp_extension case
logging.getLogger('torch.utils.cpp_extension').setLevel(logging.DEBUG)
from torch.utils.cpp_extension import _get_cuda_arch_flags
try:
    _get_cuda_arch_flags()
except:
    pass

print(f"Process {rank} completed")

if 'RANK' in os.environ:
    dist.destroy_process_group()