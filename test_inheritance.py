#!/usr/bin/env python3

"""
Simple inheritance test for Schedule1F1B and ScheduleGPipe
"""

import sys
import os

# Add the PyTorch root directory to Python path
sys.path.insert(0, '/pytorch')

try:
    from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe
    from torch.distributed.pipelining.schedules import _PipelineScheduleRuntime
    
    # Test inheritance
    print("Schedule1F1B inheritance test:")
    print(f"  Schedule1F1B inherits from _PipelineScheduleRuntime: {issubclass(Schedule1F1B, _PipelineScheduleRuntime)}")
    print(f"  Schedule1F1B MRO: {Schedule1F1B.__mro__}")
    
    print("\nScheduleGPipe inheritance test:")
    print(f"  ScheduleGPipe inherits from _PipelineScheduleRuntime: {issubclass(ScheduleGPipe, _PipelineScheduleRuntime)}")
    print(f"  ScheduleGPipe MRO: {ScheduleGPipe.__mro__}")
    
    # Test that _step_microbatches method exists
    print("\n_step_microbatches method test:")
    print(f"  Schedule1F1B has _step_microbatches: {hasattr(Schedule1F1B, '_step_microbatches')}")
    print(f"  ScheduleGPipe has _step_microbatches: {hasattr(ScheduleGPipe, '_step_microbatches')}")
    
except Exception as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)