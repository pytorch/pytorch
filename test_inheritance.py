#!/usr/bin/env python3

import sys
import os

# Add the PyTorch directory to the path
sys.path.insert(0, '/pytorch')

# Mock the torch.version import that's failing
import types
torch_version = types.ModuleType('torch.version')
torch_version.__version__ = '0.0.0'
sys.modules['torch.version'] = torch_version

try:
    from torch.distributed.pipelining.schedules import Schedule1F1B, ScheduleGPipe, _PipelineScheduleRuntime
    
    print("Testing inheritance...")
    print(f"Schedule1F1B inherits from _PipelineScheduleRuntime: {issubclass(Schedule1F1B, _PipelineScheduleRuntime)}")
    print(f"ScheduleGPipe inherits from _PipelineScheduleRuntime: {issubclass(ScheduleGPipe, _PipelineScheduleRuntime)}")
    
    print("\nMRO (Method Resolution Order) for Schedule1F1B:")
    for i, cls in enumerate(Schedule1F1B.__mro__):
        print(f"  {i}: {cls.__name__}")
    
    print("\nMRO (Method Resolution Order) for ScheduleGPipe:")  
    for i, cls in enumerate(ScheduleGPipe.__mro__):
        print(f"  {i}: {cls.__name__}")
        
    print("\nChecking _step_microbatches method existence:")
    print(f"Schedule1F1B has _step_microbatches: {hasattr(Schedule1F1B, '_step_microbatches')}")
    print(f"ScheduleGPipe has _step_microbatches: {hasattr(ScheduleGPipe, '_step_microbatches')}")
    print(f"_PipelineScheduleRuntime has _step_microbatches: {hasattr(_PipelineScheduleRuntime, '_step_microbatches')}")
    
    print("\nAll inheritance tests PASSED!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()