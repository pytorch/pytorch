#!/usr/bin/env python3

"""
Check the current inheritance and _step_microbatches implementation
"""

import sys
import os

# Add the PyTorch root directory to Python path
sys.path.insert(0, '/pytorch')

# Mock the missing modules to avoid import errors
import types

# Mock torch.version
torch_version = types.ModuleType('torch.version')
torch_version.__version__ = '2.6.0a0'
sys.modules['torch.version'] = torch_version

# Mock torch.torch_version
torch_torch_version = types.ModuleType('torch.torch_version')
torch_torch_version.__version__ = '2.6.0a0'
sys.modules['torch.torch_version'] = torch_torch_version

# Try to import just the schedules module directly to check class inheritance
try:
    print("=== CHECKING SCHEDULE CLASS INHERITANCE ===")
    
    # Import the specific modules
    from torch.distributed.pipelining.schedules import Schedule1F1B, ScheduleGPipe, _PipelineScheduleRuntime
    
    print("1. Testing Schedule1F1B inheritance:")
    print(f"   Schedule1F1B inherits from _PipelineScheduleRuntime: {issubclass(Schedule1F1B, _PipelineScheduleRuntime)}")
    print(f"   MRO: {[cls.__name__ for cls in Schedule1F1B.__mro__]}")
    
    print("\n2. Testing ScheduleGPipe inheritance:")
    print(f"   ScheduleGPipe inherits from _PipelineScheduleRuntime: {issubclass(ScheduleGPipe, _PipelineScheduleRuntime)}")
    print(f"   MRO: {[cls.__name__ for cls in ScheduleGPipe.__mro__]}")
    
    print("\n3. Testing _step_microbatches method existence:")
    print(f"   Schedule1F1B has _step_microbatches: {hasattr(Schedule1F1B, '_step_microbatches')}")
    print(f"   ScheduleGPipe has _step_microbatches: {hasattr(ScheduleGPipe, '_step_microbatches')}")
    print(f"   _PipelineScheduleRuntime has _step_microbatches: {hasattr(_PipelineScheduleRuntime, '_step_microbatches')}")
    
    # Check the actual method sources
    print("\n4. Method source analysis:")
    if hasattr(Schedule1F1B, '_step_microbatches'):
        method = getattr(Schedule1F1B, '_step_microbatches')
        print(f"   Schedule1F1B._step_microbatches defined in: {method.__qualname__}")
    
    if hasattr(ScheduleGPipe, '_step_microbatches'):
        method = getattr(ScheduleGPipe, '_step_microbatches')
        print(f"   ScheduleGPipe._step_microbatches defined in: {method.__qualname__}")
    
    if hasattr(_PipelineScheduleRuntime, '_step_microbatches'):
        method = getattr(_PipelineScheduleRuntime, '_step_microbatches')
        print(f"   _PipelineScheduleRuntime._step_microbatches defined in: {method.__qualname__}")
    
    print("\n=== ALL INHERITANCE CHECKS COMPLETED ===")
    
except Exception as e:
    print(f"Error during inheritance check: {e}")
    import traceback
    traceback.print_exc()

# Now check the schedule directory structure
try:
    print("\n=== SCHEDULE CODE STRUCTURE ===")
    
    from torch.distributed.pipelining import schedules
    print(f"Schedule module location: {schedules.__file__}")
    
    # List all classes in the schedules module
    import inspect
    for name, obj in inspect.getmembers(schedules):
        if inspect.isclass(obj) and name.startswith('Schedule'):
            print(f"Class {name}: {obj.__bases__}")
    
except Exception as e:
    print(f"Error checking module structure: {e}")
    import traceback
    traceback.print_exc()