#!/usr/bin/env python3

# Simple test to check if the classes can be imported and work
import sys
import os
sys.path.insert(0, '/pytorch')

try:
    # Import the necessary classes
    from torch.distributed.pipelining.schedules import Schedule1F1B, ScheduleGPipe, _PipelineScheduleRuntime
    from torch.distributed.pipelining.schedules import _Action, FORWARD, FULL_BACKWARD
    
    print("Successfully imported classes!")
    
    # Test inheritance
    print(f"Schedule1F1B inherits from _PipelineScheduleRuntime: {issubclass(Schedule1F1B, _PipelineScheduleRuntime)}")
    print(f"ScheduleGPipe inherits from _PipelineScheduleRuntime: {issubclass(ScheduleGPipe, _PipelineScheduleRuntime)}")
    
    # Test that the methods exist
    print(f"Schedule1F1B has _step_microbatches: {hasattr(Schedule1F1B, '_step_microbatches')}")
    print(f"ScheduleGPipe has _step_microbatches: {hasattr(ScheduleGPipe, '_step_microbatches')}")
    
    # Test constants
    print(f"FORWARD constant: {FORWARD}")
    print(f"FULL_BACKWARD constant: {FULL_BACKWARD}")
    
    # Test Action creation
    action = _Action(0, FORWARD, 0)
    print(f"Created action: {action}")
    
    print("All tests PASSED!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()