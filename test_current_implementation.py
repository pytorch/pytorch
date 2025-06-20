#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torch'))

from torch.distributed.pipelining.schedules import Schedule1F1B, ScheduleGPipe, _PipelineScheduleRuntime

# Test inheritance
print("Testing inheritance...")
print(f"Schedule1F1B MRO: {Schedule1F1B.__mro__}")
print(f"ScheduleGPipe MRO: {ScheduleGPipe.__mro__}")

print(f"Schedule1F1B inherits from _PipelineScheduleRuntime: {issubclass(Schedule1F1B, _PipelineScheduleRuntime)}")
print(f"ScheduleGPipe inherits from _PipelineScheduleRuntime: {issubclass(ScheduleGPipe, _PipelineScheduleRuntime)}")

# Check _step_microbatches method source
print("\nTesting _step_microbatches method...")
print(f"Schedule1F1B._step_microbatches: {Schedule1F1B._step_microbatches}")
print(f"ScheduleGPipe._step_microbatches: {ScheduleGPipe._step_microbatches}")
print(f"_PipelineScheduleRuntime._step_microbatches: {_PipelineScheduleRuntime._step_microbatches}")

# Check if methods call super()
import inspect
print("\nChecking method implementations...")

# Get Schedule1F1B._step_microbatches source
try:
    source_1f1b = inspect.getsource(Schedule1F1B._step_microbatches)
    print("Schedule1F1B._step_microbatches source:")
    print(source_1f1b)
    print("Contains super() call: ", "super()" in source_1f1b)
except:
    print("Could not get source for Schedule1F1B._step_microbatches")

try:
    source_gpipe = inspect.getsource(ScheduleGPipe._step_microbatches)
    print("\nScheduleGPipe._step_microbatches source:")
    print(source_gpipe)
    print("Contains super() call: ", "super()" in source_gpipe)
except:
    print("Could not get source for ScheduleGPipe._step_microbatches")

print("\nTest completed successfully!")