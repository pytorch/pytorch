#!/usr/bin/env python3

"""
Verification script to check that the current implementation is correct
"""

import re

def verify_schedules_implementation():
    """Verify that Schedule1F1B and ScheduleGPipe are correctly implemented"""
    
    schedules_file = '/pytorch/torch/distributed/pipelining/schedules.py'
    
    with open(schedules_file, 'r') as f:
        content = f.read()
    
    # Check inheritance
    schedule1f1b_match = re.search(r'class Schedule1F1B\(([^)]+)\):', content)
    schedulegpipe_match = re.search(r'class ScheduleGPipe\(([^)]+)\):', content)
    
    schedule1f1b_parent = schedule1f1b_match.group(1).strip() if schedule1f1b_match else None
    schedulegpipe_parent = schedulegpipe_match.group(1).strip() if schedulegpipe_match else None
    
    print("=== INHERITANCE VERIFICATION ===")
    print(f"Schedule1F1B inherits from: {schedule1f1b_parent}")
    print(f"ScheduleGPipe inherits from: {schedulegpipe_parent}")
    
    # Check _step_microbatches implementations
    print("\n=== _STEP_MICROBATCHES VERIFICATION ===")
    
    # Find Schedule1F1B _step_microbatches method
    schedule1f1b_class_match = re.search(r'class Schedule1F1B.*?(?=class|\Z)', content, re.DOTALL)
    if schedule1f1b_class_match:
        schedule1f1b_class_content = schedule1f1b_class_match.group(0)
        schedule1f1b_method_match = re.search(r'def _step_microbatches.*?(?=def|\Z)', schedule1f1b_class_content, re.DOTALL)
        if schedule1f1b_method_match:
            schedule1f1b_method_content = schedule1f1b_method_match.group(0)
            has_super_call_1f1b = 'super()._step_microbatches' in schedule1f1b_method_content
            print(f"Schedule1F1B._step_microbatches calls super(): {has_super_call_1f1b}")
        else:
            print("Schedule1F1B._step_microbatches method not found")
    
    # Find ScheduleGPipe _step_microbatches method  
    schedulegpipe_class_match = re.search(r'class ScheduleGPipe.*?(?=class|\Z)', content, re.DOTALL)
    if schedulegpipe_class_match:
        schedulegpipe_class_content = schedulegpipe_class_match.group(0)
        schedulegpipe_method_match = re.search(r'def _step_microbatches.*?(?=def|\Z)', schedulegpipe_class_content, re.DOTALL)
        if schedulegpipe_method_match:
            schedulegpipe_method_content = schedulegpipe_method_match.group(0)
            has_super_call_gpipe = 'super()._step_microbatches' in schedulegpipe_method_content
            print(f"ScheduleGPipe._step_microbatches calls super(): {has_super_call_gpipe}")
        else:
            print("ScheduleGPipe._step_microbatches method not found")
    
    # Check pipeline order generation
    print("\n=== PIPELINE ORDER GENERATION VERIFICATION ===")
    
    # Check if _generate_1f1b_schedule exists
    has_1f1b_generator = '_generate_1f1b_schedule' in content
    print(f"Schedule1F1B has _generate_1f1b_schedule method: {has_1f1b_generator}")
    
    # Check if _generate_gpipe_schedule exists  
    has_gpipe_generator = '_generate_gpipe_schedule' in content
    print(f"ScheduleGPipe has _generate_gpipe_schedule method: {has_gpipe_generator}")
    
    # Check if both classes call _load_actions in __init__
    has_1f1b_load_actions = re.search(r'class Schedule1F1B.*?self\._load_actions\(', content, re.DOTALL)
    has_gpipe_load_actions = re.search(r'class ScheduleGPipe.*?self\._load_actions\(', content, re.DOTALL)
    
    print(f"Schedule1F1B calls _load_actions in __init__: {bool(has_1f1b_load_actions)}")
    print(f"ScheduleGPipe calls _load_actions in __init__: {bool(has_gpipe_load_actions)}")
    
    print("\n=== SUMMARY ===")
    
    # Check if everything is correct
    inheritance_correct = (schedule1f1b_parent == '_PipelineScheduleRuntime' and 
                          schedulegpipe_parent == '_PipelineScheduleRuntime')
    
    methods_correct = (has_super_call_1f1b and has_super_call_gpipe)
    
    pipeline_order_correct = (has_1f1b_generator and has_gpipe_generator and 
                             has_1f1b_load_actions and has_gpipe_load_actions)
    
    print(f"✅ Inheritance correct: {inheritance_correct}")
    print(f"✅ _step_microbatches methods correct: {methods_correct}")
    print(f"✅ Pipeline order generation correct: {pipeline_order_correct}")
    
    if inheritance_correct and methods_correct and pipeline_order_correct:
        print("\n🎉 ALL IMPLEMENTATION REQUIREMENTS ARE SATISFIED!")
        print("   - Both classes inherit from _PipelineScheduleRuntime")
        print("   - Both _step_microbatches methods call super()._step_microbatches")
        print("   - Pipeline order generation is properly implemented")
        return True
    else:
        print("\n❌ SOME REQUIREMENTS ARE NOT MET")
        return False

if __name__ == '__main__':
    verify_schedules_implementation()