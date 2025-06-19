#!/usr/bin/env python3

"""
Simple test to check inheritance without importing torch
"""

import sys
import os

# This script is designed to check the inheritance by directly reading the source code
# since we can't actually import torch due to missing libraries

def read_class_definition(file_path, class_name):
    """Read a class definition from a Python file and return the parent class"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    import re
    pattern = rf'class {class_name}\(([^)]+)\):'
    match = re.search(pattern, content)
    if match:
        return match.group(1).strip()
    return None

def check_method_implementation(file_path, class_name, method_name):
    """Check if a class has a specific method implementation"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the class definition
    import re
    class_pattern = rf'class {class_name}\([^)]+\):'
    class_match = re.search(class_pattern, content)
    if not class_match:
        return False, "Class not found"
    
    # Find the method within the class
    class_start = class_match.end()
    
    # Find the next class definition to determine the end of our class
    next_class_pattern = r'\nclass \w+.*?:'
    next_class_match = re.search(next_class_pattern, content[class_start:])
    if next_class_match:
        class_end = class_start + next_class_match.start()
    else:
        class_end = len(content)
    
    class_content = content[class_start:class_end]
    
    # Look for the method definition
    method_pattern = rf'def {method_name}\('
    method_match = re.search(method_pattern, class_content)
    
    if method_match:
        # Check if it calls super()
        method_start = method_match.start()
        # Find the next method or end of class
        next_method_pattern = r'\n    def \w+'
        next_method_match = re.search(next_method_pattern, class_content[method_start:])
        if next_method_match:
            method_end = method_start + next_method_match.start()
        else:
            method_end = len(class_content)
        
        method_content = class_content[method_start:method_end]
        has_super_call = 'super()._step_microbatches' in method_content
        return True, f"Method found, calls super(): {has_super_call}"
    
    return False, "Method not found"

def main():
    schedules_file = '/pytorch/torch/distributed/pipelining/schedules.py'
    
    print("=== INHERITANCE CHECK ===")
    
    # Check Schedule1F1B inheritance
    schedule1f1b_parent = read_class_definition(schedules_file, 'Schedule1F1B')
    print(f"Schedule1F1B inherits from: {schedule1f1b_parent}")
    
    # Check ScheduleGPipe inheritance
    schedule_gpipe_parent = read_class_definition(schedules_file, 'ScheduleGPipe')
    print(f"ScheduleGPipe inherits from: {schedule_gpipe_parent}")
    
    print("\n=== METHOD IMPLEMENTATION CHECK ===")
    
    # Check _step_microbatches implementation
    has_method1, details1 = check_method_implementation(schedules_file, 'Schedule1F1B', '_step_microbatches')
    print(f"Schedule1F1B._step_microbatches: {has_method1} - {details1}")
    
    has_method2, details2 = check_method_implementation(schedules_file, 'ScheduleGPipe', '_step_microbatches') 
    print(f"ScheduleGPipe._step_microbatches: {has_method2} - {details2}")
    
    print("\n=== SUMMARY ===")
    
    # Determine if the inheritance and methods are correct
    expected_parent = '_PipelineScheduleRuntime'
    inheritance_ok = (schedule1f1b_parent == expected_parent and 
                     schedule_gpipe_parent == expected_parent)
    
    methods_ok = (has_method1 and 'calls super(): True' in details1 and
                  has_method2 and 'calls super(): True' in details2)
    
    print(f"Inheritance correct: {inheritance_ok}")
    print(f"Method implementations correct: {methods_ok}")
    
    if inheritance_ok and methods_ok:
        print("✅ All tests would PASS - inheritance and method implementations are correct!")
    else:
        print("❌ Some tests would FAIL")
        if not inheritance_ok:
            print(f"  - Expected both classes to inherit from {expected_parent}")
        if not methods_ok:
            print(f"  - Expected both classes to have _step_microbatches methods that call super()")

if __name__ == '__main__':
    main()