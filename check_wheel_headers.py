#!/usr/bin/env python3
"""
check_wheel_headers.py - Check what headers are currently included in PyTorch wheels
Run this BEFORE making changes to setup.py to see the current state
"""
import glob
import os
from pathlib import Path

def check_current_wheel_contents():
    """Check what headers are currently included based on existing setup.py patterns"""
    
    print("=== Current Wheel Contents Analysis ===\n")
    
    # Current problematic patterns from setup.py (around line 2800)
    current_patterns = [
        "include/*.h",
        "include/**/*.h", 
        "include/*.hpp",
        "include/**/*.hpp",
        "include/*.cuh", 
        "include/**/*.cuh"
    ]
    
    print("Checking patterns from setup.py torch_package_data:")
    
    # Check what these patterns actually include
    all_files = []
    for pattern in current_patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
        print(f"  {pattern}: {len(files)} files")
    
    # Categorize files
    pytorch_core = []
    third_party = []
    
    # Known third-party directories that shouldn't be in wheels
    third_party_dirs = ['google/', 'fmt/', 'pybind11/', 'fbgemm/', 'oneapi/', 'protobuf/', 'asmjit/', 'dnnl/', 'cpuinfo/']
    # PyTorch core directories that should be included
    pytorch_dirs = ['ATen/', 'c10/', 'caffe2/', 'torch/']
    
    for file in set(all_files):  # Remove duplicates
        file_path = file.replace('\\', '/')  # Normalize path
        
        # Check if it's third-party
        is_third_party = any(tp_dir in file_path for tp_dir in third_party_dirs)
        # Check if it's PyTorch core
        is_pytorch = any(pt_dir in file_path for pt_dir in pytorch_dirs)
        
        if is_third_party:
            third_party.append(file)
        elif is_pytorch:
            pytorch_core.append(file)
        else:
            # Other files (might be root level includes)
            pytorch_core.append(file)
    
    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Total files found: {len(set(all_files))}")
    print(f"PyTorch core files: {len(pytorch_core)}")
    print(f"Third-party files: {len(third_party)}")
    
    if third_party:
        print(f"\n❌ PROBLEM: {len(third_party)} third-party headers found in wheels!")
        print("These should NOT be included (causes diamond dependency conflicts):")
        
        # Group by directory
        tp_dirs = {}
        for tp_file in third_party:
            # Extract directory name after include/
            parts = tp_file.replace('\\', '/').split('/')
            if len(parts) > 1:
                dir_name = parts[1]  # First dir after include/
                tp_dirs[dir_name] = tp_dirs.get(dir_name, 0) + 1
        
        print(f"\nThird-party directories found:")
        for dir_name, count in sorted(tp_dirs.items()):
            print(f"   include/{dir_name}/: {count} files")
            
        print(f"\nFirst 10 third-party files:")
        for tp_file in sorted(third_party)[:10]:
            print(f"   {tp_file}")
        if len(third_party) > 10:
            print(f"   ... and {len(third_party)-10} more")
    else:
        print(f"\n✅ No third-party headers found")
    
    print(f"\n=== RECOMMENDATION ===")
    if third_party:
        print("❌ Current setup.py includes third-party headers (Issue #164883)")
        print("✅ Need to replace wildcard patterns with specific PyTorch directories")
    else:
        print("✅ Current setup.py looks good")

if __name__ == "__main__":
    if not os.path.exists("setup.py"):
        print("❌ Error: Run this script from PyTorch root directory (where setup.py exists)")
        exit(1)
    
    check_current_wheel_contents()
