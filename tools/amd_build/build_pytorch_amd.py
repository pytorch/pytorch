"""Requires the hipify-python.py script (https://github.com/ROCm-Developer-Tools/pyHIPIFY)."""
import shutil
import subprocess
import os
from shutil import copytree, ignore_patterns
from functools import reduce

amd_build_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.dirname(os.path.dirname(amd_build_dir))
out_dir = os.path.join(os.path.dirname(proj_dir), "pytorch_amd")
include_dirs = [
    "aten",
    "torch"
]

# List of operators currently disabled
yaml_file = os.path.join(amd_build_dir, "disabled_features.yaml")

# Create the pytorch_amd directory
shutil.copytree(proj_dir, out_dir)

# Apply patch files.
patch_folder = os.path.join(amd_build_dir, "patches")
for filename in os.listdir(os.path.join(amd_build_dir, "patches")):
    subprocess.Popen(["git", "apply", os.path.join(patch_folder, filename)], cwd=out_dir)

# HIPCC Compiler doesn't provide host defines - Automatically include them.
for root, _, files in os.walk(os.path.join(out_dir, "aten/src/ATen")):
    for filename in files:
        if filename.endswith(".cu") or filename.endswith(".cuh"):
            filepath = os.path.join(root, filename)

            # Add the include header!
            with open(filepath, "r+") as f:
                txt = f.read()
                result = '#include "hip/hip_runtime.h"\n%s' % txt
                f.seek(0)
                f.write(result)
                f.truncate()
                f.flush()

                # Flush to disk
                os.fsync(f)

# Make various replacements inside AMD_BUILD/torch directory
ignore_files = ["csrc/autograd/profiler.h", "csrc/autograd/profiler.cpp",
                "csrc/cuda/cuda_check.h", "csrc/jit/fusion_compiler.cpp"]
for root, _directories, files in os.walk(os.path.join(out_dir, "torch")):
    for filename in files:
        if filename.endswith(".cpp") or filename.endswith(".h"):
            source = os.path.join(root, filename)
            # Disabled files
            if reduce(lambda result, exclude: source.endswith(exclude) or result, ignore_files, False):
                continue
            # Update contents.
            with open(source, "r+", encoding="utf-8") as f:
                contents = f.read()
                contents = contents.replace("WITH_CUDA", "WITH_ROCM")
                contents = contents.replace("CUDA_VERSION", "0")
                f.seek(0)
                f.write(contents)
                f.truncate()
                f.flush()
                os.fsync(f)

# Execute the Hipify Script.
subprocess.Popen(
    ["/opt/rocm/bin/hipify-python.py",
        "--project-directory", proj_dir,
        "--output-directory", out_dir,
        "--include-dirs"] + include_dirs +
    ["--yaml-settings", yaml_file, "--add-static-casts", "True"])
