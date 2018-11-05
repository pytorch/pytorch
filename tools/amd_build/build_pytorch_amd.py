from __future__ import absolute_import, division, print_function

import os
import subprocess
import sys
from functools import reduce

from pyHIPIFY import hipify_python

amd_build_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.dirname(os.path.dirname(amd_build_dir))

includes = [
    "aten/*",
    "torch/*",
]

ignores = [
    "aten/src/ATen/core/*",
]

# List of operators currently disabled
yaml_file = os.path.join(amd_build_dir, "disabled_features.yaml")

# Apply patch files in place.
patch_folder = os.path.join(amd_build_dir, "patches")
for filename in os.listdir(os.path.join(amd_build_dir, "patches")):
    subprocess.Popen(["git", "apply", os.path.join(patch_folder, filename)], cwd=proj_dir)

# HIPCC Compiler doesn't provide host defines - Automatically include them.
for root, _, files in os.walk(os.path.join(proj_dir, "aten/src/ATen")):
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
                "csrc/cuda/cuda_check.h"]
for root, _directories, files in os.walk(os.path.join(proj_dir, "torch")):
    for filename in files:
        if filename.endswith(".cpp") or filename.endswith(".h"):
            source = os.path.join(root, filename)
            # Disabled files
            if reduce(lambda result, exclude: source.endswith(exclude) or result, ignore_files, False):
                continue
            # Update contents.
            with open(source, "r+") as f:
                contents = f.read()
                contents = contents.replace("USE_CUDA", "USE_ROCM")
                contents = contents.replace("CUDA_VERSION", "0")
                f.seek(0)
                f.write(contents)
                f.truncate()
                f.flush()
                os.fsync(f)

hipify_python.hipify(
    project_directory=proj_dir,
    output_directory=proj_dir,
    includes=includes,
    ignores=ignores,
    yaml_settings=yaml_file,
    add_static_casts_option=True,
    show_progress=False)
