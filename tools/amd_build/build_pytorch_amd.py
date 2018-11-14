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
json_file = os.path.join(amd_build_dir, "disabled_features.json")

# Apply patch files in place.
patch_folder = os.path.join(amd_build_dir, "patches")
for filename in os.listdir(os.path.join(amd_build_dir, "patches")):
    subprocess.Popen(["git", "apply", os.path.join(patch_folder, filename)], cwd=proj_dir)

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
    json_settings=json_file,
    add_static_casts_option=True,
    show_progress=False)
