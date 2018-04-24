"""Requires the hipify-python.py script (https://github.com/ROCm-Developer-Tools/pyHIPIFY)."""
import shutil
import subprocess
import os
from shutil import copytree, ignore_patterns

amd_build_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.dirname(os.path.dirname(amd_build_dir))
out_dir = os.path.join(os.path.dirname(proj_dir), "pytorch_amd")
exclude_dirs = [
    "aten/src/TH",
    "aten/src/THNN",
    "aten/src/THS",
    "caffe2",
    "third_party"
]

# List of operators currently disabled
yaml_file = os.path.join(amd_build_dir, "disabled_features.yaml")

# Create the pytorch_amd directory
shutil.copytree(proj_dir, out_dir)

# Extract (.hip) files.
for root, _directories, files in os.walk(os.path.join(amd_build_dir, "hip_files")):
    for filename in files:
        if filename.endswith(".hip"):
            source = os.path.join(root, filename)
            destination = os.path.join(out_dir, source[source.find("hip_files/") + 10:])

            # Extract the .hip file.
            shutil.copy(source, destination)

# Apply patch files.
patch_folder = os.path.join(amd_build_dir, "patches")
for filename in os.listdir(os.path.join(amd_build_dir, "patches")):
    subprocess.Popen(["git", "apply", os.path.join(patch_folder, filename)], cwd=out_dir)

# Make various replacements inside AMD_BUILD/torch directory
ignore_files =  ["csrc/autograd/profiler.h", "csrc/autograd/profiler.cpp", "csrc/cuda/cuda_check.h", "csrc/jit/fusion_compiler.h", "csrc/jit/fusion_compiler.cpp"]
for root, _directories, files in os.walk(os.path.join(amd_build_dir, "torch")):
    for filename in files:
        if filename.endswith(".cpp") or filename.endswith(".h"):
            source = os.path.join(root, filename)
            # Disabled files
            if reduce(lambda result, exclude: source.endswith(exclude) or result, ignore_files, False):
                continue

            # Update contents.
            with open(source, "r+") as f:
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
        "--exclude-dirs"] + exclude_dirs +
    ["--yaml-settings", yaml_file, "--add-static-casts", "True"])
