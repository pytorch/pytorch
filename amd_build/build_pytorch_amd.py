import shutil
import subprocess
import os

cwd = os.path.dirname(__file__)
proj_dir = os.path.dirname(cwd)
out_dir = os.path.join(os.path.dirname(proj_dir), "pytorch_amd")
exclude_dirs = [
    "aten/src/TH",
    "aten/src/THNN",
    "aten/src/THS",
    "caffe2",
    "third_party"
]

yaml_file = os.path.join(cwd, "disabled_features.yaml")

# Create the pytorch_amd directory
shutil.copytree(proj_dir, out_dir)

# Extract (.hip) files.
shutil.copy(os.path.join(cwd, "hip_files/ATen/CMakeLists.txt.hip"), os.path.join(out_dir, "aten/src/ATen/CMakeLists.txt"))
shutil.copy(os.path.join(cwd, "hip_files/_aten/CMakeLists.txt.hip"), os.path.join(out_dir, "aten/CMakeLists.txt"))
shutil.copy(os.path.join(cwd, "hip_files/THC/THCApply.cuh.hip"), os.path.join(out_dir, "aten/src/THC/THCApply.cuh"))
shutil.copy(os.path.join(cwd, "hip_files/THC/THCAsmUtils.cuh.hip"), os.path.join(out_dir, "aten/src/THC/THCAsmUtils.cuh"))
shutil.copy(os.path.join(cwd, "hip_files/THC/THCBlas.cu.hip"), os.path.join(out_dir, "aten/src/THC/THCBlas.cu.hip"))
shutil.copy(os.path.join(cwd, "hip_files/THC/THCDeviceUtils.cuh.hip"), os.path.join(out_dir, "aten/src/THC/THCDeviceUtils.cuh"))
shutil.copy(os.path.join(cwd, "hip_files/THC/THCNumerics.cuh.hip"), os.path.join(out_dir, "aten/src/THC/THCNumerics.cuh"))
shutil.copy(os.path.join(cwd, "hip_files/THC/THCTensorRandom.cu.hip"), os.path.join(out_dir, "aten/src/THC/THCTensorRandom.cu"))
shutil.copy(os.path.join(cwd, "hip_files/THC/THCTensorRandom.h.hip"), os.path.join(out_dir, "aten/src/THC/THCTensorRandom.h"))
shutil.copy(os.path.join(cwd, "hip_files/THC/generic/THCTensorRandom.cu.hip"), os.path.join(out_dir, "aten/src/THC/generic/THCTensorRandom.cu"))

# Move to avoid HCC bug.
shutil.move(os.path.join(out_dir, "aten/src/ATen/native/cudnn/Conv.cpp"), os.path.join(out_dir, "aten/src/ATen/native/cudnn/ConvCuDNN.cpp"))
shutil.move(os.path.join(out_dir, "aten/src/ATen/native/cudnn/BatchNorm.cpp"), os.path.join(out_dir, "aten/src/ATen/native/cudnn/BatchNormCuDNN.cpp"))

# Apply the Patch File.
subprocess.Popen(["git", "apply", os.path.join(cwd, "patch084e3a7.patch")], cwd=out_dir)

# Execute the Hipify Script.
subprocess.Popen(
    ["/opt/rocm/bin/hipify-python.py",
    "--project-directory", proj_dir,
    "--output-directory", out_dir,
    "--exclude-dirs"] + exclude_dirs + \
    ["--yaml-settings",  yaml_file,
    "--add-static-casts", "True"])
