import shutil
import subprocess
import os

cwd = os.getcwd()
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
