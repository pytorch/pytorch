import shutil
import subprocess
import os
import sys
from shutil import copytree, ignore_patterns
from functools import reduce

amd_build_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.dirname(os.path.dirname(amd_build_dir))

includes = [
    "aten/*",
    "torch/*"
]

# List of operators currently disabled
yaml_file = os.path.join(amd_build_dir, "disabled_features.yaml")

# Apply patch files in place.
patch_folder = os.path.join(amd_build_dir, "patches")
for filename in os.listdir(os.path.join(amd_build_dir, "patches")):
    subprocess.Popen(["git", "apply", os.path.join(patch_folder, filename)], cwd=proj_dir)

# Execute the Hipify Script.
args = (["--project-directory", proj_dir] +
        ["--output-directory", proj_dir] +
        ["--includes"] + includes +
        ["--yaml-settings", yaml_file] +
        ["--add-static-casts", "True"] +
        ["--show-progress", "False"])

subprocess.check_call([
    sys.executable,
    os.path.join(amd_build_dir, "pyHIPIFY", "hipify-python.py")
] + args)
