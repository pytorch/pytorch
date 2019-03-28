#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import os
import sys
import subprocess
import argparse
from functools import reduce
from itertools import chain

from pyHIPIFY import hipify_python

parser = argparse.ArgumentParser(description='Top-level script for HIPifying, filling in most common parameters')
parser.add_argument(
    '--out-of-place-only',
    action='store_true',
    help="Whether to only run hipify out-of-place on source files")

parser.add_argument(
    '--project-directory',
    type=str,
    default='',
    help="The root of the project.",
    required=False)

parser.add_argument(
    '--output-directory',
    type=str,
    default='',
    help="The Directory to Store the Hipified Project",
    required=False)

# Hipify using HIP-Clang launch.
parser.add_argument(
    '--hip-clang-launch',
    action='store_true',
    help=argparse.SUPPRESS)

args = parser.parse_args()

amd_build_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.join(os.path.dirname(os.path.dirname(amd_build_dir)))

if args.project_directory:
    proj_dir = args.project_directory

out_dir = proj_dir
if args.output_directory:
    out_dir = args.output_directory

includes = [
    "caffe2/operators/*",
    "caffe2/sgd/*",
    "caffe2/image/*",
    "caffe2/transforms/*",
    "caffe2/video/*",
    "caffe2/distributed/*",
    "caffe2/queue/*",
    "binaries/*",
    "caffe2/**/*_test*",
    "caffe2/core/*",
    "caffe2/db/*",
    "caffe2/utils/*",
    "c10/cuda/*",
    "c10/cuda/test/CMakeLists.txt",
    "modules/*",
    # PyTorch paths
    # Keep this synchronized with is_pytorch_file in hipify_python.py
    "aten/src/ATen/cuda/*",
    "aten/src/ATen/native/cuda/*",
    "aten/src/ATen/native/cudnn/*",
    "aten/src/ATen/native/sparse/cuda/*",
    "aten/src/THC/*",
    "aten/src/THCUNN/*",
    "aten/src/ATen/test/*",
    # CMakeLists.txt isn't processed by default, but there are a few
    # we do want to handle, so explicitly specify them
    "aten/src/THC/CMakeLists.txt",
    "aten/src/THCUNN/CMakeLists.txt",
    "torch/*",
    "tools/autograd/templates/python_variable_methods.cpp",
]

ignores = [
    "caffe2/operators/depthwise_3x3_conv_op_cudnn.cu",
    "caffe2/operators/pool_op_cudnn.cu",
    '*/hip/*',
    # These files are compatible with both cuda and hip
    "aten/src/ATen/core/*",
    "torch/csrc/autograd/engine.cpp",
    # generated files we shouldn't frob
    "torch/lib/tmp_install/*",
    "torch/include/*",
]

json_settings = os.path.join(amd_build_dir, "disabled_features.json")

if not args.out_of_place_only:
    # Apply patch files in place (PyTorch only)
    patch_folder = os.path.join(amd_build_dir, "patches")
    for filename in os.listdir(os.path.join(amd_build_dir, "patches")):
        subprocess.Popen(["git", "apply", os.path.join(patch_folder, filename)], cwd=proj_dir)

    # Make various replacements inside AMD_BUILD/torch directory
    ignore_files = [
        # These files use nvrtc, hip doesn't have equivalent
        "csrc/autograd/profiler.h",
        "csrc/autograd/profiler.cpp",
        # These files are compatible with both cuda and hip
        "csrc/autograd/engine.cpp"
    ]
    paths = ("torch", "tools")
    for root, _directories, files in chain.from_iterable(os.walk(path) for path in paths):
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
    output_directory=out_dir,
    includes=includes,
    ignores=ignores,
    out_of_place_only=args.out_of_place_only,
    json_settings=json_settings,
    hip_clang_launch=args.hip_clang_launch)
