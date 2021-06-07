#!/usr/bin/env python3


import os
import argparse
import sys
sys.path.append(os.path.realpath(os.path.join(
    __file__,
    os.path.pardir,
    os.path.pardir,
    os.path.pardir,
    'torch',
    'utils')))

from hipify import hipify_python

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
    help="The directory to store the hipified project",
    required=False)

parser.add_argument(
    '--extra-include-dir',
    type=str,
    default=[],
    nargs='+',
    help="The list of extra directories in caffe2 to hipify",
    required=False)

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
    "caffe2/contrib/aten/*",
    "binaries/*",
    "caffe2/**/*_test*",
    "caffe2/core/*",
    "caffe2/db/*",
    "caffe2/utils/*",
    "caffe2/contrib/gloo/*",
    "caffe2/contrib/nccl/*",
    "c10/cuda/*",
    "c10/cuda/test/CMakeLists.txt",
    "modules/*",
    # PyTorch paths
    # Keep this synchronized with is_pytorch_file in hipify_python.py
    "aten/src/ATen/cuda/*",
    "aten/src/ATen/native/cuda/*",
    "aten/src/ATen/native/cudnn/*",
    "aten/src/ATen/native/sparse/cuda/*",
    "aten/src/ATen/native/quantized/cuda/*",
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

for new_dir in args.extra_include_dir:
    abs_new_dir = os.path.join(proj_dir, new_dir)
    if os.path.exists(abs_new_dir):
        new_dir = os.path.join(new_dir, '**/*')
        includes.append(new_dir)

ignores = [
    "caffe2/operators/depthwise_3x3_conv_op_cudnn.cu",
    "caffe2/operators/pool_op_cudnn.cu",
    '*/hip/*',
    # These files are compatible with both cuda and hip
    "aten/src/ATen/core/*",
    "torch/csrc/jit/codegen/cuda/codegen.cpp",
    "torch/csrc/jit/codegen/cuda/runtime/block_reduction.cu",
    "torch/csrc/jit/codegen/cuda/runtime/broadcast.cu",
    "torch/csrc/jit/codegen/cuda/runtime/grid_reduction.cu",
    "torch/csrc/jit/codegen/fuser/cuda/resource_strings.h",
    "torch/csrc/jit/tensorexpr/ir_printer.cpp",
    # generated files we shouldn't frob
    "torch/lib/tmp_install/*",
    "torch/include/*",
]

# Check if the compiler is hip-clang.
def is_hip_clang():
    try:
        hip_path = os.getenv('HIP_PATH', '/opt/rocm/hip')
        return 'HIP_COMPILER=clang' in open(hip_path + '/lib/.hipInfo').read()
    except IOError:
        return False

# TODO Remove once gloo submodule is recent enough to contain upstream fix.
if is_hip_clang():
    gloo_cmake_file = "third_party/gloo/cmake/Hip.cmake"
    do_write = False
    with open(gloo_cmake_file, "r") as sources:
        lines = sources.readlines()
    newlines = [line.replace(' hip_hcc ', ' amdhip64 ') for line in lines]
    if lines == newlines:
        print("%s skipped" % gloo_cmake_file)
    else:
        with open(gloo_cmake_file, "w") as sources:
            for line in newlines:
                sources.write(line)
        print("%s updated" % gloo_cmake_file)

gloo_cmake_file = "third_party/gloo/cmake/Modules/Findrccl.cmake"
if os.path.exists(gloo_cmake_file):
    do_write = False
    with open(gloo_cmake_file, "r") as sources:
        lines = sources.readlines()
    newlines = [line.replace('RCCL_LIBRARY', 'RCCL_LIBRARY_PATH') for line in lines]
    if lines == newlines:
        print("%s skipped" % gloo_cmake_file)
    else:
        with open(gloo_cmake_file, "w") as sources:
            for line in newlines:
                sources.write(line)
        print("%s updated" % gloo_cmake_file)

# TODO Remove once gloo submodule is recent enough to contain upstream fix.
if is_hip_clang():
    gloo_cmake_file = "third_party/gloo/cmake/Dependencies.cmake"
    do_write = False
    with open(gloo_cmake_file, "r") as sources:
        lines = sources.readlines()
    newlines = [line.replace('HIP_HCC_FLAGS', 'HIP_CLANG_FLAGS') for line in lines]
    if lines == newlines:
        print("%s skipped" % gloo_cmake_file)
    else:
        with open(gloo_cmake_file, "w") as sources:
            for line in newlines:
                sources.write(line)
        print("%s updated" % gloo_cmake_file)

hipify_python.hipify(
    project_directory=proj_dir,
    output_directory=out_dir,
    includes=includes,
    ignores=ignores,
    out_of_place_only=args.out_of_place_only,
    hip_clang_launch=is_hip_clang())
