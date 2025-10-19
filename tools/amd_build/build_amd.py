#!/usr/bin/env python3


import argparse
import os
import sys
from pathlib import Path


# NOTE: `tools/amd_build/build_amd.py` could be a symlink.
# The behavior of `symlink / '..'` is different from `symlink.parent`.
# Use `pardir` three times rather than using `path.parents[2]`.
REPO_ROOT = (
    Path(__file__).absolute() / os.path.pardir / os.path.pardir / os.path.pardir
).resolve()
sys.path.append(str(REPO_ROOT / "torch" / "utils"))

from hipify import hipify_python  # type: ignore[import]


parser = argparse.ArgumentParser(
    description="Top-level script for HIPifying, filling in most common parameters"
)
parser.add_argument(
    "--out-of-place-only",
    action="store_true",
    help="Whether to only run hipify out-of-place on source files",
)

parser.add_argument(
    "--project-directory",
    type=str,
    default="",
    help="The root of the project.",
    required=False,
)

parser.add_argument(
    "--output-directory",
    type=str,
    default="",
    help="The directory to store the hipified project",
    required=False,
)

parser.add_argument(
    "--extra-include-dir",
    type=str,
    default=[],
    nargs="+",
    help="The list of extra directories in caffe2 to hipify",
    required=False,
)

args = parser.parse_args()

# NOTE: `tools/amd_build/build_amd.py` could be a symlink.
amd_build_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.dirname(os.path.dirname(amd_build_dir))

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
    "third_party/nvfuser/*",
    # PyTorch paths
    # Keep this synchronized with is_pytorch_file in hipify_python.py
    "aten/src/ATen/cuda/*",
    "aten/src/ATen/native/cuda/*",
    "aten/src/ATen/native/cudnn/*",
    "aten/src/ATen/native/quantized/cudnn/*",
    "aten/src/ATen/native/nested/cuda/*",
    "aten/src/ATen/native/sparse/cuda/*",
    "aten/src/ATen/native/quantized/cuda/*",
    "aten/src/ATen/native/transformers/cuda/attention_backward.cu",
    "aten/src/ATen/native/transformers/cuda/attention.cu",
    "aten/src/ATen/native/transformers/cuda/sdp_utils.cpp",
    "aten/src/ATen/native/transformers/cuda/sdp_utils.h",
    "aten/src/ATen/native/transformers/cuda/mem_eff_attention/debug_utils.h",
    "aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h",
    "aten/src/ATen/native/transformers/cuda/mem_eff_attention/pytorch_utils.h",
    "aten/src/THC/*",
    "aten/src/ATen/test/*",
    # CMakeLists.txt isn't processed by default, but there are a few
    # we do want to handle, so explicitly specify them
    "aten/src/THC/CMakeLists.txt",
    "torch/*",
    "tools/autograd/templates/python_variable_methods.cpp",
    "torch/csrc/stable/*",
]

includes = [os.path.join(proj_dir, include) for include in includes]

for new_dir in args.extra_include_dir:
    abs_new_dir = os.path.join(proj_dir, new_dir)
    if os.path.exists(abs_new_dir):
        abs_new_dir = os.path.join(abs_new_dir, "**/*")
        includes.append(abs_new_dir)

ignores = [
    "caffe2/operators/depthwise_3x3_conv_op_cudnn.cu",
    "caffe2/operators/pool_op_cudnn.cu",
    "*/hip/*",
    # These files are compatible with both cuda and hip
    "aten/src/ATen/core/*",
    # Correct path to generate HIPConfig.h:
    #   CUDAConfig.h.in -> (amd_build) HIPConfig.h.in -> (cmake) HIPConfig.h
    "aten/src/ATen/cuda/CUDAConfig.h",
    "third_party/nvfuser/csrc/codegen.cpp",
    "third_party/nvfuser/runtime/block_reduction.cu",
    "third_party/nvfuser/runtime/block_sync_atomic.cu",
    "third_party/nvfuser/runtime/block_sync_default_rocm.cu",
    "third_party/nvfuser/runtime/broadcast.cu",
    "third_party/nvfuser/runtime/grid_reduction.cu",
    "third_party/nvfuser/runtime/helpers.cu",
    "torch/csrc/jit/codegen/fuser/cuda/resource_strings.h",
    "torch/csrc/jit/tensorexpr/ir_printer.cpp",
    "torch/csrc/jit/ir/ir.h",
    # generated files we shouldn't frob
    "torch/lib/tmp_install/*",
    "torch/include/*",
]

ignores = [os.path.join(proj_dir, ignore) for ignore in ignores]


# Check if the compiler is hip-clang.
#
# This used to be a useful function but now we can safely always assume hip-clang.
# Leaving the function here avoids bc-linter errors.
def is_hip_clang() -> bool:
    return True


# TODO Remove once the following submodules are updated
hip_platform_files = [
    "third_party/fbgemm/fbgemm_gpu/CMakeLists.txt",
    "third_party/fbgemm/fbgemm_gpu/cmake/Hip.cmake",
    "third_party/fbgemm/fbgemm_gpu/codegen/embedding_backward_dense_host.cpp",
    "third_party/fbgemm/fbgemm_gpu/codegen/embedding_backward_split_host_template.cpp",
    "third_party/fbgemm/fbgemm_gpu/codegen/embedding_backward_split_template.cu",
    "third_party/fbgemm/fbgemm_gpu/codegen/embedding_forward_quantized_split_lookup.cu",
    "third_party/fbgemm/fbgemm_gpu/include/fbgemm_gpu/utils/cuda_prelude.cuh",
    "third_party/fbgemm/fbgemm_gpu/include/fbgemm_gpu/utils/stochastic_rounding.cuh",
    "third_party/fbgemm/fbgemm_gpu/include/fbgemm_gpu/utils/vec4.cuh",
    "third_party/fbgemm/fbgemm_gpu/include/fbgemm_gpu/utils/weight_row.cuh",
    "third_party/fbgemm/fbgemm_gpu/include/fbgemm_gpu/sparse_ops.cuh",
    "third_party/fbgemm/fbgemm_gpu/src/jagged_tensor_ops.cu",
    "third_party/fbgemm/fbgemm_gpu/src/quantize_ops.cu",
    "third_party/fbgemm/fbgemm_gpu/src/sparse_ops.cu",
    "third_party/fbgemm/fbgemm_gpu/src/split_embeddings_cache_cuda.cu",
    "third_party/fbgemm/fbgemm_gpu/src/topology_utils.cpp",
    "third_party/fbgemm/src/EmbeddingSpMDM.cc",
    "third_party/gloo/cmake/Dependencies.cmake",
    "third_party/gloo/gloo/cuda.cu",
    "third_party/kineto/libkineto/CMakeLists.txt",
    "third_party/nvfuser/CMakeLists.txt",
    "third_party/tensorpipe/cmake/Hip.cmake",
]


def remove_hcc(line: str) -> str:
    line = line.replace("HIP_PLATFORM_HCC", "HIP_PLATFORM_AMD")
    line = line.replace("HIP_HCC_FLAGS", "HIP_CLANG_FLAGS")
    return line


for hip_platform_file in hip_platform_files:
    do_write = False
    if os.path.exists(hip_platform_file):
        with open(hip_platform_file) as sources:
            lines = sources.readlines()
        newlines = [remove_hcc(line) for line in lines]
        if lines == newlines:
            print(f"{hip_platform_file} skipped")
        else:
            with open(hip_platform_file, "w") as sources:
                for line in newlines:
                    sources.write(line)
            print(f"{hip_platform_file} updated")

# NOTE: fbgemm sources needing hipify
# fbgemm is its own project with its own build system. pytorch uses fbgemm as
# a submodule to acquire some gpu source files but compiles only those sources
# instead of using fbgemm's own build system. One of the source files refers
# to a header file that is the result of running hipify, but fbgemm uses
# slightly different hipify settings than pytorch. fbgemm normally hipifies
# and renames tuning_cache.cuh to tuning_cache_hip.cuh, but pytorch's settings
# for hipify puts it into its own 'hip' directory. After hipify runs below with
# the added fbgemm file, we move it to its expected location.
fbgemm_dir = "third_party/fbgemm/fbgemm_gpu/experimental/gen_ai/src/quantize/common/include/fbgemm_gpu/quantize"
fbgemm_original = f"{fbgemm_dir}/tuning_cache.cuh"
fbgemm_move_src = f"{fbgemm_dir}/hip/tuning_cache.cuh"
fbgemm_move_dst = f"{fbgemm_dir}/tuning_cache_hip.cuh"

hipify_python.hipify(
    project_directory=proj_dir,
    output_directory=out_dir,
    includes=includes,
    ignores=ignores,
    extra_files=[
        "torch/_inductor/codegen/cuda/device_op_overrides.py",
        "torch/_inductor/codegen/cpp_wrapper_cpu.py",
        "torch/_inductor/codegen/cpp_wrapper_gpu.py",
        "torch/_inductor/codegen/wrapper.py",
        fbgemm_original,
    ],
    out_of_place_only=args.out_of_place_only,
    hip_clang_launch=is_hip_clang(),
)

# only update the file if it changes or doesn't exist
do_write = True
src_lines = None
with open(fbgemm_move_src) as src:
    src_lines = src.readlines()
if os.path.exists(fbgemm_move_dst):
    dst_lines = None
    with open(fbgemm_move_dst) as dst:
        dst_lines = dst.readlines()
    if src_lines == dst_lines:
        print(f"{fbgemm_move_dst} skipped")
        do_write = False
if do_write:
    with open(fbgemm_move_dst, "w") as dst:
        for line in src_lines:
            dst.write(line)
    print(f"{fbgemm_move_dst} updated")
