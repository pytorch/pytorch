#!/usr/bin/env python

import os
import sys
import subprocess

amd_build_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.join(os.path.dirname(os.path.dirname(amd_build_dir)))

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
    "caffe2/core/THCCachingAllocator*",
    "caffe2/db/*",
]

ignores = [
    "caffe2/operators/depthwise_3x3_conv_op.cu",
    "caffe2/operators/depthwise_3x3_conv_op_cudnn.cu",
    "caffe2/operators/top_k.cu",
    "caffe2/operators/top_k_radix_selection.cuh",
    "caffe2/operators/top_k_heap_selection.cuh",
    "caffe2/operators/pool_op_cudnn.cu",
    "caffe2/operators/roi_align_op_gpu_test.cc",
    '**/hip/**',
]

file_extensions = ['.cc', '.cu', '.h', '.cuh']

# Execute the Hipify Script.
args = [
    "--project-directory", proj_dir,
    "--output-directory", proj_dir,
    "--includes"] + includes + \
    ["--extensions"] + file_extensions + \
    ["--ignores"] + ignores + \
    ["--hipify_caffe2", "True"] + \
    ["--add-static-casts", "True"]

subprocess.check_call([
    sys.executable,
    os.path.join(amd_build_dir, "pyHIPIFY", "hipify-python.py"),
] + args)
