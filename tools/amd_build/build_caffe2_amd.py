#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import argparse
import os
import sys

from pyHIPIFY import hipify_python

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
    "caffe2/core/*",
    "caffe2/db/*",
    "modules/detectron/*",
    "caffe2/utils/*",
]

ignores = [
    "caffe2/operators/depthwise_3x3_conv_op.cu",
    "caffe2/operators/depthwise_3x3_conv_op_cudnn.cu",
    "caffe2/operators/pool_op_cudnn.cu",
    '**/hip/**',
]

file_extensions = ['.cc', '.cu', '.h', '.cuh']

parser = argparse.ArgumentParser(
    description="The Script to Hipify Caffe2")

parser.add_argument(
    '--hip-suffix',
    type=str,
    default='cc',
    help="The suffix for the hipified files",
    required=False)

args = parser.parse_args()

hipify_python.hipify(
    project_directory=proj_dir,
    output_directory=proj_dir,
    includes=includes,
    extensions=file_extensions,
    ignores=ignores,
    hipify_caffe2=True,
    add_static_casts_option=True,
    hip_suffix=args.hip_suffix,
    extensions_to_hip_suffix=['.cc', '.cu'])
