#!/usr/bin/env python

import os

amd_build_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.join(os.path.dirname(os.path.dirname(amd_build_dir)), "caffe2")

include_dirs = ["operators",
                "sgd",
                "image",
                "transforms",
                "video",
                "distributed"]

output_dir = os.path.join(os.path.dirname(os.path.dirname(amd_build_dir)), "caffe2_hip")

file_extensions = ['cc','cu','h','cuh']

ignore_file_list = ["depthwise_3x3_conv_op.cu",
					"depthwise_3x3_conv_op_cudnn.cu",
                    "top_k.cu",
					"top_k_radix_selection.cuh",
					"top_k_heap_selection.cuh",
                    "pool_op_cudnn.cu",
                    "utility_ops.cu",
                    "max_pool_with_index.cu"]#REVIST THIS FILE

# Execute the Hipify Script.
args = ["--project-directory", proj_dir,
        "--output-directory", output_dir,
        "--include-dirs"] + include_dirs + \
        ["--extensions"] + file_extensions + \
        ["--ignore_files"] + ignore_file_list + \
        ["--hipify_caffe2", "True"] + \
        ["--add-static-casts", "True"]

os.system("python " + os.path.join(amd_build_dir, "pyHIPIFY", "hipify-python.py") + " " + " ".join(args))

