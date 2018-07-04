#!/usr/bin/env python

"""Requires the hipify-python.py script (https://github.com/ROCm-Developer-Tools/pyHIPIFY)."""
import shutil
import subprocess
import os
import sys
import glob
#from shutil import copytree, ignore_patterns, copyfile, rmtree
from functools import reduce
import pdb

amd_build_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.join(os.path.dirname(os.path.dirname(amd_build_dir)), "caffe2")
include_dirs = [
    "operators"
]
output_dir = os.path.join(os.path.dirname(os.path.dirname(amd_build_dir)), "caffe2_hip")

file_extensions = ['cc','cu','h']

# Make various replacements inside AMD_BUILD/torch directory
#ignore_files = ["csrc/autograd/profiler.h", "csrc/autograd/profiler.cpp",
 #               "csrc/cuda/cuda_check.h", "csrc/jit/fusion_compiler.cpp"]
pdb.set_trace()
# Execute the Hipify Script.
args = ["--project-directory", proj_dir,
        "--output-directory", output_dir,
        "--include-dirs"] + include_dirs + ["--extensions"] + file_extensions
#os.execv(os.path.join(amd_build_dir, "pyHIPIFY", "hipify-python.py"), ['python'] + args)
os.system("python " + os.path.join(amd_build_dir, "pyHIPIFY", "hipify-python.py") + " " + " ".join(args))
## copy files in hip directories
pdb.set_trace()
for dir in include_dirs:
    for file in glob.glob(os.path.join(output_dir,dir)+"/*"):
        basename = os.path.basename(file)
        dest_dir = os.path.join(proj_dir,dir,"hip")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        if basename.endswith("hip.cc") or basename.endswith("hip.h"):
            dest_filepath = os.path.join(dest_dir,basename)
            if not os.path.exists(dest_filepath):
                shutil.copyfile(file,dest_filepath)
pdb.set_trace()
shutil.rmtree(output_dir)
"""
import os
import glob
import subprocess

caffe2_root = "../../caffe2"
hipify_tool = "../tools/amd_build/hipify-perl"

os.chdir(caffe2_root)

ignore_file_list = {"depthwise_3x3_conv_op.cu",
					"top_k.cu",
					"top_k_radix_selection.cuh",
					"top_k_heap_selection.cuh",
                    "pool_op_cudnn.cu",
                    "utility_ops.cu",
                    "max_pool_with_index.cu"}#REVIST THIS FILE

for folder in ["operators", "sgd", "image", "transforms", "video", "distributed"]:
    for extension in ["/*.cu", "/*.cuh", "/*gpu.cc", "/*gpu.h"]:
        target = folder + extension
        for file in glob.glob(target):
        	if file.split('/')[-1] not in ignore_file_list:
        		os.system(hipify_tool + ' --inplace ' + file)
        os.chdir(folder)
        if glob.glob('*hip.cc'):
            os.system("mv -n *hip.cc hip")
            os.system("rm -f *hip.cc")
        os.chdir(caffe2_root)

"""

