#!/usr/bin/env python

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
                    "pool_op_cudnn.cu"}

for folder in ["operators", "sgd", "image", "transforms", "video"]:
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

