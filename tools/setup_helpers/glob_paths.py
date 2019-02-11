#!/usr/bin/env python3

import os
import glob

PACKAGE_DATA_PATHS = [
                'bin/*',
                'test/*',
                '__init__.pyi',
                'lib/*.so*',
                'lib/*.dylib*',
                'lib/*.dll',
                'lib/*.lib',
                'lib/*.pdb',
                'lib/torch_shm_manager',
                'lib/*.h',
                'include/ATen/*.h',
                'include/ATen/cpu/*.h',
                'include/ATen/core/*.h',
                'include/ATen/cuda/*.cuh',
                'include/ATen/cuda/*.h',
                'include/ATen/cuda/detail/*.cuh',
                'include/ATen/cuda/detail/*.h',
                'include/ATen/cudnn/*.h',
                'include/ATen/detail/*.h',
                'include/caffe2/utils/*.h',
                'include/c10/*.h',
                'include/c10/macros/*.h',
                'include/c10/core/*.h',
                'include/ATen/core/dispatch/*.h',
                'include/c10/core/impl/*.h',
                'include/ATen/core/opschema/*.h',
                'include/c10/util/*.h',
                'include/c10/cuda/*.h',
                'include/c10/cuda/impl/*.h',
                'include/c10/hip/*.h',
                'include/c10/hip/impl/*.h',
                'include/caffe2/**/*.h',
                'include/torch/*.h',
                'include/torch/csrc/*.h',
                'include/torch/csrc/api/include/torch/*.h',
                'include/torch/csrc/api/include/torch/data/*.h',
                'include/torch/csrc/api/include/torch/data/dataloader/*.h',
                'include/torch/csrc/api/include/torch/data/datasets/*.h',
                'include/torch/csrc/api/include/torch/data/detail/*.h',
                'include/torch/csrc/api/include/torch/data/samplers/*.h',
                'include/torch/csrc/api/include/torch/data/transforms/*.h',
                'include/torch/csrc/api/include/torch/detail/*.h',
                'include/torch/csrc/api/include/torch/detail/ordered_dict.h',
                'include/torch/csrc/api/include/torch/nn/*.h',
                'include/torch/csrc/api/include/torch/nn/modules/*.h',
                'include/torch/csrc/api/include/torch/nn/parallel/*.h',
                'include/torch/csrc/api/include/torch/optim/*.h',
                'include/torch/csrc/api/include/torch/serialize/*.h',
                'include/torch/csrc/autograd/*.h',
                'include/torch/csrc/autograd/generated/*.h',
                'include/torch/csrc/cuda/*.h',
                'include/torch/csrc/jit/*.h',
                'include/torch/csrc/jit/generated/*.h',
                'include/torch/csrc/jit/passes/*.h',
                'include/torch/csrc/jit/script/*.h',
                'include/torch/csrc/utils/*.h',
                'include/pybind11/*.h',
                'include/pybind11/detail/*.h',
                'include/TH/*.h*',
                'include/TH/generic/*.h*',
                'include/THC/*.cuh',
                'include/THC/*.h*',
                'include/THC/generic/*.h',
                'include/THCUNN/*.cuh',
                'include/THCUNN/generic/*.h',
                'include/THNN/*.h',
                'include/THNN/generic/*.h',
                'share/cmake/ATen/*.cmake',
                'share/cmake/Caffe2/*.cmake',
                'share/cmake/Caffe2/public/*.cmake',
                'share/cmake/Caffe2/Modules_CUDA_fix/*.cmake',
                'share/cmake/Caffe2/Modules_CUDA_fix/upstream/*.cmake',
                'share/cmake/Caffe2/Modules_CUDA_fix/upstream/FindCUDA/*.cmake',
                'share/cmake/Gloo/*.cmake',
                'share/cmake/Torch/*.cmake',
            ]

# For testing, execute this script from the toplevel "pytorch" clone as:
# ./tools/setup_helpers/glob_paths.py
if __name__ == "__main__":

    all_found_paths = set()
    glob_sources_by_file = {}

    for glob_path in PACKAGE_DATA_PATHS:

        found_files = glob.glob(os.path.join("build/lib.linux-x86_64-3.5/torch", glob_path), recursive=True)
        all_found_paths.update(found_files)

        for f in found_files:
            glob_sources_by_file.setdefault(f, set()).add(glob_path)

        if not found_files:
            print("No files matching this glob:", glob_path)
        else:
            print("Found %d files matching this glob: %s" % (len(found_files), glob_path))

    print("Glob path count:", len(PACKAGE_DATA_PATHS))
#    print("Found file count:", len(all_found_paths))

    print("="*20)
    for f, gs in sorted(glob_sources_by_file.items()):
        if len(gs) > 1:
            print('File "%s" found by multiple glob patterns:' % f)
            for gp in sorted(gs):
                print("\t", gp)
