This folder contains a number of scripts which are used as
part of the PyTorch build process.  This directory also doubles
as a Python module hierarchy (thus the `__init__.py`).

## Overview

Modern infrastructure:

* [autograd](autograd) - Code generation for autograd.  This
  includes definitions of all our derivatives.
* [jit](jit) - Code generation for JIT
* [shared](shared) - Generic infrastructure that scripts in
  tools may find useful.
  * [module_loader.py](shared/module_loader.py) - Makes it easier
    to import arbitrary Python files in a script, without having to add
    them to the PYTHONPATH first.

Legacy infrastructure (we should kill this):

* [nnwrap](nnwrap) - Generates the THNN/THCUNN wrappers which make
  legacy functionality available.  (TODO: What exactly does this
  implement?)
* [cwrap](cwrap) - Implementation of legacy code generation for THNN/THCUNN.
  This is used by nnwrap.

Build system pieces:

* [setup_helpers](setup_helpers) - Helper code for searching for
  third-party dependencies on the user system.
* [build_pytorch_libs.sh](build_pytorch_libs.sh) - Script that
  builds all of the constituent libraries of PyTorch, but not the
  PyTorch Python extension itself.  We are working on eliminating this
  script in favor of a unified cmake build.
* [build_pytorch_libs.bat](build_pytorch_libs.bat) - Same as
  above, but for Windows.
* [build_libtorch.py](build_libtorch.py) - Script for building
  libtorch, a standalone C++ library without Python support.  This
  build script is tested in CI.

Developer tools which you might find useful:

* [clang_tidy.py](clang_tidy.py) - Script for running clang-tidy
  on lines of your script which you changed.
* [git_add_generated_dirs.sh](git_add_generated_dirs.sh) and
  [git_reset_generated_dirs.sh](git_reset_generated_dirs.sh) -
  Use this to force add generated files to your Git index, so that you
  can conveniently run diffs on them when working on code-generation.
  (See also [generated_dirs.txt](generated_dirs.txt) which
  specifies the list of directories with generated files.)

Important if you want to run on AMD GPU:

* [amd_build](amd_build) - HIPify scripts, for transpiling CUDA
  into AMD HIP.  Right now, PyTorch and Caffe2 share logic for how to
  do this transpilation, but have separate entry-points for transpiling
  either PyTorch or Caffe2 code.
  * [build_caffe2_amd.py](amd_build/build_caffe2_amd.py) - Script
    for HIPifying the Caffe2 codebase.
  * [build_pytorch_amd.py](amd_build/build_pytorch_amd.py) - Script
    for HIPifying the PyTorch codebase.

Tools which are only situationally useful:

* [aten_mirror.sh](aten_mirror.sh) - Mirroring script responsible
  for keeping https://github.com/zdevito/ATen up-to-date.
* [docker](docker) - Dockerfile for running (but not developing)
  PyTorch, using the official conda binary distribution.  Context:
  https://github.com/pytorch/pytorch/issues/1619
* [download_mnist.py](download_mnist.py) - Download the MNIST
  dataset; this is necessary if you want to run the C++ API tests.
* [run-clang-tidy-in-ci.sh](run-clang-tidy-in-ci.sh) - Responsible
  for checking that C++ code is clang-tidy clean in CI on Travis
