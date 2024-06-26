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

Build system pieces:

* [setup_helpers](setup_helpers) - Helper code for searching for
  third-party dependencies on the user system.
* [build_pytorch_libs.py](build_pytorch_libs.py) - cross-platform script that
  builds all of the constituent libraries of PyTorch,
  but not the PyTorch Python extension itself.
* [build_libtorch.py](build_libtorch.py) - Script for building
  libtorch, a standalone C++ library without Python support.  This
  build script is tested in CI.

Developer tools which you might find useful:

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
  * [build_amd.py](amd_build/build_amd.py) - Top-level entry
    point for HIPifying our codebase.

Tools which are only situationally useful:

* [docker](docker) - Dockerfile for running (but not developing)
  PyTorch, using the official conda binary distribution.  Context:
  https://github.com/pytorch/pytorch/issues/1619
* [download_mnist.py](download_mnist.py) - Download the MNIST
  dataset; this is necessary if you want to run the C++ API tests.

[actions/github-script]: https://github.com/actions/github-script
[flake8]: https://flake8.pycqa.org/en/latest/
[github actions expressions]: https://docs.github.com/en/actions/reference/context-and-expression-syntax-for-github-actions#about-contexts-and-expressions
[pytorch/add-annotations-github-action]: https://github.com/pytorch/add-annotations-github-action
[shellcheck]: https://github.com/koalaman/shellcheck
