---
title: Installation
---

# Installation

The Caffe2 libray's dependency is largely similar to that of Caffe's. Thus, if you have installed Caffe in the past, you should most likely be good to go. Otherwise, please check the prerequisites and specific platforms' guides.

Caffe2 uses a homebrew build script so that we can deal with multiple targets as well as optional dependencies. The format is similar to build systems like [Bazel](http://bazel.io) and [Buck](https://buckbuild.com/) with some custom flavors. It is based on python, so you will need to have Python installed.

- [Prerequisites](#prerequisites)
- [Compilation](#compilation)
- [Docker](#docker)
- Platforms: [Ubuntu](#ubuntu), [OS X](OSX)

When updating Caffe2, it's best to `make clean` before re-compiling.

## Prerequisites

Caffe2 has several dependencies.

* A C++ compiler that supports C++11.
* [CUDA](https://developer.nvidia.com/cuda-zone) is required for GPU mode.
    * library version above 6.5 are needed for C++11 support, and 7.0 is recommended.
* `protobuf`, `glog`, `gflags`, `eigen3`

In addition, Caffe2 has several optional dependencies: not having these will not cause problems, but some components will not work. Note that strictly speaking, CUDA is also an optional dependency. You can compile a purely CPU-based Caffe2 by not having CUDA. However, since CUDA is critical in achieving high-performance computation, you may want to consider it a necessary dependency.

* [OpenCV](http://opencv.org/), which is needed for image-related operations. If you work with images, you most likely want this.
* [OpenMPI](http://www.open-mpi.org/), needed for MPI-related Caffe2 operators.
* `leveldb`, needed for Caffe2's LevelDB IO backend. LevelDB also depends on `snappy`.
* `lmdb`, needed for Caffe2's LMDB IO backend.
* [ZeroMQ](http://zeromq.org/), needed for Caffe2's ZmqDB IO backend (serving data through a socket).
* [cuDNN](https://developer.nvidia.com/cudnn), needed for Caffe2's cuDNN operators.

If you do not install some of the dependencies, when you compile Caffe2, you will receive warning message that some components are not correctly built. This is fine - you will be able to use the other components without problem.

Pycaffe2 has its own natural needs, mostly on the Python side: `numpy (>= 1.7)` and `protobuf` are needed. We also recommend installing the following packages: `flask`, `ipython`, `matplotlib`, `notebook`, `pydot`, `python-nvd3`, `scipy`, `tornado`, and `scikit-image`.

We suggest first installing the [Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution, which provides most of the necessary packages, as well as the `hdf5` library dependency.

## Compilation

Now that you have the prerequisites, you should be good to go. Caffe's build environment should be able to figure out everything itself. However, there may be cases when you need to manually specify some paths of the build toolchain. In that case, go to `build_env.py`, and modify the lines in the Env class (look for line `class Env(object)`) accordingly. Then, simply run

    make

The build script should tell you what got built and what did not get built.

## Docker

If you have docker installed on your machine, you may want to use the provided Docker build files for simpler set up. Please check the `contrib/docker*` folders for details.

Running these Docker images with CUDA GPUs is currently only supported on Linux hosts, as far as I can tell. You will need to make sure that your host driver is also 346.46, and you will need to invoke docker with

    docker run -t -i --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia0 [other cuda cards] ...

## Ubuntu

For ubuntu 14.04 users, the Docker script may be a good example on the steps of building Caffe2. Please check `contrib/docker-ubuntu-14.04/Dockerfile` for details. For ubuntu 12.04, use `contrib/docker-ubuntu-12.04/Dockerfile`.

## OSX

Detailed instruction for OSX is to be written. For now, check [Caffe's OSX guide](http://caffe.berkeleyvision.org/install_osx.html) for details.