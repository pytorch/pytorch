# Docker images for Jenkins

This directory contains everything needed to build the Docker images
that are used in our Jenkins setup. These images provide a variety of
build environments for which we want to ensure that Caffe2 is
compatible.

The Dockerfiles located in subdirectories are parameterized to
conditionally run build stages depending on build arguments passed to
`docker build`. This lets us use only a few Dockerfiles for many
images. The different configurations are identified by a freeform
string that we call a _build environment_. This string is persisted in
each image as the `BUILD_ENVIRONMENT` environment variable.

Examples of valid build environments are:

* `py2-cuda9.0-cudnn7-ubuntu16.04`
* `py3-mkl-ubuntu16.04`
* `py3-gcc7-ubuntu16.04`
* `py3-cuda8.0-cudnn7-centos7`

See `build.sh` for a full list of terms that are extracted from the
build environment into parameters for the image build.

## Contents

* `build.sh` -- dispatch script to launch all builds
* `common` -- scripts used to execute individual Docker build stages
* `centos` -- Dockerfile for CentOS image
* `centos-cuda` -- Dockerfile for CentOS image with CUDA support for nvidia-docker
* `ubuntu` -- Dockerfile for Ubuntu image
* `ubuntu-cuda` -- Dockerfile for Ubuntu image with CUDA support for nvidia-docker



