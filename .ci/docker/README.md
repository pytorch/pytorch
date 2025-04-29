# Docker images for GitHub CI and CD

This directory contains everything needed to build the Docker images
that are used in our CI.

The Dockerfiles located in subdirectories are parameterized to
conditionally run build stages depending on build arguments passed to
`docker build`. This lets us use only a few Dockerfiles for many
images. The different configurations are identified by a freeform
string that we call a _build environment_. This string is persisted in
each image as the `BUILD_ENVIRONMENT` environment variable.

See `build.sh` for valid build environments (it's the giant switch).

## Docker CI builds

* `build.sh` -- dispatch script to launch all builds
* `common` -- scripts used to execute individual Docker build stages
* `ubuntu` -- Dockerfile for Ubuntu image for CPU build and test jobs
* `ubuntu-cuda` -- Dockerfile for Ubuntu image with CUDA support for nvidia-docker
* `ubuntu-rocm` -- Dockerfile for Ubuntu image with ROCm support
* `ubuntu-xpu` -- Dockerfile for Ubuntu image with XPU support

### Docker CD builds

* `conda` - Dockerfile and build.sh to build Docker images used in nightly conda builds
* `manywheel` - Dockerfile and build.sh to build Docker images used in nightly manywheel builds
* `libtorch` - Dockerfile and build.sh to build Docker images used in nightly libtorch builds

## Usage

```bash
# Build a specific image
./build.sh pytorch-linux-bionic-py3.8-gcc9 -t myimage:latest

# Set flags (see build.sh) and build image
sudo bash -c 'PROTOBUF=1 ./build.sh pytorch-linux-bionic-py3.8-gcc9 -t myimage:latest
```
