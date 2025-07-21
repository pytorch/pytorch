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
sudo bash -c 'TRITON=1 ./build.sh pytorch-linux-bionic-py3.8-gcc9 -t myimage:latest
```


## [Guidance] Add external lib for build and testing
To add external libs for the docker build, follow the steps below
### Add pinned commit (if applies)
We use pinned commit for test stabiblity, the nightly.yaml file check and update pinned commit for certain repo dependency daily.
If the lib you introduce is needed to install for specicic pinned commit/ build from scratch from a repo
1. add the repo you want to watch in nightly.yml and merge-rules.yml:[exmaple: pinned vllm](https://github.com/pytorch/pytorch/pull/158591/files#diff-0d5658b415099a82c11c03a06ca4ec765b4003a1f4b2f3f1943980a882cf8aa6)
2. add initial pinned commit in .ci/docker/ci_commiy_pins/. the txt name should match the one defined in step 1

### Add docker image and build logics

.ci/docker/build.sh


use linux as example:

