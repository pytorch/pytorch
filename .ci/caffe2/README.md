# Jenkins

The scripts in this directory are the entrypoint for testing Caffe2.

The environment variable `BUILD_ENVIRONMENT` is expected to be set to
the build environment you intend to test. It is a hint for the build
and test scripts to configure Caffe2 a certain way and include/exclude
tests. Docker images, they equal the name of the image itself. For
example: `py2-cuda9.0-cudnn7-ubuntu16.04`. The Docker images that are
built on Jenkins and are used in triggered builds already have this
environment variable set in their manifest. Also see
`./docker/jenkins/*/Dockerfile` and search for `BUILD_ENVIRONMENT`.

Our Jenkins installation is located at https://ci.pytorch.org/jenkins/.
