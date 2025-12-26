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
## Fix typos and clarify sections

In this commit, several typos in the README file have been corrected.
Additionally, the following sections were updated to make them clearer:
- Installation instructions
- Usage examples
- Contributing guidelines

These changes aim to provide more clarity for users and new contributors.
## Fix typos and clarify sections

In this commit, several typos in the README file have been corrected.
Additionally, the following sections were updated to make them clearer:
- Installation instructions
- Usage examples
- Contributing guidelines

These changes aim to provide more clarity for users and new contributors.
