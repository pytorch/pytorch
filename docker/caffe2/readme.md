# Docker & Caffe2

**Note: use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run all GPU builds.**

To get the latest source, rerun the docker builds using the Dockerfiles.

Docker images at https://hub.docker.com/r/caffe2ai/caffe2/ are a few months old, but will be refreshed soon.

**Build like:** `docker build -t caffe2:cuda8-cudnn6-all-options .`

**Run like:** `nvidia-docker run --rm -it caffe2:cuda8-cudnn6-all-options python -m caffe2.python.operator_test.relu_op_test`

For Docker on USB related instructions you can find some help on the gh-pages branch [here](https://github.com/caffe2/caffe2/tree/gh-pages/docker/caffe2-docker-usb)
