# Caffe2 & TensorRT integration

[![Jenkins Build Status](https://ci.pytorch.org/jenkins/job/caffe2-master/lastCompletedBuild/badge/icon)](https://ci.pytorch.org/jenkins/job/caffe2-master)

This directory contains the code implementing `TensorRTOp` Caffe2 operator as well as Caffe2 model converter (using `ONNX` model as an intermediate format).
To enable this functionality in your PyTorch build please set

`USE_TENSORRT=1 ... python setup.py ...`
 
 or if you use CMake directly
 
 `-DUSE_TENSORRT=ON`

For further information please explore `caffe2/python/trt/test_trt.py` test showing all possible use cases.

## Questions and Feedback

Please use GitHub issues (https://github.com/pytorch/pytorch/issues) to ask questions, report bugs, and request new features.
