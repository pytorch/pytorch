import warnings


try:
    from caffe2.proto import caffe2_pb2
except ImportError:
    warnings.warn("Caffe2 was not built with this PyTorch build. "
                  "Please enable Caffe2 by building PyTorch from source with `BUILD_CAFFE2=1` flag.")
