import warnings
from torch.onnx import _CAFFE2_ATEN_FALLBACK

if not _CAFFE2_ATEN_FALLBACK:
    warnings.warn("Caffe2 support is not fully enabled in this PyTorch build. "
                  "Please enable Caffe2 by building PyTorch from source with `BUILD_CAFFE2=1` flag.", stacklevel=TO_BE_DETERMINED)
