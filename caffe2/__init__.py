import warnings
from torch.onnx import _CAFFE2_ATEN_FALLBACK

if not _CAFFE2_ATEN_FALLBACK:
    warnings.warn("Caffe2 support is no longer present in PyTorch.")
