from ..utils import has_triton

if has_triton():
    from .conv import _conv, conv
    from .conv1x1 import _conv1x1, conv1x1

    __all__ = ["_conv", "conv", "_conv1x1", "conv1x1"]
