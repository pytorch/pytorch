import tensorrt as trt

if hasattr(trt, "__version__"):
    from .converters import *  # noqa: F403
