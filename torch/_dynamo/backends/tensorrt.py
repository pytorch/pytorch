import importlib
import logging

from torch._dynamo import register_backend

logger = logging.getLogger(__name__)


@register_backend
def tensorrt(*args, **kwargs):
    try:
        from torch_tensorrt.dynamo.backend import torch_tensorrt_backend
    except ImportError:
        logger.exception(
            "Unable to import Torch-TensorRT. Please install Torch-TensorRT. "
            "See https://github.com/pytorch/TensorRT for more details."
        )
        raise

    return torch_tensorrt_backend(*args, **kwargs)


def has_torch_tensorrt():
    try:
        importlib.import_module("torch_tensorrt.dynamo.backend")
        return True
    except ImportError:
        return False
