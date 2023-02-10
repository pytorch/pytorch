import importlib
import os
import tempfile

import torch
from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend

'''
Placeholder for TensorRT backend for dynamo via torch-tensorrt
'''

# @register_backend
# def tensorrt(gm, example_inputs):
#    import torch_tensorrt # type: ignore[import]
#    pass
