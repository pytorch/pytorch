from ._lower_to_native_backend import _lower_to_native_backend
from .graph_module import QuantizedGraphModule
from typing import Dict
import torch

def lower_to_qnnpack(model: QuantizedGraphModule, modules: Dict[str, torch.nn.Module]) -> QuantizedGraphModule:
    """ Lower a quantized reference model (with reference quantized operator patterns)
    to qnnpack
    """
    return _lower_to_native_backend(model, modules)
