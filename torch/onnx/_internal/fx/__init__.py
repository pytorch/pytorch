from .patcher import ONNXTorchPatcher
from .serialization import save_model_with_external_data


__all__ = [
    "save_model_with_external_data",
    "ONNXTorchPatcher",
]
