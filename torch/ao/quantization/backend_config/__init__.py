from .tensorrt import get_tensorrt_backend_config_dict
from .native import get_native_backend_config_dict

# TODO: add more validations
def validate_backend_config_dict(backend_config_dict):
    return "configs" in backend_config_dict

__all__ = [
    "get_native_backend_config_dict",
    "get_tensorrt_backend_config_dict",
]
