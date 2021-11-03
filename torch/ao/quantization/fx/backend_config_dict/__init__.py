from .tensorrt import get_tensorrt_backend_config_dict

def validate_backend_config_dict(backend_config_dict):
    return "configs" in backend_config_dict
