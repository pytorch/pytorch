from .fbgemm import get_fbgemm_backend_config_dict
from .tensorrt import get_tensorrt_backend_config_dict

def validate_backend_config_dict(backend_config_dict):
    return "quant_patterns" in backend_config_dict
