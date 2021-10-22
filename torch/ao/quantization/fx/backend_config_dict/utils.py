from .quantize_handler import get_quantize_handler_cls
from typing import Dict, Any
from ..quantization_types import Pattern, QuantizerCls

def get_pattern_to_quantize_handlers(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, QuantizerCls]:
    pattern_to_quantize_handlers = {}
    for config in backend_config_dict["configs"]:
        pattern = config["pattern"]
        observation_type = config["observation_type"]
        dtype_configs = config["dtype_configs"]
        pattern_to_quantize_handlers[pattern] = \
            get_quantize_handler_cls(observation_type, dtype_configs)

    return pattern_to_quantize_handlers


def get_pattern_to_dtype_configs(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Dict[str, Any]]:
    pattern_to_dtype_configs: Dict[str, Any] = {}
    for config in backend_config_dict["configs"]:
        pattern = config["pattern"]
        dtype_configs = config["dtype_configs"]
        pattern_to_dtype_configs[pattern] = dtype_configs
    return pattern_to_dtype_configs
