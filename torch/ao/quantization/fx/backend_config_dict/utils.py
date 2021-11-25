import torch
from .quantize_handler import get_quantize_handler_cls
from typing import Dict, Any, List, Callable
from ..quantization_types import Pattern, QuantizerCls

def get_pattern_to_quantize_handlers(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, QuantizerCls]:
    """
    Note: Quantize handler is just a holder for some check methods like
    (should_insert_observer_for_output), maybe this can be a enum as well,
    we can refactor this after we convert the path for fbgemm/qnnpack fully to the
    new path, this is not exposed to backend developers
    """
    pattern_to_quantize_handlers = dict()
    for config in backend_config_dict["configs"]:
        pattern = config["pattern"]
        observation_type = config["observation_type"]
        dtype_configs = config["dtype_configs"]
        pattern_to_quantize_handlers[pattern] = \
            get_quantize_handler_cls(observation_type, dtype_configs)

    return pattern_to_quantize_handlers


def get_pattern_to_dtype_configs(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, List[Dict[str, torch.dtype]]]:
    pattern_to_dtype_configs: Dict[Pattern, List[Dict[str, torch.dtype]]] = dict()
    for config in backend_config_dict["configs"]:
        pattern = config["pattern"]
        dtype_configs = config["dtype_configs"]
        pattern_to_dtype_configs[pattern] = dtype_configs
    return pattern_to_dtype_configs

def get_pattern_to_input_type_to_index(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Dict[str, int]]:
    pattern_to_input_type_to_index: Dict[Pattern, Dict[str, int]] = dict()
    for config in backend_config_dict["configs"]:
        pattern = config["pattern"]
        input_type_to_index = config.get("input_type_to_index", {})
        pattern_to_input_type_to_index[pattern] = input_type_to_index
    return pattern_to_input_type_to_index

def get_quantized_reference_module_mapping(
        backend_config_dict: Dict[str, Any]) -> Dict[Callable, Callable]:
    mapping: Dict[Callable, Callable] = dict()
    for config in backend_config_dict["configs"]:
        if "root_module" in config and "reference_quantized_module_for_root" in config:
            mapping[config["root_module"]] = config["reference_quantized_module_for_root"]
    return mapping
