import torch
import torch.nn as nn
from .quantize_handler import get_quantize_handler_cls
from .fuse_handler import get_fuse_handler_cls
from typing import Dict, Any, List, Callable, Union
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
    for config in backend_config_dict.get("configs", []):
        pattern = config["pattern"]
        observation_type = config["observation_type"]
        dtype_configs = config["dtype_configs"]
        pattern_to_quantize_handlers[pattern] = \
            get_quantize_handler_cls(observation_type, dtype_configs)

    return pattern_to_quantize_handlers

def get_pattern_to_dtype_configs(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, List[Dict[str, torch.dtype]]]:
    pattern_to_dtype_configs: Dict[Pattern, List[Dict[str, torch.dtype]]] = dict()
    for config in backend_config_dict.get("configs", []):
        pattern = config["pattern"]
        dtype_configs = config["dtype_configs"]
        pattern_to_dtype_configs[pattern] = dtype_configs
    return pattern_to_dtype_configs

def get_pattern_to_input_type_to_index(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Dict[str, int]]:
    pattern_to_input_type_to_index: Dict[Pattern, Dict[str, int]] = dict()
    for config in backend_config_dict.get("configs", []):
        pattern = config["pattern"]
        input_type_to_index = config.get("input_type_to_index", {})
        pattern_to_input_type_to_index[pattern] = input_type_to_index
    return pattern_to_input_type_to_index

def get_quantized_reference_module_mapping(
        backend_config_dict: Dict[str, Any]) -> Dict[Callable, Callable]:
    mapping: Dict[Callable, Callable] = dict()
    for config in backend_config_dict.get("configs", []):
        if "root_module" in config and "reference_quantized_module_for_root" in config:
            mapping[config["root_module"]] = config["reference_quantized_module_for_root"]
    return mapping

def get_fusion_pattern_to_fuse_handler_cls(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Callable]:
    fusion_pattern_to_fuse_handlers = dict()
    for config in backend_config_dict.get("configs", []):
        if "fuser_method" in config:
            pattern = config["pattern"]
            fusion_pattern_to_fuse_handlers[pattern] = \
                get_fuse_handler_cls()

    return fusion_pattern_to_fuse_handlers

def get_fuser_method_mapping(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Union[nn.Sequential, Callable]]:
    fuser_method_mapping : Dict[Pattern, Union[nn.Sequential, Callable]] = dict()
    for config in backend_config_dict.get("configs", []):
        if "fuser_method" in config:
            pattern = config["pattern"]
            fuser_method = config["fuser_method"]
            fuser_method_mapping[pattern] = fuser_method

    return fuser_method_mapping

def get_module_to_qat_module(
        backend_config_dict: Dict[str, Any]) -> Dict[Callable, Callable]:
    module_to_qat_module: Dict[Callable, Callable] = dict()
    for config in backend_config_dict.get("configs", []):
        if "pattern" in config and "qat_module" in config:
            pattern = config["pattern"]
            qat_module = config["qat_module"]
            module_to_qat_module[pattern] = qat_module

    return module_to_qat_module
