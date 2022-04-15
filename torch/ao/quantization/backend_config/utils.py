from typing import Dict, Any, List, Callable, Union, Tuple

import torch
import torch.nn as nn
from ..quantization_types import Pattern

def get_pattern_to_dtype_configs(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, List[Dict[str, Any]]]:
    pattern_to_dtype_configs: Dict[Pattern, List[Dict[str, torch.dtype]]] = dict()
    for config in backend_config_dict.get("configs", []):
        pattern = config["pattern"]
        dtype_configs = config["dtype_configs"]
        pattern_to_dtype_configs[pattern] = dtype_configs
    return pattern_to_dtype_configs

def get_qat_module_classes(
        backend_config_dict: Dict[str, Any]) -> Tuple[type, ...]:
    qat_module_classes = []
    for config in backend_config_dict.get("configs", []):
        pattern = config["pattern"]
        qat_module = config.get("qat_module", None)
        if qat_module is not None:
            qat_module_classes.append(qat_module)
    return tuple(set(qat_module_classes))

def get_fused_module_classes(
        backend_config_dict: Dict[str, Any]) -> Tuple[type, ...]:
    fused_module_classes = []
    for config in backend_config_dict.get("configs", []):
        pattern = config["pattern"]
        fused_module = config.get("fused_module", None)
        if fused_module is not None:
            fused_module_classes.append(fused_module)
    return tuple(set(fused_module_classes))

def get_pattern_to_input_type_to_index(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Dict[str, int]]:
    pattern_to_input_type_to_index: Dict[Pattern, Dict[str, int]] = dict()
    for config in backend_config_dict.get("configs", []):
        pattern = config["pattern"]
        input_type_to_index = config.get("input_type_to_index", {})
        pattern_to_input_type_to_index[pattern] = input_type_to_index
    return pattern_to_input_type_to_index

def get_root_module_to_quantized_reference_module(
        backend_config_dict: Dict[str, Any]) -> Dict[Callable, Callable]:
    mapping: Dict[Callable, Callable] = dict()
    for config in backend_config_dict.get("configs", []):
        if "root_module" in config and "reference_quantized_module_for_root" in config:
            mapping[config["root_module"]] = config["reference_quantized_module_for_root"]
    return mapping

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

def get_fusion_pattern_to_root_node_getter(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Callable]:
    """ Get a map from fusion pattern to a function that returns the root node
    from the fusion pattern, e.g. the most common one is:
    def get_root_node(node_pattern):
        while not isinstance(node_pattern[-1], Node):
            node_pattern = node_pattern[-1]
        return node_pattern[-1]
    This can work for all patterns whose root node is the "last node" in the pattern,
    e.g. (torch.add, MatchAllNode, (torch.ReLU, torch.Conv2d))
    """
    root_node_getter_mapping: Dict[Pattern, Callable] = dict()
    for config in backend_config_dict.get("configs", []):
        if "root_node_getter" in config:
            pattern = config["pattern"]
            root_node_getter = config["root_node_getter"]
            root_node_getter_mapping[pattern] = root_node_getter

    return root_node_getter_mapping

def get_fusion_pattern_to_extra_inputs_getter(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Callable]:
    """ Get a map from fusion pattern to a function that returns extra input nodes
    from the fusion pattern, in the order required by the root node. This is optional,
    if not specified, we will not copy over any extra inputs for the root node.
    Example:
    # Let's say we have the pattern (torch.add, MatchAllNode, (torch.nn.BatchNorm2d, torch.nn.Conv2d))
    # and root node is torch.nn.Conv2d, and the node in MatchAllNode would be an extra
    # argument to the fused module, we can unpack the pattern and return the node at
    # MatchAllNode here
    # we can implement extra_inputs_getter as follows:
    def extra_inputs_getter(pattern) -> List[Any]:
        add, extra_input, conv_pattern = pattern
        return [extra_input]
    """
    extra_inputs_getter_mapping: Dict[Pattern, Callable] = dict()
    for config in backend_config_dict.get("configs", []):
        if "extra_inputs_getter" in config:
            pattern = config["pattern"]
            extra_inputs_getter = config["extra_inputs_getter"]
            extra_inputs_getter_mapping[pattern] = extra_inputs_getter

    return extra_inputs_getter_mapping
