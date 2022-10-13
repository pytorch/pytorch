from typing import Dict, Any, List, Callable, Union, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import BackendConfig, DTypeConfig
from ..quantization_types import Pattern

__all__ = [
    "get_pattern_to_dtype_configs",
    "get_qat_module_classes",
    "get_fused_module_classes",
    "get_pattern_to_input_type_to_index",
    "get_root_module_to_quantized_reference_module",
    "get_fuser_method_mapping",
    "get_module_to_qat_module",
    "get_fusion_pattern_to_root_node_getter",
    "get_fusion_pattern_to_extra_inputs_getter",
    "remove_boolean_dispatch_from_name",
    "pattern_to_human_readable",
    "entry_to_pretty_str",
]

def get_pattern_to_dtype_configs(backend_config: BackendConfig) -> Dict[Pattern, List[DTypeConfig]]:
    pattern_to_dtype_configs: Dict[Pattern, List[DTypeConfig]] = {}
    for pattern, config in backend_config.configs.items():
        pattern_to_dtype_configs[pattern] = config.dtype_configs
    return pattern_to_dtype_configs

def get_qat_module_classes(backend_config: BackendConfig) -> Tuple[type, ...]:
    qat_module_classes = []
    for config in backend_config.configs.values():
        if config.qat_module is not None:
            qat_module_classes.append(config.qat_module)
    return tuple(set(qat_module_classes))

def get_fused_module_classes(backend_config: BackendConfig) -> Tuple[type, ...]:
    fused_module_classes = []
    for config in backend_config.configs.values():
        if config.fused_module is not None:
            fused_module_classes.append(config.fused_module)
    return tuple(set(fused_module_classes))

def get_pattern_to_input_type_to_index(backend_config: BackendConfig) -> Dict[Pattern, Dict[str, int]]:
    pattern_to_input_type_to_index: Dict[Pattern, Dict[str, int]] = {}
    for pattern, config in backend_config.configs.items():
        pattern_to_input_type_to_index[pattern] = config._input_type_to_index
    return pattern_to_input_type_to_index

def get_root_module_to_quantized_reference_module(
        backend_config: BackendConfig) -> Dict[Type[torch.nn.Module], Type[torch.nn.Module]]:
    mapping: Dict[Type[torch.nn.Module], Type[torch.nn.Module]] = {}
    for config in backend_config.configs.values():
        if config.root_module is not None and config.reference_quantized_module is not None:
            mapping[config.root_module] = config.reference_quantized_module
    return mapping

def get_fuser_method_mapping(backend_config: BackendConfig) -> Dict[Pattern, Union[nn.Sequential, Callable]]:
    fuser_method_mapping : Dict[Pattern, Union[nn.Sequential, Callable]] = {}
    for pattern, config in backend_config.configs.items():
        if config.fuser_method is not None:
            fuser_method_mapping[pattern] = config.fuser_method
    return fuser_method_mapping

def get_module_to_qat_module(backend_config: BackendConfig) -> Dict[Pattern, Type[torch.nn.Module]]:
    module_to_qat_module: Dict[Pattern, Type[torch.nn.Module]] = {}
    for pattern, config in backend_config.configs.items():
        if config.qat_module is not None:
            module_to_qat_module[pattern] = config.qat_module
    return module_to_qat_module

def get_fusion_pattern_to_root_node_getter(backend_config: BackendConfig) -> Dict[Pattern, Callable]:
    """ Get a map from fusion pattern to a function that returns the root node
    from the fusion pattern, e.g. the most common one is:
    def get_root_node(node_pattern):
        while not isinstance(node_pattern[-1], Node):
            node_pattern = node_pattern[-1]
        return node_pattern[-1]
    This can work for all patterns whose root node is the "last node" in the pattern,
    e.g. (torch.add, MatchAllNode, (torch.ReLU, torch.Conv2d))
    """
    root_node_getter_mapping: Dict[Pattern, Callable] = {}
    for pattern, config in backend_config.configs.items():
        if config._root_node_getter is not None:
            root_node_getter_mapping[pattern] = config._root_node_getter
    return root_node_getter_mapping

def get_fusion_pattern_to_extra_inputs_getter(backend_config: BackendConfig) -> Dict[Pattern, Callable]:
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
    extra_inputs_getter_mapping: Dict[Pattern, Callable] = {}
    for pattern, config in backend_config.configs.items():
        if config._extra_inputs_getter is not None:
            extra_inputs_getter_mapping[pattern] = config._extra_inputs_getter
    return extra_inputs_getter_mapping

def remove_boolean_dispatch_from_name(p) -> Any:
    """
    Some ops have a default string representation such as
    '<function boolean_dispatch.<locals>.fn at 0x7ff1106bf280>',
    this function replaces them with the hardcoded function names.
    """
    if p is F.fractional_max_pool2d:
        return "torch.nn.functional.fractional_max_pool2d"
    elif p is F.fractional_max_pool3d:
        return "torch.nn.functional.fractional_max_pool3d"
    elif p is F.max_pool1d:
        return "torch.nn.functional.max_pool1d"
    elif p is F.max_pool2d:
        return "torch.nn.functional.max_pool2d"
    elif p is F.max_pool3d:
        return "torch.nn.functional.max_pool3d"
    elif p is F.adaptive_max_pool1d:
        return "torch.nn.functional.adaptive_max_pool1d"
    elif p is F.adaptive_max_pool2d:
        return "torch.nn.functional.adaptive_max_pool2d"
    elif p is F.adaptive_max_pool3d:
        return "torch.nn.functional.adaptive_max_pool3d"
    assert "boolean_dispatch" not in str(p), \
        f"{p} does not have a human readable representation in " + \
        "quantization documentation"
    return p

def pattern_to_human_readable(p) -> Any:
    if isinstance(p, tuple):
        # nested patterns, recurse
        return tuple(pattern_to_human_readable(inner_p) for inner_p in p)
    elif isinstance(p, str):
        # method names are already human readable
        return p
    else:
        p = remove_boolean_dispatch_from_name(p)
        return p

# TODO(future PR): move backend_config_dict to use dataclass and move this logic to
# the corresponding __str__ function
def entry_to_pretty_str(entry) -> str:
    """
    Given a backend_config_dict entry, returns a string with the human readable
    representation of it.
    """
    s = "{\n"

    # always output the pattern first
    if "pattern" in entry:
        pattern_str = pattern_to_human_readable(entry["pattern"])

        s += f"  'pattern': {pattern_str},\n"

    # custom output for dtype_configs to make it look nice
    if "dtype_configs" in entry:
        s += "  'dtype_configs': [\n"
        for dtype_config in entry["dtype_configs"]:
            s += "    {\n"
            for k, v in dtype_config.items():
                s += f"      '{k}': {v},\n"
            s += "    },\n"
        s += "  ],\n"

    # custom output for num_tensor_args_to_observation_type to make it look nice
    if "num_tensor_args_to_observation_type" in entry:
        s += "  'num_tensor_args_to_observation_type': {\n"
        for k, v in entry["num_tensor_args_to_observation_type"].items():
            s += f"    {k}: {v},\n"
        s += "  },\n"

    # output all the other fields
    custom_handled_fields = [
        "pattern",
        "dtype_configs",
        "num_tensor_args_to_observation_type",
    ]
    for field_name in entry:
        if field_name in custom_handled_fields:
            continue
        s += f"  '{field_name}': {entry[field_name]},\n"

    s += "}"
    return s
