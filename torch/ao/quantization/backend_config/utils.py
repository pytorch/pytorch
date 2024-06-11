# mypy: allow-untyped-defs
from typing import Dict, Any, List, Callable, Union, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
)
from ..utils import Pattern
from ..fuser_method_mappings import (
    _reverse2,
    _reverse3,
)

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
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        pattern_to_dtype_configs[pattern] = config.dtype_configs
    return pattern_to_dtype_configs

def get_qat_module_classes(backend_config: BackendConfig) -> Tuple[type, ...]:
    qat_module_classes = []
    for config in backend_config.configs:
        if config.qat_module is not None:
            qat_module_classes.append(config.qat_module)
    return tuple(set(qat_module_classes))

def get_fused_module_classes(backend_config: BackendConfig) -> Tuple[type, ...]:
    fused_module_classes = []
    for config in backend_config.configs:
        if config.fused_module is not None:
            fused_module_classes.append(config.fused_module)
    return tuple(set(fused_module_classes))

def get_pattern_to_input_type_to_index(backend_config: BackendConfig) -> Dict[Pattern, Dict[str, int]]:
    pattern_to_input_type_to_index: Dict[Pattern, Dict[str, int]] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        pattern_to_input_type_to_index[pattern] = config._input_type_to_index
    return pattern_to_input_type_to_index

def get_root_module_to_quantized_reference_module(
        backend_config: BackendConfig) -> Dict[Type[torch.nn.Module], Type[torch.nn.Module]]:
    mapping: Dict[Type[torch.nn.Module], Type[torch.nn.Module]] = {}
    for config in backend_config.configs:
        if config.root_module is not None and config.reference_quantized_module is not None:
            mapping[config.root_module] = config.reference_quantized_module
    return mapping

def get_fuser_method_mapping(backend_config: BackendConfig) -> Dict[Pattern, Union[nn.Sequential, Callable]]:
    fuser_method_mapping : Dict[Pattern, Union[nn.Sequential, Callable]] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        if config.fuser_method is not None:
            # Note: both the fuser method and the pattern are specified in forward order in the
            # BackendConfig, but the internal pattern matching code uses the reversed nested tuple
            # format, so we need to convert both to the internal format
            fuser_method = _get_fuser_method_in_reversed_nested_tuple_format(config)
            fuser_method_mapping[pattern] = fuser_method
    return fuser_method_mapping

def get_module_to_qat_module(backend_config: BackendConfig) -> Dict[Pattern, Type[torch.nn.Module]]:
    module_to_qat_module: Dict[Pattern, Type[torch.nn.Module]] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
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
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
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
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
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

def _get_pattern_in_reversed_nested_tuple_format(config: BackendPatternConfig) -> Pattern:
    """
    Return the pattern specified in the given config in the reversed nested tuple format
    used internally in the quantization pattern matching code.

    If the pattern is not a tuple, or the pattern is already specified in the reversed
    nested tuple format, return the pattern as is. Otherwise:

    For 2-tuples (a, b), return (b, a).
    For 3-tuples (a, b, c), return (c, (b, a)).

    For example:
        * Given nn.Linear, return nn.Linear
        * Given (nn.Linear, nn.ReLU), return (nn.ReLU, nn.Linear)
        * Given (nn.Conv2d, nn.BatchNorm2d, nn.ReLU), return
          (nn.ReLU, (nn.BatchNorm2d, nn.Conv2d))

    For context, the reason why this is needed is the user-facing BackendConfig
    API accepts the flat 2-or-3-tuple format in forward order. While this simple
    format handles the vast majority of use cases, it does not handle the more
    complex ones, and so the internal pattern matching code for quantization uses
    the following, more general reversed nested tuple format instead:

        operator = module_type | functional | torch op | native op | MatchAllNode
        Pattern = (operator, Pattern, Pattern, ...) | operator

    In the future, we expect to replace the above complex format with the one used
    by the subgraph rewriter in torch.fx, so we don't have to maintain our own
    complex pattern matching code. Then we won't need this helper function anymore.
    """
    if config._pattern_complex_format is not None:
        return config._pattern_complex_format
    if config.pattern is None:
        raise ValueError("Either 'pattern' or 'pattern_complex_format' must be specified")
    if not isinstance(config.pattern, tuple):
        return config.pattern

    # Pattern is specified in the simple tuple format, need to convert
    if len(config.pattern) == 2:
        (a, b) = config.pattern
        return (b, a)
    elif len(config.pattern) == 3:
        (a, b, c) = config.pattern
        return (c, (b, a))
    else:
        raise ValueError("Expected a tuple with 2 or 3 elements, got: ", config.pattern)

def _get_fuser_method_in_reversed_nested_tuple_format(config: BackendPatternConfig) -> Callable:
    """
    Return the fuser method specified in the given config in the reversed nested
    tuple format used internally in the quantization pattern matching code.

    If pattern is specified in the reversed nested tuple format, we assume the
    fuser method is also specified in this format and simply return it as is.
    Otherwise, we convert the fuser method as follows:

        * Given f(is_qat, conv, relu), return f'(is_qat, relu, conv)
        * Given f(is_qat, conv, bn, relu), return f'(is_qat, relu, bn_conv),
          where bn_conv is a 2-tuple (bn, conv)

    The first argument of a fuser method is always `is_qat` and is not affected
    in the conversion. We currently only support functions with 3 or 4 arguments.
    """
    assert config.fuser_method is not None
    if config._pattern_complex_format is not None:
        return config.fuser_method
    if not isinstance(config.pattern, tuple):
        raise ValueError("Expected pattern to be a tuple, got: ", config.pattern)

    # Pattern is specified in the simple tuple format, need to convert
    if len(config.pattern) == 2:
        return _reverse2(config.fuser_method)
    elif len(config.pattern) == 3:
        return _reverse3(config.fuser_method)
    else:
        raise ValueError("Expected a tuple with 2 or 3 elements, got: ", config.pattern)
