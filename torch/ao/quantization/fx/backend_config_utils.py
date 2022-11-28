import torch
from torch.ao.quantization.fx.pattern_utils import get_default_quant_patterns, sorted_patterns_dict
from torch.ao.quantization.backend_config import (
    get_native_backend_config,
    ObservationType,
)
from torch.ao.quantization.utils import (
    get_combined_dict,
    Pattern,
    NodePattern,
    QuantizerCls,
)

from ..backend_config import BackendConfig
from ..fuser_method_mappings import get_fuser_method_new
from ..utils import _parent_name, NodePattern, Pattern
from .quantization_patterns import QuantizeHandler
from .fusion_patterns import FuseHandler
from .match_utils import MatchAllNode
from .custom_config import FuseCustomConfig
from torch.nn.utils.parametrize import type_before_parametrizations

from typing import Callable, Dict

class DefaultFuseHandler(FuseHandler):
    def __init__(self, node: Node):
        super().__init__(node)

    def fuse(self,
             load_arg: Callable,
             named_modules: Dict[str, torch.nn.Module],
             fused_graph: Graph,
             root_node: Node,
             extra_inputs: List[Any],
             matched_node_pattern: NodePattern,
             fuse_custom_config: FuseCustomConfig,
             fuser_method_mapping: Optional[Dict[Pattern, Union[torch.nn.Sequential, Callable]]],
             is_qat: bool) -> Node:
        assert root_node.op == "call_module", "Expecting module node to be a call_module Node"
        root_module = named_modules[str(root_node.target)]

        def get_modules(pattern):
            """ Given a node pattern, extract the corresponding modules
            e.g. input: (relu_node, (bn_node, conv_node))
                 output: (relu_module, (bn_module, conv_module))
            """
            if isinstance(pattern, (tuple, list)):
                n, *args = pattern
                modules: List[torch.nn.Module] = []
                modules.append(get_modules(n))
                for a in args:
                    modules.append(get_modules(a))
                return tuple(modules)
            else:
                n = pattern
                if n.op == "call_module":
                    return named_modules[n.target]
                elif n.op == "call_function" and n.target == torch.nn.functional.relu:
                    relu = torch.nn.ReLU()
                    relu.training = root_module.training
                    return relu
                elif n.op == "call_function" or n.op == "call_method":
                    return n.target
                else:
                    return MatchAllNode

        # since relu can be used multiple times, we'll need to create a relu module for each match
        matched_modules = get_modules(matched_node_pattern)

        def get_matched_types(m):
            if isinstance(m, tuple):
                return tuple(map(get_matched_types, m))
            if isinstance(m, torch.nn.Module):
                return type_before_parametrizations(m)
            return m

        matched_module_types = get_matched_types(matched_modules)
        module_parent_name, module_name = _parent_name(root_node.target)
        fuser_method = get_fuser_method_new(matched_module_types, fuser_method_mapping)
        # TODO: change the signature for fuser_method to take matched module patterns
        # as input
        fused_module = fuser_method(is_qat, *matched_modules)
        setattr(named_modules[module_parent_name], module_name, fused_module)
        extra_args = []
        for input in extra_inputs:
            extra_args.append(load_arg(input))
        node = fused_graph.node_copy(root_node, load_arg)
        args = list(node.args)
        args.extend(extra_args)
        node.args = tuple(args)
        return node

def get_quantize_handler_cls(
        observation_type,
        dtype_configs,
        num_tensor_args_to_observation_type,
        input_output_observed):

    class ConfigurableQuantizeHandler(QuantizeHandler):
        def __init__(
                self,
                node_pattern: NodePattern,
                modules: Dict[str, torch.nn.Module],
                root_node_getter: Callable = None):
            super().__init__(node_pattern, modules, root_node_getter)
            if num_tensor_args_to_observation_type:
                assert self.num_tensor_args in num_tensor_args_to_observation_type, \
                    f"Must provide observation_type config for tensor number {self.num_tensor_args}" \
                    f" in num_tensor_args_to_observation_type for {node_pattern}"
                self.observation_type = num_tensor_args_to_observation_type[self.num_tensor_args]
            else:
                self.observation_type = observation_type
            self.dtype_configs = dtype_configs
            self.input_output_observed_ = input_output_observed

        def is_general_tensor_value_op(self) -> bool:
            return self.observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT

        # This is temporary, and will be removed soon
        def input_output_observed(self):
            return self.input_output_observed_


    return ConfigurableQuantizeHandler

def get_pattern_to_quantize_handlers(backend_config: BackendConfig) -> Dict[Pattern, QuantizerCls]:
    """
    Note: Quantize handler is just a holder for some check methods like
    (should_insert_observer_for_output), maybe this can be a enum as well,
    we can refactor this after we convert the path for fbgemm/qnnpack fully to the
    new path, this is not exposed to backend developers
    """
    pattern_to_quantize_handlers = {}
    for pattern, config in backend_config.configs.items():
        observation_type = config.observation_type
        dtype_configs = config.dtype_configs
        num_tensor_args_to_observation_type = config._num_tensor_args_to_observation_type
        input_output_observed = config._input_output_observed
        if input_output_observed is None:
            input_output_observed = True
        pattern_to_quantize_handlers[pattern] = \
            get_quantize_handler_cls(
                observation_type,
                dtype_configs,
                num_tensor_args_to_observation_type,
                input_output_observed)

    return pattern_to_quantize_handlers

# TODO: move this to torch/ao/quantization/backend_config/utils.py
def get_fusion_pattern_to_fuse_handler_cls(
        backend_config: BackendConfig) -> Dict[Pattern, Callable]:
    fusion_pattern_to_fuse_handlers: Dict[Pattern, Callable] = {}
    for pattern, config in backend_config.configs.items():
        if config.fuser_method is not None:
            # TODO: is this logic right?
            fusion_pattern_to_fuse_handlers[pattern] = DefaultFuseHandler

    return fusion_pattern_to_fuse_handlers

# TODO: remove when all uses are changed to backend_config
def get_native_quant_patterns(additional_quant_patterns: Dict[Pattern, QuantizerCls] = None) -> Dict[Pattern, QuantizerCls]:
    """
    Return a map from pattern to quantize handlers based on the default patterns and the native backend_config.
    The returned map is sorted such that longer patterns will be encountered first when iterating through it.
    """
    patterns = get_default_quant_patterns()
    if additional_quant_patterns is not None:
        patterns = get_combined_dict(patterns, additional_quant_patterns)
    # TODO: currently we just extend the quantize handlers generated from
    # `get_native_backend_config`
    # in the future we can just assign backend_config when everything is defined
    for pattern, quantize_handler in get_pattern_to_quantize_handlers(get_native_backend_config()).items():
        patterns[pattern] = quantize_handler
    return sorted_patterns_dict(patterns)

get_fusion_pattern_to_fuse_handler_cls.__module__ = "torch.ao.quantization.fx.backend_config_utils"
get_native_quant_patterns.__module__ = "torch.ao.quantization.fx.backend_config_utils"
get_pattern_to_quantize_handlers.__module__ = "torch.ao.quantization.fx.backend_config_utils"

__all__ = [
    "get_quantize_handler_cls",
    "get_fusion_pattern_to_fuse_handler_cls",
    "get_native_quant_patterns",
    "get_pattern_to_quantize_handlers",
]
