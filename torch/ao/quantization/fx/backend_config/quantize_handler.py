import torch
from typing import Dict, Callable
from .observation_type import ObservationType
from ..quantization_patterns import QuantizeHandler
from ..quantization_types import NodePattern
from ..utils import all_node_args_have_no_tensors
from torch.fx import Node


def _binary_op_observation_type_getter(quantize_handler):
    # determine how many of the first two args are Tensors (versus scalars)
    # this distinguishes things like "x + y" from "x + 2" or "2 + x"
    num_tensor_args = 0
    if isinstance(quantize_handler.root_node, Node):
        cache_for_no_tensor_check: Dict[Node, bool] = dict()
        for arg_idx in range(len(quantize_handler.root_node.args)):
            arg = quantize_handler.root_node.args[arg_idx]
            if isinstance(arg, Node) and (
                    not all_node_args_have_no_tensors(
                        arg, quantize_handler.modules, cache_for_no_tensor_check)):
                num_tensor_args += 1
    if num_tensor_args == 1:
        return ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    else:
        return ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT

def get_quantize_handler_cls(
        observation_type, dtype_configs, is_binary_op_with_binary_scalar_op_variant):

    class ConfigurableQuantizeHandler(QuantizeHandler):
        def __init__(
                self,
                node_pattern: NodePattern,
                modules: Dict[str, torch.nn.Module],
                root_node_getter: Callable = None):
            super().__init__(node_pattern, modules, root_node_getter)
            if is_binary_op_with_binary_scalar_op_variant:
                self.observation_type = _binary_op_observation_type_getter(self)
            else:
                self.observation_type = observation_type
            self.dtype_configs = dtype_configs

        def is_general_tensor_value_op(self) -> bool:
            return self.observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT

    return ConfigurableQuantizeHandler
