import torch
from typing import Dict, Callable
from .observation_type import ObservationType
from ..quantization_patterns import QuantizeHandler
from ..quantization_types import NodePattern

def get_quantize_handler_cls(
        observation_type, dtype_configs, num_tensor_args_to_observation_type):

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

        def is_general_tensor_value_op(self) -> bool:
            return self.observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT

    return ConfigurableQuantizeHandler
