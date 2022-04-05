import torch
from typing import Dict, Callable
from .observation_type import ObservationType
from ..quantization_patterns import QuantizeHandler
from ..quantization_types import NodePattern

def get_quantize_handler_cls(observation_type, dtype_configs):

    class ConfigurableQuantizeHandler(QuantizeHandler):
        def __init__(
                self,
                node_pattern: NodePattern,
                modules: Dict[str, torch.nn.Module],
                root_node_getter: Callable = None):
            super().__init__(node_pattern, modules, root_node_getter)
            self.observation_type = observation_type
            self.dtype_configs = dtype_configs

        def is_general_tensor_value_op(self) -> bool:
            return observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT

    return ConfigurableQuantizeHandler
