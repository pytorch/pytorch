import torch
from typing import Dict
from torch.fx.graph import Node
from .observation_type import ObservationType
from ..quantization_patterns import QuantizeHandler

def get_quantize_handler_cls(observation_type, dtype_configs):

    class ConfigurableQuantizeHandler(QuantizeHandler):
        def __init__(self, node: Node, modules: Dict[str, torch.nn.Module]):
            super().__init__(node, modules)
            self.observation_type = observation_type
            self.dtype_configs = dtype_configs

        def is_general_tensor_value_op(self) -> bool:
            return observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT

    return ConfigurableQuantizeHandler
