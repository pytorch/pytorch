import torch
from typing import Dict
from torch.fx.graph import Node
from .observation_type import ObservationType
from ..quantization_patterns import QuantizeHandler

def get_quantize_handler_cls(observation_type, pattern_configs):
    assert observation_type == ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    "Only OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT is supported right now"

    class ConfigurableQuantizeHandler(QuantizeHandler):
        def __init__(self, node: Node, modules: Dict[str, torch.nn.Module]):
            super().__init__(node, modules)
            self.pattern_configs = pattern_configs

    return ConfigurableQuantizeHandler
