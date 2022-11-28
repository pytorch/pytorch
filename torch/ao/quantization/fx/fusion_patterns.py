import torch
from torch.fx.graph import Node, Graph
from ..utils import NodePattern, Pattern
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union, List
from .custom_config import FuseCustomConfig

__all__ = [
    "FuseHandler",
    "DefaultFuseHandler",
]

# ----------------------------
# Fusion Pattern Registrations
# ----------------------------

# Base Pattern Handler
class FuseHandler(ABC):
    """ Base handler class for the fusion patterns
    """
    def __init__(self, node: Node):
        pass

    @abstractmethod
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
        pass

# TODO: remove
class DefaultFuseHandler(FuseHandler):
    pass
