# mypy: allow-untyped-defs
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Union

import torch
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.fuser_method_mappings import get_fuser_method_new
from torch.ao.quantization.utils import _parent_name, NodePattern, Pattern
from torch.fx.graph import Graph, Node
from torch.nn.utils.parametrize import type_before_parametrizations

from .custom_config import FuseCustomConfig
from .match_utils import MatchAllNode


__all__ = [
    "DefaultFuseHandler",
    "FuseHandler",
]


# ----------------------------
# Fusion Pattern Registrations
# ----------------------------


# Base Pattern Handler
class FuseHandler(ABC):
    """Base handler class for the fusion patterns"""

    @abstractmethod
    def __init__(self, node: Node):
        pass

    @abstractmethod
    def fuse(
        self,
        load_arg: Callable,
        named_modules: dict[str, torch.nn.Module],
        fused_graph: Graph,
        root_node: Node,
        extra_inputs: list[Any],
        matched_node_pattern: NodePattern,
        fuse_custom_config: FuseCustomConfig,
        fuser_method_mapping: dict[Pattern, Union[torch.nn.Sequential, Callable]],
        is_qat: bool,
    ) -> Node:
        pass


class DefaultFuseHandler(FuseHandler):
    def __init__(self, node: Node):
        super().__init__(node)  # type:ignore[safe-super]

    def fuse(
        self,
        load_arg: Callable,
        named_modules: dict[str, torch.nn.Module],
        fused_graph: Graph,
        root_node: Node,
        extra_inputs: list[Any],
        matched_node_pattern: NodePattern,
        fuse_custom_config: FuseCustomConfig,
        fuser_method_mapping: dict[Pattern, Union[torch.nn.Sequential, Callable]],
        is_qat: bool,
    ) -> Node:
        if root_node.op != "call_module":
            raise AssertionError("Expecting module node to be a call_module Node")
        root_module = named_modules[str(root_node.target)]

        def get_modules(pattern):
            """Given a node pattern, extract the corresponding modules
            e.g. input: (relu_node, (bn_node, conv_node))
                 output: (relu_module, (bn_module, conv_module))
            """
            if isinstance(pattern, (tuple, list)):
                n, *args = pattern
                modules: list[torch.nn.Module] = []
                modules.append(get_modules(n))
                modules.extend(get_modules(a) for a in args)
                return tuple(modules)
            else:
                n = pattern
                if n.op == "call_module":
                    return named_modules[n.target]
                elif n.op == "call_function" and n.target is torch.nn.functional.relu:
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
        extra_args = [load_arg(input) for input in extra_inputs]
        node = fused_graph.node_copy(root_node, load_arg)
        args = list(node.args)
        args.extend(extra_args)
        node.args = tuple(args)
        return node


def _get_fusion_pattern_to_fuse_handler_cls(
    backend_config: BackendConfig,
) -> dict[Pattern, Callable]:
    fusion_pattern_to_fuse_handlers: dict[Pattern, Callable] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        if config.fuser_method is not None:
            # TODO: is this logic right?
            fusion_pattern_to_fuse_handlers[pattern] = DefaultFuseHandler
    return fusion_pattern_to_fuse_handlers
