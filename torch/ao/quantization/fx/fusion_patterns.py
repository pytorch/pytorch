import torch
from torch.fx.graph import Node
from .pattern_utils import (
    register_fusion_pattern,
)
from ..utils import _parent_name
from .quantization_types import QuantizerCls, NodePattern, Pattern
from ..fuser_method_mappings import get_fuser_method_new
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union, List
from .match_utils import MatchAllNode

# ----------------------------
# Fusion Pattern Registrations
# ----------------------------

# Base Pattern Handler
class FuseHandler(ABC):
    """ Base handler class for the fusion patterns
    """
    def __init__(self, quantizer: QuantizerCls, node: Node):
        pass

    @abstractmethod
    def fuse(self,
             quantizer: QuantizerCls,
             load_arg: Callable,
             root_node: Node,
             matched_node_pattern: NodePattern,
             fuse_custom_config_dict: Dict[str, Any],
             fuser_method_mapping: Optional[Dict[Pattern, Union[torch.nn.Sequential, Callable]]],
             is_qat: bool) -> Node:
        pass

@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Linear))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Linear))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.BatchNorm2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.BatchNorm2d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.BatchNorm3d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.BatchNorm3d))
@register_fusion_pattern((torch.nn.BatchNorm1d, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.BatchNorm2d, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.BatchNorm3d, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.BatchNorm1d, torch.nn.Linear))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm1d, torch.nn.Conv1d)))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm3d, torch.nn.Conv3d)))
@register_fusion_pattern((torch.nn.functional.relu, (torch.nn.BatchNorm1d, torch.nn.Conv1d)))
@register_fusion_pattern((torch.nn.functional.relu, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
@register_fusion_pattern((torch.nn.functional.relu, (torch.nn.BatchNorm3d, torch.nn.Conv3d)))
@register_fusion_pattern((torch.nn.BatchNorm1d, torch.nn.ConvTranspose1d))
@register_fusion_pattern((torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d))
@register_fusion_pattern((torch.nn.BatchNorm3d, torch.nn.ConvTranspose3d))
class DefaultFuseHandler(FuseHandler):
    def __init__(
            self,
            quantizer: QuantizerCls,
            node: Node):
        super().__init__(quantizer, node)

    def fuse(self,
             quantizer: QuantizerCls,
             load_arg: Callable,
             root_node: Node,
             matched_node_pattern: NodePattern,
             fuse_custom_config_dict: Dict[str, Any],
             fuser_method_mapping: Optional[Dict[Pattern, Union[torch.nn.Sequential, Callable]]],
             is_qat: bool) -> Node:
        additional_fuser_method_mapping = fuse_custom_config_dict.get("additional_fuser_method_mapping", {})
        assert root_node.op == "call_module", "Expecting module node to be a call_module Node"
        root_module = quantizer.modules[root_node.target]
        assert len(additional_fuser_method_mapping) == 0, "Fusion implementation is "
        "undergoing changes, additoinal_fuser_method_mapping is not supported currently."
        def get_modules(pattern, modules):
            """ Given a node pattern, extract the corresponding modules
            e.g. input: (relu_node, (bn_node, conv_node))
                 output: (relu_module, (bn_module, conv_module))
            """
            if isinstance(pattern, (tuple, list)):
                n, *args = pattern
                get_modules(n, modules)
                arg_modules: List[torch.nn.Module] = []
                for a in args:
                    get_modules(a, arg_modules)
                arg_modules = tuple(arg_modules) if len(arg_modules) > 1 else arg_modules[0]  # type: ignore[assignment]
                modules.append(arg_modules)
            else:
                n = pattern
                if n.op == "call_module":
                    modules.append(quantizer.modules[n.target])
                elif n.op == "call_function" and n.target == torch.nn.functional.relu:
                    relu = torch.nn.ReLU()
                    relu.training = root_module.training
                    modules.append(relu)
                else:
                    modules.append(MatchAllNode)
            return tuple(modules)

        # since relu can be used multiple times, we'll need to create a relu module for each match
        matched_modules = get_modules(matched_node_pattern, [])

        def get_matched_types(m):
            if isinstance(m, tuple):
                return tuple(map(get_matched_types, m))
            return type(m)

        matched_module_types = get_matched_types(matched_modules)
        module_parent_name, module_name = _parent_name(root_node.target)
        fuser_method = get_fuser_method_new(matched_module_types, fuser_method_mapping)
        # TODO: change the signature for fuser_method to take matched module patterns
        # as input
        fused_module = fuser_method(is_qat, *matched_modules)
        # TODO: maybe add a pass to cleanup bn modules?
        setattr(quantizer.modules[module_parent_name], module_name, fused_module)
        return quantizer.fused_graph.node_copy(root_node, load_arg)
