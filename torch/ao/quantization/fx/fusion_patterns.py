import torch
from torch.fx.graph import Node
from .pattern_utils import (
    register_fusion_pattern,
)
from .utils import _parent_name
from .quantization_types import QuantizerCls, NodePattern
from ..fuser_method_mappings import get_fuser_method
from ..fuser_method_mappings import get_fuser_method_new
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict
from .match_utils import MatchAllNode

# ---------------------
# Fusion Pattern Registrations
# ---------------------

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
             fuse_custom_config_dict: Dict[str, Any]) -> Node:
        pass

@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.BatchNorm1d, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.BatchNorm2d, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.BatchNorm3d, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm1d, torch.nn.Conv1d)))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm3d, torch.nn.Conv3d)))
@register_fusion_pattern((torch.nn.functional.relu, (torch.nn.BatchNorm1d, torch.nn.Conv1d)))
@register_fusion_pattern((torch.nn.functional.relu, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
@register_fusion_pattern((torch.nn.functional.relu, (torch.nn.BatchNorm3d, torch.nn.Conv3d)))
@register_fusion_pattern((torch.nn.BatchNorm1d, torch.nn.Linear))
class ConvOrLinearBNReLUFusion(FuseHandler):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node)
        self.relu_node = None
        self.bn_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and type(quantizer.modules[node.target]) == torch.nn.ReLU):
            self.relu_node = node
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == 'call_module'
        if type(quantizer.modules[node.target]) in [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == 'call_module'
        self.conv_or_linear_node = node
        self.conv_or_linear = quantizer.modules[self.conv_or_linear_node.target]

    def fuse(self,
             quantizer: QuantizerCls,
             load_arg: Callable,
             root_node: Node,
             matched_node_pattern: NodePattern,
             fuse_custom_config_dict: Dict[str, Any]) -> Node:
        additional_fuser_method_mapping = fuse_custom_config_dict.get("additional_fuser_method_mapping", {})
        op_list = []
        if self.relu_node is not None:
            # since relu can be used multiple times, we'll need to create a relu module for each match
            if self.relu_node.op == 'call_module':
                relu = torch.nn.ReLU(quantizer.modules[self.relu_node.target].inplace)
            else:
                # TODO: get inplace argument from functional
                relu = torch.nn.ReLU()
            op_list.append(relu)
            relu.training = self.conv_or_linear.training
            if self.bn_node is not None:
                op_list.append(self.bn)
            op_list.append(self.conv_or_linear)
        else:
            assert self.bn_node is not None
            op_list.append(self.bn)
            op_list.append(self.conv_or_linear)

        # the modules are added in order of relu - bn - conv_or_linear
        # so we need to correct it
        op_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        conv_or_linear_parent_name, conv_or_linear_name = _parent_name(self.conv_or_linear_node.target)
        fuser_method = get_fuser_method(op_type_list, additional_fuser_method_mapping)
        if fuser_method is None:
            raise NotImplementedError("Cannot fuse modules: {}".format(op_type_list))
        fused = fuser_method(*op_list)
        setattr(quantizer.modules[conv_or_linear_parent_name], conv_or_linear_name, fused)

        # TODO: do we need to make sure bn is only used once?
        if self.bn_node is not None:
            parent_name, name = _parent_name(self.bn_node.target)
            setattr(quantizer.modules[parent_name], name, torch.nn.Identity())
        # relu may be used multiple times, so we don't set relu to identity
        return quantizer.fused_graph.node_copy(self.conv_or_linear_node, load_arg)

@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Linear))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Linear))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.BatchNorm2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.BatchNorm2d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.BatchNorm3d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.BatchNorm3d))
class ModuleReLUFusion(FuseHandler):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node)
        self.relu_node = node
        assert isinstance(node.args[0], Node)
        node = node.args[0]
        assert node.op == 'call_module'
        self.module_node = node
        self.module = quantizer.modules[self.module_node.target]

    def fuse(self, quantizer: QuantizerCls,
             load_arg: Callable,
             root_node: Node,
             matched_node_pattern: NodePattern,
             fuse_custom_config_dict: Dict[str, Any]) -> Node:
        additional_fuser_method_mapping = fuse_custom_config_dict.get("additional_fuser_method_mapping", {})
        assert len(additional_fuser_method_mapping) == 0, "Fusion implementation is "
        "undergoing changes, additoinal_fuser_method_mapping is not supported currently."
        def get_module(n):
            if n.op == "call_module":
                return quantizer.modules[n.target]
            elif n.op == "call_function" and n.target == torch.nn.functional.relu:
                relu = torch.nn.ReLU()
                relu.training = self.module.training
                return relu
            return MatchAllNode

        print("matched node pattern", matched_node_pattern)
        matched_modules = tuple(map(get_module, matched_node_pattern))
        # since relu can be used multiple times, we'll need to create a relu module for each match

        def get_type(m):
            return type(m)

        matched_module_types = tuple(map(get_type, matched_modules))
        module_parent_name, module_name = _parent_name(self.module_node.target)
        print("module parent name, module name", module_parent_name, module_name)
        print("matched module types:", matched_module_types)
        fuser_method = get_fuser_method_new(matched_module_types)
        print("fuser method:", fuser_method)
        # TODO: change the signature for fuser_method to take matched module patterns
        # as input
        fused_module = fuser_method(*matched_modules)
        print(fused_module)
        setattr(quantizer.modules[module_parent_name], module_name, fused_module)
        return quantizer.fused_graph.node_copy(self.module_node, load_arg)
