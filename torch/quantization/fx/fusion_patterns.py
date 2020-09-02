import torch
from .pattern_utils import (
    register_fusion_pattern,
)
from .utils import _parent_name
from ..fuser_method_mappings import get_fuser_method

# ---------------------
# Fusion Patterns
# ---------------------

@register_fusion_pattern((torch.nn.BatchNorm2d, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
@register_fusion_pattern((torch.nn.functional.relu, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
class ConvBNReLUFusion():
    def __init__(self, quantizer, node):
        super().__init__()
        self.relu_node = None
        self.bn_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and type(quantizer.modules[node.target]) == torch.nn.ReLU):
            self.relu_node = node
            node = node.args[0]
        assert node.op == 'call_module'
        if type(quantizer.modules[node.target]) in [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            node = node.args[0]
        assert node.op == 'call_module'
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]

    def fuse(self, quantizer, load_arg):
        op_list = []
        if self.relu_node is not None:
            # since relu can be used multiple times, we'll need to create a relu module for each match
            if self.relu_node.op == 'call_module':
                relu = torch.nn.ReLU(quantizer.modules[self.relu_node.target].inplace)
            else:
                # TODO: get inplace argument from functional
                relu = torch.nn.ReLU()
            op_list.append(relu)
            relu.training = self.conv.training
            if self.bn_node is not None:
                op_list.append(self.bn)
            op_list.append(self.conv)
        else:
            assert self.bn_node is not None
            op_list.append(self.bn)
            op_list.append(self.conv)

        # the modules are added in order of relu - bn - conv
        # so we need to correct it
        op_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        conv_parent_name, conv_name = _parent_name(self.conv_node.target)
        fuser_method = get_fuser_method(op_type_list)
        if fuser_method is None:
            raise NotImplementedError("Cannot fuse modules: {}".format(types))
        setattr(quantizer.modules[conv_parent_name], conv_name, fuser_method(*op_list))

        # TODO: do we need to make sure bn is only used once?
        if self.bn_node is not None:
            parent_name, name = _parent_name(self.bn_node.target)
            setattr(quantizer.modules[parent_name], name, torch.nn.Identity())
        # relu may be used multiple times, so we don't set relu to identity
        return quantizer.fused_graph.node_copy(self.conv_node, load_arg)

@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Linear))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Linear))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.BatchNorm1d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.BatchNorm1d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.BatchNorm2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.BatchNorm2d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.BatchNorm3d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.BatchNorm3d))
class ModuleReLUFusion():
    def __init__(self, quantizer, node):
        super().__init__()
        self.relu_node = node
        node = node.args[0]
        assert node.op == 'call_module'
        self.module_node = node
        self.module = quantizer.modules[self.module_node.target]

    def fuse(self, quantizer, load_arg):
        op_list = []
        # since relu can be used multiple times, we'll need to create a relu module for each match
        if self.relu_node.op == 'call_module':
            relu = torch.nn.ReLU(quantizer.modules[self.relu_node.target].inplace)
        else:
            # TODO: get inplace argument from functional
            relu = torch.nn.ReLU()
        relu.training = self.module.training
        op_list.append(relu)
        op_list.append(self.module)

        op_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        module_parent_name, module_name = _parent_name(self.module_node.target)
        fuser_method = OP_LIST_TO_FUSER_METHOD.get(op_type_list, None)
        if fuser_method is None:
            raise NotImplementedError("Cannot fuse modules: {}".format(types))
        setattr(quantizer.modules[module_parent_name], module_name, fuser_method(*op_list))
        return quantizer.fused_graph.node_copy(self.module_node, load_arg)
