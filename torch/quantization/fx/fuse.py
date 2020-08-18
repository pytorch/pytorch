import torch
from torch.quantization.fuse_modules import (
    fuse_conv_bn,
    fuse_conv_bn_relu,
)

from torch.fx import (
    GraphModule,
)

from torch.fx.graph import (
    Graph,
    map_arg,
)

from .pattern_utils import (
    matches,
    register_fusion_pattern,
    get_fusion_patterns,
)

from .utils import _parent_name

import copy

# Fusion Patterns
@register_fusion_pattern((torch.nn.BatchNorm2d, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv2d))
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
        if isinstance(quantizer.modules[node.target], torch.nn.BatchNorm2d):
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            node = node.args[0]
        assert node.op == 'call_module' and type(quantizer.modules[node.target]) == torch.nn.modules.Conv2d
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]

    def fuse(self, quantizer, load_arg):
        conv_parent_name, conv_name = _parent_name(self.conv_node.target)
        if self.relu_node is not None:
            # since relu can be used multiple times, we'll need to create a relu module for each match
            if self.relu_node.op == 'call_module':
                relu = torch.nn.ReLU(quantizer.modules[self.relu_node.target].inplace)
            else:
                # TODO: get inplace argument from functional
                relu = torch.nn.ReLU()
            relu.training = self.conv.training
            if self.bn_node is not None:
                setattr(quantizer.modules[conv_parent_name], conv_name, fuse_conv_bn_relu(self.conv, self.bn, relu))
            else:
                # conv_relu
                setattr(quantizer.modules[conv_parent_name], conv_name, torch.nn.intrinsic.ConvReLU2d(self.conv, relu))
        else:
            assert self.bn_node is not None
            setattr(quantizer.modules[conv_parent_name], conv_name, fuse_conv_bn(self.conv, self.bn))

        # TODO: do we need to make sure bn is only used once?
        if self.bn_node is not None:
            parent_name, name = _parent_name(self.bn_node.target)
            setattr(quantizer.modules[parent_name], name, torch.nn.Identity())
        return quantizer.fused_graph.node_copy(self.conv_node, load_arg)

@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Linear))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Linear))
class LinearReLUFusion():
    def __init__(self, quantizer, node):
        super().__init__()
        self.relu_node = node
        node = node.args[0]
        assert node.op == 'call_module'
        assert isinstance(quantizer.modules[node.target], torch.nn.modules.Linear)
        self.linear_node = node
        self.linear = quantizer.modules[self.linear_node.target]

    def fuse(self, quantizer, load_arg):
        # since relu can be used multiple times, we'll need to create a relu module for each match
        if self.relu_node.op == 'call_module':
            relu = torch.nn.ReLU(quantizer.modules[self.relu_node.target].inplace)
        else:
            # TODO: get inplace argument from functional
            relu = torch.nn.ReLU()
        relu.training = self.linear.training
        # linear_relu
        linear_parent_name, linear_name = _parent_name(self.linear_node.target)
        setattr(quantizer.modules[linear_parent_name], linear_name, torch.nn.intrinsic.LinearReLU(self.linear, relu))
        return quantizer.fused_graph.node_copy(self.linear_node, load_arg)

class Fuser:
    def fuse_conv_bn(self, model, inplace=False):
        input_root = model.root
        if not inplace:
            input_root = copy.deepcopy(input_root)
        input_graph = model.graph
        self.modules = dict(input_root.named_modules())

        fusion_patterns = get_fusion_patterns()
        # find conv-bn pairs
        conv_bn_pairs = self._find_matches(input_root, input_graph, fusion_patterns)
        self.fused_graph = Graph()
        env = {}

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])

        for node in input_graph.nodes:
            root_node, obj = conv_bn_pairs.get(node.name, (None, None))
            if root_node is node:
                env[node.name] = obj.fuse(self, load_arg)
            elif root_node is None:
                env[node.name] = self.fused_graph.node_copy(node, load_arg)
            # node matched in patterns and is not root is removed here

        self.fused_graph.output(load_arg(input_graph.result))
        return GraphModule(input_root, self.fused_graph)

    def _find_matches(self, root, graph, patterns):
        modules = dict(root.named_modules())
        match_map = {}  # node name -> (root_node, match_value?)

        def apply_match(pattern, node, match):
            if isinstance(pattern, tuple):
                s, *args = pattern
                apply_match(s, node, match)
                for subpattern, arg in zip(args, node.args):
                    apply_match(subpattern, arg, match)
            else:
                match_map[node.name] = match

        for node in reversed(graph.nodes):
            if node.name not in match_map:
                for pattern, value in patterns.items():
                    if matches(modules, node, pattern):
                        apply_match(pattern, node, (node, value(self, node)))

        return match_map

def fuse(graph_module, inplace=False):
    fuser = Fuser()
    return fuser.fuse_conv_bn(graph_module, inplace)
