r"""
**This file is EXPERIMENTAL and is mostly used for testing purposes! Do not
rely on it for anything!**
"""
import operator
import sys

import torch
from torch.fx import Graph, GraphModule
from torch.fx.graph import map_arg
from torch.fx.proxy import Proxy
from torch.nn.utils import fuse_conv_bn_weights


# can be a
#  module type, a builtin function, or a string to match target


def _minmax_scale_zeropoint(
    min_val, max_val, qmin=-127, qmax=128, eps=torch.finfo(torch.float32).eps
):
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)
    if max_val == min_val:
        return 1.0, 0
    else:
        scale = (max_val - min_val) / float(qmax - qmin)
        scale = max(scale, eps)
        zero_point = qmin - round(min_val / scale)
        zero_point = max(qmin, zero_point)
        zero_point = min(qmax, zero_point)
        zero_point = int(zero_point)
        return scale, zero_point


class MinMaxObserver:
    def __init__(self, quantizer, node):
        self.min, self.max = float("inf"), float("-inf")
        self.all_tensors = True

    def observe(self, node, env):
        v = env[node.name]
        if not isinstance(v, torch.Tensor):
            self.all_tensors = False
            return
        self.max = max(self.max, float(v.max()))
        self.min = min(self.min, float(v.min()))

    def scale_zeropoint(self):
        return _minmax_scale_zeropoint(self.min, self.max, qmin=0, qmax=255)


class NoObserver:
    def __init__(self, quantizer, node):
        pass

    def observe(self, node, env):
        pass


_DEFAULT_QUANTIZATION_PATTERNS = {}


def register_pattern(pattern):
    def insert(fn):
        _DEFAULT_QUANTIZATION_PATTERNS[pattern] = fn
        return fn

    return insert


@register_pattern(operator.add)
class Add(MinMaxObserver):
    def quantize(self, quantizer, node, load_arg):
        if not self.all_tensors:
            return NotImplemented
        scale, zeropoint = self.scale_zeropoint()
        return quantizer.quantized_graph.create_node(
            "call_function",
            torch.ops.quantized.add,
            load_arg(node.args),
            {"scale": scale, "zero_point": zeropoint},
        )


class Relu(NoObserver):
    def quantize(self, quantizer, node, load_arg):
        return torch.relu(
            load_arg(node.args[0])
        )  # torch.relu works directly on quantized tensors?


# these ops have quantized equivalents that do not need any extra information
@register_pattern(torch.nn.ReLU)
@register_pattern(torch.nn.AvgPool2d)
@register_pattern(torch.nn.MaxPool2d)
@register_pattern(torch.nn.AdaptiveAvgPool2d)
class CopyNode(NoObserver):
    def quantize(self, quantizer, node, load_arg):
        return quantizer.quantized_graph.node_copy(node, load_arg)


class IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x


# handle conv, maybe followed by bn, maybe followed by relu
@register_pattern(torch.nn.modules.conv.Conv2d)
@register_pattern((torch.nn.ReLU, torch.nn.modules.conv.Conv2d))
@register_pattern(
    (torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.conv.Conv2d)
)
@register_pattern(
    (
        torch.nn.ReLU,
        (torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.conv.Conv2d),
    )
)
class ConvNormRelu(MinMaxObserver):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        self.relu_node, self.bn_node = None, None
        if isinstance(quantizer.modules[node.target], torch.nn.ReLU):
            self.relu_node = node
            node = node.args[0]
        if isinstance(quantizer.modules[node.target], torch.nn.BatchNorm2d):
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            node = node.args[0]
        assert isinstance(quantizer.modules[node.target], torch.nn.modules.Conv2d)
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]

    def quantize(self, quantizer, node, load_arg):
        mod = self.conv
        weight, bias = mod.weight, mod.bias

        if self.bn_node is not None:
            weight, bias = fuse_conv_bn_weights(
                weight,
                bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias,
            )

        min_val, max_val = float(weight.min()), float(weight.max())

        act_scale, act_zp = self.scale_zeropoint()

        weight_scale, weight_zp = _minmax_scale_zeropoint(min_val, max_val)
        qweight = torch.quantize_per_tensor(
            weight, weight_scale, weight_zp, torch.qint8
        )

        ctor = (
            torch.ao.nn.intrinsic.quantized.ConvReLU2d
            if self.relu_node is not None
            else torch.ao.nn.quantized.Conv2d
        )

        qconv = ctor(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            mod.padding_mode,
        )

        qconv.set_weight_bias(qweight, bias)
        qconv.scale = float(act_scale)
        qconv.zero_point = int(act_zp)
        parent_name, name = _parent_name(self.conv_node.target)
        setattr(quantizer.modules[parent_name], name, qconv)
        if self.bn_node is not None:
            _, bn_name = _parent_name(self.bn_node.target)
            # we can't just delete this because submodules's forwards (which are not longer use)
            # try to call it, so replace with something that does nothing.
            setattr(quantizer.modules[parent_name], bn_name, IdentityModule())

        return quantizer.quantized_graph.create_node(
            "call_module",
            self.conv_node.target,
            (load_arg(self.conv_node.args[0]),),
            {},
        )


# turn foo.bar -> ['foo', 'bar']
def _parent_name(target):
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]


class DefaultQuant(MinMaxObserver):
    def quantize(self, input):
        assert self.all_tensors
        scale, zeropoint = self.scale_zeropoint()
        return torch.quantize_per_tensor(
            Proxy(input), scale, zeropoint, torch.quint8
        ).node


def matches(modules, node, pattern, max_uses=sys.maxsize):
    if isinstance(pattern, tuple):
        self_match, *arg_matches = pattern
    else:
        self_match = pattern
        arg_matches = None

    if len(node.users) > max_uses:
        return False

    if isinstance(self_match, type) and issubclass(self_match, torch.nn.Module):
        if node.op != "call_module":
            return False
        if not isinstance(modules[node.target], self_match):
            return False
    elif callable(self_match):
        if node.op != "call_function" or node.target is not self_match:
            return False
    elif node.target != self_match:
        return False

    if not arg_matches:
        return True

    if len(arg_matches) != len(node.args):
        return False

    return all(
        matches(modules, node, arg_match, max_uses=1)
        for node, arg_match in zip(node.args, arg_matches)
    )


class Quantizer:
    def __init__(
        self, mod, patterns=_DEFAULT_QUANTIZATION_PATTERNS, quant_ctor=DefaultQuant
    ):
        self.root = mod
        self.graph = mod.graph
        self.quant_ctor = quant_ctor

        # cached information for observe
        self.state_dict = self.root.state_dict()
        self.modules = dict(self.root.named_modules())

        # match the patterns that will get quantized
        self.matches = self._find_matches(patterns)
        # find _inputs_ to matched nodes that are not quantized, these
        # have to be quantized, which requires measuring stats,
        # initialize an quant_ctor object for each
        self.quants = self._find_quants(quant_ctor)

    def observe(self, args):
        # most of this function is just an interpreter for the graph
        # it would be possible to put this in some abstraction, but
        # it is pretty nice to just be able to see exactly what is happening here
        # and hack on it.
        # maybe we should just provide an example interpreter that people copy/paste
        # then edit.
        args_iter = iter(args)
        env = {}

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])

        for node in self.graph.nodes:
            if node.op == "placeholder":
                result = next(args_iter)
            elif node.op == "get_attr":
                result = self.state_dict[node.target]
            elif node.op == "call_function":
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == "call_method":
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == "call_module":
                result = self.modules[node.target](
                    *load_arg(node.args), **load_arg(node.kwargs)
                )
            elif node.op == "output":
                return load_arg(node.args[0])

            env[node.name] = result
            root_node, obj = self.matches.get(node.name, (None, None))
            if root_node is node:
                obj.observe(node, env)
            if node.name in self.quants:
                self.quants[node.name].observe(node, env)

        raise RuntimeError("Graph had no output node!")

    def quantize(self):
        self.quantized_graph = Graph()

        env = {}
        quant_env = {}

        def load_arg(n, quantized):
            if not quantized:
                if n.name not in env and n.name in quant_env:
                    env[n.name] = Proxy(quant_env[n.name]).dequantize().node
                return env[n.name]
            else:
                if n.name not in quant_env and n.name in env:
                    quant_env[n.name] = self.quants[n.name].quantize(env[n.name])
                return quant_env[n.name]

        def copy_recursive(node):
            r = env[node.name] = self.quantized_graph.node_copy(
                node, lambda n: load_arg(n, quantized=False)
            )
            return r

        for node in self.graph.nodes:
            root_node, obj = self.matches.get(node.name, (None, None))
            if root_node is None:
                # not quantized just copy it
                env[node.name] = self.quantized_graph.node_copy(
                    node, lambda n: load_arg(n, quantized=False)
                )

            elif root_node is node:
                r = obj.quantize(
                    self,
                    node,
                    lambda a: map_arg(a, lambda n: load_arg(n, quantized=True)),
                )
                if r is NotImplemented:
                    # quantizer choose to to quantize the node take the entire match, and just copy it over
                    env[node.name] = copy_recursive(node)
                else:
                    quant_env[node.name] = r

        return GraphModule(self.root, self.quantized_graph)

    def _find_matches(self, patterns):
        modules = dict(self.root.named_modules())
        match_map = {}  # node name -> (root_node, match_value?)

        def apply_match(pattern, node, match):
            if isinstance(pattern, tuple):
                s, *args = pattern
                apply_match(s, node, match)
                for subpattern, arg in zip(args, node.args):
                    apply_match(subpattern, arg, match)
            else:
                match_map[node.name] = match

        for node in reversed(self.graph.nodes):
            if node.name not in match_map:
                for pattern, value in patterns.items():
                    if matches(modules, node, pattern):
                        apply_match(pattern, node, (node, value(self, node)))

        return match_map

    def _find_quants(self, quant_ctor):
        quants = {}

        def visit_arg(n):
            # note: we have to measure quantization information
            # even for nodes where we might not use it because it is already
            # quantized. This is because each match has the option to
            # say NotImplemented (if for instance, it is an __add__ and the data type is not appropriate)
            if n.name not in quants:
                quants[n.name] = quant_ctor(self, n)

        for node in self.graph.nodes:
            if node.name in self.matches:
                map_arg(node.args, visit_arg)
                map_arg(node.kwargs, visit_arg)
        return quants
