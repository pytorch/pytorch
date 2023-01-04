import copy
import itertools
import operator
from typing import Optional

import numpy

import torch
import torch.nn as nn
from torch._dynamo.utils import fake_mode_from_tensors
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class UnaryAttr(object):
    def __init__(self, op_name: str, scalars_attr=None, algorithm_attr=None):
        self.op_name = op_name
        self.scalars_attr = scalars_attr if scalars_attr else []
        self.algorithm_attr = algorithm_attr if algorithm_attr else ""
        super(UnaryAttr, self).__init__()

    def __call__(self, unary_module: nn.Module):
        if type(unary_module) is nn.ReLU6:
            unary_module = nn.Hardtanh(min_val=0, max_val=6)
        assert all(hasattr(unary_module, item) for item in self.scalars_attr)
        scalars = [getattr(unary_module, item) for item in self.scalars_attr]

        algorithm = ""
        if self.algorithm_attr:
            assert hasattr(unary_module, self.algorithm_attr)
            algorithm = getattr(unary_module, self.algorithm_attr)

        return self.op_name, scalars, algorithm


def is_bfloat16_module(m):
    weight_is_bf16 = m.weight.dtype == torch.bfloat16
    bias_is_bf16 = m.bias is None or m.bias.dtype == torch.bfloat16
    return weight_is_bf16 and bias_is_bf16


def check_node_kind(current_node, modules, node_kind):
    if not isinstance(current_node, torch.fx.Node):
        return False
    if current_node.op != "call_module":
        return False
    if not isinstance(current_node.target, str):
        return False
    if current_node.target not in modules:
        return False
    if type(modules[current_node.target]) is not node_kind:
        return False
    return True


def check_node_is_binary(node):
    return (
        (node.op == "call_function" and node.target in [torch.add, torch.sub])
        or (
            node.op == "call_function"
            and node.target
            in [operator.add, operator.iadd, operator.sub, operator.isub]
        )
        or (node.op == "call_method" and node.target in ["add", "add_", "sub", "sub_"])
    )


def check_binary_op_kwargs_is_default(node):
    # For binary op, we hope the kwargs values are the default value:
    # torch.sub(add)(input, other, *, alpha=1, out=None).
    if len(node.args) > 2:
        return False
    if len(node.kwargs) > 0:
        if "out" in node.kwargs and node.kwargs["out"] is not None:
            return False
        if "alpha" in node.kwargs and node.kwargs["alpha"] != 1.0:
            return False
    return True


def check_node_is_add_inplace(node):
    return (node.op == "call_function" and node.target in [operator.iadd]) or (
        node.op == "call_method" and node.target in ["add_"]
    )


class ConvUnary2d(nn.Conv2d):
    def __init__(
        self,
        conv: nn.Module,
        unary: Optional[nn.Module],
        input_size: list,
    ):
        super(ConvUnary2d, self).__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            conv.weight.device,
            conv.weight.dtype,
        )
        self._update_module_params(conv, unary, input_size)

    def _update_module_params(self, conv, unary, input_size):
        self.__dict__ = copy.deepcopy(conv.__dict__)
        self.attr = "none"
        self.scalars = []
        self.algorithm = ""
        if unary is not None:
            self.attr, self.scalars, self.algorithm = unary_modules_map[
                unary.__class__
            ](unary)
        self.weight = torch.nn.Parameter(
            torch._C._nn.mkldnn_reorder_conv2d_weight(
                self.weight.to_mkldnn(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
                input_size,
            ),
            requires_grad=self.weight.requires_grad,
        )

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return torch.ops.mkldnn._convolution_pointwise(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                _pair(0),
                self.stride,
                self.dilation,
                self.groups,
                self.attr,
                self.scalars,
                self.algorithm,
            )
        return torch.ops.mkldnn._convolution_pointwise(
            input,
            weight,
            bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            self.attr,
            self.scalars,
            self.algorithm,
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)


class ConvBinary2d(nn.Conv2d):
    def __init__(
        self,
        conv: nn.Module,
        binary_op_name: str,
        input_size: list,
    ):
        super(ConvBinary2d, self).__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            conv.weight.device,
            conv.weight.dtype,
        )
        self._update_module_params(conv, binary_op_name, input_size)

    def _update_module_params(self, conv, binary_op_name, input_size):
        self.__dict__ = copy.deepcopy(conv.__dict__)
        self.binary_attr = binary_op_name
        self.binary_alpha = None
        self.unary_attr = None
        self.unary_scalars = []
        self.unary_algorithm = None
        self.weight = torch.nn.Parameter(
            torch._C._nn.mkldnn_reorder_conv2d_weight(
                self.weight.to_mkldnn(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
                input_size,
            ),
            requires_grad=self.weight.requires_grad,
        )

    def _update_unary_params(self, unary):
        self.unary_attr, self.unary_scalars, self.unary_algorithm = unary_modules_map[
            unary.__class__
        ](unary)

    def _conv_forward(self, input, other, weight, bias):
        if self.padding_mode != "zeros":
            return torch.ops.mkldnn._convolution_pointwise(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                other,
                weight,
                bias,
                _pair(0),
                self.stride,
                self.dilation,
                self.groups,
                self.binary_attr,
                self.binary_alpha,
                self.unary_attr,
                self.unary_scalars,
                self.unary_algorithm,
            )
        return torch.ops.mkldnn._convolution_pointwise(
            input,
            other,
            weight,
            bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            self.binary_attr,
            self.binary_alpha,
            self.unary_attr,
            self.unary_scalars,
            self.unary_algorithm,
        )

    def forward(self, input, other):
        return self._conv_forward(input, other, self.weight, self.bias)


class ConvBinaryInplace2d(nn.Conv2d):
    def __init__(
        self,
        conv: nn.Module,
        binary_op_name: str,
        input_size: list,
    ):
        super(ConvBinaryInplace2d, self).__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            conv.weight.device,
            conv.weight.dtype,
        )
        self._update_module_params(conv, binary_op_name, input_size)

    def _update_module_params(self, conv, binary_op_name, input_size):
        self.__dict__ = copy.deepcopy(conv.__dict__)
        self.binary_attr = binary_op_name
        self.binary_alpha = None
        self.unary_attr = None
        self.unary_scalars = []
        self.unary_algorithm = None
        self.weight = torch.nn.Parameter(
            torch._C._nn.mkldnn_reorder_conv2d_weight(
                self.weight.to_mkldnn(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
                input_size,
            ),
            requires_grad=self.weight.requires_grad,
        )

    def _update_unary_params(self, unary):
        self.unary_attr, self.unary_scalars, self.unary_algorithm = unary_modules_map[
            unary.__class__
        ](unary)

    def _conv_forward(self, input, other, weight, bias):
        if self.padding_mode != "zeros":
            return torch.ops.mkldnn._convolution_pointwise_(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                other,
                weight,
                bias,
                _pair(0),
                self.stride,
                self.dilation,
                self.groups,
                self.binary_attr,
                self.binary_alpha,
                self.unary_attr,
                self.unary_scalars,
                self.unary_algorithm,
            )
        return torch.ops.mkldnn._convolution_pointwise_(
            input,
            other,
            weight,
            bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            self.binary_attr,
            self.binary_alpha,
            self.unary_attr,
            self.unary_scalars,
            self.unary_algorithm,
        )

    def forward(self, input, other):
        return self._conv_forward(input, other, self.weight, self.bias)


class PackedLinear(nn.Linear):
    def __init__(self, linear: nn.Module, input_size: list):
        super(PackedLinear, self).__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype,
        )
        self._update_module_params(linear, input_size)

    def _update_module_params(self, linear, input_size):
        self.__dict__ = copy.deepcopy(linear.__dict__)
        self.batch_size = int(numpy.prod(input_size) / input_size[-1])
        self.packed_weight = torch.nn.Parameter(
            torch.ops.mkl._mkl_reorder_linear_weight(
                self.weight.to_mkldnn(), self.batch_size
            ),
            requires_grad=self.weight.requires_grad,
        )

    def forward(self, input):
        y = torch.ops.mkl._mkl_linear(
            input, self.packed_weight, self.weight, self.bias, self.batch_size
        )
        return y


class LinearUnary(nn.Linear):
    def __init__(
        self,
        linear: nn.Module,
        unary: nn.Module,
    ):
        super(LinearUnary, self).__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype,
        )
        self._update_module_params(linear, unary)

    def _update_module_params(self, linear, unary):
        self.__dict__ = copy.deepcopy(linear.__dict__)
        self.attr, self.scalars, self.algorithm = unary_modules_map[unary.__class__](
            unary
        )

    def forward(self, input):
        y = torch.ops.mkldnn._linear_pointwise(
            input, self.weight, self.bias, self.attr, self.scalars, self.algorithm
        )
        return y


class LinearBinary(nn.Linear):
    def __init__(self, linear: nn.Module, binary_op_name: str):
        super(LinearBinary, self).__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype,
        )
        self._update_module_params(linear, binary_op_name)

    def _update_module_params(self, linear, binary_op_name):
        self.__dict__ = copy.deepcopy(linear.__dict__)

        self.attr = binary_op_name

    def forward(self, input, other):
        y = torch.ops.mkldnn._linear_pointwise(
            input, other, self.weight, self.bias, self.attr
        )
        return y


def packed_conv_eval(conv: nn.Module, input_size: list):
    assert not (conv.training), "Fusion only for eval!"
    return ConvUnary2d(
        conv,
        None,
        input_size,
    )


def fused_conv_unary_eval(conv: nn.Module, unary: nn.Module, input_size: list):
    assert not (conv.training), "Fusion only for eval!"
    return ConvUnary2d(
        conv,
        unary,
        input_size,
    )


def fused_conv_binary_eval(conv: nn.Module, binary_op_name: str, input_size: list):
    assert not (conv.training), "Fusion only for eval!"
    return ConvBinary2d(
        conv,
        binary_op_name,
        input_size,
    )


def fused_conv_binary_inplace_eval(
    conv: nn.Module, binary_op_name: str, input_size: list
):
    assert not (conv.training), "Fusion only for eval!"
    return ConvBinaryInplace2d(
        conv,
        binary_op_name,
        input_size,
    )


def fused_conv_binary_unary_eval(
    conv_binary: nn.Module, unary: nn.Module, input_size: list
):
    assert not (conv_binary.training), "Fusion only for eval!"
    # reuse origin conv module, and just update its' unary attr.
    conv_binary._update_unary_params(unary)
    return conv_binary


def packed_linear_eval(linear: nn.Module, input_size: list):
    assert not (linear.training), "Fusion only for eval!"
    return PackedLinear(linear, input_size)


def fused_linear_unary_eval(linear: nn.Module, unary: nn.Module, input_size: list):
    assert not (linear.training), "Fusion only for eval!"
    return LinearUnary(
        linear,
        unary,
    )


def fused_linear_binary_eval(linear: nn.Module, attr: str, input_size: list):
    assert not (linear.training), "Fusion only for eval!"
    linear_binary = LinearBinary(
        linear,
        attr,
    )
    return linear_binary


def mkldnn_fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    is_cpu = all(
        example_input.device == torch.device("cpu") for example_input in example_inputs
    )

    # make sure the autograd is disabled.
    if torch.is_grad_enabled():
        return gm
    if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
        return gm
    if not is_cpu:
        return gm
    # For binary fusion, we need to check inputs info to make sure
    # the binary inputs have same tensor info(device, dtype, and layout).

    fake_mode = fake_mode_from_tensors(example_inputs)
    ShapeProp(gm, fake_mode=fake_mode).propagate(*example_inputs)
    gm = fuse_unary(gm)
    gm = fuse_binary_inplace(gm)
    gm = fuse_binary(gm)
    # why re-run fuse_unary? we want to enable conv+binary+unary fusion,
    # such as conv+add+relu for vision model.
    gm = fuse_unary(gm)
    gm = pack_module(gm)
    return gm


def fuse_unary(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())

    for (unary_module, _), (computation_module, fuse_func,) in itertools.product(
        unary_modules_map.items(), computation_op_unary_op_fusion_map.items()
    ):
        pattern = (computation_module, unary_module)
        for node in gm.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if (
                    len(node.args[0].users) > 1
                ):  # Output of computation_node is used by other nodes
                    continue
                computation_node = modules[node.args[0].target]
                unary_node = modules[node.target]
                eval_mode = all(not n.training for n in [computation_node, unary_node])
                if not eval_mode:
                    continue
                # TODO: support padding str input("valid", "same").
                if type(computation_node) in [nn.Conv2d] and isinstance(
                    computation_node.padding, str
                ):
                    continue
                # TODO: support more conv+binary+unary fusion.
                if type(computation_node) in [
                    ConvBinary2d,
                    ConvBinaryInplace2d,
                ] and type(unary_node) not in [nn.ReLU]:
                    continue
                # only fuse for linear when the dtype is bf16
                if type(computation_node) in [nn.Linear] and not is_bfloat16_module(
                    computation_node
                ):
                    continue
                computation_node_input_size = (
                    node.args[0].args[0].meta.get("tensor_meta").shape
                )
                fused_module = fuse_func(
                    computation_node, unary_node, computation_node_input_size
                )
                replace_node_module(node.args[0], modules, fused_module)

                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
                gm.graph.lint()
    gm.recompile()
    return gm


def replace_and_fuse_for_binary(
    computation_node, node, fuse_func, attr, modules, index_node, index_pointwise
):
    computation_node_input_size = (
        node.args[index_node].args[0].meta.get("tensor_meta").shape
    )
    fused_module = fuse_func(computation_node, attr, computation_node_input_size)
    replace_node_module(node.args[index_node], modules, fused_module)
    node.args[index_node].args = node.args[index_node].args + (
        node.args[index_pointwise],
    )
    node.replace_all_uses_with(node.args[index_node])


def binary_inputs_meta_is_same(binary_node):
    tensor0_meta = binary_node.args[0].meta.get("tensor_meta")
    tensor1_meta = binary_node.args[1].meta.get("tensor_meta")
    if not tensor0_meta or not tensor1_meta:
        return False
    if (
        tensor0_meta.shape != tensor1_meta.shape
        or tensor0_meta.stride != tensor1_meta.stride
        or tensor0_meta.dtype != tensor1_meta.dtype
    ):
        return False

    return True


def fuse_binary(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if check_node_is_binary(node) and check_binary_op_kwargs_is_default(node):
            for node_kind, fuse_func in computation_op_binary_op_fusion_map.items():
                if not isinstance(node.args[0], torch.fx.Node) or not isinstance(
                    node.args[1], torch.fx.Node
                ):
                    continue
                if not binary_inputs_meta_is_same(node):
                    continue
                attr = binary_attr[node.target]
                index_list = supported_index_list[attr]
                for index_dict in index_list:
                    index_node = index_dict["index_computation"]
                    index_pointwise = index_dict["index_pointwise"]
                    if check_node_kind(node.args[index_node], modules, node_kind):
                        if len(node.args[index_node].users) > 1:
                            continue
                        computation_node = modules[node.args[index_node].target]
                        # TODO: support padding str input("valid", "same").
                        if type(computation_node) in [nn.Conv2d] and isinstance(
                            computation_node.padding, str
                        ):
                            continue
                        # only fuse for linear when the dtype is bf16
                        if type(computation_node) in [
                            nn.Linear
                        ] and not is_bfloat16_module(computation_node):
                            continue
                        replace_and_fuse_for_binary(
                            computation_node,
                            node,
                            fuse_func,
                            attr if attr != "iadd" else "add",
                            modules,
                            index_node,
                            index_pointwise,
                        )
                        # Make sure the fused node is post node of node's inputs nodes.
                        node.append(node.args[index_node])
                        gm.graph.erase_node(node)
                        gm.graph.lint()
                        break

    gm.recompile()
    return gm


def fuse_binary_inplace(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if check_node_is_add_inplace(node) and check_binary_op_kwargs_is_default(node):
            for (
                node_kind,
                fuse_func,
            ) in computation_op_binary_op_fusion_inplace_map.items():
                if not isinstance(node.args[0], torch.fx.Node) or not isinstance(
                    node.args[1], torch.fx.Node
                ):
                    continue
                if not binary_inputs_meta_is_same(node):
                    continue
                if check_node_kind(node.args[1], modules, node_kind):
                    if len(node.args[1].users) > 1:
                        continue
                    # make sure the output and input are not same tensor.
                    if node.args[1].args[0] == node.args[0]:
                        continue
                    computation_node = modules[node.args[1].target]
                    # TODO: support padding str input("valid", "same").
                    if type(computation_node) in [nn.Conv2d] and isinstance(
                        computation_node.padding, str
                    ):
                        continue
                    replace_and_fuse_for_binary(
                        computation_node,
                        node,
                        fuse_func,
                        "add",
                        modules,
                        1,  # conv module index
                        0,  # binary op index
                    )
                    # Make sure the fused node is post node of node's inputs nodes.
                    node.append(node.args[1])
                    gm.graph.erase_node(node)
                    gm.graph.lint()
                    break

    gm.recompile()
    return gm


def pack_module(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module":
            assert isinstance(node.target, str)
            cur_module = modules[node.target]
            if type(cur_module) in computation_op_packed_map:
                computation_node_input_meta = node.args[0].meta.get("tensor_meta")
                if computation_node_input_meta.dtype != torch.float32:
                    continue
                if type(cur_module) in [torch.nn.Linear] and not torch._C.has_mkl:
                    continue
                computation_node_input_size = computation_node_input_meta.shape
                if type(cur_module) in [nn.Conv2d] and isinstance(
                    cur_module.padding, str
                ):
                    continue
                new_module = computation_op_packed_map[type(cur_module)](
                    cur_module, computation_node_input_size
                )
                assert isinstance(new_module, nn.Module)
                replace_node_module(node, modules, new_module)
                gm.graph.lint()
    gm.recompile()
    return gm


computation_op_unary_op_fusion_map = {
    nn.Conv2d: fused_conv_unary_eval,
    nn.Linear: fused_linear_unary_eval,
    ConvBinary2d: fused_conv_binary_unary_eval,
    ConvBinaryInplace2d: fused_conv_binary_unary_eval,
}


unary_modules_map = {
    nn.ReLU: UnaryAttr("relu"),
    nn.Sigmoid: UnaryAttr("sigmoid"),
    nn.Tanh: UnaryAttr("tanh"),
    nn.Hardswish: UnaryAttr("hardswish"),
    nn.LeakyReLU: UnaryAttr("leaky_relu", scalars_attr=["negative_slope"]),
    nn.Hardtanh: UnaryAttr("hardtanh", scalars_attr=["min_val", "max_val"]),
    nn.GELU: UnaryAttr("gelu", algorithm_attr="approximate"),
    nn.ReLU6: UnaryAttr("hardtanh", scalars_attr=["min_val", "max_val"]),
    nn.SiLU: UnaryAttr("swish"),
}


binary_attr = {
    torch.add: "add",  # node.op == "call_function"
    "add": "add",  # node.op == "call_method"
    "add_": "iadd",  # node.op == "call_method"
    operator.add: "add",  # node.op == "call_function"
    operator.iadd: "iadd",  # node.op == "call_function"
    torch.sub: "sub",  # node.op == "call_function"
    "sub": "sub",  # node.op == "call_method"
    "sub_": "sub",  # node.op == "call_method"
    operator.sub: "sub",  # node.op == "call_function"
    operator.isub: "sub",  # node.op == "call_function"
}


computation_op_binary_op_fusion_map = {
    nn.Conv2d: fused_conv_binary_eval,
    nn.Linear: fused_linear_binary_eval,
}


computation_op_binary_op_fusion_inplace_map = {
    nn.Conv2d: fused_conv_binary_inplace_eval,
}


computation_op_packed_map = {
    nn.Linear: packed_linear_eval,
    nn.Conv2d: packed_conv_eval,
}


# For add: we support conv/linear + other and other + conv
# For sub/add_/sub_, we only support conv/linear - other
# or conv/linear +(-)= other
supported_index_list = {
    "add": [
        {"index_computation": 0, "index_pointwise": 1},
        {"index_computation": 1, "index_pointwise": 0},
    ],
    "iadd": [{"index_computation": 0, "index_pointwise": 1}],
    "sub": [{"index_computation": 0, "index_pointwise": 1}],
}
