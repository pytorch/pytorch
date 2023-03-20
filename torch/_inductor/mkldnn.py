import copy
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch._dynamo.utils import fake_mode_from_tensors
from torch.fx.experimental.optimization import replace_node_module
from torch.fx.experimental.symbolic_shapes import guard_int
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn.modules.utils import _pair
from . import config


def is_group_depthwise_conv_transpose(m):
    return (
        type(m) in [nn.ConvTranspose2d] and m.groups > 1 and m.groups == m.in_channels
    )


class PackedConv2d(nn.Conv2d):
    def __init__(
        self,
        conv: nn.Module,
        input_size: list,
    ):
        super().__init__(
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
        self._update_module_params(conv, input_size)

    def _update_module_params(self, conv, input_size):
        self.__dict__ = copy.deepcopy(conv.__dict__)
        self.weight = torch.nn.Parameter(
            torch._C._nn.mkldnn_reorder_conv2d_weight(
                self.weight.to_mkldnn(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
                tuple(guard_int(x) for x in input_size),
            ),
            requires_grad=self.weight.requires_grad,
        )

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return torch.ops.mkldnn._convolution(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                _pair(0),
                self.stride,
                self.dilation,
                self.groups,
            )
        return torch.ops.mkldnn._convolution(
            input,
            weight,
            bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)


class PackedLinear(nn.Linear):
    def __init__(self, linear: nn.Module, input_size: list):
        super().__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype,
        )
        self._update_module_params(linear, input_size)

    def _update_module_params(self, linear, input_size):
        self.__dict__ = copy.deepcopy(linear.__dict__)
        self.batch_size = reduce(lambda x, y: x * y, input_size[:-1])
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


class PackedConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        conv_transpose: nn.Module,
        input_size: list,
    ):
        super().__init__(
            conv_transpose.in_channels,
            conv_transpose.out_channels,
            conv_transpose.kernel_size,
            conv_transpose.stride,
            conv_transpose.padding,
            conv_transpose.output_padding,
            conv_transpose.groups,
            conv_transpose.bias is not None,
            conv_transpose.dilation,
            conv_transpose.padding_mode,
            conv_transpose.weight.device,
            conv_transpose.weight.dtype,
        )
        self._update_module_params(conv_transpose, input_size)

    def _update_module_params(self, conv_transpose, input_size):
        self.__dict__ = copy.deepcopy(conv_transpose.__dict__)
        packed_weight = torch.ops.mkldnn._reorder_convolution_transpose_weight(
            self.weight.to_mkldnn(),
            self.padding,
            self.output_padding,
            self.stride,
            self.dilation,
            self.groups,
            input_size,
        )
        self.weight = torch.nn.Parameter(
            packed_weight,
            requires_grad=self.weight.requires_grad,
        )

    def _conv_transpose_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return torch.ops.mkldnn._convolution_transpose(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                _pair(0),
                self.output_padding,
                self.stride,
                self.dilation,
                self.groups,
            )
        return torch.ops.mkldnn._convolution_transpose(
            input,
            weight,
            bias,
            self.padding,
            self.output_padding,
            self.stride,
            self.dilation,
            self.groups,
        )

    def forward(self, input):
        return self._conv_transpose_forward(input, self.weight, self.bias)


def packed_conv_eval(conv: nn.Module, input_size: list):
    assert not (conv.training), "Fusion only for eval!"
    return PackedConv2d(
        conv,
        input_size,
    )


def packed_conv_transpose_eval(conv_transpose: nn.Module, input_size: list):
    assert not (conv_transpose.training), "Fusion only for eval!"
    return PackedConvTranspose2d(conv_transpose, input_size)


def packed_linear_eval(linear: nn.Module, input_size: list):
    assert not (linear.training), "Fusion only for eval!"
    return PackedLinear(linear, input_size)


def mkldnn_fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    is_cpu = all(
        example_input.device == torch.device("cpu")
        for example_input in example_inputs
        if isinstance(example_input, torch.Tensor)
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
    if config.cpp.weight_prepack:
        gm = pack_module(gm)
    return gm


def convert_outplace_to_inplace(gm: torch.fx.GraphModule):
    if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
        return gm
    # This function is about replace outplace with inplace for better performance(external call),
    # which happen after AOTAutograd.
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in [
            torch.ops.mkldnn._convolution_pointwise.binary
        ]:
            # args[0] and args[1] is _convolution_pointwise.binary's input,
            # need to check whether args[1] can be written or not.
            if node.args[1].op in ["placeholder", "output"]:
                continue
            # TODO: node.args[1].users > 1, but node.args[1] never be used after current node.
            if len(node.args[1].users) > 1:
                continue
            if node.args[1] == node.args[0]:
                continue
            binary_attr = node.args[8]
            unary_attr = node.args[10]
            if binary_attr != "add" or unary_attr not in ["", "relu"]:
                continue
            node.target = torch.ops.mkldnn._convolution_pointwise_.binary
    gm.graph.lint()
    gm.recompile()
    return gm


def pack_module(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module":
            assert isinstance(node.target, str)
            cur_module = modules[node.target]
            if type(cur_module) in computation_op_packed_map:
                if cur_module.training:
                    continue
                computation_node_input_meta = node.args[0].meta.get("tensor_meta")
                if computation_node_input_meta.dtype != torch.float32:
                    continue
                if type(cur_module) in [torch.nn.Linear] and not torch._C.has_mkl:
                    continue
                computation_node_input_size = computation_node_input_meta.shape
                if (
                    type(cur_module) in [torch.nn.Linear]
                    and len(computation_node_input_size) < 2
                ):
                    continue
                if type(cur_module) in [nn.Conv2d] and isinstance(
                    cur_module.padding, str
                ):
                    continue
                # TODO: remove this when group depthwise ConvTranspose is supported
                if is_group_depthwise_conv_transpose(cur_module):
                    continue
                new_module = computation_op_packed_map[type(cur_module)](
                    cur_module, computation_node_input_size
                )
                assert isinstance(new_module, nn.Module)
                replace_node_module(node, modules, new_module)
    gm.graph.lint()
    gm.recompile()
    return gm


computation_op_packed_map = {
    nn.Linear: packed_linear_eval,
    nn.Conv2d: packed_conv_eval,
    nn.ConvTranspose2d: packed_conv_transpose_eval,
}
