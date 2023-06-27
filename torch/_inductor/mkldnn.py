import copy
import itertools
from functools import reduce
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch._dynamo.utils import detect_fake_mode
from torch.fx.experimental.optimization import replace_node_module
from torch.fx.experimental.symbolic_shapes import free_symbols
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
        input_size: Optional[list],
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
                input_size,
            )
            if input_size is not None
            else self.weight.to_mkldnn(),
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
                "none",
                [],
                "",
            )
        return torch.ops.mkldnn._convolution_pointwise(
            input,
            weight,
            bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            "none",
            [],
            "",
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)


class PackedLinearFP32(nn.Linear):
    def __init__(self, linear: nn.Module, input_size: Optional[list]):
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


class PackedLinearBF16(nn.Linear):
    def __init__(self, linear: nn.Module, input_size: Optional[list]):
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
        self.batch_size = (
            reduce(lambda x, y: x * y, input_size[:-1])
            if input_size is not None
            else None
        )
        self.packed_weight = torch.nn.Parameter(
            torch.ops.mkldnn._reorder_linear_weight(
                self.weight.to_mkldnn(),
                self.batch_size,
            ),
            requires_grad=self.weight.requires_grad,
        )

    def forward(self, input):
        y = torch.ops.mkldnn._linear_pointwise(
            input,
            self.packed_weight,
            self.bias,
            "none",
            [],
            "",
        )
        return y


class PackedConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        conv_transpose: nn.Module,
        input_size: Optional[list],
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
        packed_weight = (
            torch.ops.mkldnn._reorder_convolution_transpose_weight(
                self.weight.to_mkldnn(),
                self.padding,
                self.output_padding,
                self.stride,
                self.dilation,
                self.groups,
                input_size,
            )
            if input_size is not None
            else self.weight.transpose(0, 1).to_mkldnn()
        )
        self.weight = torch.nn.Parameter(
            packed_weight,
            requires_grad=self.weight.requires_grad,
        )

    def _conv_transpose_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for PackedConvTranspose2d"
            )
        return torch.ops.mkldnn._convolution_transpose_pointwise(
            input,
            weight,
            bias,
            self.padding,
            self.output_padding,
            self.stride,
            self.dilation,
            self.groups,
            "none",
            [],
            "",
        )

    def forward(self, input):
        return self._conv_transpose_forward(input, self.weight, self.bias)


class PackedLSTM(nn.LSTM):
    def __init__(
        self,
        lstm: nn.Module,
        input_size: Optional[list],
    ):
        super().__init__(
            lstm.input_size,
            lstm.hidden_size,
            lstm.num_layers,
            lstm.bias,
            lstm.batch_first,
            lstm.dropout,
            lstm.bidirectional,
            lstm.proj_size,
            lstm.weight_ih_l0.device,
            lstm.weight_ih_l0.dtype,
        )
        self._update_module_params(lstm, input_size)
        self.forward_op = torch.ops.mkldnn._lstm

    def _update_module_params(self, lstm, input_size):
        self.__dict__ = copy.deepcopy(lstm.__dict__)
        packed_flat_weights = torch.ops.mkldnn._reorder_lstm_weight(
            self._flat_weights,
            self.input_size,
            self.hidden_size,
            self.bias,
            self.num_layers,
            self.bidirectional,
            self.batch_first,
            input_size,
        )
        assert len(packed_flat_weights) == len(self._flat_weights_names)
        for i, (name, tensor) in enumerate(
            zip(self._flat_weights_names, packed_flat_weights)
        ):
            setattr(
                self,
                name,
                torch.nn.Parameter(
                    tensor, requires_grad=self._flat_weights[i].requires_grad
                ),
            )


def packed_conv_eval(conv: nn.Module, input_size: Optional[list]):
    assert not (conv.training), "Fusion only for eval!"
    return PackedConv2d(
        conv,
        input_size,
    )


def packed_conv_transpose_eval(conv_transpose: nn.Module, input_size: Optional[list]):
    assert not (conv_transpose.training), "Fusion only for eval!"
    return PackedConvTranspose2d(conv_transpose, input_size)


def packed_linear_eval(linear: nn.Module, input_size: Optional[list]):
    assert not (linear.training), "Fusion only for eval!"
    if linear.weight.dtype == torch.bfloat16:
        return PackedLinearBF16(linear, input_size)
    return PackedLinearFP32(linear, input_size)


def packed_lstm_eval(lstm: nn.Module, input_size: Optional[list]):
    assert not (lstm.training), "Fusion only for eval!"
    return PackedLSTM(lstm, input_size)


def mkldnn_fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    is_cpu = all(
        example_input.device == torch.device("cpu")
        for example_input in example_inputs
        if isinstance(example_input, torch.Tensor)
    )

    # make sure the autograd and autocast are disabled.
    if torch.is_grad_enabled() or torch.is_autocast_cpu_enabled():
        return gm
    if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
        return gm
    if not is_cpu:
        return gm
    fake_mode = detect_fake_mode(example_inputs)
    # NB: free_symbols test here is a BIG hammer.  ShapeProp doesn't
    # work with symbolic shapes though, see
    # https://github.com/pytorch/pytorch/pull/103512
    if config.cpp.weight_prepack:
        if not any(free_symbols(e) for e in example_inputs):
            ShapeProp(gm, fake_mode=fake_mode).propagate(*example_inputs)
        gm = pack_module(gm)
    return gm


def pack_module(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module":
            assert isinstance(node.target, str)
            cur_module = modules[node.target]
            if type(cur_module) in computation_op_packed_map:
                if isinstance(cur_module, nn.LSTM):
                    devices = {w.device for w in cur_module._flat_weights}
                    assert (
                        len(devices) == 1
                    ), "Expect lstm weight to be on the same device"
                    device = devices.pop()

                    dtypes = {w.dtype for w in cur_module._flat_weights}
                    assert len(dtypes) == 1, "Expect lstm weight to be the same dtype"
                    dtype = dtypes.pop()

                    shapes = itertools.chain(
                        [w.shape for w in cur_module._flat_weights]
                    )
                else:
                    device = cur_module.weight.device
                    dtype = cur_module.weight.dtype
                    shapes = cur_module.weight.shape

                if (
                    device != torch.device("cpu")
                    or dtype not in [torch.bfloat16, torch.float32]
                    or any(size == 0 for size in shapes)
                ):
                    continue
                if cur_module.training:
                    continue
                if (
                    dtype == torch.bfloat16
                    and not torch.ops.mkldnn._is_mkldnn_bf16_supported()
                ):
                    continue
                if node.args[0].meta.get("tensor_meta") is None:
                    computation_node_input_size = None
                    # Conv2d and ConvTranspose2d weight format are dependent on input size,
                    # but ShapeProp may be failed to get the input size, so we skip them.
                    if not (
                        (
                            type(cur_module) in [torch.nn.Linear]
                            and dtype == torch.bfloat16
                        )
                        or type(cur_module) in [torch.nn.LSTM]
                    ):
                        continue
                else:
                    computation_node_input_size = tuple(
                        int(x) for x in node.args[0].meta.get("tensor_meta").shape
                    )
                    if any(size == 0 for size in computation_node_input_size):
                        continue
                    if type(cur_module) in [torch.nn.Linear]:
                        # for fp32 linear, only packed when has mkl.
                        if (dtype == torch.float32 and (not torch._C.has_mkl)) or len(
                            computation_node_input_size
                        ) < 2:
                            continue
                    elif type(cur_module) in [nn.LSTM]:
                        # pack_padded_sequence input is not supported.
                        # For pack_padded_sequence input, the len(computation_node_input_size) == 4
                        if len(computation_node_input_size) not in [2, 3]:
                            continue
                    else:
                        if len(computation_node_input_size) != 4:
                            continue
                if type(cur_module) in [nn.Conv2d] and isinstance(
                    cur_module.padding, str
                ):
                    continue
                # TODO: remove this when group depthwise ConvTranspose is supported
                if type(cur_module) in [nn.ConvTranspose2d] and (
                    is_group_depthwise_conv_transpose(cur_module)
                    or len(node.args) > 1
                    or len(node.kwargs) > 0
                    or any(
                        not isinstance(output_padding, int)
                        or not isinstance(stride, int)
                        or output_padding >= stride
                        for output_padding, stride in zip(
                            cur_module.output_padding, cur_module.stride
                        )
                    )  # Port from: aten/src/ATen/native/Convolution.cpp:is_output_padding_big
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


computation_op_packed_map = {
    nn.Linear: packed_linear_eval,
    nn.Conv2d: packed_conv_eval,
    nn.ConvTranspose2d: packed_conv_transpose_eval,
    nn.LSTM: packed_lstm_eval,
}
