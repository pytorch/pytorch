# Owner(s): ["oncall: quantization"]
import copy
import itertools
from enum import Enum

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torch.nn as nn
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization import ObserverBase
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    QUANT_ANNOTATION_KEY,
    X86InductorQuantizer,
)
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    skipIfNoInductorSupport,
    skipIfNoX86,
)
from torch.testing._internal.common_quantized import override_quantized_engine
from torch.testing._internal.common_utils import skipIfTorchDynamo


class NodePosType(Enum):
    left = 1
    right = 2
    both = 3


class TestHelperModules:
    class SingleConv2dModule(torch.nn.Module):
        def __init__(self, with_bn=False) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 6, (2, 2), stride=(1, 1), padding=(1, 1))
            self.bn = torch.nn.BatchNorm2d(6)
            self.with_bn = with_bn

        def forward(self, x):
            x = self.conv(x)
            if self.with_bn:
                x = self.bn(x)
            return x

    class Conv2dUnaryModule(torch.nn.Module):
        def __init__(self, post_op, use_bias: bool = False, with_bn=False) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                3, 6, (2, 2), stride=(1, 1), padding=(1, 1), bias=use_bias
            )
            self.post_op = post_op
            self.bn = torch.nn.BatchNorm2d(6)
            self.with_bn = with_bn
            self.maxpool = torch.nn.MaxPool2d((3, 3))

        def forward(self, x):
            x = self.conv(x)
            if self.with_bn:
                x = self.bn(x)
            x = self.post_op(x)
            x = self.maxpool(x)
            return x

    class Conv2dAddModule(torch.nn.Module):
        def __init__(
            self,
            inplace_add: bool = False,
            conv2d_type: NodePosType = NodePosType.left,
            use_bias: bool = False,
            with_bn: bool = False,
        ) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            self.conv2 = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            self.relu = nn.ReLU()
            self.inplace_add = inplace_add
            self.conv2d_type = conv2d_type
            self.bn = torch.nn.BatchNorm2d(3)
            self.with_bn = with_bn

        def forward(self, x):
            if self.conv2d_type == NodePosType.left:
                if self.inplace_add:
                    tmp = self.conv(x)
                    if self.with_bn:
                        tmp = self.bn(tmp)
                    tmp += self.relu(x)
                    return tmp
                else:
                    tmp = self.conv(x)
                    if self.with_bn:
                        tmp = self.bn(tmp)
                    return tmp + self.relu(x)
            elif self.conv2d_type == NodePosType.right:
                if self.inplace_add:
                    tmp = self.relu(x)
                    tmp += self.conv(x)
                    return tmp
                else:
                    return self.relu(x) + self.conv(x)
            elif self.conv2d_type == NodePosType.both:
                if self.inplace_add:
                    tmp = self.conv(x)
                    tmp += self.conv2(x)
                    return tmp
                else:
                    return self.conv(x) + self.conv2(x)

    class Conv2dAddReLUModule(torch.nn.Module):
        def __init__(
            self,
            inplace_add: bool = False,
            conv2d_type: NodePosType = NodePosType.left,
            inplace_relu: bool = False,
            use_bias: bool = False,
            with_bn: bool = False,
        ) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            self.conv2 = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            self.relu = nn.ReLU()
            self.inplace_add = inplace_add
            self.conv2d_type = conv2d_type
            self.relu2 = nn.ReLU(inplace=inplace_relu)
            self.bn = torch.nn.BatchNorm2d(3)
            self.with_bn = with_bn

        def forward(self, x):
            if self.conv2d_type == NodePosType.left:
                if self.inplace_add:
                    tmp = self.conv(x)
                    if self.with_bn:
                        tmp = self.bn(tmp)
                    tmp += self.relu(x)
                    return self.relu2(tmp)
                else:
                    tmp = self.conv(x)
                    if self.with_bn:
                        tmp = self.bn(tmp)
                    return self.relu2(tmp + self.relu(x))
            elif self.conv2d_type == NodePosType.right:
                if self.inplace_add:
                    tmp = self.relu(x)
                    tmp += self.conv(x)
                    return self.relu2(tmp)
                else:
                    return self.relu2(self.relu(x) + self.conv(x))
            elif self.conv2d_type == NodePosType.both:
                if self.inplace_add:
                    tmp = self.conv(x)
                    tmp += self.conv2(x)
                    return self.relu2(tmp)
                else:
                    return self.relu2(self.conv(x) + self.conv2(x))

    class Conv2dSingleOpPowModule(nn.Module):
        def __init__(self, single_op):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 1)
            self.single_op = single_op

        def forward(self, x):
            x = self.conv(x)
            x = self.single_op(x)
            return torch.pow(x, 2)

    class SerialsConv2dAddReLUModule(torch.nn.Module):
        """Serials of 2 Conv2d -> Add -> ReLU Pattern."""

        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            self.conv2 = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            self.conv3 = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            self.conv4 = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            self.relu = nn.ReLU()
            self.relu2 = nn.ReLU()

        def forward(self, x):
            x1 = self.conv(x)
            res1 = self.relu(self.conv2(x1) + self.conv3(x1))
            res2 = self.relu2(self.conv4(res1) + res1)
            return res2

    class Conv2dCatMaxpool2d(torch.nn.Module):
        def __init__(
            self,
        ):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                3, 16, 7, bias=True, stride=2, padding=3, dilation=1
            )
            self.conv2 = torch.nn.Conv2d(
                3, 16, 7, bias=True, stride=2, padding=3, dilation=1
            )
            self.relu = torch.nn.ReLU()
            self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
            self.conv3 = torch.nn.Conv2d(
                32, 32, 7, bias=True, stride=2, padding=3, dilation=1
            )

        def forward(self, x):
            temp1 = self.relu(self.conv(x))
            temp2 = self.conv2(x + 1)
            temp3 = torch.cat((temp1, temp2), 1)
            temp4 = self.maxpool(temp3)
            temp5 = self.conv3(temp4)
            return temp5

    class Conv2dAvgPool2d(torch.nn.Module):
        def __init__(
            self,
        ):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                3, 16, 7, bias=True, stride=2, padding=3, dilation=1
            )
            self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)

        def forward(self, x):
            temp1 = self.avgpool(self.conv(x))
            return temp1

    class Conv2dCatSameInputs(torch.nn.Module):
        def __init__(
            self,
        ):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                3, 16, 7, bias=True, stride=2, padding=3, dilation=1
            )
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            temp1 = self.relu(self.conv(x))
            temp3 = torch.cat((temp1, temp1), 1)
            return temp3

    class Conv2dCatSingleInput(torch.nn.Module):
        def __init__(
            self,
        ):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                3, 16, 7, bias=True, stride=2, padding=3, dilation=1
            )
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            temp1 = self.relu(self.conv(x))
            temp3 = torch.cat((temp1,), 1)
            return temp3

    class SingleLinearModule(torch.nn.Module):
        def __init__(self, use_bias) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=use_bias)

        def forward(self, x):
            return self.linear(x)

    class LinearUnaryModule(torch.nn.Module):
        def __init__(
            self, use_bias, postop, inplace_postop=False, post_op_algo="none"
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=use_bias)
            if postop == nn.GELU:
                self.postop = postop(approximate=post_op_algo)
            else:
                self.postop = postop(inplace=inplace_postop)

        def forward(self, x):
            return self.postop(self.linear(x))

    class LinearAddModule(torch.nn.Module):
        def __init__(
            self,
            inplace_add: bool = False,
            linear_pos: NodePosType = NodePosType.left,
            use_bias: bool = False,
        ) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(
                in_features=16, out_features=16, bias=use_bias
            )
            self.linear2 = torch.nn.Linear(
                in_features=16, out_features=16, bias=use_bias
            )
            self.relu = nn.ReLU()
            self.inplace_add = inplace_add
            self.linear_pos = linear_pos

        def forward(self, x):
            if self.linear_pos == NodePosType.left:
                if self.inplace_add:
                    tmp = self.linear(x)
                    tmp += self.relu(x)
                    return tmp
                else:
                    tmp = self.linear(x)
                    return tmp + self.relu(x)
            elif self.linear_pos == NodePosType.right:
                if self.inplace_add:
                    tmp = self.relu(x)
                    tmp += self.linear(x)
                    return tmp
                else:
                    return self.relu(x) + self.linear(x)
            elif self.linear_pos == NodePosType.both:
                if self.inplace_add:
                    tmp = self.linear(x)
                    tmp += self.linear2(x)
                    return tmp
                else:
                    return self.linear(x) + self.linear2(x)

    class LinearAddReLUModule(torch.nn.Module):
        def __init__(
            self,
            inplace_add: bool = False,
            linear_pos: NodePosType = NodePosType.left,
            inplace_relu: bool = False,
            use_bias: bool = False,
        ) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(
                in_features=16, out_features=16, bias=use_bias
            )
            self.linear2 = torch.nn.Linear(
                in_features=16, out_features=16, bias=use_bias
            )
            self.relu = nn.ReLU()
            self.inplace_add = inplace_add
            self.linear_pos = linear_pos
            self.relu2 = nn.ReLU(inplace=inplace_relu)

        def forward(self, x):
            if self.linear_pos == NodePosType.left:
                if self.inplace_add:
                    tmp = self.linear(x)
                    tmp += self.relu(x)
                    return self.relu2(tmp)
                else:
                    tmp = self.linear(x)
                    return self.relu2(tmp + self.relu(x))
            elif self.linear_pos == NodePosType.right:
                if self.inplace_add:
                    tmp = self.relu(x)
                    tmp += self.linear(x)
                    return self.relu2(tmp)
                else:
                    return self.relu2(self.relu(x) + self.linear(x))
            elif self.linear_pos == NodePosType.both:
                if self.inplace_add:
                    tmp = self.linear(x)
                    tmp += self.linear2(x)
                    return self.relu2(tmp)
                else:
                    return self.relu2(self.linear(x) + self.linear2(x))

    class SerialsLinearAddReLUModule(torch.nn.Module):
        """Serials of 2 Linear -> Add -> ReLU Pattern."""

        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_features=16, out_features=16, bias=True)
            self.linear2 = torch.nn.Linear(in_features=16, out_features=16, bias=True)
            self.linear3 = torch.nn.Linear(in_features=16, out_features=16, bias=True)
            self.linear4 = torch.nn.Linear(in_features=16, out_features=16, bias=True)
            self.relu = nn.ReLU()
            self.relu2 = nn.ReLU()

        def forward(self, x):
            x1 = self.linear(x)
            res1 = self.relu(self.linear2(x1) + self.linear3(x1))
            res2 = self.relu2(self.linear4(res1) + res1)
            return res2

    class LinearAddModule2(torch.nn.Module):
        def __init__(
            self,
            inplace_add: bool = False,
        ) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_features=16, out_features=16, bias=True)
            self.linear2 = torch.nn.Linear(in_features=16, out_features=16, bias=True)
            self.inplace_add = inplace_add

        def forward(self, x):
            if self.inplace_add:
                tmp = self.linear(x)
                tmp += self.linear2(tmp)
                return tmp
            else:
                tmp = self.linear(x)
                return tmp + self.linear2(tmp)

    class Conv2dAddModule2(torch.nn.Module):
        def __init__(
            self,
            inplace_add: bool = False,
        ) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
            )
            self.conv2 = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
            )
            self.inplace_add = inplace_add
            self.bn = torch.nn.BatchNorm2d(3)
            self.bn2 = torch.nn.BatchNorm2d(3)

        def forward(self, x):
            if self.inplace_add:
                tmp = self.bn(self.conv(x))
                tmp += self.bn2(self.conv2(tmp))
                return tmp
            else:
                tmp = self.bn(self.conv(x))
                return tmp + self.bn2(self.conv2(tmp))

    class SelfAttnLikeModule(torch.nn.Module):
        def __init__(
            self,
            input_dim,
            transpose_for_score=False,
            num_attention_heads=None,
            attention_head_size=None,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.q_proj = nn.Linear(input_dim, input_dim, bias=False)
            self.k_proj = nn.Linear(input_dim, input_dim, bias=False)
            self.v_proj = nn.Linear(input_dim, input_dim, bias=False)
            self.softmax = nn.Softmax(dim=-1)
            self.transpose_for_score = transpose_for_score
            if self.transpose_for_score:
                assert num_attention_heads is not None
                assert attention_head_size is not None
                self.num_attention_heads = num_attention_heads
                self.attention_head_size = attention_head_size

        def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (
                self.num_attention_heads,
                self.attention_head_size,
            )
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            if self.transpose_for_score:
                q = self.transpose_for_scores(q)
                k = self.transpose_for_scores(k)
                v = self.transpose_for_scores(v)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.input_dim**0.5)
            attention = self.softmax(scores)
            weighted = torch.matmul(attention, v)
            return weighted


class X86InductorQuantTestCase(QuantizationTestCase):
    def _test_quantizer(
        self,
        model,
        example_inputs,
        quantizer,
        expected_node_occurrence,
        expected_node_list=None,
        is_qat=False,
        debug=False,
    ):
        m_eager = model.train() if is_qat else model.eval()

        # program capture
        m = copy.deepcopy(m_eager)
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )

        # QAT Model failed to deepcopy
        export_model = m if is_qat else copy.deepcopy(m)
        m = prepare_qat_pt2e(m, quantizer) if is_qat else prepare_pt2e(m, quantizer)
        # Calibrate
        m(*example_inputs)
        prepare_model = copy.deepcopy(m)
        m = convert_pt2e(m)
        convert_model = copy.deepcopy(m)
        if debug:
            convert_model.print_readable(True)
        pt2_quant_output = m(*example_inputs)
        node_occurrence = {
            ns.call_function(k): v for k, v in expected_node_occurrence.items()
        }
        if expected_node_list is None:
            expected_node_list = []
        node_list = [ns.call_function(n) for n in expected_node_list]
        self.checkGraphModuleNodes(
            m, expected_node_occurrence=node_occurrence, expected_node_list=node_list
        )
        return export_model, prepare_model, convert_model


@skipIfNoInductorSupport
class TestQuantizePT2EX86Inductor(X86InductorQuantTestCase):
    @skipIfNoX86
    def test_conv2d(self):
        """
        Test pattern of single conv2d with X86InductorQuantizer.
        """
        with override_quantized_engine("x86"), torch.no_grad():
            m = TestHelperModules.SingleConv2dModule().eval()
            example_inputs = (torch.randn(2, 3, 16, 16),)
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            node_occurrence = {
                # one for input and weight of the conv
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                # note: quantize op for weights are const propagated
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
            }
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
            ]
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )

    @skipIfNoX86
    def test_conv2d_unary(self):
        """
        Test pattern of conv2d with unary post ops (such as relu, hardtanh, hardswish, relu6) with X86InductorQuantizer.
        """
        unary_map = {
            "relu": [torch.nn.ReLU(inplace=False), torch.ops.aten.relu.default],
            "relu_inplace": [torch.nn.ReLU(inplace=True), torch.ops.aten.relu_.default],
            "hardtanh": [
                torch.nn.Hardtanh(min_val=0.0, max_val=6.0, inplace=False),
                torch.ops.aten.hardtanh.default,
            ],
            "hardtanh_inplace": [
                torch.nn.Hardtanh(min_val=0.0, max_val=6.0, inplace=True),
                torch.ops.aten.hardtanh_.default,
            ],
            "relu6": [torch.nn.ReLU6(inplace=False), torch.ops.aten.hardtanh.default],
            "relu6_inplace": [
                torch.nn.ReLU6(inplace=True),
                torch.ops.aten.hardtanh_.default,
            ],
            "hardswish": [
                torch.nn.Hardswish(inplace=False),
                torch.ops.aten.hardswish.default,
            ],
            "hardswish_inplace": [
                torch.nn.Hardswish(inplace=True),
                torch.ops.aten.hardswish_.default,
            ],
            "swish": [torch.nn.SiLU(inplace=False), torch.ops.aten.silu.default],
            "swish_inplace": [
                torch.nn.SiLU(inplace=True),
                torch.ops.aten.silu_.default,
            ],
        }
        use_bias_list = [True, False]
        with override_quantized_engine("x86"), torch.no_grad():
            for unary_op, use_bias in itertools.product(
                unary_map.keys(), use_bias_list
            ):
                m = TestHelperModules.Conv2dUnaryModule(
                    unary_map[unary_op][0], use_bias=use_bias
                ).eval()
                example_inputs = (torch.randn(2, 3, 16, 16),)
                quantizer = X86InductorQuantizer().set_global(
                    xiq.get_default_x86_inductor_quantization_config()
                )
                node_occurrence = {
                    # one for input and weight of the conv
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
                    # note: quantize op for weights are const propagated
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.conv2d.default,
                    unary_map[unary_op][1],
                ]
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    @skipIfNoX86
    def test_conv2d_binary(self):
        """
        Test pattern of conv2d with binary post ops (such as add) with X86InductorQuantizer.
        Currently, only add as binary post op is supported.
        """
        conv2d_type_list = [NodePosType.left, NodePosType.both]
        example_inputs = (torch.randn(2, 3, 6, 6),)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        with override_quantized_engine("x86"), torch.no_grad():
            for conv2d_type in conv2d_type_list:
                m = TestHelperModules.Conv2dAddModule(conv2d_type=conv2d_type).eval()
                if conv2d_type != NodePosType.both:
                    node_occurrence = {
                        # one for input and weight of the conv
                        # one for extra input node of add
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
                        # quantize_per_channel for weights are const propagated
                        torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                    }
                else:
                    node_occurrence = {
                        # one for input of the conv
                        # one for input of another conv
                        # 2 conv will share same input quant/dequant
                        # one for extra input node of add
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
                        # quantize_per_channel for weights are const propagated
                        torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                    }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.conv2d.default,
                    torch.ops.aten.add.Tensor,
                ]
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    @skipIfNoX86
    def test_conv2d_binary2(self):
        """
        Test Pattern:
            tmp = conv2d_1(x)
            tmp2 = conv2d_2(tmp)
            return tmp + tmp2
        Since conv2d_1 has 2 users, we should annotate conv2d_2 for binary fusion instead of conv2d_1
        """
        example_inputs = (torch.randn(2, 3, 6, 6),)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        inplace_add_list = [True, False]
        with override_quantized_engine("x86"), torch.no_grad():
            for inplace_add in inplace_add_list:
                m = TestHelperModules.Conv2dAddModule2(inplace_add=inplace_add).eval()
                node_occurrence = {
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
                    # quantize_per_channel for weights are const propagated
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.conv2d.default,
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    (
                        torch.ops.aten.add_.Tensor
                        if inplace_add
                        else torch.ops.aten.add.Tensor
                    ),
                ]
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    @skipIfNoX86
    def test_conv2d_binary_unary(self):
        """
        Test pattern of conv2d with binary + unary post ops (such as add + relu) with X86InductorQuantizer.
        Currently, only add as binary post op and relu as unary post op are supported.
        """
        conv2d_type_list = [NodePosType.left, NodePosType.both]
        example_inputs = (torch.randn(2, 3, 6, 6),)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        with override_quantized_engine("x86"), torch.no_grad():
            for conv2d_type in conv2d_type_list:
                m = TestHelperModules.Conv2dAddReLUModule(
                    conv2d_type=conv2d_type,
                ).eval()
                if conv2d_type != NodePosType.both:
                    node_occurrence = {
                        # one for input for conv
                        # one for extra input node of add
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
                        # note: quantize op for weights are const propagated
                        torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                    }
                else:
                    node_occurrence = {
                        # one for input of the conv
                        # one for input of another conv
                        # 2 conv will share same input quant/dequant
                        # one for extra input node of add
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
                        # note: quantize op for weights are const propagated
                        torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                    }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.conv2d.default,
                    torch.ops.aten.add.Tensor,
                ]
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    @skipIfNoX86
    def test_conv2d_serials_binary_unary(self):
        """
        Test pattern of 2 following up conv2d add relu with X86InductorQuantizer.
        """
        with override_quantized_engine("x86"), torch.no_grad():
            m = TestHelperModules.SerialsConv2dAddReLUModule().eval()
            example_inputs = (torch.randn(2, 3, 16, 16),)
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            node_occurrence = {
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 4,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 6,
                # quantize_per_channel for weights are const propagated
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 4,
            }
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )

    def _single_op_share_observer_recipe_test_helper(self, m, x, single_op):
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        example_inputs = (x,)
        node_occurrence = {
            # one for input and weight of the conv, two for input/output for the maxpool2d
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            single_op,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        _, prepare_model, _ = self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )
        # Check Maxpool2d has share observer at input and output
        for node in prepare_model.graph.nodes:
            if node.op == "call_function" and node.target is single_op:
                single_op_node = node
                input_obs_of_single_op = getattr(
                    prepare_model, single_op_node.args[0].target
                )
                output_obs_of_single_op = getattr(
                    prepare_model, next(iter(single_op_node.users)).target
                )
            elif (
                node.op == "call_function"
                and node.target is torch.ops.aten.conv2d.default
            ):
                conv_node = node
                input_obs_of_conv = getattr(prepare_model, conv_node.args[0].target)
        self.assertTrue(isinstance(input_obs_of_single_op, ObserverBase))
        self.assertTrue(isinstance(output_obs_of_single_op, ObserverBase))
        self.assertTrue(isinstance(input_obs_of_conv, ObserverBase))
        self.assertTrue(input_obs_of_single_op is output_obs_of_single_op)
        self.assertTrue(input_obs_of_single_op is not input_obs_of_conv)

    @skipIfNoX86
    def test_maxpool2d_recipe(self):
        r"""
        Test pattern: int8_in_int8_out_ops(maxpool) - non_quantizable op(pow)
        Since maxpool is a int8_in_int8_out_op, there is obs between maxpool and pow.
        """
        self._single_op_share_observer_recipe_test_helper(
            TestHelperModules.Conv2dSingleOpPowModule(nn.MaxPool2d(1, 1)).eval(),
            torch.rand(1, 2, 14, 14),
            torch.ops.aten.max_pool2d.default,
        )

    @skipIfNoX86
    def test_adaptive_avg_pool2d_recipe(self):
        r"""
        Test pattern: int8_in_int8_out_ops(adaptive_avg_pool2d) - non_quantizable op(pow)
        Since adaptive_avg_pool2d is a int8_in_int8_out_op, there is obs between adaptive_avg_pool2d and pow.
        """
        self._single_op_share_observer_recipe_test_helper(
            TestHelperModules.Conv2dSingleOpPowModule(
                nn.AdaptiveAvgPool2d((1, 1))
            ).eval(),
            torch.rand(1, 2, 14, 14),
            torch.ops.aten.adaptive_avg_pool2d.default,
        )

    @skipIfNoX86
    def test_flatten_recipe(self):
        r"""
        Test pattern: int8_in_int8_out_ops(flatten) - non_quantizable op(pow)
        Since flatten is a int8_in_int8_out_op, there is obs between flatten and pow.
        """
        self._single_op_share_observer_recipe_test_helper(
            TestHelperModules.Conv2dSingleOpPowModule(
                lambda x: torch.flatten(x, 1)
            ).eval(),
            torch.rand(1, 2, 14, 14),
            torch.ops.aten.flatten.using_ints,
        )

    @skipIfNoX86
    def test_cat_recipe(self):
        r"""
        Test pattern: conv -> cat -> maxpool2d
        Since cat, maxpool is a int8_in_int8_out_op, the inputs and outputs should with same observer.
        """
        m = TestHelperModules.Conv2dCatMaxpool2d().eval()
        x = torch.randn(16, 3, 16, 16).contiguous(memory_format=torch.channels_last)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        example_inputs = (x,)
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 6,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 6,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
        }
        node_list = [
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.cat.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.max_pool2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        _, prepare_model, _ = self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )
        # Check Cat/Maxpool2d has share observer at input and output
        for node in prepare_model.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.cat.default:
                cat_act_obs0 = getattr(prepare_model, node.all_input_nodes[0].target)
                cat_act_obs1 = getattr(prepare_model, node.all_input_nodes[1].target)
                cat_out_obs = getattr(prepare_model, next(iter(node.users)).target)
            elif (
                node.op == "call_function"
                and node.target is torch.ops.aten.max_pool2d.default
            ):
                maxpool_node = node
                input_obs_of_maxpool = getattr(
                    prepare_model, maxpool_node.args[0].target
                )
                output_obs_of_maxpool = getattr(
                    prepare_model, next(iter(maxpool_node.users)).target
                )
        self.assertTrue(isinstance(cat_act_obs0, ObserverBase))
        self.assertTrue(isinstance(cat_act_obs1, ObserverBase))
        self.assertTrue(isinstance(cat_out_obs, ObserverBase))
        self.assertTrue(isinstance(input_obs_of_maxpool, ObserverBase))
        self.assertTrue(isinstance(output_obs_of_maxpool, ObserverBase))
        self.assertTrue(cat_act_obs0 is cat_act_obs1)
        self.assertTrue(cat_act_obs0 is cat_out_obs)
        self.assertTrue(cat_out_obs is input_obs_of_maxpool)
        self.assertTrue(input_obs_of_maxpool is output_obs_of_maxpool)

    @skipIfNoX86
    def test_cat_recipe_same_inputs(self):
        r"""
        Test pattern: conv -> cat([input0, input0])
        Since cat has 2 input node of same tensor, they should also be with same observer.
        """
        m = TestHelperModules.Conv2dCatSameInputs().eval()
        x = torch.randn(16, 3, 16, 16).contiguous(memory_format=torch.channels_last)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        example_inputs = (x,)
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.cat.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        _, prepare_model, _ = self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )
        # Check Cat has share observer at input and output
        for node in prepare_model.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.cat.default:
                cat_act_obs0 = getattr(prepare_model, node.args[0][0].target)
                cat_act_obs1 = getattr(prepare_model, node.args[0][1].target)
                cat_out_obs = getattr(prepare_model, next(iter(node.users)).target)
        self.assertTrue(isinstance(cat_act_obs0, ObserverBase))
        self.assertTrue(isinstance(cat_act_obs1, ObserverBase))
        self.assertTrue(isinstance(cat_out_obs, ObserverBase))
        self.assertTrue(cat_act_obs0 is cat_act_obs1)
        self.assertTrue(cat_act_obs0 is cat_out_obs)

    @skipIfNoX86
    def test_cat_recipe_single_input(self):
        r"""
        Test pattern: conv -> cat([input0,])
        Since cat has 1 input node, they should also be with same observer.
        """
        m = TestHelperModules.Conv2dCatSingleInput().eval()
        x = torch.randn(16, 3, 16, 16).contiguous(memory_format=torch.channels_last)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        example_inputs = (x,)
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.cat.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        _, prepare_model, _ = self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )
        # Check Cat has share observer at input and output
        for node in prepare_model.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.cat.default:
                cat_act_obs0 = getattr(prepare_model, node.args[0][0].target)
                cat_out_obs = getattr(prepare_model, next(iter(node.users)).target)
        self.assertTrue(isinstance(cat_act_obs0, ObserverBase))
        self.assertTrue(isinstance(cat_out_obs, ObserverBase))
        self.assertTrue(cat_act_obs0 is cat_out_obs)

    @skipIfNoX86
    def test_avg_pool2d_recipe(self):
        r"""
        Test pattern: conv -> AvgPool2d
        Since AvgPool2d is a int8_in_int8_out_op, the inputs and outputs should with same observer.
        """
        m = TestHelperModules.Conv2dAvgPool2d().eval()
        x = torch.randn(16, 3, 16, 16).contiguous(memory_format=torch.channels_last)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        example_inputs = (x,)
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.avg_pool2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        _, prepare_model, _ = self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )
        for node in prepare_model.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is torch.ops.aten.avg_pool2d.default
            ):
                avgpool_node = node
                input_obs_of_avgpool = getattr(
                    prepare_model, avgpool_node.args[0].target
                )
                output_obs_of_avgpool = getattr(
                    prepare_model, next(iter(avgpool_node.users)).target
                )
            elif (
                node.op == "call_function"
                and node.target is torch.ops.aten.conv2d.default
            ):
                conv_node = node
                output_obs_of_conv = getattr(
                    prepare_model, next(iter(conv_node.users)).target
                )
        self.assertTrue(isinstance(input_obs_of_avgpool, ObserverBase))
        self.assertTrue(isinstance(output_obs_of_avgpool, ObserverBase))
        self.assertTrue(isinstance(output_obs_of_conv, ObserverBase))
        self.assertTrue(input_obs_of_avgpool is output_obs_of_avgpool)
        self.assertTrue(input_obs_of_avgpool is output_obs_of_conv)

    @skipIfNoX86
    def test_linear(self):
        """
        Test pattern of single linear with X86InductorQuantizer.
        """
        with override_quantized_engine("x86"), torch.no_grad():
            for use_bias in [True, False]:
                m = TestHelperModules.SingleLinearModule(use_bias).eval()
                example_inputs = (torch.randn(2, 4),)
                quantizer = X86InductorQuantizer().set_global(
                    xiq.get_default_x86_inductor_quantization_config()
                )
                node_occurrence = {
                    # one for input and weight, one for output
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                    # quantize_per_channel for weights are const propagated
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.linear.default,
                ]
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    @skipIfNoX86
    def test_linear_unary(self):
        """
        Test pattern of linear with unary post ops (e.g. relu) with X86InductorQuantizer.
        """
        use_bias_list = [True, False]
        inplace_list = [True, False]
        postop_list = [nn.ReLU, nn.LeakyReLU]  # only test two to save time
        cases = itertools.product(use_bias_list, inplace_list, postop_list)
        post_op_map = {
            nn.ReLU: [torch.ops.aten.relu_.default, torch.ops.aten.relu.default],
            nn.LeakyReLU: [
                torch.ops.aten.leaky_relu_.default,
                torch.ops.aten.leaky_relu.default,
            ],
        }
        with override_quantized_engine("x86"), torch.no_grad():
            for use_bias, inplace, postop in cases:
                m = TestHelperModules.LinearUnaryModule(
                    use_bias=use_bias, postop=postop, inplace_postop=inplace
                ).eval()
                example_inputs = (torch.randn(2, 4),)
                quantizer = X86InductorQuantizer().set_global(
                    xiq.get_default_x86_inductor_quantization_config()
                )
                node_occurrence = {
                    # one for input and weight of the conv, one for output for the relu
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                    # quantize_per_channel for weights are const propagated
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.linear.default,
                    post_op_map[postop][0 if inplace else 1],
                ]
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    @skipIfNoX86
    def test_linear_unary_gelu(self):
        """
        Test pattern of linear with unary post ops (e.g. gelu) with X86InductorQuantizer.
        """
        use_bias_list = [True, False]
        postop = nn.GELU
        post_op_algorithm = ["none", "tanh"]
        cases = itertools.product(use_bias_list, post_op_algorithm)
        with override_quantized_engine("x86"), torch.no_grad():
            for use_bias, post_op_algo in cases:
                m = TestHelperModules.LinearUnaryModule(
                    use_bias=use_bias, postop=postop, post_op_algo=post_op_algo
                ).eval()
                example_inputs = (torch.randn(2, 4),)
                quantizer = X86InductorQuantizer().set_global(
                    xiq.get_default_x86_inductor_quantization_config()
                )
                node_occurrence = {
                    # one for input and weight of the conv, one for output for the gelu
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                    # quantize_per_channel for weights are const propagated
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.linear.default,
                    torch.ops.aten.gelu.default,
                ]
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    def _check_annotation_stat(self, gm, expected_stat_dict):
        # Check expected annotation statistics to ensure the annotation is correct

        def _check_annotation(node):
            annot = node.meta.get(QUANT_ANNOTATION_KEY, None)
            if annot is None:
                return False, False
            return annot._annotated, annot._is_output_of_quantized_pattern

        for node in gm.graph.nodes:
            if node.target in expected_stat_dict.keys():
                annotated, is_quant_out = _check_annotation(node)
                expected_stat_dict[node.target]["annotated"] -= annotated
                expected_stat_dict[node.target]["is_quant_out"] -= is_quant_out
        for op_stat in expected_stat_dict.values():
            assert all(v == 0 for v in op_stat.values())

    @skipIfNoX86
    def test_linear_binary(self):
        """
        Test pattern of linear with binary post ops (such as add) with X86InductorQuantizer.
        Currently, only add as binary post op is supported.
        """
        linear_pos_list = [NodePosType.left, NodePosType.right, NodePosType.both]
        # TODO test for inplace add after refactoring of capture_pre_autograd_graph
        inplace_add_list = [False]
        example_inputs = (torch.randn(2, 16),)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        cases = itertools.product(linear_pos_list, inplace_add_list)
        with override_quantized_engine("x86"), torch.no_grad():
            for linear_pos, inplace_add in cases:
                m = TestHelperModules.LinearAddModule(
                    inplace_add=inplace_add, linear_pos=linear_pos
                ).eval()
                if linear_pos != NodePosType.both:
                    node_occurrence = {
                        # Only one 1 q-dq for input of the linear
                        # No q-dq for extra input node of add
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                        # quantize_per_channel for weights are const propagated
                        torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                    }
                else:
                    node_occurrence = {
                        # One quantize_per_tensor for both linear nodes (shared)
                        # Two dequantize_per_tensor for two linear nodes
                        # No q-dq for extra input node of add
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
                        # quantize_per_channel for weights are const propagated
                        torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                    }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.linear.default,
                    (
                        torch.ops.aten.add_.Tensor
                        if inplace_add
                        else torch.ops.aten.add.Tensor
                    ),
                ]
                fq_m = self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )[-1]
                # One linear and add are fused. The other linear is quantized alone if present
                aten = torch.ops.aten
                add_op = aten.add_.Tensor if inplace_add else aten.add.Tensor
                expected_annotation_stat = {
                    aten.linear.default: {
                        "annotated": 2 if linear_pos == NodePosType.both else 1,
                        "is_quant_out": 1 if linear_pos == NodePosType.both else 0,
                    },
                    add_op: {"annotated": 1, "is_quant_out": 1},
                }
                self._check_annotation_stat(fq_m, expected_annotation_stat)

    @skipIfNoX86
    def test_linear_binary2(self):
        """
        Test Pattern:
            tmp = linear_1(x)
            tmp2 = linear_2(tmp)
            return tmp + tmp2
        Since linear_1 has 2 users, we should annotate linear_2 for binary fusion instead of linear_1
        """
        example_inputs = (torch.randn(2, 16),)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        # TODO test for inplace add after refactoring of capture_pre_autograd_graph
        inplace_add_list = [False]
        with override_quantized_engine("x86"), torch.no_grad():
            for inplace_add in inplace_add_list:
                m = TestHelperModules.LinearAddModule2(inplace_add=inplace_add).eval()
                # Two q-dq nodes for inputs of linear nodes
                # No q-dq for extra input node of add
                node_occurrence = {
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
                    # quantize_per_channel for weights are const propagated
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.linear.default,
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    (
                        torch.ops.aten.add_.Tensor
                        if inplace_add
                        else torch.ops.aten.add.Tensor
                    ),
                ]
                fq_m = self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )[-1]
                # One linear and add are fused. The other linear is quantized alone if present
                aten = torch.ops.aten
                add_op = aten.add_.Tensor if inplace_add else aten.add.Tensor
                expected_annotation_stat = {
                    aten.linear.default: {
                        "annotated": 2,
                        "is_quant_out": 1,
                    },
                    add_op: {"annotated": 1, "is_quant_out": 1},
                }
                self._check_annotation_stat(fq_m, expected_annotation_stat)

    @skipIfNoX86
    def test_linear_binary_unary(self):
        """
        Test pattern of linear with binary + unary post ops (such as add + relu) with X86InductorQuantizer.
        Currently, only add as binary post op and relu as unary post op are supported.
        """
        linear_pos_list = [NodePosType.left, NodePosType.right, NodePosType.both]
        # TODO test for inplace add after refactoring of capture_pre_autograd_graph
        inplace_add_list = [False]
        # TODO test for inplace relu after refactoring of capture_pre_autograd_graph
        inplace_relu_list = [False]
        example_inputs = (torch.randn(2, 16),)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        cases = itertools.product(linear_pos_list, inplace_add_list, inplace_relu_list)
        with override_quantized_engine("x86"), torch.no_grad():
            for linear_pos, inplace_add, inplace_relu in cases:
                m = TestHelperModules.LinearAddReLUModule(
                    inplace_add=inplace_add,
                    linear_pos=linear_pos,
                    inplace_relu=inplace_relu,
                ).eval()
                if linear_pos != NodePosType.both:
                    node_occurrence = {
                        # Only one q-dq node for input of the linear
                        # No q-dq node for extra input node of add
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                        # note: quantize op for weights are const propagated
                        torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                    }
                else:
                    node_occurrence = {
                        # One quantize_per_tensor for both linear nodes (shared)
                        # Two dequantize_per_tensor for two linear nodes
                        # No q-dq for extra input node of add
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
                        # note: quantize op for weights are const propagated
                        torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                    }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.linear.default,
                    (
                        torch.ops.aten.add_.Tensor
                        if inplace_add
                        else torch.ops.aten.add.Tensor
                    ),
                ]
                fq_m = self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )[-1]
                # linear, add, relu are fused
                # The other linear is quantized alone if present
                aten = torch.ops.aten
                add_op = aten.add_.Tensor if inplace_add else aten.add.Tensor
                relu_op = aten.relu_.default if inplace_relu else aten.relu.default
                expected_annotation_stat = {
                    aten.linear.default: {
                        "annotated": 2 if linear_pos == NodePosType.both else 1,
                        "is_quant_out": 1 if linear_pos == NodePosType.both else 0,
                    },
                    add_op: {"annotated": 1, "is_quant_out": 0},
                    relu_op: {"annotated": 1, "is_quant_out": 1},
                }
                self._check_annotation_stat(fq_m, expected_annotation_stat)

    @skipIfNoX86
    def test_linear_binary_unary_serials(self):
        """
        Test pattern of 2 following up linear add relu with X86InductorQuantizer.
        """
        with override_quantized_engine("x86"), torch.no_grad():
            m = TestHelperModules.SerialsLinearAddReLUModule().eval()
            example_inputs = (torch.randn(2, 16),)
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            node_occurrence = {
                # quantize_per_tensor: 1 for linear_1, 1 for linear_2/3 (shared), 1 for linear_4
                # dequantize_per_tensor: 1 for each linear
                # No q-dq for extra input node of add
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4,
                # quantize_per_channel for weights are const propagated
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 4,
            }
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.linear.default,
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.linear.default,
                torch.ops.aten.linear.default,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
            fq_m = self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )[-1]
            # Two linear nodes are quantized alone
            # The other two are fused with add and relu
            aten = torch.ops.aten
            expected_annotation_stat = {
                aten.linear.default: {
                    "annotated": 4,
                    "is_quant_out": 2,
                },
                aten.add.Tensor: {"annotated": 2, "is_quant_out": 0},
                aten.relu.default: {"annotated": 2, "is_quant_out": 2},
            }
            self._check_annotation_stat(fq_m, expected_annotation_stat)

    @skipIfTorchDynamo("very slow")
    @skipIfNoX86
    def test_qat_conv2d(self):
        """
        Test QAT pattern of conv2d_bn with X86InductorQuantizer.
        """
        with override_quantized_engine("x86"):
            m = TestHelperModules.SingleConv2dModule(with_bn=True)
            example_inputs = (torch.randn(2, 3, 16, 16),)
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config(is_qat=True)
            )
            node_occurrence = {
                # one for input and weight of the conv, one for output for the conv
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
                # note: quantize op for weights are const propagated
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                # BN should be folded into Conv
                torch.ops.aten._native_batch_norm_legit.default: 0,
            }
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            ]
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
                is_qat=True,
            )

    @skipIfTorchDynamo("very slow")
    @skipIfNoX86
    def test_qat_conv2d_unary(self):
        """
        Test QAT pattern of conv2d_bn with unary post ops (such as relu, sigmoid) with X86InductorQuantizer.
        Currently, only relu as unary post op is supported.
        """
        unary_map = {
            "relu": [torch.nn.ReLU(inplace=False), torch.ops.aten.relu.default],
            "relu_inplace": [torch.nn.ReLU(inplace=True), torch.ops.aten.relu_.default],
            "hardtanh": [
                torch.nn.Hardtanh(min_val=0.0, max_val=6.0, inplace=False),
                torch.ops.aten.hardtanh.default,
            ],
            "hardtanh_inplace": [
                torch.nn.Hardtanh(min_val=0.0, max_val=6.0, inplace=True),
                torch.ops.aten.hardtanh_.default,
            ],
            "relu6": [torch.nn.ReLU6(inplace=False), torch.ops.aten.hardtanh.default],
            "relu6_inplace": [
                torch.nn.ReLU6(inplace=True),
                torch.ops.aten.hardtanh_.default,
            ],
            "hardswish": [
                torch.nn.Hardswish(inplace=False),
                torch.ops.aten.hardswish.default,
            ],
            "hardswish_inplace": [
                torch.nn.Hardswish(inplace=True),
                torch.ops.aten.hardswish_.default,
            ],
            "swish": [torch.nn.SiLU(inplace=False), torch.ops.aten.silu.default],
            "swish_inplace": [
                torch.nn.SiLU(inplace=True),
                torch.ops.aten.silu_.default,
            ],
        }

        with override_quantized_engine("x86"):
            for unary_op in unary_map.keys():
                m = TestHelperModules.Conv2dUnaryModule(
                    unary_map[unary_op][0], with_bn=True
                )
                example_inputs = (torch.randn(2, 3, 16, 16),)
                quantizer = X86InductorQuantizer().set_global(
                    xiq.get_default_x86_inductor_quantization_config(is_qat=True)
                )
                node_occurrence = {
                    # one for input and weight of the conv, one for output for the relu
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
                    # note: quantize op for weights are const propagated
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                    # BN should be folded into Conv
                    torch.ops.aten._native_batch_norm_legit.default: 0,
                }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.conv2d.default,
                    unary_map[unary_op][1],
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                ]
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                    is_qat=True,
                )

    @skipIfTorchDynamo("very slow")
    @skipIfNoX86
    def test_qat_conv2d_binary(self):
        """
        Test qat pattern of conv2d_bn with binary post ops (such as add) with X86InductorQuantizer.
        Currently, only add as binary post op is supported.
        """
        example_inputs = (torch.randn(2, 3, 6, 6),)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config(is_qat=True)
        )
        with override_quantized_engine("x86"):
            for inplace_add in [True, False]:
                m = TestHelperModules.Conv2dAddModule(
                    inplace_add=inplace_add, with_bn=True
                )
                node_occurrence = {
                    # one for input and weight of the conv
                    # one for output for the add
                    # one for extra input node of add
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
                    # quantize_per_channel for weights are const propagated
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                    # BN should be folded into Conv
                    torch.ops.aten._native_batch_norm_legit.default: 0,
                }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.conv2d.default,
                    (
                        torch.ops.aten.add_.Tensor
                        if inplace_add
                        else torch.ops.aten.add.Tensor
                    ),
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                ]
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                    is_qat=True,
                )

    @skipIfTorchDynamo("very slow")
    @skipIfNoX86
    def test_qat_conv2d_binary2(self):
        """
        Test qat Pattern:
            tmp = bn1(conv2d_1(x))
            tmp2 = bn2(conv2d_2(tmp))
            return tmp + tmp2
        Since conv2d_1 has 2 users, we should annotate conv2d_2 for binary fusion instead of conv2d_1
        """
        example_inputs = (torch.randn(2, 3, 6, 6),)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config(is_qat=True)
        )
        inplace_add_list = [True, False]
        with override_quantized_engine("x86"), torch.no_grad():
            for inplace_add in inplace_add_list:
                m = TestHelperModules.Conv2dAddModule2(inplace_add=inplace_add)
                node_occurrence = {
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4,
                    # quantize_per_channel for weights are const propagated
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                    # BN should be folded into Conv
                    torch.ops.aten._native_batch_norm_legit.default: 0,
                }
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.conv2d.default,
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    (
                        torch.ops.aten.add_.Tensor
                        if inplace_add
                        else torch.ops.aten.add.Tensor
                    ),
                ]
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                    is_qat=True,
                )

    @skipIfTorchDynamo("very slow")
    @skipIfNoX86
    def test_qat_conv2d_binary_unary(self):
        """
        Test QAT pattern of conv2d_bn with binary + unary post ops (such as add + relu) with X86InductorQuantizer.
        Currently, only add as binary post op and relu as unary post op are supported.
        """
        example_inputs = (torch.randn(2, 3, 6, 6),)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config(is_qat=True)
        )
        with override_quantized_engine("x86"):
            m = TestHelperModules.Conv2dAddReLUModule(with_bn=True)
            node_occurrence = {
                # one for input for conv
                # one for output for the relu
                # one for extra input node of add
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
                # note: quantize op for weights are const propagated
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                # BN should be folded into Conv
                torch.ops.aten._native_batch_norm_legit.default: 0,
            }
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.add.Tensor,
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            ]
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
                is_qat=True,
            )

    @skipIfNoX86
    def test_dynamic_quant_linear(self):
        """
        Test pattern of dynamic quantization of linear with X86InductorQuantizer.
        """
        with override_quantized_engine("x86"), torch.no_grad():
            m = TestHelperModules.SelfAttnLikeModule(input_dim=64).eval()
            example_inputs = (torch.randn(1, 4, 64),)
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config(is_dynamic=True)
            )
            node_occurrence = {
                torch.ops.quantized_decomposed.choose_qparams.tensor: 1,
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
                torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
                # quantize_per_channel for weights are const propagated
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
            }
            node_list = [
                torch.ops.quantized_decomposed.choose_qparams.tensor,
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
                torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
                torch.ops.aten.linear.default,
            ]
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )

    @skipIfNoX86
    def test_qat_dynamic_quant_linear(self):
        """
        Test pattern of qat dynamic quantization of linear with X86InductorQuantizer.
        """
        with override_quantized_engine("x86"), torch.no_grad():
            m = TestHelperModules.SelfAttnLikeModule(input_dim=64).eval()
            example_inputs = (torch.randn(1, 4, 64),)
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config(
                    is_qat=True, is_dynamic=True
                )
            )
            node_occurrence = {
                torch.ops.quantized_decomposed.choose_qparams.tensor: 1,
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
                torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
                # quantize_per_channel for weights are const propagated
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
            }
            node_list = [
                torch.ops.quantized_decomposed.choose_qparams.tensor,
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
                torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
                torch.ops.aten.linear.default,
            ]
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
                is_qat=True,
            )

    @skipIfNoX86
    def test_set_module_name_qconfig(self):
        """Test case for quantizing a specific submodule by configuring `set_module_name_qconfig`.

        Expect that all linear layers within the submodule `sub` are quantized.
        """

        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 10)
                self.relu1 = torch.nn.ReLU(inplace=False)
                self.linear2 = torch.nn.Linear(10, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu1(x)
                x = self.linear2(x)
                return x

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                self.sub = Sub()

            def forward(self, x):
                x = self.linear(x)
                x = self.sub(x)
                return x

        m = M().eval()
        example_inputs = (torch.randn(3, 5),)
        # Set global to `None` and then default config for a specific submodule.
        quantizer = X86InductorQuantizer()
        quantizer.set_module_name_qconfig(
            "sub", xiq.get_default_x86_inductor_quantization_config()
        )
        node_occurrence = {
            torch.ops.aten.linear.default: 3,
            # quantize and dequantize the input of two linear layers from `sub`
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            # dequantize the weight of two linear layers from `sub`
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        node_list = [
            # first linear is not quantized
            torch.ops.aten.linear.default,
            # two  Q/DQ pairs for two linear layers from `sub`
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
        ]
        self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    @skipIfNoX86
    def test_set_module_name_qconfig_with_underscores(self) -> None:
        """Test that if a module name has an underscore, we can still quantize it."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # This module name has underscores, which can be part of a mangled name.
                self.foo_bar = torch.nn.Linear(2, 2)
                self.baz = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.baz(self.foo_bar(x))

        # Set global to no quantization and then default config for a specific submodule whose name includes an underscore.
        quantizer = X86InductorQuantizer()
        quantizer.set_module_name_qconfig(
            "foo_bar", xiq.get_default_x86_inductor_quantization_config()
        )
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = capture_pre_autograd_graph(m, example_inputs)
        m = prepare_pt2e(m, quantizer)
        # Use a linear count instead of names because the names might change, but
        # the order should be the same.
        count = 0
        for n in m.graph.nodes:
            if n.op == "call_function" and n.target == torch.ops.aten.linear.default:
                # Get the weight observer to see the per-channel vs per-tensor.
                weight_observer_node = n.args[1]
                if count == 0:
                    # for foo_bar.
                    self.assertEqual(
                        weight_observer_node.op,
                        "call_module",
                        f"The op of linear({count})'s weight_observer_node is {weight_observer_node.op} instead call_module",
                    )
                    observer_instance = getattr(m, weight_observer_node.target)
                    self.assertEqual(
                        observer_instance.qscheme, torch.per_channel_symmetric
                    )
                else:
                    # For baz it should have no observer at all.
                    self.assertNotEqual(
                        weight_observer_node.op,
                        "call_module",
                        f"The op of linear({count})'s weight_observer_node is {weight_observer_node.op} instead call_module",
                    )
                count += 1

    @skipIfNoX86
    def test_set_module_name_and_module_type_case1(self):
        """Test that set `module_name_qconfig` and `module_type_qconfig` at the same time.

        Expect that all linear layers are not quantized except the last one.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 10)
                self.linear2 = torch.nn.Linear(10, 5)
                self.sub = torch.nn.Linear(5, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sub(x)
                return x

        m = M().eval()
        example_inputs = (torch.randn(3, 5),)
        # Set `sub` with default config and then `None` for all `Linear`.
        # The config set by `set_module_name_qconfig` has higher priority than `set_module_type_qconfig`.
        quantizer = X86InductorQuantizer()
        quantizer.set_module_name_qconfig(
            "sub", xiq.get_default_x86_inductor_quantization_config()
        ).set_module_type_qconfig(torch.nn.Linear, None)

        node_occurrence = {
            torch.ops.aten.linear.default: 3,
            # quantize and dequantize the input of the last linear
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
            # dequantize the weight of the last linear
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        node_list = [
            # first and second linear are not quantized
            torch.ops.aten.linear.default,
            torch.ops.aten.linear.default,
            # last linear is quantized
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
        ]
        self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    @skipIfNoX86
    def test_set_module_name_and_module_type_case2(self):
        """Test that set `module_name_qconfig` and `module_type_qconfig` at the same time.

        Expect that all linear layers are quantized except the last one.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 10)
                self.linear2 = torch.nn.Linear(10, 5)
                self.sub = torch.nn.Linear(5, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sub(x)
                return x

        m = M().eval()
        example_inputs = (torch.randn(3, 5),)
        # Set `sub` with None and then default config for a all `Linear`.
        quantizer = X86InductorQuantizer()
        quantizer.set_module_name_qconfig("sub", None).set_module_type_qconfig(
            torch.nn.Linear, xiq.get_default_x86_inductor_quantization_config()
        )

        node_occurrence = {
            torch.ops.aten.linear.default: 3,
            # quantize and dequantize the input and output of the first and second linear
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            # dequantize the weight of the first and second linear
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        node_list = [
            # Q/DQ for first lienar
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
            # Q/DQ for second lienar
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
            # last linear is not quantized
            torch.ops.aten.linear.default,
        ]
        self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    @skipIfNoX86
    def test_set_module_name_qconfig_for_dynamic_quant(self):
        """Test that quantize a specific submodule for dynamic quantization."""

        with override_quantized_engine("x86"), torch.no_grad():
            for is_qat in [False, True]:
                m = TestHelperModules.SelfAttnLikeModule(input_dim=64).eval()
                example_inputs = (torch.randn(1, 4, 64),)
                # only quantize `q_proj` `v_proj`
                dynamic_config = xiq.get_default_x86_inductor_quantization_config(
                    is_dynamic=True, is_qat=is_qat
                )
                quantizer = (
                    X86InductorQuantizer()
                    .set_module_name_qconfig("q_proj", dynamic_config)
                    .set_module_name_qconfig("v_proj", dynamic_config)
                )
                node_occurrence = {
                    # quantize and dequantize the input
                    torch.ops.quantized_decomposed.choose_qparams.tensor: 1,
                    torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
                    # dequantize the weight of q_proj and v_proj
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                }
                node_list = [
                    # quantize and dequantize the input
                    torch.ops.quantized_decomposed.choose_qparams.tensor,
                    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
                    # q_proj
                    torch.ops.aten.linear.default,
                    # k_proj
                    torch.ops.aten.linear.default,
                    # v_proj
                    torch.ops.aten.linear.default,
                ]
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                    is_qat=is_qat,
                )

    @skipIfNoX86
    def test_set_module_name_with_mixed_configs(self):
        """Test case for setting module names with mixed static/dynamic or QAT/non-QAT configurations.

        The config for 'v_proj' will always be ignored and raise a warning.
        """
        with override_quantized_engine("x86"), torch.no_grad():
            with self.assertWarns(UserWarning) as context:
                for q_is_dynamic, v_is_dynamic, q_is_qat, v_is_qat in itertools.product(
                    [False, True], repeat=4
                ):
                    if q_is_dynamic == v_is_dynamic and q_is_qat == v_is_qat:
                        continue
                    m = TestHelperModules.SelfAttnLikeModule(input_dim=64).eval()
                    example_inputs = (torch.randn(1, 4, 64),)
                    quantizer = (
                        X86InductorQuantizer()
                        .set_module_name_qconfig(
                            "q_proj",
                            xiq.get_default_x86_inductor_quantization_config(
                                is_qat=q_is_qat, is_dynamic=q_is_dynamic
                            ),
                        )
                        .set_module_name_qconfig(
                            "v_proj",
                            xiq.get_default_x86_inductor_quantization_config(
                                is_qat=v_is_qat, is_dynamic=v_is_dynamic
                            ),
                        )
                    )
                    quant_op = (
                        torch.ops.quantized_decomposed.quantize_per_tensor.tensor
                        if q_is_dynamic
                        else torch.ops.quantized_decomposed.quantize_per_tensor.default
                    )
                    dequant_op = (
                        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor
                        if q_is_dynamic
                        else torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    )
                    node_occurrence = {
                        # quantize and dequantize the input
                        quant_op: 1,
                        dequant_op: 1,
                        # only `q_proj` was quantized, dequantize its weight
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                    }
                    node_list = [
                        # quantize and dequantize the input
                        quant_op,
                        dequant_op,
                        # q_proj
                        torch.ops.aten.linear.default,
                        # k_proj/v_proj
                        torch.ops.aten.linear.default,
                        torch.ops.aten.linear.default,
                    ]
                    self._test_quantizer(
                        m,
                        example_inputs,
                        quantizer,
                        node_occurrence,
                        node_list,
                        is_qat=q_is_qat,
                    )
                    warning_msg = (
                        "Mixed QAT and Non-QAT"
                        if q_is_qat != v_is_qat
                        else "Mixed dynamic and static"
                    )
                    self.assertTrue(
                        any(
                            warning_msg in msg
                            for msg in [str(w.message) for w in context.warnings]
                        )
                    )

    @skipIfNoX86
    def test_set_module_name_and_module_type_with_mixed_configs(self):
        """Test that set `module_name_qconfig` and `module_type_qconfig` at the same time with mixed the configs.

        Expect that only the last linear(`sub`) is quantized using static quantization.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 10)
                self.linear2 = torch.nn.Linear(10, 5)
                self.sub = torch.nn.Linear(5, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sub(x)
                return x

        m = M().eval()
        example_inputs = (torch.randn(3, 5),)
        # Set `sub` with static config and then dynamic config for a all `Linear`(ignored).
        quantizer = X86InductorQuantizer()
        quantizer.set_module_name_qconfig(
            "sub", xiq.get_default_x86_inductor_quantization_config(is_dynamic=False)
        ).set_module_type_qconfig(
            torch.nn.Linear,
            xiq.get_default_x86_inductor_quantization_config(is_dynamic=True),
        )

        node_occurrence = {
            torch.ops.aten.linear.default: 3,
            # quantize and dequantize the input of the last linear
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
            # dequantize the weight of the last linear
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        node_list = [
            # first and second linear are not quantized
            torch.ops.aten.linear.default,
            torch.ops.aten.linear.default,
            # Q/DQ pairs for the last linear
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
        ]
        self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    @skipIfNoX86
    def test_filter_conv2d_recipe(self):
        """
        Test removing conv2d from default recipe of X86InductorQuantizer.
        """
        with override_quantized_engine("x86"), torch.no_grad():
            m = TestHelperModules.Conv2dUnaryModule(torch.nn.ReLU(inplace=False)).eval()
            example_inputs = (torch.randn(2, 3, 16, 16),)
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            quantizer.set_module_type_qconfig(torch.nn.Conv2d, None)
            node_occurrence = {
                # one for input and weight of the conv
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 0,
                # note: quantize op for weights are const propagated
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 0,
            }
            node_list = [
                torch.ops.aten.conv2d.default,
                torch.ops.aten.relu.default,
            ]
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )

    @skipIfNoX86
    def test_filter_linear_recipe(self):
        """
        Test removing linear from default recipe of X86InductorQuantizer.
        """
        with override_quantized_engine("x86"), torch.no_grad():
            m = TestHelperModules.LinearUnaryModule(
                use_bias=True,
                postop=nn.ReLU,
            ).eval()
            example_inputs = (torch.randn(2, 4),)
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            quantizer.set_function_type_qconfig(torch.nn.functional.linear, None)
            node_occurrence = {
                # one for input and weight of the conv
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 0,
                # note: quantize op for weights are const propagated
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 0,
            }
            node_list = [
                torch.ops.aten.linear.default,
                torch.ops.aten.relu.default,
            ]
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )

    @skipIfNoX86
    def test_filter_maxpool2d_recipe(self):
        """
        Test removing maxpool2d from default recipe of X86InductorQuantizer.
        """
        with override_quantized_engine("x86"), torch.no_grad():
            m = TestHelperModules.Conv2dUnaryModule(torch.nn.ReLU(inplace=False)).eval()
            example_inputs = (torch.randn(2, 3, 16, 16),)
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            quantizer.set_function_type_qconfig(torch.nn.functional.max_pool2d, None)
            node_occurrence = {
                # one for input and weight of the conv
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                # note: quantize op for weights are const propagated
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
            }
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.relu.default,
                torch.ops.aten.max_pool2d.default,
            ]
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )

    @skipIfNoX86
    def test_attention_block(self):
        """
        Test pattern of Attention like Block with X86InductorQuantizer.
        """
        for annotate_matmul in [False, True]:
            with override_quantized_engine("x86"), torch.no_grad():
                m = TestHelperModules.SelfAttnLikeModule(
                    input_dim=64 * 16,
                    transpose_for_score=True,
                    num_attention_heads=16,
                    attention_head_size=64,
                ).eval()
                example_inputs = (torch.randn(2, 384, 1024),)

                m(*example_inputs)

                quantizer = X86InductorQuantizer().set_global(
                    xiq.get_default_x86_inductor_quantization_config()
                )

                if annotate_matmul:
                    quantizer.set_function_type_qconfig(
                        torch.matmul, quantizer.get_global_quantization_config()
                    )

                node_occurrence = {
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: (
                        5 if annotate_matmul else 1
                    ),
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: (
                        7 if annotate_matmul else 3
                    ),
                    # quantize_per_channel for weights are const propagated
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
                }
                if annotate_matmul:
                    node_list = [
                        torch.ops.quantized_decomposed.quantize_per_tensor.default,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                        torch.ops.aten.linear.default,
                        torch.ops.aten.view.default,
                        torch.ops.aten.permute.default,
                        torch.ops.quantized_decomposed.quantize_per_tensor.default,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                        torch.ops.aten.matmul.default,
                        torch.ops.aten.div.Tensor,
                        torch.ops.aten.softmax.int,
                    ]
                else:
                    node_list = [
                        torch.ops.quantized_decomposed.quantize_per_tensor.default,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                        torch.ops.aten.linear.default,
                        torch.ops.aten.view.default,
                        torch.ops.aten.permute.default,
                        torch.ops.aten.matmul.default,
                        torch.ops.aten.div.Tensor,
                        torch.ops.aten.softmax.int,
                    ]
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )
