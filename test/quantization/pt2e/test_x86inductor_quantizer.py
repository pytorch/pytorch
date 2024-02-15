# Owner(s): ["oncall: quantization"]
import copy
import torch
import torch.nn as nn
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
)
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    skipIfNoX86,
    skipIfNoInductorSupport,
)
from torch.testing._internal.common_utils import skipIfTorchDynamo
from torch.testing._internal.common_quantized import override_quantized_engine
from enum import Enum
import itertools
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization import ObserverBase
from torch._export import capture_pre_autograd_graph

class Conv2DType(Enum):
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
            self.conv = nn.Conv2d(3, 6, (2, 2), stride=(1, 1), padding=(1, 1), bias=use_bias)
            self.post_op = post_op
            self.bn = torch.nn.BatchNorm2d(6)
            self.with_bn = with_bn

        def forward(self, x):
            x = self.conv(x)
            if self.with_bn:
                x = self.bn(x)
            x = self.post_op(x)
            return x

    class Conv2dAddModule(torch.nn.Module):
        def __init__(self,
                     inplace_add: bool = False,
                     conv2d_type: Conv2DType = Conv2DType.left,
                     use_bias: bool = False,
                     with_bn: bool = False,
                     ) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=use_bias
            )
            self.conv2 = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=use_bias
            )
            self.relu = nn.ReLU()
            self.inplace_add = inplace_add
            self.conv2d_type = conv2d_type
            self.bn = torch.nn.BatchNorm2d(3)
            self.with_bn = with_bn

        def forward(self, x):
            if self.conv2d_type == Conv2DType.left:
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
            elif self.conv2d_type == Conv2DType.right:
                if self.inplace_add:
                    tmp = self.relu(x)
                    tmp += self.conv(x)
                    return tmp
                else:
                    return self.relu(x) + self.conv(x)
            elif self.conv2d_type == Conv2DType.both:
                if self.inplace_add:
                    tmp = self.conv(x)
                    tmp += self.conv2(x)
                    return tmp
                else:
                    return self.conv(x) + self.conv2(x)

    class Conv2dAddReLUModule(torch.nn.Module):
        def __init__(self,
                     inplace_add: bool = False,
                     conv2d_type: Conv2DType = Conv2DType.left,
                     inplace_relu: bool = False,
                     use_bias: bool = False,
                     with_bn: bool = False,
                     ) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=use_bias
            )
            self.conv2 = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=use_bias
            )
            self.relu = nn.ReLU()
            self.inplace_add = inplace_add
            self.conv2d_type = conv2d_type
            self.relu2 = nn.ReLU(inplace=inplace_relu)
            self.bn = torch.nn.BatchNorm2d(3)
            self.with_bn = with_bn

        def forward(self, x):
            if self.conv2d_type == Conv2DType.left:
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
            elif self.conv2d_type == Conv2DType.right:
                if self.inplace_add:
                    tmp = self.relu(x)
                    tmp += self.conv(x)
                    return self.relu2(tmp)
                else:
                    return self.relu2(self.relu(x) + self.conv(x))
            elif self.conv2d_type == Conv2DType.both:
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
        """ Serials of 2 Conv2d -> Add -> ReLU Pattern.
        """
        def __init__(self, ) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True
            )
            self.conv2 = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True
            )
            self.conv3 = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True
            )
            self.conv4 = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True
            )
            self.relu = nn.ReLU()
            self.relu2 = nn.ReLU()

        def forward(self, x):
            x1 = self.conv(x)
            res1 = self.relu(self.conv2(x1) + self.conv3(x1))
            res2 = self.relu2(self.conv4(res1) + res1)
            return res2

    class Conv2dCatMaxpool2d(torch.nn.Module):
        def __init__(self,):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 7, bias=True, stride=2, padding=3, dilation=1)
            self.conv2 = torch.nn.Conv2d(3, 16, 7, bias=True, stride=2, padding=3, dilation=1)
            self.relu = torch.nn.ReLU()
            self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
            self.conv3 = torch.nn.Conv2d(32, 32, 7, bias=True, stride=2, padding=3, dilation=1)

        def forward(self, x):
            temp1 = self.relu(self.conv(x))
            temp2 = self.conv2(x + 1)
            temp3 = torch.cat((temp1, temp2), 1)
            temp4 = self.maxpool(temp3)
            temp5 = self.conv3(temp4)
            return temp5

    class Conv2dAvgPool2d(torch.nn.Module):
        def __init__(self,):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 7, bias=True, stride=2, padding=3, dilation=1)
            self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)

        def forward(self, x):
            temp1 = self.avgpool(self.conv(x))
            return temp1

    class Conv2dCatSameInputs(torch.nn.Module):
        def __init__(self,):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 7, bias=True, stride=2, padding=3, dilation=1)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            temp1 = self.relu(self.conv(x))
            temp3 = torch.cat((temp1, temp1), 1)
            return temp3

    class Conv2dCatSingleInput(torch.nn.Module):
        def __init__(self,):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 7, bias=True, stride=2, padding=3, dilation=1)
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
        def __init__(self, use_bias, postop, inplace_postop) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=use_bias)
            self.postop = postop(inplace=inplace_postop)

        def forward(self, x):
            return self.postop(self.linear(x))

    class Conv2dAddModule2(torch.nn.Module):
        def __init__(self,
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
        def __init__(self, input_dim) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.q_proj = nn.Linear(input_dim, input_dim, bias=False)
            self.k_proj = nn.Linear(input_dim, input_dim, bias=False)
            self.v_proj = nn.Linear(input_dim, input_dim, bias=False)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            scores = torch.bmm(q, k.transpose(1, 2)) / (self.input_dim ** 0.5)
            attention = self.softmax(scores)
            weighted = torch.bmm(attention, v)
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
            "hardtanh": [torch.nn.Hardtanh(min_val=0.0, max_val=6.0, inplace=False), torch.ops.aten.hardtanh.default],
            "hardtanh_inplace": [torch.nn.Hardtanh(min_val=0.0, max_val=6.0, inplace=True), torch.ops.aten.hardtanh_.default],
            "relu6": [torch.nn.ReLU6(inplace=False), torch.ops.aten.hardtanh.default],
            "relu6_inplace": [torch.nn.ReLU6(inplace=True), torch.ops.aten.hardtanh_.default],
            "hardswish": [torch.nn.Hardswish(inplace=False), torch.ops.aten.hardswish.default],
            "hardswish_inplace": [torch.nn.Hardswish(inplace=True), torch.ops.aten.hardswish_.default]
        }
        use_bias_list = [True, False]
        with override_quantized_engine("x86"), torch.no_grad():
            for unary_op, use_bias in itertools.product(unary_map.keys(), use_bias_list):
                m = TestHelperModules.Conv2dUnaryModule(unary_map[unary_op][0], use_bias=use_bias).eval()
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
        conv2d_type_list = [Conv2DType.left, Conv2DType.both]
        example_inputs = (torch.randn(2, 3, 6, 6),)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        with override_quantized_engine("x86"), torch.no_grad():
            for conv2d_type in conv2d_type_list:
                m = TestHelperModules.Conv2dAddModule(conv2d_type=conv2d_type).eval()
                if conv2d_type != Conv2DType.both:
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
                    torch.ops.aten.add_.Tensor if inplace_add else torch.ops.aten.add.Tensor,
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
        conv2d_type_list = [Conv2DType.left, Conv2DType.both]
        example_inputs = (torch.randn(2, 3, 6, 6),)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        with override_quantized_engine("x86"), torch.no_grad():
            for conv2d_type in conv2d_type_list:
                m = TestHelperModules.Conv2dAddReLUModule(
                    conv2d_type=conv2d_type,
                ).eval()
                if conv2d_type != Conv2DType.both:
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
            quantizer = X86InductorQuantizer().set_global(xiq.get_default_x86_inductor_quantization_config())
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
            if (
                node.op == "call_function"
                and node.target is single_op
            ):
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
            TestHelperModules.Conv2dSingleOpPowModule(nn.AdaptiveAvgPool2d((1, 1))).eval(),
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
            TestHelperModules.Conv2dSingleOpPowModule(lambda x: torch.flatten(x, 1)).eval(),
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
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.cat.default
            ):
                cat_act_obs0 = getattr(
                    prepare_model, node.all_input_nodes[0].target
                )
                cat_act_obs1 = getattr(
                    prepare_model, node.all_input_nodes[1].target
                )
                cat_out_obs = getattr(
                    prepare_model, next(iter(node.users)).target
                )
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
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.cat.default
            ):
                cat_act_obs0 = getattr(
                    prepare_model, node.args[0][0].target
                )
                cat_act_obs1 = getattr(
                    prepare_model, node.args[0][1].target
                )
                cat_out_obs = getattr(
                    prepare_model, next(iter(node.users)).target
                )
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
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.cat.default
            ):
                cat_act_obs0 = getattr(
                    prepare_model, node.args[0][0].target
                )
                cat_out_obs = getattr(
                    prepare_model, next(iter(node.users)).target
                )
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
                output_obs_of_conv = getattr(prepare_model, next(iter(conv_node.users)).target)
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
            nn.LeakyReLU: [torch.ops.aten.leaky_relu_.default, torch.ops.aten.leaky_relu.default],
        }
        with override_quantized_engine("x86"), torch.no_grad():
            for use_bias, inplace, postop in cases:
                m = TestHelperModules.LinearUnaryModule(use_bias=use_bias, postop=postop, inplace_postop=inplace).eval()
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
            "hardtanh": [torch.nn.Hardtanh(min_val=0.0, max_val=6.0, inplace=False), torch.ops.aten.hardtanh.default],
            "hardtanh_inplace": [torch.nn.Hardtanh(min_val=0.0, max_val=6.0, inplace=True), torch.ops.aten.hardtanh_.default],
            "relu6": [torch.nn.ReLU6(inplace=False), torch.ops.aten.hardtanh.default],
            "relu6_inplace": [torch.nn.ReLU6(inplace=True), torch.ops.aten.hardtanh_.default],
            "hardswish": [torch.nn.Hardswish(inplace=False), torch.ops.aten.hardswish.default],
            "hardswish_inplace": [torch.nn.Hardswish(inplace=True), torch.ops.aten.hardswish_.default]
        }

        with override_quantized_engine("x86"):
            for unary_op in unary_map.keys():
                m = TestHelperModules.Conv2dUnaryModule(unary_map[unary_op][0], with_bn=True)
                example_inputs = (torch.randn(2, 3, 16, 16),)
                quantizer = X86InductorQuantizer().set_global(
                    xiq.get_default_x86_inductor_quantization_config(is_qat=True)
                )
                node_occurrence = {
                    # one for input and weight of the conv, one for output for the relu
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
                m = TestHelperModules.Conv2dAddModule(inplace_add=inplace_add, with_bn=True)
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
                    torch.ops.aten.add_.Tensor if inplace_add else torch.ops.aten.add.Tensor,
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
                    torch.ops.aten.add_.Tensor if inplace_add else torch.ops.aten.add.Tensor,
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
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
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
                    is_qat=True,
                    is_dynamic=True
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
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
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
