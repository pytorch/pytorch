# Owner(s): ["oncall: quantization"]
import copy
import operator
import unittest
from typing import Any, List, Optional, Tuple

import torch
import torch._dynamo as torchdynamo
from torch.ao.ns.fx.utils import compute_sqnr
from torch.ao.quantization import (
    FusedMovingAvgObsFakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    observer,
    QConfigMapping,
)
from torch.ao.quantization._pt2e.quantizer import (
    OperatorConfig,
    QNNPackQuantizer,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
from torch.ao.quantization._quantize_pt2e import (
    _convert_to_reference_decomposed_fx,
    convert_pt2e,
    prepare_pt2e_quantizer,
    prepare_qat_pt2e_quantizer,
)
from torch.ao.quantization.backend_config import get_qnnpack_backend_config

from torch.ao.quantization.qconfig import (
    default_per_channel_symmetric_qnnpack_qat_qconfig,
    default_per_channel_symmetric_qnnpack_qconfig,
    default_symmetric_qnnpack_qat_qconfig,
)
from torch.ao.quantization.quantize_fx import (
    convert_to_reference_fx,
    prepare_fx,
    prepare_qat_fx,
)
from torch.fx import Node
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
)
from torch.testing._internal.common_quantized import override_quantized_engine


@skipIfNoQNNPACK
class TestQuantizePT2E(QuantizationTestCase):
    def test_simple_quantizer(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.convolution.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_weight_observer,
                        )
                        bias_qspec = QuantizationSpec(
                            dtype=torch.float32,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

            @classmethod
            def get_supported_operators(cls) -> List[OperatorConfig]:
                pass

        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        m = prepare_pt2e_quantizer(m, BackendAQuantizer())
        m(*example_inputs)
        m = convert_pt2e(m)
        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 3,
        }
        node_list = [
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.aten.convolution.default),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_max_pool2d_quantizer(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 1)
                self.pool = torch.nn.MaxPool2d(1, 1)

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                return x

        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                act_qspec = QuantizationSpec(
                    dtype=torch.uint8,
                    quant_min=0,
                    quant_max=255,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_observer,
                )
                weight_qspec = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-128,
                    quant_max=127,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_weight_observer,
                )
                bias_qspec = QuantizationSpec(
                    dtype=torch.float32,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                )
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.convolution.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )
                    if (
                        node.op == "call_function"
                        and node.target == operator.getitem
                        and node.args[1] == 0
                    ):
                        getitem_node = node
                        maxpool_node = getitem_node.args[0]
                        input_act = maxpool_node.args[0]
                        assert isinstance(input_act, Node)
                        maxpool_node.meta[
                            "quantization_annotation"
                        ] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                            },
                            _annotated=True,
                        )
                        getitem_node.meta[
                            "quantization_annotation"
                        ] = QuantizationAnnotation(
                            output_qspec=SharedQuantizationSpec(
                                (input_act, maxpool_node)
                            ),
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

            @classmethod
            def get_supported_operators(cls) -> List[OperatorConfig]:
                pass

        m = M()
        m_pt2e = copy.deepcopy(m)
        x = torch.rand(1, 2, 14, 14)
        example_inputs = (x,)
        # program capture
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        m = prepare_pt2e_quantizer(m, BackendAQuantizer())
        m(*example_inputs)
        m = convert_pt2e(m)
        node_occurrence = {
            # two for input of maxpool
            # one for input for maxpool
            # one for output of maxpool
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 4,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 4,
        }
        node_list = [
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.aten.convolution.default),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.aten.max_pool2d_with_indices.default),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_qnnpack_quantizer_conv(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

        quantizer = QNNPackQuantizer()
        operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )

        m = prepare_pt2e_quantizer(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 2,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 2,
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 1,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 1,
        }
        node_list = [
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel),
            ns.call_function(torch.ops.aten.convolution.default),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_qnnpack_quantizer_linear(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(8, 16, bias=False)
                self.linear2 = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.linear2(self.linear1(x))

        import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

        quantizer = QNNPackQuantizer()
        operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        m_eager = M().eval()

        # Test with 2d inputs
        example_inputs_2d = (torch.randn(9, 8),)
        example_inputs_3d = (torch.randn(9, 10, 8),)
        example_inputs_4d = (torch.randn(9, 10, 11, 8),)
        for example_inputs in [example_inputs_2d, example_inputs_3d, example_inputs_4d]:
            # program capture
            m = m_eager
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            m = prepare_pt2e_quantizer(m, quantizer)
            # Calibrate
            m(*example_inputs)
            m = convert_pt2e(m)
            pt2_quant_output = m(*example_inputs)
            node_occurrence = {
                # input and output are using quantize_per_tensor and weight is using quantize_per_channel
                ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
                ns.call_function(
                    torch.ops.quantized_decomposed.dequantize_per_tensor
                ): 3,
                ns.call_function(
                    torch.ops.quantized_decomposed.quantize_per_channel
                ): 2,
                ns.call_function(
                    torch.ops.quantized_decomposed.dequantize_per_channel
                ): 2,
            }
            self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
            qconfig = default_per_channel_symmetric_qnnpack_qconfig
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            backend_config = get_qnnpack_backend_config()
            m_copy = copy.deepcopy(m_eager)
            m_fx = prepare_fx(
                m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
            )
            m_fx(*example_inputs)
            m_fx = _convert_to_reference_decomposed_fx(
                m_fx, backend_config=backend_config
            )
            m_fx, _ = torchdynamo.export(
                m_fx,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )
            fx_quant_output = m_fx(*example_inputs)
            self.assertTrue(torch.allclose(fx_quant_output, pt2_quant_output))

    def test_qnnpack_quantizer_conv_linear_no_permute(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3)
                self.linear1 = torch.nn.Linear(64, 8, bias=False)
                self.linear2 = torch.nn.Linear(8, 8)

            def forward(self, x):
                conv_out = self.conv(x)
                reshape_out = torch.reshape(conv_out, (2, 64))
                return self.linear2(self.linear1(reshape_out))

        import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

        quantizer = QNNPackQuantizer()
        operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        m_eager = M().eval()

        # Test with 2d inputs
        example_inputs = (torch.randn(2, 3, 4, 4),)
        # program capture
        m = m_eager
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
            tracing_mode="real",
        )

        m = prepare_pt2e_quantizer(m, quantizer)
        # Calibrate
        m(*example_inputs)
        m = convert_pt2e(m)
        pt2_quant_output = m(*example_inputs)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 5,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 5,
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 3,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 3,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        backend_config = get_qnnpack_backend_config()
        m_copy = copy.deepcopy(m_eager)
        m_fx = prepare_fx(
            m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
        )
        m_fx(*example_inputs)
        m_fx = _convert_to_reference_decomposed_fx(m_fx, backend_config=backend_config)
        fx_quant_output = m_fx(*example_inputs)
        self.assertTrue(torch.allclose(fx_quant_output, pt2_quant_output))

    @unittest.skip(
        "Skip due to linear traces into a different pattern. See test comment."
    )
    def test_qnnpack_quantizer_conv_linear(self):
        """
        This test fails because linear decompositon changes due to the presence of
        permute node. In the below linear 1 is decomposed as
        %t_default : [#users=1] = call_function[target=torch.ops.aten.t.default](args = (%_param_constant2,), kwargs = {})
        %clone_default : [#users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_default,), kwargs = {memory_format: torch.contiguous_format})  # noqa: B950
        %_unsafe_view_default : [#users=1] = call_function[target=torch.ops.aten._unsafe_view.default](args = (%clone_default, [8, 16]), kwargs = {})  # noqa: B950
        %mm_default : [#users=1] = call_function[target=torch.ops.aten.mm.default](args = (%_unsafe_view_default, %t_default), kwargs = {})  # noqa: B950
        %view_default : [#users=1] = call_function[target=torch.ops.aten.view.default](args = (%mm_default, [2, 2, 2, 8]), kwargs = {})  # noqa: B950

        Note the presence of cline and unsafe_view. This is due to permute
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3)
                self.linear1 = torch.nn.Linear(16, 8, bias=False)
                self.linear2 = torch.nn.Linear(8, 8)

            def forward(self, x):
                conv_out = self.conv(x)
                permute_out = torch.permute(conv_out, (0, 2, 3, 1))
                return self.linear2(self.linear1(permute_out))

        import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

        quantizer = QNNPackQuantizer()
        operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        m_eager = M().eval()

        # Test with 2d inputs
        example_inputs = (torch.randn(2, 3, 4, 4),)
        # program capture
        m = m_eager
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
            tracing_mode="real",
        )

        m = prepare_pt2e_quantizer(m, quantizer)
        # Calibrate
        m(*example_inputs)
        m = convert_pt2e(m)
        pt2_quant_output = m(*example_inputs)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 3,
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 2,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 2,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        backend_config = get_qnnpack_backend_config()
        m_copy = copy.deepcopy(m)
        m_fx = prepare_fx(
            m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
        )
        m_fx = convert_to_reference_fx(m_fx, backend_config=backend_config)
        fx_quant_output = m_fx(*example_inputs)
        self.assertTrue(torch.allclose(fx_quant_output, pt2_quant_output))

    def test_qnnpack_quantizer_obs_sharing_ops(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.hardtanh = torch.nn.Hardtanh()
                self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.conv(x)
                x = self.adaptive_avg_pool2d(x)
                x = self.hardtanh(x)
                x = torch.mean(x)
                return x

        import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

        quantizer = QNNPackQuantizer()
        operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )

        m = prepare_pt2e_quantizer(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 5,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 5,
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 1,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 1,
        }
        node_list = [
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel),
            ns.call_function(torch.ops.aten.convolution.default),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.aten.mean.dim),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.aten.hardtanh.default),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.aten.mean.default),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_prepare_qat_conv_bn_fusion(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        example_inputs = (torch.randn(1, 3, 5, 5),)
        self._verify_symmetric_qnnpack_qat_graph(
            M(), example_inputs, is_per_channel=False, has_relu=False
        )
        self._verify_symmetric_qnnpack_qat_graph(
            M(), example_inputs, is_per_channel=True, has_relu=False
        )

    def test_prepare_qat_conv_bn_fusion_constant_args(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3, stride=(2, 2), padding=(4, 4))
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        example_inputs = (torch.randn(1, 3, 5, 5),)
        # stride, padding, dilation, transposed, output_padding, groups
        conv_args = ((2, 2), (4, 4), (1, 1), False, (0, 0), 1)
        self._verify_symmetric_qnnpack_qat_graph(
            M(),
            example_inputs,
            is_per_channel=False,
            has_relu=False,
            expected_conv_constant_args=conv_args,
        )
        self._verify_symmetric_qnnpack_qat_graph(
            M(),
            example_inputs,
            is_per_channel=True,
            has_relu=False,
            expected_conv_constant_args=conv_args,
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M(), example_inputs, is_per_channel=False
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M(), example_inputs, is_per_channel=True
        )

    def test_prepare_qat_conv_bn_fusion_no_conv_bias(self):
        class M1(torch.nn.Module):
            """
            Single conv + BN with no conv bias.
            """

            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                return x

        class M2(torch.nn.Module):
            """
            Mixed conv + BN with and without conv bias.
            """

            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3, bias=True)
                self.bn2 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                return x

        example_inputs = (torch.randn(3, 3, 5, 5),)
        self._verify_symmetric_qnnpack_qat_graph(
            M1(), example_inputs, is_per_channel=False, has_relu=False, has_bias=False
        )
        self._verify_symmetric_qnnpack_qat_graph(
            M1(), example_inputs, is_per_channel=True, has_relu=False, has_bias=False
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M1(), example_inputs, is_per_channel=False
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M1(), example_inputs, is_per_channel=True
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M2(), example_inputs, is_per_channel=False
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M2(), example_inputs, is_per_channel=True
        )

    def test_prepare_qat_conv_bn_relu_fusion(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        example_inputs = (torch.randn(1, 3, 5, 5),)
        self._verify_symmetric_qnnpack_qat_graph(
            M(), example_inputs, is_per_channel=False, has_relu=True
        )
        self._verify_symmetric_qnnpack_qat_graph(
            M(), example_inputs, is_per_channel=True, has_relu=True
        )

    def test_prepare_qat_conv_bn_fusion_getitem_placeholder(self):
        """
        Test this special case seen in resnet18:

          maxpool -> maxpool_getitem -> conv -> bn -> conv_bn_getitem

        We want the metadata to be copied from the `conv_bn_getitem` node, not `maxpool_getitem`.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.maxpool = torch.nn.MaxPool2d(kernel_size=1)
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.maxpool(x)
                x = self.conv(x)
                x = self.bn(x)
                return x

        def _get_getitem_nodes(m: torch.fx.GraphModule):
            """
            Return a 2-tuple of (maxpool_getitem_node, conv_bn_getitem_node) from the graph.
            """
            maxpool_getitem_node, conv_bn_getitem_node = None, None
            for node in m.graph.nodes:
                if node.target != operator.getitem:
                    continue
                if (
                    node.args[0].target
                    == torch.ops.aten.max_pool2d_with_indices.default
                ):
                    maxpool_getitem_node = node
                elif (
                    node.args[0].target
                    == torch.ops.aten._native_batch_norm_legit.default
                ):
                    conv_bn_getitem_node = node
                else:
                    raise ValueError("Unexpected getitem node ", node, node.args)
            assert (
                maxpool_getitem_node is not None
            ), "did not find maxpool getitem node, bad test setup"
            assert (
                conv_bn_getitem_node is not None
            ), "did not find conv bn getitem node, bad test setup"
            return (maxpool_getitem_node, conv_bn_getitem_node)

        # Program capture
        example_inputs = (torch.randn(1, 3, 5, 5),)
        m, guards = torchdynamo.export(
            M(),
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        m.graph.eliminate_dead_code()
        m.recompile()
        (_, original_conv_bn_getitem_node) = _get_getitem_nodes(m)

        # Prepare QAT
        import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

        quantizer = QNNPackQuantizer()
        quantizer.set_global(
            qq.get_symmetric_quantization_config(is_per_channel=False, is_qat=True)
        )
        m = prepare_qat_pt2e_quantizer(m, quantizer)
        (maxpool_getitem_node, conv_bn_getitem_node) = _get_getitem_nodes(m)

        # Verify that the metadata was copied from `conv_bn_getitem`, not `maxpool_getitem`
        original_conv_bn_getitem_meta = original_conv_bn_getitem_node.meta[
            "quantization_annotation"
        ]
        maxpool_getitem_meta = maxpool_getitem_node.meta["quantization_annotation"]
        conv_bn_getitem_meta = conv_bn_getitem_node.meta["quantization_annotation"]
        self.assertEqual(conv_bn_getitem_meta, original_conv_bn_getitem_meta)
        self.assertNotEqual(conv_bn_getitem_meta, maxpool_getitem_meta)

    def _verify_symmetric_qnnpack_qat_graph(
        self,
        m: torch.fx.GraphModule,
        example_inputs: Tuple[Any, ...],
        is_per_channel: bool,
        has_relu: bool,
        has_bias: bool = True,
        expected_conv_constant_args: Optional[Tuple[Any, ...]] = None,
    ):
        """
        Verify that the graph module matches the fused QAT [conv - bn (- relu)] pattern
        with fake quantizes inserted into the correct places.
        # TODO: also verify that metadata is copied over to the new nodes.
        """
        import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

        quantizer = QNNPackQuantizer()
        quantizer.set_global(
            qq.get_symmetric_quantization_config(is_per_channel, is_qat=True)
        )
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
            tracing_mode="real",
        )
        m = prepare_qat_pt2e_quantizer(m, quantizer)
        m(*example_inputs)

        # Verify: getitem output activation fake quantize
        output_node = list(m.graph.nodes)[-1]
        output_fq_node = output_node.args[0][0]
        self.assertTrue(output_fq_node.target.startswith("activation_post_process_"))
        output_fq_mod = getattr(m, output_fq_node.target)
        self.assertEqual(type(output_fq_mod), FusedMovingAvgObsFakeQuantize)
        self.assertEqual(
            type(output_fq_mod.activation_post_process), MovingAverageMinMaxObserver
        )
        self.assertEqual(output_fq_mod.dtype, torch.qint8)
        self.assertEqual(output_fq_mod.quant_min, -128)
        self.assertEqual(output_fq_mod.quant_max, 127)

        # Verify: getitem(bn, 0) or relu(getitem(bn, 0))
        if has_relu:
            relu_node = output_fq_node.args[0]
            getitem_node = relu_node.args[0]
            self.assertEqual(relu_node.target, torch.ops.aten.relu.default)
        else:
            relu_node = None
            getitem_node = output_fq_node.args[0]
        bn_node = getitem_node.args[0]
        self.assertEqual(getitem_node.target, operator.getitem)
        self.assertEqual(
            bn_node.target, torch.ops.aten._native_batch_norm_legit.default
        )

        # Verify: conv / scale_factor.reshape [+ bias.reshape]
        if has_bias:
            add_bias_node = bn_node.args[0]
            (div_scale_factor_node, bias_reshape_node) = add_bias_node.args
            self.assertEqual(add_bias_node.target, torch.ops.aten.add.Tensor)
            self.assertEqual(bias_reshape_node.target, torch.ops.aten.view.default)
        else:
            div_scale_factor_node = bn_node.args[0]
        (conv_node, scale_factor_reshape_node) = div_scale_factor_node.args
        self.assertEqual(div_scale_factor_node.target, torch.ops.aten.div.Tensor)
        self.assertEqual(conv_node.target, torch.ops.aten.convolution.default)
        self.assertEqual(scale_factor_reshape_node.target, torch.ops.aten.view.default)

        # Verify: conv constant args
        if expected_conv_constant_args is not None:
            assert (
                len(expected_conv_constant_args) == 6
            ), "wrong num conv args, bad test setup"
            for i in range(6):
                self.assertEqual(conv_node.args[i + 3], expected_conv_constant_args[i])

        # Verify: conv input activation fake quantize
        conv_input_fq_node = conv_node.args[0]
        conv_input_node = conv_input_fq_node.args[0]
        self.assertTrue(
            conv_input_fq_node.target.startswith("activation_post_process_")
        )
        conv_input_fq_mod = getattr(m, conv_input_fq_node.target)
        self.assertEqual(type(conv_input_fq_mod), FusedMovingAvgObsFakeQuantize)
        self.assertEqual(
            type(conv_input_fq_mod.activation_post_process), MovingAverageMinMaxObserver
        )
        self.assertEqual(conv_input_fq_mod.dtype, torch.qint8)
        self.assertEqual(conv_input_fq_mod.quant_min, -128)
        self.assertEqual(conv_input_fq_mod.quant_max, 127)
        self.assertTrue(conv_input_node.op, "placeholder")

        # Verify: conv weight fake quantize
        conv_weight_fq_node = conv_node.args[1]
        self.assertTrue(
            conv_weight_fq_node.target.startswith("activation_post_process_")
        )
        conv_weight_fq_mod = getattr(m, conv_weight_fq_node.target)
        if is_per_channel:
            expected_weight_observer_type = MovingAveragePerChannelMinMaxObserver
        else:
            expected_weight_observer_type = MovingAverageMinMaxObserver
        self.assertEqual(type(conv_weight_fq_mod), FusedMovingAvgObsFakeQuantize)
        self.assertEqual(
            type(conv_weight_fq_mod.activation_post_process),
            expected_weight_observer_type,
        )
        self.assertEqual(conv_weight_fq_mod.dtype, torch.qint8)
        self.assertEqual(conv_weight_fq_mod.quant_min, -127)
        self.assertEqual(conv_weight_fq_mod.quant_max, 127)

        # Verify: conv(fq(input), fq(weight * scale_factor.reshape), zero_bias)
        zero_bias_node = conv_node.args[2]
        mul_weight_scale_factor_node = conv_weight_fq_node.args[0]
        (
            conv_weight_fq_node,
            scale_factor_reshape_node,
        ) = mul_weight_scale_factor_node.args
        if has_bias:
            self.assertEqual(zero_bias_node.target, torch.ops.aten.zeros_like.default)
        else:
            self.assertTrue(zero_bias_node is None)
        self.assertEqual(mul_weight_scale_factor_node.target, torch.ops.aten.mul.Tensor)
        self.assertEqual(scale_factor_reshape_node.target, torch.ops.aten.view.default)

        # Verify: scale_factor = bn_weight / sqrt(bn_running_var + eps)
        scale_factor_node = scale_factor_reshape_node.args[0]
        (bn_weight_node, sqrt_node) = scale_factor_node.args
        bn_running_var_add_node = sqrt_node.args[0]
        (bn_running_var_node, eps) = bn_running_var_add_node.args
        self.assertEqual(scale_factor_node.target, torch.ops.aten.div.Tensor)
        self.assertTrue("param_constant" in bn_weight_node.target)
        self.assertEqual(sqrt_node.target, torch.ops.aten.sqrt.default)
        self.assertEqual(bn_running_var_add_node.target, torch.ops.aten.add.Tensor)
        self.assertTrue("tensor_constant" in bn_running_var_node.target)
        self.assertEqual(eps, 1e-5)

    # TODO: merge these numerics tests with the graph tests above
    def test_prepare_qat_conv_bn_numerics(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        example_inputs = (torch.randn(1, 3, 5, 5),)
        self._verify_symmetric_qnnpack_qat_numerics(
            M(), example_inputs, is_per_channel=False
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M(), example_inputs, is_per_channel=True
        )

    def test_prepare_qat_conv_bn_relu_numerics(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        example_inputs = (torch.randn(1, 3, 5, 5),)
        self._verify_symmetric_qnnpack_qat_numerics(
            M(), example_inputs, is_per_channel=False
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M(), example_inputs, is_per_channel=True
        )

    def _verify_symmetric_qnnpack_qat_numerics(
        self,
        model: torch.nn.Module,
        example_inputs: Tuple[Any, ...],
        is_per_channel: bool,
        verify_convert: bool = False,
    ):
        """
        Helper method to verify that the QAT numerics for PT2E quantization match those of
        FX graph mode quantization for symmetric qnnpack.
        """
        # PT2 export
        import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

        model_pt2e = copy.deepcopy(model)
        quantizer = QNNPackQuantizer()
        quantizer.set_global(
            qq.get_symmetric_quantization_config(
                is_per_channel=is_per_channel, is_qat=True
            )
        )
        model_pt2e, guards = torchdynamo.export(
            model_pt2e,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        model_pt2e = prepare_qat_pt2e_quantizer(model_pt2e, quantizer)
        after_prepare_result_pt2e = model_pt2e(*example_inputs)

        # FX
        # Note: In order to match the PT2E numerics exactly, we need to feed the
        # example inputs to the model once before calling prepare, since this is
        # what torchdynamo.export does. Otherwise, the BN running mean and variance
        # would diverge in the two flows and this test would fail. For more detail,
        # see https://github.com/pytorch/pytorch/issues/95900.
        model_fx = copy.deepcopy(model)
        model_fx(*example_inputs)
        if is_per_channel:
            default_qconfig = default_per_channel_symmetric_qnnpack_qat_qconfig
        else:
            default_qconfig = default_symmetric_qnnpack_qat_qconfig
        qconfig_mapping = QConfigMapping().set_global(default_qconfig)
        backend_config = get_qnnpack_backend_config()
        model_fx = prepare_qat_fx(
            model_fx, qconfig_mapping, example_inputs, backend_config=backend_config
        )
        after_prepare_result_fx = model_fx(*example_inputs)

        # Verify that numerics match
        self.assertEqual(after_prepare_result_pt2e, after_prepare_result_fx)

        if verify_convert:
            model_pt2e = convert_pt2e(model_pt2e)
            quant_result_pt2e = model_pt2e(*example_inputs)

            model_fx = _convert_to_reference_decomposed_fx(
                model_fx, backend_config=backend_config
            )
            quant_result_fx = model_fx(*example_inputs)
            self.assertEqual(after_prepare_result_pt2e, after_prepare_result_fx)

    def test_convert_qat_conv_bn_numerics(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        example_inputs = (torch.randn(1, 3, 5, 5),)
        self._verify_symmetric_qnnpack_qat_numerics(
            M(), example_inputs, is_per_channel=False
        )
        # TODO: enable in a separate PR
        # self._verify_symmetric_qnnpack_qat_numerics(M(), example_inputs, is_per_channel=True)


class TestQuantizePT2EModels(QuantizationTestCase):
    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    def test_resnet18_with_quantizer_api(self):
        import torchvision

        with override_quantized_engine("qnnpack"):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = torchvision.models.resnet18().eval()
            m_copy = copy.deepcopy(m)
            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
            )

            before_fusion_result = m(*example_inputs)
            import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

            quantizer = QNNPackQuantizer()
            operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
            quantizer.set_global(operator_config)
            m = prepare_pt2e_quantizer(m, quantizer)
            # checking that we inserted observers correctly for maxpool operator (input and
            # output share observer instance)
            self.assertEqual(
                id(m.activation_post_process_3), id(m.activation_post_process_2)
            )
            after_prepare_result = m(*example_inputs)
            m = convert_pt2e(m)

            after_quant_result = m(*example_inputs)

            # comparing with existing fx graph mode quantization reference flow
            qconfig = default_per_channel_symmetric_qnnpack_qconfig
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            backend_config = get_qnnpack_backend_config()
            m_fx = prepare_fx(
                m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
            )
            after_prepare_result_fx = m_fx(*example_inputs)
            m_fx = convert_to_reference_fx(m_fx, backend_config=backend_config)

            after_quant_result_fx = m_fx(*example_inputs)

            # the result matches exactly after prepare
            # Note: this currently will always be true since we are inserting observers
            # the check becomes useful when we add qat examples
            # but we can still manully inspect the printed observers to make sure
            # it matches
            self.assertEqual(after_prepare_result, after_prepare_result_fx)
            self.assertEqual(
                compute_sqnr(after_prepare_result, after_prepare_result_fx),
                torch.tensor(float("inf")),
            )
            # there are slight differences after convert due to different implementations
            # of quant/dequant
            self.assertTrue(
                torch.max(after_quant_result - after_quant_result_fx) < 1e-1
            )
            self.assertTrue(
                compute_sqnr(after_quant_result, after_quant_result_fx) > 35
            )
