# Owner(s): ["oncall: quantization"]
import copy
import unittest
from typing import List

import torch
import torch._dynamo as torchdynamo
from torch.ao.ns.fx.utils import compute_sqnr
from torch.ao.quantization import observer, QConfigMapping
from torch.ao.quantization._pt2e.quantizer import (
    OperatorConfig,
    QNNPackQuantizer,
    Quantizer,
)
from torch.ao.quantization._quantize_pt2e import convert_pt2e, prepare_pt2e_quantizer
from torch.ao.quantization.backend_config import get_qnnpack_backend_config

from torch.ao.quantization.qconfig import default_per_channel_symmetric_qnnpack_qconfig
from torch.ao.quantization.quantize_fx import convert_to_reference_fx, prepare_fx
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
                _DEFAULT_TARGET_DTYPE_INFO = {
                    "input_act_obs_or_fq_ctr": observer.PlaceholderObserver.with_args(
                        dtype=torch.float
                    ),
                    "output_act_obs_or_fq_ctr": observer.PlaceholderObserver.with_args(
                        dtype=torch.float
                    ),
                }
                for node in model.graph.nodes:
                    node.meta["target_dtype_info"] = copy.deepcopy(
                        _DEFAULT_TARGET_DTYPE_INFO
                    )
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.convolution.default
                    ):
                        node.meta["target_dtype_info"] = {
                            "input_act_obs_or_fq_ctr": observer.default_observer,
                            "weight_obs_or_fq_ctr": observer.default_weight_observer,
                            "bias_obs_or_fq_ctr": observer.PlaceholderObserver.with_args(
                                dtype=torch.float
                            ),
                            "output_act_obs_or_fq_ctr": observer.default_observer,
                            "weight_index": 1,
                            "bias_index": 2,
                        }

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
            tracing_mode="real",
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
            tracing_mode="real",
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
            m_copy = copy.deepcopy(m)
            m_fx = prepare_fx(
                m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
            )
            m_fx = convert_to_reference_fx(m_fx, backend_config=backend_config)
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
        m_copy = copy.deepcopy(m)
        m_fx = prepare_fx(
            m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
        )
        m_fx = convert_to_reference_fx(m_fx, backend_config=backend_config)
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
        %clone_default : [#users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_default,), kwargs = {memory_format: torch.contiguous_format})
        %_unsafe_view_default : [#users=1] = call_function[target=torch.ops.aten._unsafe_view.default](args = (%clone_default, [8, 16]), kwargs = {})
        %mm_default : [#users=1] = call_function[target=torch.ops.aten.mm.default](args = (%_unsafe_view_default, %t_default), kwargs = {})
        %view_default : [#users=1] = call_function[target=torch.ops.aten.view.default](args = (%mm_default, [2, 2, 2, 8]), kwargs = {})

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
            tracing_mode="real",
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
                tracing_mode="real",
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
            self.assertEqual(after_quant_result, after_quant_result_fx)
            self.assertTrue(
                compute_sqnr(after_quant_result, after_quant_result_fx)
                == torch.tensor(float("inf"))
            )
