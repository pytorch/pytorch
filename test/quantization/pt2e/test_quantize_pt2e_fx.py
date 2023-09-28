# Owner(s): ["oncall: quantization"]
import copy

import torch
import torch._dynamo as torchdynamo
import torch.nn as nn
from torch.ao.ns.fx.utils import compute_sqnr
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
    default_per_channel_symmetric_qnnpack_qconfig,
)
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    _prepare_pt2e_deprecated,
)
from torch.ao.quantization.backend_config import get_qnnpack_backend_config
from torch.ao.quantization.backend_config._qnnpack_pt2e import (
    get_qnnpack_pt2e_backend_config,
)
from torch.ao.quantization.quantize_fx import (
    convert_to_reference_fx,
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)

from torch.testing._internal.common_utils import (
    IS_WINDOWS,
)
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
)
from torch.testing._internal.common_quantized import override_quantized_engine
import unittest


# TODO: remove after quantizer API is more mature
@unittest.skip("TODO: delete")
@skipIfNoQNNPACK
class TestQuantizePT2EFX(QuantizationTestCase):
    def test_qconfig_none(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        with override_quantized_engine("qnnpack"):
            m = M().eval()
            example_inputs = (torch.randn(1, 1, 1, 1),)
            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            qconfig = get_default_qconfig("qnnpack")
            qconfig_mapping = (
                QConfigMapping().set_global(qconfig).set_module_name("conv2", None)
            )
            backend_config = get_qnnpack_pt2e_backend_config()
            m = _prepare_pt2e_deprecated(m, qconfig_mapping, example_inputs, backend_config)
            m(*example_inputs)
            m = convert_pt2e(m)
            m(*example_inputs)

            # first conv is quantized, second conv is not quantized
            node_occurrence = {
                # two for input of the first conv, one for output for the first conv
                ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 3,
                ns.call_function(
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default
                ): 3,
            }
            node_list = [
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
                ns.call_function(torch.ops.aten.convolution.default),
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
                ns.call_function(torch.ops.aten.convolution.default),
            ]
            self.checkGraphModuleNodes(
                m,
                expected_node_list=node_list,
                expected_node_occurrence=node_occurrence,
            )

    def test_qconfig_module_type(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)
                self.linear = nn.Linear(9, 3)

            def forward(self, x):
                x = self.conv(x)
                x = x.reshape((1, -1))
                x = self.linear(x)
                return x

        with override_quantized_engine("qnnpack"):
            m = M().eval()
            example_inputs = (torch.randn(1, 1, 3, 3),)

            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            qconfig = get_default_qconfig("qnnpack")
            qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Conv2d, qconfig)
            backend_config = get_qnnpack_pt2e_backend_config()
            m = _prepare_pt2e_deprecated(m, qconfig_mapping, example_inputs, backend_config)
            m(*example_inputs)
            m = convert_pt2e(m)
            m(*example_inputs)
            # conv is quantized, linear is not quantized
            node_occurrence = {
                # two for input and weight of the conv, one for output for the conv
                ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 3,
                ns.call_function(
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default
                ): 3,
            }
            node_list = [
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
                ns.call_function(torch.ops.aten.convolution.default),
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
                ns.call_function(torch.ops.aten.addmm.default),
            ]
            self.checkGraphModuleNodes(m, expected_node_list=node_list)

    def test_transposed_conv_bn_fusion(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_trans = torch.nn.ConvTranspose2d(10, 20, 3)
                # channels for batchnorm is the same as the out_channels for convtranspose
                self.bn = torch.nn.BatchNorm2d(20)

            def forward(self, x):
                return self.bn(self.conv_trans(x))

        with override_quantized_engine("qnnpack"):
            m = M().eval()
            example_inputs = (torch.randn(10, 10, 10, 10),)
            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            node_occurrence = {
                ns.call_function(torch.ops.aten.convolution.default): 1,
                ns.call_function(
                    torch.ops.aten._native_batch_norm_legit_no_training.default
                ): 1,
            }
            self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

            qconfig = get_default_qconfig("qnnpack")
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            backend_config = get_qnnpack_pt2e_backend_config()
            m = _prepare_pt2e_deprecated(m, qconfig_mapping, example_inputs, backend_config)
            # make sure it runs
            m(*example_inputs)

            # make sure bn is fused into conv
            node_occurrence = {
                ns.call_function(torch.ops.aten.convolution.default): 1,
                ns.call_function(
                    torch.ops.aten._native_batch_norm_legit_no_training.default
                ): 0,
            }
            self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    # TODO(jerryzh168): move all _convert_to_reference_decomposed_fx tests here
    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on Windows")
    def test__convert_to_reference_decomposed_fx_per_channel_quant_module(self):
        """ Test the result for per channel weight quant for reference modules
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        m = M().eval()
        qconfig_mapping = QConfigMapping().set_global(default_per_channel_symmetric_qnnpack_qconfig)
        example_inputs = (torch.randn(1, 3, 10, 10),)
        m = prepare_fx(m, qconfig_mapping, example_inputs, backend_config=get_qnnpack_backend_config())
        m(*example_inputs)
        m_ref = copy.deepcopy(m)
        m_ref = convert_to_reference_fx(m_ref, backend_config=get_qnnpack_backend_config())
        m = _convert_to_reference_decomposed_fx(m, backend_config=get_qnnpack_backend_config())
        expected_occurrence = {
            # for input and output activations
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 2,
            # weight is per channel quantized
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel.default): 1,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel.default): 1,
        }
        import torch._dynamo as torchdynamo
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
            tracing_mode="real",
        )
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence)
        # make sure it runs
        res_ref = m_ref(*example_inputs)
        res = m(*example_inputs)
        self.assertEqual(res, res_ref)
        # check the qmin/qmax for per channel quant
        for n in m.graph.nodes:
            if n.op == "call_function" and \
               n.target == torch.ops.quantized_decomposed.quantize_per_channel.default:
                _QUANT_MIN_INDEX = 4
                _QUANT_MAX_INDEX = 5
                self.assertEqual(n.args[_QUANT_MIN_INDEX], -127)
                self.assertEqual(n.args[_QUANT_MAX_INDEX], 127)

@unittest.skip("TODO: delete")
class TestQuantizePT2EFXModels(QuantizationTestCase):
    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    def test_resnet18(self):
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

            backend_config = get_qnnpack_pt2e_backend_config()
            # TODO: define qconfig_mapping specifically for executorch
            qconfig = get_default_qconfig("qnnpack")
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            before_fusion_result = m(*example_inputs)

            m = _prepare_pt2e_deprecated(m, qconfig_mapping, example_inputs, backend_config)

            # checking that we inserted observers correctly for maxpool operator (input and
            # output share observer instance)
            self.assertEqual(
                id(m.activation_post_process_3), id(m.activation_post_process_2)
            )
            after_prepare_result = m(*example_inputs)
            m = convert_pt2e(m)

            after_quant_result = m(*example_inputs)

            # comparing with existing fx graph mode quantization reference flow
            backend_config = get_qnnpack_backend_config()
            m_fx = prepare_fx(
                m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
            )
            after_prepare_result_fx = m_fx(*example_inputs)
            m_fx = _convert_to_reference_decomposed_fx(m_fx, backend_config=backend_config)

            after_quant_result_fx = m_fx(*example_inputs)

            # the result matches exactly after prepare
            self.assertEqual(after_prepare_result, after_prepare_result_fx)
            self.assertEqual(
                compute_sqnr(after_prepare_result, after_prepare_result_fx),
                torch.tensor(float("inf")),
            )
            # there are slight differences after convert due to different implementations
            # of quant/dequant
            self.assertTrue(torch.max(after_quant_result - after_quant_result_fx) < 1e-1)
            self.assertTrue(compute_sqnr(after_quant_result, after_quant_result_fx) > 35)
