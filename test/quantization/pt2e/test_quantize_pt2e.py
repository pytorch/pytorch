# Owner(s): ["oncall: quantization"]
import copy
import operator
import itertools
from typing import List

import torch
import torch._dynamo as torchdynamo
import torch.nn as nn
from torch.ao.ns.fx.utils import compute_sqnr
from torch.ao.quantization import (
    FusedMovingAvgObsFakeQuantize,
    observer,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    QConfigMapping,
)
from torch.ao.quantization._pt2e.quantizer import (
    OperatorConfig,
    QNNPackQuantizer,
    Quantizer,
    X86InductorQuantizer,
)
from torch.ao.quantization._quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e_quantizer,
    prepare_qat_pt2e_quantizer,
)
from torch.ao.quantization.backend_config import get_qnnpack_backend_config

from torch.ao.quantization.qconfig import (
    default_per_channel_symmetric_qnnpack_qat_qconfig,
    default_per_channel_symmetric_qnnpack_qconfig,
)
from torch.ao.quantization.quantize_fx import (
    convert_to_reference_fx,
    prepare_fx,
    prepare_qat_fx,
)
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
    skipIfNoX86,
    skipIfNoDynamoSupport,
)
from torch.testing._internal.common_quantized import override_quantized_engine
from enum import Enum


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

        import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq
        quantizer = QNNPackQuantizer()
        quantizer.set_global(qq.get_symmetric_quantization_config(is_per_channel=True, is_qat=True))
        m = M()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
            tracing_mode="real",
        )

        m = prepare_qat_pt2e_quantizer(m, quantizer)
        m(*example_inputs)

        # TODO: also verify that metadata is copied over to the new nodes

        # Verify: getitem output activation fake quantize
        output_node = list(m.graph.nodes)[-1]
        getitem_fq_node = output_node.args[0][0]
        self.assertTrue(getitem_fq_node.target.startswith("activation_post_process_"))
        getitem_fq_mod = getattr(m, getitem_fq_node.target)
        self.assertEqual(type(getitem_fq_mod), FusedMovingAvgObsFakeQuantize)
        self.assertEqual(type(getitem_fq_mod.activation_post_process), MovingAverageMinMaxObserver)
        self.assertEqual(getitem_fq_mod.dtype, torch.qint8)
        self.assertEqual(getitem_fq_mod.quant_min, -128)
        self.assertEqual(getitem_fq_mod.quant_max, 127)

        # Verify: getitem(bn, 0)
        getitem_node = getitem_fq_node.args[0]
        bn_node = getitem_node.args[0]
        self.assertEqual(getitem_node.target, operator.getitem)
        self.assertEqual(bn_node.target, torch.ops.aten._native_batch_norm_legit.default)

        # Verify: conv / scale_factor.reshape + bias.reshape
        add_bias_node = bn_node.args[0]
        (div_scale_factor_node, bias_reshape_node) = add_bias_node.args
        (conv_node, scale_factor_reshape_node) = div_scale_factor_node.args
        self.assertEqual(add_bias_node.target, torch.ops.aten.add.Tensor)
        self.assertEqual(div_scale_factor_node.target, torch.ops.aten.div.Tensor)
        self.assertEqual(bias_reshape_node.target, torch.ops.aten.view.default)
        self.assertEqual(conv_node.target, torch.ops.aten.convolution.default)
        self.assertEqual(scale_factor_reshape_node.target, torch.ops.aten.view.default)

        # Verify: conv input activation fake quantize
        conv_input_fq_node = conv_node.args[0]
        conv_input_node = conv_input_fq_node.args[0]
        self.assertTrue(conv_input_fq_node.target.startswith("activation_post_process_"))
        conv_input_fq_mod = getattr(m, conv_input_fq_node.target)
        self.assertEqual(type(conv_input_fq_mod), FusedMovingAvgObsFakeQuantize)
        self.assertEqual(type(conv_input_fq_mod.activation_post_process), MovingAverageMinMaxObserver)
        self.assertEqual(conv_input_fq_mod.dtype, torch.qint8)
        self.assertEqual(conv_input_fq_mod.quant_min, -128)
        self.assertEqual(conv_input_fq_mod.quant_max, 127)
        self.assertTrue(conv_input_node.op, "placeholder")

        # Verify: conv weight fake quantize
        conv_weight_fq_node = conv_node.args[1]
        self.assertTrue(conv_weight_fq_node.target.startswith("activation_post_process_"))
        conv_weight_fq_mod = getattr(m, conv_weight_fq_node.target)
        self.assertEqual(type(conv_weight_fq_mod), FusedMovingAvgObsFakeQuantize)
        self.assertEqual(type(conv_weight_fq_mod.activation_post_process), MovingAveragePerChannelMinMaxObserver)
        self.assertEqual(conv_weight_fq_mod.dtype, torch.qint8)
        self.assertEqual(conv_weight_fq_mod.quant_min, -127)
        self.assertEqual(conv_weight_fq_mod.quant_max, 127)

        # Verify: conv(fq(input), fq(weight * scale_factor.reshape), zero_bias)
        zero_bias_node = conv_node.args[2]
        mul_weight_scale_factor_node = conv_weight_fq_node.args[0]
        (conv_weight_fq_node, scale_factor_reshape_node) = mul_weight_scale_factor_node.args
        self.assertEqual(zero_bias_node.target, torch.ops.aten.zeros_like.default)
        self.assertEqual(mul_weight_scale_factor_node.target, torch.ops.aten.mul.Tensor)
        self.assertEqual(zero_bias_node.target, torch.ops.aten.zeros_like.default)
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

        import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq
        quantizer = QNNPackQuantizer()
        quantizer.set_global(qq.get_symmetric_quantization_config(is_per_channel=True, is_qat=True))
        m = M()
        m_fx = copy.deepcopy(m)
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # PT2 export
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        m = prepare_qat_pt2e_quantizer(m, quantizer)
        result = m(*example_inputs)

        # FX
        qconfig_mapping = QConfigMapping().set_global(default_per_channel_symmetric_qnnpack_qat_qconfig)
        backend_config = get_qnnpack_backend_config()
        m_fx = prepare_qat_fx(m_fx, qconfig_mapping, example_inputs, backend_config=backend_config)
        result_fx = m_fx(*example_inputs)

        # Verify that numerics match
        self.assertEqual(result, result_fx)


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
            self.assertEqual(after_quant_result, after_quant_result_fx)
            self.assertTrue(compute_sqnr(after_quant_result, after_quant_result_fx) == torch.tensor(float("inf")))


@skipIfNoDynamoSupport
class TestX86InductorQuantizePT2E(QuantizationTestCase):
    @skipIfNoX86
    def test_conv2d_with_quantizer_api(self):
        class Mod(torch.nn.Module):
            def __init__(self, ) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 6, (2, 2), stride=(1, 1), padding=(1, 1))

            def forward(self, x):
                return self.conv(x)

        with override_quantized_engine("x86"):
            with torch.no_grad():
                m = Mod().eval()
                m_copy = copy.deepcopy(m)
                example_inputs = (torch.randn(2, 3, 16, 16),)
                # program capture
                m, guards = torchdynamo.export(
                    m,
                    *copy.deepcopy(example_inputs),
                    aten_graph=True,
                )

                before_fusion_result = m(*example_inputs)
                import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
                quantizer = X86InductorQuantizer()
                operator_config = xiq.get_default_x86_inductor_quantization_config()
                quantizer.set_global(operator_config)
                # Insert Observer
                m = prepare_pt2e_quantizer(m, quantizer)
                after_prepare_result = m(*example_inputs)
                m = convert_pt2e(m)
                node_occurrence = {
                    # one for input and weight of the conv, one for output for the conv
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 2,
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 2,
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 1,
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 1,
                }
                node_list = [
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                    ns.call_function(torch.ops.aten.convolution.default),
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ]
                self.checkGraphModuleNodes(m,
                                           expected_node_occurrence=node_occurrence,
                                           expected_node_list=node_list)

    @skipIfNoX86
    def test_conv2d_unary_with_quantizer_api(self):
        class Mod(torch.nn.Module):
            def __init__(self, inplace_relu: bool = False, use_bias: bool = False) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 6, (2, 2), stride=(1, 1), padding=(1, 1), bias=use_bias)
                self.relu = nn.ReLU(inplace=inplace_relu)

            def forward(self, x):
                return self.relu(self.conv(x))

        inplace_relu_list = [True, False]
        use_bias_list = [True, False]
        with override_quantized_engine("x86"):
            with torch.no_grad():
                for inplace_relu, use_bias in itertools.product(inplace_relu_list, use_bias_list):
                    m = Mod(inplace_relu=inplace_relu, use_bias=use_bias).eval()
                    m_copy = copy.deepcopy(m)
                    example_inputs = (torch.randn(2, 3, 16, 16),)
                    # program capture
                    m, guards = torchdynamo.export(
                        m,
                        *copy.deepcopy(example_inputs),
                        aten_graph=True,
                    )

                    before_fusion_result = m(*example_inputs)
                    import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
                    quantizer = X86InductorQuantizer()
                    operator_spec = xiq.get_default_x86_inductor_quantization_config()
                    quantizer.set_global(operator_spec)
                    # Insert Observer
                    m = prepare_pt2e_quantizer(m, quantizer)
                    after_prepare_result = m(*example_inputs)
                    m = convert_pt2e(m)
                    node_occurrence = {
                        # one for input and weight of the conv, one for output for the relu
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 2,
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 2,
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 1,
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 1,
                    }
                    node_list = [
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                        ns.call_function(torch.ops.aten.convolution.default),
                        ns.call_function(torch.ops.aten.relu_.default if inplace_relu else torch.ops.aten.relu.default),
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                    ]
                    self.checkGraphModuleNodes(m,
                                               expected_node_occurrence=node_occurrence,
                                               expected_node_list=node_list)

    @skipIfNoX86
    def test_conv2d_binary_with_quantizer_api(self):
        class Conv2DType(Enum):
            left = 1
            right = 2
            both = 3

        class Mod(torch.nn.Module):
            def __init__(self,
                         inplace_add: bool = False,
                         conv2d_type: Conv2DType = Conv2DType.left,
                         use_bias: bool = False,
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

            def forward(self, x):
                if self.conv2d_type == Conv2DType.left:
                    if self.inplace_add:
                        tmp = self.conv(x)
                        tmp += self.relu(x)
                        return tmp
                    else:
                        return self.conv(x) + self.relu(x)
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


        inplace_add_list = [True, False]
        conv2d_type_list = [Conv2DType.left, Conv2DType.right, Conv2DType.both]
        use_bias_list = [True, False]
        with override_quantized_engine("x86"):
            with torch.no_grad():
                for inplace_add, conv2d_type, use_bias in itertools.product(inplace_add_list, conv2d_type_list, use_bias_list):
                    m = Mod(inplace_add=inplace_add, conv2d_type=conv2d_type, use_bias=use_bias).eval()
                    m_copy = copy.deepcopy(m)
                    example_inputs = (torch.randn(2, 3, 16, 16),)
                    # program capture
                    m, guards = torchdynamo.export(
                        m,
                        *copy.deepcopy(example_inputs),
                        aten_graph=True,
                    )

                    before_fusion_result = m(*example_inputs)
                    import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
                    quantizer = X86InductorQuantizer()
                    operator_spec = xiq.get_default_x86_inductor_quantization_config()
                    quantizer.set_global(operator_spec)
                    # Insert Observer
                    m = prepare_pt2e_quantizer(m, quantizer)
                    after_prepare_result = m(*example_inputs)
                    m = convert_pt2e(m)
                    if conv2d_type != Conv2DType.both:
                        node_occurrence = {
                            # one for input and weight of the conv
                            # one for output for the add
                            # one for extra input node of add
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 3,
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 1,
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 1,
                        }
                    else:
                        node_occurrence = {
                            # one for input and weight of the conv
                            # one for output for the add
                            # 2 conv will share same input quant/dequant
                            # one for extra input node of add
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 3,
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 2,
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 2,
                        }
                    node_list = [
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                        ns.call_function(torch.ops.aten.convolution.default),
                        ns.call_function(torch.ops.aten.add_.Tensor if inplace_add else torch.ops.aten.add.Tensor),
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                    ]
                    self.checkGraphModuleNodes(m,
                                               expected_node_occurrence=node_occurrence,
                                               expected_node_list=node_list)

    @skipIfNoX86
    def test_conv2d_binary_unary_with_quantizer_api(self):
        class Conv2DType(Enum):
            left = 1
            right = 2
            both = 3

        class Mod(torch.nn.Module):
            def __init__(self,
                         inplace_add: bool = False,
                         conv2d_type: Conv2DType = Conv2DType.left,
                         inplace_relu: bool = False,
                         use_bias: bool = False,
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

            def forward(self, x):
                if self.conv2d_type == Conv2DType.left:
                    if self.inplace_add:
                        tmp = self.conv(x)
                        tmp += self.relu(x)
                        return self.relu2(tmp)
                    else:
                        return self.relu2(self.conv(x) + self.relu(x))
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

        inplace_add_list = [True, False]
        conv2d_type_list = [Conv2DType.left, Conv2DType.right, Conv2DType.both]
        inplace_relu_list = [True, False]
        use_bias_list = [True, False]
        with override_quantized_engine("x86"):
            with torch.no_grad():
                for inplace_add, conv2d_type, inplace_relu, use_bias in itertools.product(
                        inplace_add_list,
                        conv2d_type_list,
                        inplace_relu_list,
                        use_bias_list,
                ):
                    m = Mod(inplace_add=inplace_add, conv2d_type=conv2d_type, inplace_relu=inplace_relu, use_bias=use_bias).eval()
                    m_copy = copy.deepcopy(m)
                    example_inputs = (torch.randn(2, 3, 16, 16),)
                    # program capture
                    m, guards = torchdynamo.export(
                        m,
                        *copy.deepcopy(example_inputs),
                        aten_graph=True,
                    )

                    before_fusion_result = m(*example_inputs)
                    import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
                    quantizer = X86InductorQuantizer()
                    operator_spec = xiq.get_default_x86_inductor_quantization_config()
                    quantizer.set_global(operator_spec)
                    # Insert Observer
                    m = prepare_pt2e_quantizer(m, quantizer)
                    after_prepare_result = m(*example_inputs)
                    m = convert_pt2e(m)
                    if conv2d_type != Conv2DType.both:
                        node_occurrence = {
                            # one for input and weight of the conv
                            # one for output for the relu
                            # one for extra input node of add
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 3,
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 1,
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 1,
                        }
                    else:
                        node_occurrence = {
                            # one for input and weight of the conv
                            # one for output for the relu
                            # 2 conv will share same input quant/dequant
                            # one for extra input node of add
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 3,
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 2,
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 2,
                        }
                    node_list = [
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                        ns.call_function(torch.ops.aten.convolution.default),
                        ns.call_function(torch.ops.aten.add_.Tensor if inplace_add else torch.ops.aten.add.Tensor),
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                    ]
                    self.checkGraphModuleNodes(m,
                                               expected_node_occurrence=node_occurrence,
                                               expected_node_list=node_list)

    @skipIfNoX86
    def test_conv2d_serials_binary_unary_with_quantizer_api(self):
        class Mod(torch.nn.Module):
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

        with override_quantized_engine("x86"):
            with torch.no_grad():
                m = Mod().eval()
                m_copy = copy.deepcopy(m)
                example_inputs = (torch.randn(2, 3, 16, 16),)
                # program capture
                m, guards = torchdynamo.export(
                    m,
                    *copy.deepcopy(example_inputs),
                    aten_graph=True,
                )

                before_fusion_result = m(*example_inputs)
                import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
                quantizer = X86InductorQuantizer()
                operator_config = xiq.get_default_x86_inductor_quantization_config()
                quantizer.set_global(operator_config)
                # Insert Observer
                m = prepare_pt2e_quantizer(m, quantizer)
                after_prepare_result = m(*example_inputs)
                m = convert_pt2e(m)
                node_occurrence = {
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 5,
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 5,
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 4,
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 4,
                }
                node_list = [
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                    ns.call_function(torch.ops.aten.convolution.default),
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                    ns.call_function(torch.ops.aten.convolution.default),
                    ns.call_function(torch.ops.aten.convolution.default),
                    ns.call_function(torch.ops.aten.add.Tensor),
                    ns.call_function(torch.ops.aten.relu.default),
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ]
                self.checkGraphModuleNodes(m,
                                           expected_node_occurrence=node_occurrence,
                                           expected_node_list=node_list)
