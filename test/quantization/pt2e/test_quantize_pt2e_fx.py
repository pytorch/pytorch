# Owner(s): ["oncall: quantization"]
import copy
import itertools

import torch
import torch._dynamo as torchdynamo
import torch.nn as nn
from torch._inductor.compile_fx import compile_fx
from torch.ao.ns.fx.utils import compute_sqnr
from torch.ao.quantization import (
    get_default_qconfig,
    observer,
    QConfigMapping,
    default_per_channel_symmetric_qnnpack_qconfig,
)
from torch.ao.quantization._quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)
from torch.ao.quantization.backend_config import get_qnnpack_backend_config
from torch.ao.quantization.backend_config._qnnpack_pt2e import (
    get_qnnpack_pt2e_backend_config,
)
from torch.ao.quantization.backend_config._x86_inductor_pt2e import (
    get_x86_inductor_pt2e_backend_config,
)
from torch.ao.quantization.backend_config.x86 import get_x86_backend_config
from torch.ao.quantization.quantize_fx import (
    convert_fx,
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
    skipIfNoX86,
)
from torch.testing._internal.common_quantized import override_quantized_engine
import unittest


# TODO: remove after quantizer API is more mature
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
            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
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
            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
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

    def test_rearrange_weight_observer_for_decomposed_linear(self):
        """
        Check whether weight observer is correctly rearranged for decomposed linear.
        before:
            weight - t - observer \
              input - observer - addmm/mm
        after:
            weight - observer - t \
              input - observer - addmm/mm
        """

        class M(torch.nn.Module):
            def __init__(self, with_bias, use_relu):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=with_bias)
                self.relu = nn.ReLU()
                self.use_relu = use_relu

            def forward(self, x):
                x = self.linear(x)
                return self.relu(x) if self.use_relu else x

        with_bias_list = [True, False]
        use_relu_list = [True, False]
        cases = itertools.product(with_bias_list, use_relu_list)
        for with_bias, use_relu in cases:
            m = M(with_bias, use_relu).eval()
            example_inputs = (torch.randn(1, 4),)

            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            qconfig = get_default_qconfig("qnnpack")
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            backend_config = get_qnnpack_pt2e_backend_config()
            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)

            # 1. Check graph nodes:
            # - args[0] of t should be the weight observer
            # - args[-1] of addmm/mm should be t
            error_msg = (
                "Weight observer is not correctly rearranged for decomposed linear"
            )
            for node in m.graph.nodes:
                if node.target == torch.ops.aten.t.default:
                    target = node.args[0].target
                    self.assertTrue(
                        isinstance(getattr(m, target), observer.ObserverBase), error_msg
                    )
                elif node.target in (
                    torch.ops.aten.addmm.default,
                    torch.ops.aten.mm.default,
                ):
                    target = node.args[-1].target
                    self.assertTrue(target == torch.ops.aten.t.default, error_msg)

            # 2. Check m.code to ensure `m.recompile()` is called.
            # If weight observer is rearranged in graph but `m.recompile()` is not called,
            # m.code would be wrong.
            code_before_recompile = m.code
            m.recompile()
            code_after_recompile = m.code
            self.assertTrue(code_before_recompile == code_after_recompile, error_msg)

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
            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
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

@skipIfNoQNNPACK
class TestQuantizePT2EFXX86Inductor(QuantizationTestCase):
    @skipIfNoX86
    def test_inductor_backend_config_conv(self):
        class M(torch.nn.Module):
            def __init__(self, use_relu: bool = False, inplace_relu: bool = False):
                super().__init__()
                self.use_relu = use_relu
                self.conv1 = nn.Conv2d(3, 6, (2, 2), stride=(1, 1), padding=(1, 1))
                self.relu = nn.ReLU(inplace=inplace_relu)

            def forward(self, x):
                x = self.conv1(x)
                return self.relu(x) if self.use_relu else x

        use_relu_list = [True, False]
        inplace_relu_list = [True, False]
        with override_quantized_engine("x86"):
            with torch.no_grad():
                for use_relu, inplace_relu in itertools.product(
                    use_relu_list, inplace_relu_list
                ):
                    m = M(use_relu=use_relu, inplace_relu=inplace_relu).eval()
                    example_inputs = (torch.randn(2, 3, 4, 4),)
                    # program capture
                    # **TODO** Add testcase for tracing_mode="symbolic" after fix issue:
                    # https://github.com/pytorch/pytorch/issues/96274
                    export_module, guards = torchdynamo.export(
                        m,
                        *copy.deepcopy(example_inputs),
                        aten_graph=True,
                        tracing_mode="real",
                    )

                    qconfig = get_default_qconfig("x86")
                    qconfig_mapping = QConfigMapping().set_global(qconfig)
                    backend_config = get_x86_inductor_pt2e_backend_config()
                    prepare_module = prepare_pt2e(
                        export_module, qconfig_mapping, example_inputs, backend_config
                    )
                    prepare_module(*example_inputs)
                    convert_module = convert_pt2e(prepare_module)
                    convert_module(*example_inputs)

                    # Fake quant should only be inserted at start and end
                    node_occurrence = {
                        # one for input and weight of the conv, one for output for the conv
                        ns.call_function(
                            torch.ops.quantized_decomposed.quantize_per_tensor.default
                        ): 2,
                        ns.call_function(
                            torch.ops.quantized_decomposed.quantize_per_channel.default
                        ): 1,
                        ns.call_function(
                            torch.ops.quantized_decomposed.dequantize_per_channel.default
                        ): 1,
                        ns.call_function(
                            torch.ops.quantized_decomposed.dequantize_per_tensor.default
                        ): 2,
                    }
                    if use_relu:
                        node_list = [
                            ns.call_function(
                                torch.ops.quantized_decomposed.quantize_per_tensor.default
                            ),
                            ns.call_function(
                                torch.ops.quantized_decomposed.dequantize_per_tensor.default
                            ),
                            ns.call_function(torch.ops.aten.convolution.default),
                            ns.call_function(
                                torch.ops.aten.relu_.default
                                if inplace_relu
                                else torch.ops.aten.relu.default
                            ),
                            ns.call_function(
                                torch.ops.quantized_decomposed.quantize_per_tensor.default
                            ),
                            ns.call_function(
                                torch.ops.quantized_decomposed.dequantize_per_tensor.default
                            ),
                        ]
                    else:
                        node_list = [
                            ns.call_function(
                                torch.ops.quantized_decomposed.quantize_per_tensor.default
                            ),
                            ns.call_function(
                                torch.ops.quantized_decomposed.dequantize_per_tensor.default
                            ),
                            ns.call_function(torch.ops.aten.convolution.default),
                            ns.call_function(
                                torch.ops.quantized_decomposed.quantize_per_tensor.default
                            ),
                            ns.call_function(
                                torch.ops.quantized_decomposed.dequantize_per_tensor.default
                            ),
                        ]
                    self.checkGraphModuleNodes(
                        convert_module,
                        expected_node_occurrence=node_occurrence,
                        expected_node_list=node_list,
                    )

                    # Step1: Ref result in 1.X fx path
                    backend_config_1_x = get_x86_backend_config()
                    m_copy = copy.deepcopy(m)
                    m_prepare_fx = prepare_fx(
                        m_copy,
                        qconfig_mapping,
                        example_inputs,
                        backend_config=backend_config_1_x,
                    )
                    after_prepare_result_fx = m_prepare_fx(*example_inputs)
                    m_convert_fx = convert_fx(
                        m_prepare_fx, backend_config=backend_config_1_x
                    )
                    ref_result = m_convert_fx(*example_inputs)

                    # Step2: Start to lowering into Inductor
                    run = compile_fx(convert_module, example_inputs)
                    # Inductor first run
                    inductor_res = run(*example_inputs)
                    # Inductor second run
                    inductor_res = run(*example_inputs)
                    self.assertEqual(ref_result, inductor_res, atol=5e-2, rtol=5e-2)

    @skipIfNoX86
    def test_inductor_qconv_lowering(self):
        dim_to_module = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d
        }

        class M(torch.nn.Module):
            def __init__(self, dim: int, bias: bool):
                super().__init__()
                self.conv = dim_to_module[dim](3, 6, 2, stride=2, padding=0, dilation=1, bias=bias)

            def forward(self, x):
                return nn.functional.gelu(self.conv(x))

        conv_dims = [1, 2, 3]
        use_bias_list = [True, False]
        with override_quantized_engine("x86"):
            with torch.no_grad():
                cases = itertools.product(conv_dims, use_bias_list)
                for dim, use_bias in cases:
                    m = M(dim, use_bias).eval()
                    input_shape = (2, 3, *([6] * dim))
                    example_inputs = (torch.randn(input_shape),)
                    # program capture
                    exported_model, guards = torchdynamo.export(
                        m,
                        *copy.deepcopy(example_inputs),
                        aten_graph=True,
                        tracing_mode="real",
                    )

                    qconfig = get_default_qconfig("x86")
                    qconfig_mapping = QConfigMapping().set_global(qconfig)
                    backend_config_inductor = get_x86_inductor_pt2e_backend_config()
                    prepared_model = prepare_pt2e(
                        exported_model,
                        qconfig_mapping,
                        example_inputs,
                        backend_config_inductor
                    )
                    prepared_model(*example_inputs)
                    converted_model = convert_pt2e(prepared_model)

                    run = compile_fx(converted_model, example_inputs)
                    result_inductor = run(*example_inputs)

                    m_copy = copy.deepcopy(m)
                    backend_config_fx = get_x86_backend_config()
                    prepared_model_fx = prepare_fx(
                        m_copy,
                        qconfig_mapping,
                        example_inputs,
                        backend_config=backend_config_fx,
                    )
                    prepared_model_fx(*example_inputs)
                    converted_model_fx = convert_fx(
                        prepared_model_fx, backend_config=backend_config_fx
                    )
                    result_fx = converted_model_fx(*example_inputs)

                    self.assertEqual(result_inductor, result_fx)


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

            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)

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
