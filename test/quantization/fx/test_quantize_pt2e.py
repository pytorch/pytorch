# Owner(s): ["oncall: quantization"]
import torch
import torch.nn as nn
import torch._dynamo as torchdynamo
from torch.testing._internal.common_utils import xfailIfPython311
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
    skipIfNoX86,
    skipIfNoDynamoSupport,
)
from torch.testing._internal.common_quantization import NodeSpec as ns
from torch.testing._internal.common_quantized import (
    override_quantized_engine,
)
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
    observer,
)
from torch.ao.quantization.qconfig import default_per_channel_symmetric_qnnpack_qconfig
from torch.ao.quantization.backend_config import (
    get_qnnpack_backend_config,
)
from torch.ao.quantization.backend_config._qnnpack_pt2e import get_qnnpack_pt2e_backend_config
from torch.ao.quantization.backend_config._x86_inductor_pt2e import get_x86_inductor_pt2e_backend_config
from torch.ao.quantization.backend_config.x86 import get_x86_backend_config
from torch.ao.quantization.quantize_fx import prepare_fx, convert_to_reference_fx, convert_fx
from torch.ao.quantization._pt2e.quantizer import Quantizer
from torch.ao.quantization._pt2e.quantizer import QNNPackQuantizer
from torch.ao.quantization._pt2e.quantizer import X86InductorQuantizer
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e, prepare_pt2e_quantizer
from torch.ao.ns.fx.utils import (
    compute_sqnr,
)
import copy
import itertools
from torch._inductor.compile_fx import compile_fx
from enum import Enum


@skipIfNoQNNPACK
class TestQuantizePT2E(QuantizationTestCase):
    @xfailIfPython311
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
            qconfig_mapping = QConfigMapping().set_global(qconfig) \
                                              .set_module_name("conv2", None)
            backend_config = get_qnnpack_pt2e_backend_config()
            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
            m(*example_inputs)
            m = convert_pt2e(m)
            m(*example_inputs)

            # first conv is quantized, second conv is not quantized
            node_occurrence = {
                # two for input of the first conv, one for output for the first conv
                ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 3,
            }
            node_list = [
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ns.call_function(torch.ops.aten.convolution.default),
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ns.call_function(torch.ops.aten.convolution.default),
            ]
            self.checkGraphModuleNodes(
                m, expected_node_list=node_list, expected_node_occurrence=node_occurrence)

    @xfailIfPython311
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
                ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 3,
            }
            node_list = [
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ns.call_function(torch.ops.aten.convolution.default),
                ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                ns.call_function(torch.ops.aten.addmm.default),
            ]
            self.checkGraphModuleNodes(m, expected_node_list=node_list)

    @xfailIfPython311
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
                    "input_act_obs_or_fq_ctr": observer.PlaceholderObserver.with_args(dtype=torch.float),
                    "output_act_obs_or_fq_ctr": observer.PlaceholderObserver.with_args(dtype=torch.float),
                }
                for node in model.graph.nodes:
                    node.meta["target_dtype_info"] = copy.deepcopy(_DEFAULT_TARGET_DTYPE_INFO)
                for node in model.graph.nodes:
                    if node.op == "call_function" and node.target == torch.ops.aten.convolution.default:
                        node.meta["target_dtype_info"] = {
                            "input_act_obs_or_fq_ctr": observer.default_observer,
                            "weight_obs_or_fq_ctr": observer.default_weight_observer,
                            "bias_obs_or_fq_ctr": observer.PlaceholderObserver.with_args(dtype=torch.float),
                            "output_act_obs_or_fq_ctr": observer.default_observer,
                            "weight_index": 1,
                            "bias_index": 2,
                        }

            def validate(self, model: torch.fx.GraphModule) -> None:
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
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence)

    @xfailIfPython311
    def test_qnnpack_quantizer_conv(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq
        quantizer = QNNPackQuantizer()
        operator_spec = qq.get_default_per_channel_symmetric_qnnpack_operator_spec()
        quantizer.set_global(operator_spec)
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
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence)

    @xfailIfPython311
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
        operator_spec = qq.get_default_per_channel_symmetric_qnnpack_operator_spec()
        quantizer.set_global(operator_spec)
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
        print("after prepare:", m)
        m(*example_inputs)
        m = convert_pt2e(m)
        print("m:", m)
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
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence)

    @xfailIfPython311
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

            qconfig = get_default_qconfig('qnnpack')
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            backend_config = get_qnnpack_pt2e_backend_config()
            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)

            # 1. Check graph nodes:
            # - args[0] of t should be the weight observer
            # - args[-1] of addmm/mm should be t
            error_msg = 'Weight observer is not correctly rearranged for decomposed linear'
            for node in m.graph.nodes:
                if node.target == torch.ops.aten.t.default:
                    target = node.args[0].target
                    self.assertTrue(isinstance(getattr(m, target), observer.ObserverBase), error_msg)
                elif node.target in (torch.ops.aten.addmm.default, torch.ops.aten.mm.default):
                    target = node.args[-1].target
                    self.assertTrue(target == torch.ops.aten.t.default, error_msg)

            # 2. Check m.code to ensure `m.recompile()` is called.
            # If weight observer is rearranged in graph but `m.recompile()` is not called,
            # m.code would be wrong.
            code_before_recompile = m.code
            m.recompile()
            code_after_recompile = m.code
            self.assertTrue(code_before_recompile == code_after_recompile, error_msg)

    @xfailIfPython311
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
                ns.call_function(torch.ops.aten._native_batch_norm_legit_no_training.default): 1,
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
                ns.call_function(torch.ops.aten._native_batch_norm_legit_no_training.default): 0,
            }
            self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

@skipIfNoQNNPACK
class TestQuantizePT2EX86Inductor(QuantizationTestCase):
    @skipIfNoX86
    @xfailIfPython311
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
                for use_relu, inplace_relu in itertools.product(use_relu_list, inplace_relu_list):
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
                    prepare_module = prepare_pt2e(export_module, qconfig_mapping, example_inputs, backend_config)
                    prepare_module(*example_inputs)
                    convert_module = convert_pt2e(prepare_module)
                    convert_module(*example_inputs)

                    # Fake quant should only be inserted at start and end
                    node_occurrence = {
                        # one for input and weight of the conv, one for output for the conv
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 2,
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 1,
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 1,
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 2,
                    }
                    if use_relu:
                        node_list = [
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                            ns.call_function(torch.ops.aten.convolution.default),
                            ns.call_function(torch.ops.aten.relu_.default if inplace_relu else torch.ops.aten.relu.default),
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                        ]
                    else:
                        node_list = [
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                            ns.call_function(torch.ops.aten.convolution.default),
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                        ]
                    self.checkGraphModuleNodes(convert_module,
                                               expected_node_occurrence=node_occurrence,
                                               expected_node_list=node_list)

                    # Step1: Ref result in 1.X fx path
                    backend_config_1_x = get_x86_backend_config()
                    m_copy = copy.deepcopy(m)
                    m_prepare_fx = prepare_fx(m_copy, qconfig_mapping, example_inputs, backend_config=backend_config_1_x)
                    after_prepare_result_fx = m_prepare_fx(*example_inputs)
                    m_convert_fx = convert_fx(m_prepare_fx, backend_config=backend_config_1_x)
                    ref_result = m_convert_fx(*example_inputs)

                    # Step2: Start to lowering into Inductor
                    run = compile_fx(convert_module, example_inputs)
                    # Inductor first run
                    inductor_res = run(*example_inputs)
                    # Inductor second run
                    inductor_res = run(*example_inputs)
                    self.assertEqual(ref_result, inductor_res, atol=5e-2, rtol=5e-2)

class TestQuantizePT2EModels(QuantizationTestCase):
    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    @xfailIfPython311
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
            self.assertEqual(id(m.activation_post_process_3), id(m.activation_post_process_2))
            after_prepare_result = m(*example_inputs)
            m = convert_pt2e(m)

            after_quant_result = m(*example_inputs)

            # comparing with existing fx graph mode quantization reference flow
            backend_config = get_qnnpack_backend_config()
            m_fx = prepare_fx(m_copy, qconfig_mapping, example_inputs, backend_config=backend_config)
            after_prepare_result_fx = m_fx(*example_inputs)
            m_fx = convert_to_reference_fx(m_fx, backend_config=backend_config)

            after_quant_result_fx = m_fx(*example_inputs)

            # the result matches exactly after prepare
            self.assertEqual(after_prepare_result, after_prepare_result_fx)
            self.assertEqual(compute_sqnr(after_prepare_result, after_prepare_result_fx), torch.tensor(float("inf")))
            # there are slight differences after convert due to different implementations
            # of quant/dequant
            self.assertTrue(torch.max(after_quant_result - after_quant_result_fx) < 1e-1)
            self.assertTrue(compute_sqnr(after_quant_result, after_quant_result_fx) > 35)


    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    @xfailIfPython311
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
            operator_spec = qq.get_default_per_channel_symmetric_qnnpack_operator_spec()
            quantizer.set_global(operator_spec)
            m = prepare_pt2e_quantizer(m, quantizer)
            # checking that we inserted observers correctly for maxpool operator (input and
            # output share observer instance)
            self.assertEqual(id(m.activation_post_process_3), id(m.activation_post_process_2))
            after_prepare_result = m(*example_inputs)
            m = convert_pt2e(m)

            after_quant_result = m(*example_inputs)

            # comparing with existing fx graph mode quantization reference flow
            qconfig = default_per_channel_symmetric_qnnpack_qconfig
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            backend_config = get_qnnpack_backend_config()
            m_fx = prepare_fx(m_copy, qconfig_mapping, example_inputs, backend_config=backend_config)
            after_prepare_result_fx = m_fx(*example_inputs)
            m_fx = convert_to_reference_fx(m_fx, backend_config=backend_config)

            after_quant_result_fx = m_fx(*example_inputs)

            # the result matches exactly after prepare
            # Note: this currently will always be true since we are inserting observers
            # the check becomes useful when we add qat examples
            # but we can still manully inspect the printed observers to make sure
            # it matches
            self.assertEqual(after_prepare_result, after_prepare_result_fx)
            self.assertEqual(compute_sqnr(after_prepare_result, after_prepare_result_fx), torch.tensor(float("inf")))
            # there are slight differences after convert due to different implementations
            # of quant/dequant
            self.assertTrue(torch.max(after_quant_result - after_quant_result_fx) < 1e-1)
            self.assertTrue(compute_sqnr(after_quant_result, after_quant_result_fx) > 35)

@skipIfNoDynamoSupport
class TestX86InductorQuantizePT2EModels(QuantizationTestCase):
    @skipIfNoX86
    @xfailIfPython311
    def test_conv2d_with_quantizer_api(self):
        class Mod(torch.nn.Module):
            def __init__(self, use_bias: bool = False) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 6, (2, 2), stride=(1, 1), padding=(1, 1), bias=use_bias)

            def forward(self, x):
                return self.conv(x)

        with override_quantized_engine("x86"):
            with torch.no_grad():
                for use_bias in [True, False]:
                    m = Mod(use_bias=use_bias).eval()
                    m_copy = copy.deepcopy(m)
                    example_inputs = (torch.randn(2, 3, 16, 16),)
                    # program capture
                    m, guards = torchdynamo.export(
                        m,
                        *copy.deepcopy(example_inputs),
                        aten_graph=True,
                        tracing_mode="real",
                    )

                    before_fusion_result = m(*example_inputs)
                    import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
                    quantizer = X86InductorQuantizer()
                    operator_spec = xiq.get_default_x86_inductor_operator_spec()
                    quantizer.set_global(operator_spec)
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
    @xfailIfPython311
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
                        tracing_mode="real",
                    )

                    before_fusion_result = m(*example_inputs)
                    import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
                    quantizer = X86InductorQuantizer()
                    operator_spec = xiq.get_default_x86_inductor_operator_spec()
                    quantizer.set_global(operator_spec)
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
                        ns.call_function(torch.ops.aten.relu_.default if inplace_relu else torch.ops.aten.relu.default),
                        ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
                        ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
                    ]
                    self.checkGraphModuleNodes(m,
                                               expected_node_occurrence=node_occurrence,
                                               expected_node_list=node_list)

    @skipIfNoX86
    @xfailIfPython311
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
                    print("conv2d_type is: {}".format(conv2d_type), flush=True)
                    m = Mod(inplace_add=inplace_add, conv2d_type=conv2d_type, use_bias=use_bias).eval()
                    m_copy = copy.deepcopy(m)
                    example_inputs = (torch.randn(2, 3, 16, 16),)
                    # program capture
                    m, guards = torchdynamo.export(
                        m,
                        *copy.deepcopy(example_inputs),
                        aten_graph=True,
                        tracing_mode="real",
                    )

                    before_fusion_result = m(*example_inputs)
                    import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
                    quantizer = X86InductorQuantizer()
                    operator_spec = xiq.get_default_x86_inductor_operator_spec()
                    quantizer.set_global(operator_spec)
                    # Insert Observer
                    m = prepare_pt2e_quantizer(m, quantizer)
                    print("m after prepare_pt2e_quantizer is: {}".format(m), flush=True)
                    after_prepare_result = m(*example_inputs)
                    m = convert_pt2e(m)
                    print("m after convert_pt2e is: {}".format(m), flush=True)
                    # from torch.fx.passes.graph_drawer import FxGraphDrawer
                    # g = FxGraphDrawer(m, "resnet50")
                    # g.get_dot_graph().write_svg("/root/test_x86_inductor_quantizer.svg")
                    if conv2d_type != Conv2DType.both:
                        node_occurrence = {
                            # one for input and weight of the conv
                            # one for output for the conv
                            # one for extra input node of add
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 3,
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 1,
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 1,
                        }
                    else:
                        node_occurrence = {
                            # one for input and weight of the conv
                            # one for output for the conv
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
    @xfailIfPython311
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
                    print("conv2d_type is: {}".format(conv2d_type), flush=True)
                    m = Mod(inplace_add=inplace_add, conv2d_type=conv2d_type, inplace_relu=inplace_relu, use_bias=use_bias).eval()
                    m_copy = copy.deepcopy(m)
                    example_inputs = (torch.randn(2, 3, 16, 16),)
                    # program capture
                    m, guards = torchdynamo.export(
                        m,
                        *copy.deepcopy(example_inputs),
                        aten_graph=True,
                        tracing_mode="real",
                    )

                    before_fusion_result = m(*example_inputs)
                    import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
                    quantizer = X86InductorQuantizer()
                    operator_spec = xiq.get_default_x86_inductor_operator_spec()
                    quantizer.set_global(operator_spec)
                    # Insert Observer
                    m = prepare_pt2e_quantizer(m, quantizer)
                    print("m after prepare_pt2e_quantizer is: {}".format(m), flush=True)
                    after_prepare_result = m(*example_inputs)
                    m = convert_pt2e(m)
                    print("m after convert_pt2e is: {}".format(m), flush=True)
                    # from torch.fx.passes.graph_drawer import FxGraphDrawer
                    # g = FxGraphDrawer(m, "resnet50")
                    # g.get_dot_graph().write_svg("/root/test_x86_inductor_quantizer.svg")
                    if conv2d_type != Conv2DType.both:
                        node_occurrence = {
                            # one for input and weight of the conv
                            # one for output for the conv
                            # one for extra input node of add
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 3,
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 3,
                            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 1,
                            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 1,
                        }
                    else:
                        node_occurrence = {
                            # one for input and weight of the conv
                            # one for output for the conv
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
    @xfailIfPython311
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
                    tracing_mode="real",
                )

                before_fusion_result = m(*example_inputs)
                import torch.ao.quantization._pt2e.quantizer.x86_inductor_quantizer as xiq
                quantizer = X86InductorQuantizer()
                operator_spec = xiq.get_default_x86_inductor_operator_spec()
                quantizer.set_global(operator_spec)
                # Insert Observer
                m = prepare_pt2e_quantizer(m, quantizer)
                print("m after prepare_pt2e_quantizer is: {}".format(m), flush=True)
                after_prepare_result = m(*example_inputs)
                m = convert_pt2e(m)
                print("m after convert_pt2e is: {}".format(m), flush=True)
                # from torch.fx.passes.graph_drawer import FxGraphDrawer
                # g = FxGraphDrawer(m, "resnet50")
                # g.get_dot_graph().write_svg("/root/test_x86_inductor_quantizer.svg")
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
