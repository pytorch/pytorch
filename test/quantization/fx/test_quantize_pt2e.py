# Owner(s): ["oncall: quantization"]
import torch
import torch.nn as nn
import torch._dynamo as torchdynamo
from torch.testing._internal.common_utils import xfailIfPython311
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
    skipIfNoONEDNN,
)
from torch.testing._internal.common_quantization import NodeSpec as ns
from torch.testing._internal.common_quantized import (
    override_quantized_engine,
)
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
)
from torch.ao.quantization.backend_config import (
    get_qnnpack_backend_config,
)
from torch.ao.quantization.backend_config._qnnpack_pt2e import get_qnnpack_pt2e_backend_config
from torch.ao.quantization.backend_config._inductor_pt2e import get_inductor_pt2e_backend_config
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, convert_to_reference_fx
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.ns.fx.utils import (
    compute_sqnr,
)
import copy
from torch._inductor.compile_fx import compile_fx_quantization
import itertools

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

    @skipIfNoONEDNN
    def test_int8_conv_with_cpu_tensors(self):
        bs, in_channels, out_channels = 1, 3, 16
        feature_size, kernel_size = 224, 3
        conv_functionals = {
            1 : torch.ao.nn.quantized.functional.conv1d,
            2 : torch.ao.nn.quantized.functional.conv2d,
            3 : torch.ao.nn.quantized.functional.conv3d,
        }
        dimension_list = [1, 2, 3]
        per_channel_quantize_list = [True, False]
        use_bias_list = [True, False]
        cases = itertools.product(dimension_list, per_channel_quantize_list, use_bias_list)
        for dimension, per_channel_quantize, use_bias in cases:
            x_shape = (bs, in_channels, *(feature_size,)*dimension)
            w_shape = (out_channels, in_channels, *(kernel_size,)*dimension)
            x = torch.ones(x_shape)
            w = torch.ones(w_shape)
            b = torch.ones(out_channels) if use_bias else None
            stride = [1] * dimension
            padding = [1] * dimension
            dilation = [1] * dimension
            groups = 1
            x_scale, x_zp = 0.2, 1
            o_scale, o_zp = 1.2, 2
            qx = torch.quantize_per_tensor(x, scale=x_scale, zero_point=x_zp, dtype=torch.quint8)
            if per_channel_quantize:
                w_scales = torch.tensor([0.5] * out_channels)
                w_zps = torch.tensor([0] * out_channels)
                qw = torch.quantize_per_channel(w, scales=w_scales, zero_points=w_zps, axis=0, dtype=torch.qint8)
            else:
                w_scales = torch.tensor([0.5])
                w_zps = torch.tensor([0])
                qw = torch.quantize_per_tensor(w, scale=w_scales, zero_point=w_zps, dtype=torch.qint8)
            qx_cpu = torch.tensor(qx.int_repr().tolist(), dtype=torch.uint8)
            qw_cpu = torch.tensor(qw.int_repr().tolist(), dtype=torch.int8)
            # prepack + compute for CPU tensor
            result = torch.ops.quantized.conv_int8_cpu_tensor(
                qx_cpu, x_scale, x_zp, qw_cpu, w_scales, w_zps, b,
                stride, padding, dilation, groups, o_scale, o_zp
            )
            # Result for reference by functional qconv
            result_ref = conv_functionals[dimension](
                qx, qw, b, stride, padding, dilation, groups, scale=o_scale, zero_point=o_zp
            )
            self.assertEqual(result, result_ref.int_repr())

    def test_inductor_backend(self):
        '''
        Inductor as a quantization backend. For experiment.
        '''
        import copy
        from torch import _dynamo, _inductor
        from torch._inductor import config
        import logging

        torch._dynamo.config.log_level = logging.DEBUG
        torch._dynamo.config.verbose = True
        torch._inductor.config.trace.enabled = True
        torch._inductor.config.debug = True

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x + x)

        with override_quantized_engine("x86"):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = Mod().eval()
            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            backend_config = get_inductor_pt2e_backend_config()
            qconfig = get_default_qconfig("x86")
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            before_fusion_result = m(*example_inputs)

            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
            after_prepare_result = m(*example_inputs)

            m = convert_pt2e(m)
            after_quant_result = m(*example_inputs)

            run = compile_fx_quantization(m, example_inputs)

            # first run
            inductor_result = run(*example_inputs)

            module_result = m(*example_inputs)
            self.assertEqual(inductor_result, module_result)

            # second run
            inductor_result = run(*example_inputs)

    def _test_inductor_backend_helper(self, mod: torch.nn.Module, input_shape: tuple):
        import copy
        from torch import _dynamo, _inductor
        import logging
        from torch.ao.quantization import get_default_qconfig_mapping

        _dynamo.config.log_level = logging.DEBUG
        _dynamo.config.verbose = True
        _inductor.config.trace.enabled = True
        _inductor.config.debug = True

        # Found some weird accuracy issue, especially with x86 backend.
        # Maybe because x86 uses reduced range, range of uint8 is 0-127.
        # But it does not make sense. Onednn backend also faces the issue.
        # If we use small shapes of input, checks pass.
        qengine = 'x86'
        with override_quantized_engine(qengine):
            input_format = torch.contiguous_format
            if len(input_shape) == 4:
                input_format = torch.channels_last
            elif len(input_shape) == 5:
                input_format = torch.channels_last_3d
            example_inputs = (torch.randn(input_shape).to(memory_format=input_format),)
            m = copy.deepcopy(mod.eval())
            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )

            backend_config = get_inductor_pt2e_backend_config()
            qconfig = get_default_qconfig(qengine)
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            before_fusion_result = m(*example_inputs)

            m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
            after_prepare_result = m(*example_inputs)

            m = convert_pt2e(m)
            after_quant_result = m(*example_inputs)

            run = compile_fx_quantization(m, example_inputs)

            inductor_result = run(*example_inputs)

            # FX quantization path
            m2 = copy.deepcopy(mod.eval())
            qconfig_mapping = get_default_qconfig_mapping(qengine)
            m2 = prepare_fx(m2, qconfig_mapping, *example_inputs)
            m2(*example_inputs)
            m2 = convert_fx(m2)
            eager_result = m2(*example_inputs)

            # Results should match. inductor_result is a tuple
            self.assertEqual(inductor_result[0], eager_result)

            # second run
            inductor_result = run(*example_inputs)
            eager_result = m2(*example_inputs)
            self.assertEqual(inductor_result[0], eager_result)

    def test_conv1d_inductor_backend(self):
        '''
        Quantize and lower convolution 1d + relu with Inductor quantization backend.
        For experiment.
        '''
        class Mod(torch.nn.Module):
            def __init__(self, with_bias: bool, use_relu: bool) -> None:
                super().__init__()
                self.conv = torch.nn.Conv1d(
                    in_channels=3,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=with_bias
                )
                self.relu = torch.nn.ReLU()
                self.use_relu = use_relu

            def forward(self, x):
                x = self.conv(x)
                return self.relu(x) if self.use_relu else x

        input_shape = (1, 3, 224)
        with_bias_list = [True, False]
        use_relu_list = [True, False]
        cases = itertools.product(with_bias_list, use_relu_list)
        for with_bias, use_relu in cases:
            self._test_inductor_backend_helper(Mod(with_bias, use_relu), input_shape)

    def test_conv2d_inductor_backend(self):
        '''
        Quantize and lower convolution 2d + relu with Inductor quantization backend.
        For experiment.
        '''
        class Mod(torch.nn.Module):
            def __init__(self, with_bias: bool, use_relu: bool) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=with_bias
                )
                self.relu = torch.nn.ReLU()
                self.use_relu = use_relu

            def forward(self, x):
                x = self.conv(x)
                return self.relu(x) if self.use_relu else x

        input_shape = (1, 3, 16, 16)
        with_bias_list = [True, False]
        use_relu_list = [True, False]
        cases = itertools.product(with_bias_list, use_relu_list)
        for with_bias, use_relu in cases:
            self._test_inductor_backend_helper(Mod(with_bias, use_relu), input_shape)

    def test_conv3d_inductor_backend(self):
        '''
        Quantize and lower convolution 3d + relu with Inductor quantization backend.
        For experiment.
        '''
        class Mod(torch.nn.Module):
            def __init__(self, with_bias: bool, use_relu: bool) -> None:
                super().__init__()
                self.conv = torch.nn.Conv3d(
                    in_channels=3,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=with_bias
                )
                self.relu = torch.nn.ReLU()
                self.use_relu = use_relu

            def forward(self, x):
                x = self.conv(x)
                return self.relu(x) if self.use_relu else x

        input_shape = (1, 3, 6, 6, 6)
        with_bias_list = [True, False]
        use_relu_list = [True, False]
        cases = itertools.product(with_bias_list, use_relu_list)
        for with_bias, use_relu in cases:
            self._test_inductor_backend_helper(Mod(with_bias, use_relu), input_shape)

    def test_linear_inductor_backend(self):
        '''
        Quantize and lower linear (+ relu) with Inductor quantization backend.
        For experiment.
        '''
        class Mod(torch.nn.Module):
            def __init__(self, with_bias: bool, use_relu: bool) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_features=16,
                    out_features=8,
                    bias=with_bias
                )
                self.relu = torch.nn.ReLU()
                self.use_relu = use_relu

            def forward(self, x):
                x = self.linear(x)
                return self.relu(x) if self.use_relu else x

        input_shape = (1, 16)
        with_bias_list = [True, False]
        use_relu_list = [True, False]
        cases = itertools.product(with_bias_list, use_relu_list)
        for with_bias, use_relu in cases:
            self._test_inductor_backend_helper(Mod(with_bias, use_relu), input_shape)
