# Owner(s): ["oncall: quantization"]
import torch
import torch.nn as nn
import torch._dynamo as torchdynamo
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
    RNNDynamicModel,
)
from torch.testing._internal.common_quantization import NodeSpec as ns
from torch.testing._internal.common_quantized import (
    override_quantized_engine,
)
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
    per_channel_dynamic_qconfig,
    default_dynamic_qconfig,
    float16_dynamic_qconfig,
    quantize_dynamic,
)
from torch.ao.quantization.backend_config import (
    get_qnnpack_backend_config,
)
from torch.ao.quantization.backend_config._qnnpack_pt2e import get_qnnpack_pt2e_backend_config
from torch.ao.quantization.quantize_fx import prepare_fx, convert_to_reference_fx
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.ns.fx.utils import (
    compute_sqnr,
)
import copy
import itertools

class TestQuantizePT2E(QuantizationTestCase):
    @skipIfNoQNNPACK
    def test_qconfig_none(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
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

class TestQuantizePT2EOps(QuantizationTestCase):
    def _test_rnn_impl(self, qconfigs, M, module_type_strs, module_types, sample_input):
        options = itertools.product(qconfigs, module_type_strs)
        for qconfig, module_type_str in options:
            model_eager = M(module_type_str).eval()
            model_graph = copy.deepcopy(model_eager)
            if torch.backends.quantized.engine == 'qnnpack' and \
               qconfig is float16_dynamic_qconfig:
                continue
                # fp16 dynamic quant is not supported for qnnpack

            eager_qconfig_dict = {x : qconfig for x in module_types}
            model_eager = quantize_dynamic(model_eager, qconfig_spec=eager_qconfig_dict)

            graph_qconfig_mapping = QConfigMapping()
            # TODO: need to support this
            # for x in module_types:
            #     graph_qconfig_mapping.set_object_type(x, qconfig)
            # use global config for now
            graph_qconfig_mapping.set_global(qconfig)
            backend_config = get_qnnpack_pt2e_backend_config()
            example_inputs = (sample_input,)
            model_graph, guards = torchdynamo.export(
                model_graph,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )
            model_graph = prepare_pt2e(model_graph, graph_qconfig_mapping, example_inputs, backend_config)
            model_graph = convert_pt2e(model_graph)
            print("model_graph:", model_graph)
            self.assertEqual(model_eager(sample_input), model_graph(sample_input))
            # self.checkScriptable(model_graph, [[sample_input]], True)

    @skipIfNoQNNPACK
    def test_lstm(self):
        with override_quantized_engine("qnnpack"):
            qconfigs = [per_channel_dynamic_qconfig, default_dynamic_qconfig, float16_dynamic_qconfig]
            module_type_strs = ["LSTM"]
            module_types = [torch.nn.LSTM]
            niter = 10
            sample_input = torch.tensor([[100, -155],
                                         [-155, 100],
                                         [100, -155]], dtype=torch.float).unsqueeze(0).repeat(niter, 1, 1)
            self._test_rnn_impl(qconfigs, RNNDynamicModel, module_type_strs, module_types, sample_input)

    def test_gru(self):
        with override_quantized_engine("qnnpack"):
            qconfigs = [per_channel_dynamic_qconfig, default_dynamic_qconfig, float16_dynamic_qconfig]
            module_type_strs = ["GRU"]
            module_types = [torch.nn.GRU]
            niter = 10
            sample_input = torch.tensor([[100, -155],
                                         [-155, 100],
                                         [100, -155]], dtype=torch.float).unsqueeze(0).repeat(niter, 1, 1)
            self._test_rnn_impl(qconfigs, RNNDynamicModel, module_type_strs, module_types, sample_input)

class TestQuantizePT2EModels(QuantizationTestCase):
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
