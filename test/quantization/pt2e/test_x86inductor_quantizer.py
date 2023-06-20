# Owner(s): ["oncall: quantization"]
import copy
import torch
import torch._dynamo as torchdynamo
import torch.nn as nn
from torch.ao.quantization._pt2e.quantizer import (
    X86InductorQuantizer,
)
from torch.ao.quantization._quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e_quantizer,
)
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    skipIfNoX86,
    skipIfNoDynamoSupport,
)
from torch.testing._internal.common_quantized import override_quantized_engine

@skipIfNoDynamoSupport
class TestQuantizePT2EX86Inductor(QuantizationTestCase):
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
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2,
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 2,
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel.default): 1,
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel.default): 1,
                }
                node_list = [
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default),
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
                    ns.call_function(torch.ops.aten.convolution.default),
                    ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default),
                    ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
                ]
                self.checkGraphModuleNodes(m,
                                           expected_node_occurrence=node_occurrence,
                                           expected_node_list=node_list)
