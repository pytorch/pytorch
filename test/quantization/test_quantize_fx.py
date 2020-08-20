import torch
import torch.nn.functional as F

# symbolic trace
from torch.fx import symbolic_trace

# graph mode quantization based on fx
from torch.quantization._quantize_fx import (
    Quantizer,
    fuse,
)

# eager mode quantization
from torch.quantization import default_qconfig

# test utils
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skipIfNoFBGEMM
)

class TestQuantizeFx(QuantizationTestCase):
    @skipIfNoFBGEMM
    def test_functional(self):
        """ Test quantizing functional conv and linear
        """
        class Conv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x, weight):
                return F.conv2d(x, weight, None, self.stride, self.padding, self.dilation, self.groups)

        conv_input = torch.rand(1, 3, 224, 224)
        conv_weight = torch.rand(3, 3, 3, 3)

        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, weight):
                return F.linear(x, weight)

        linear_input = torch.rand(8, 5)
        linear_weight = torch.rand(10, 5)

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        linear_module_input = torch.rand(8, 5)

        tests = [
            (False, Conv, (conv_input, conv_weight), ('call_function', torch.ops.quantized.conv2d)),
            (True, Linear, (linear_input, linear_weight), ('call_function', torch.ops.quantized.linear_dynamic)),
            (False, Linear, (linear_input, linear_weight), ('call_function', torch.ops.quantized.linear)),
            (True, LinearModule, (linear_module_input,), ('call_module', torch.nn.quantized.dynamic.Linear)),
            (False, LinearModule, (linear_module_input,), ('call_module', torch.nn.quantized.Linear)),
        ]

        for is_dynamic, M, inputs, quantized_node in tests:
            m = M().eval()
            qconfig = default_qconfig

            graph = symbolic_trace(m)
            script = torch.jit.script(graph)

            a = m(*inputs)
            b = graph(*inputs)
            c = script(*inputs)
            assert (a - b).abs().max() == 0
            assert (a - c).abs().max() == 0
            assert torch.allclose(a, b)
            assert torch.allclose(a, c)


            graph = fuse(graph)

            quantizer = Quantizer()
            qconfig_dict = {'': qconfig}
            if is_dynamic:
                prepared = quantizer.prepare_dynamic(graph, qconfig_dict)
            else:
                prepared = quantizer.prepare(graph, qconfig_dict)

            prepared(*inputs)

            qgraph = quantizer.convert(prepared)
            qgraph_debug = quantizer.convert(prepared, debug=True)
            qgraph.eval()
            qgraph_debug.eval()
            qgraph_script = torch.jit.script(qgraph)

            d = qgraph(*inputs)
            d_debug = qgraph_debug(*inputs)
            e = qgraph_script(*inputs)
            e_debug = qgraph_debug(*inputs)

            found = False
            modules = dict(qgraph.root.named_modules())
            for node in qgraph.graph.nodes:
                if node.op == 'call_function':
                    found = found or node.op == quantized_node[0] and node.target == quantized_node[1]
                elif node.op == 'call_module':
                    found = found or node.op == quantized_node[0] and type(modules[node.target]) == quantized_node[1]
            assert found, 'Expected to find quantized node:' + str(quantized_op)
            # assert (a-d).abs().max() < 2
            assert torch.allclose(d, e)
            assert (d - d_debug).abs().max() == 0
            assert (e - e_debug).abs().max() == 0
