import torch
import torch.nn.functional as F

# symbolic trace
from torch.fx import symbolic_trace

# graph mode quantization based on fx
from torch.quantization._quantize_fx import (
    Quantizer,
    QuantType,
    fuse,
)

import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.intrinsic.quantized as nniq

# eager mode quantization
from torch.quantization import default_qconfig

# test utils
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
)

import itertools
import unittest

class TestQuantizeFx(QuantizationTestCase):
    """ Unit tests for functionalities
    """
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

            self.checkGraphModuleHasNode(qgraph, quantized_node)
            assert torch.allclose(d, e)
            assert (d - d_debug).abs().max() == 0
            assert (e - e_debug).abs().max() == 0


class TestQuantizeFxOps(QuantizationTestCase):
    """Unit tests for individual ops
    """
    @unittest.skip("Enable after tensor is supported in fx")
    def test_linear(self):
        class ModuleLinear(torch.nn.Module):
            def __init__(self, has_relu=False, f_relu=False):
                super(ModuleLinear, self).__init__()
                self.linear = torch.nn.Linear(30, 4).float()
                if has_relu:
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()
                else:
                    self.relu = torch.nn.Identity()

            def forward(self, x):
                return self.relu(self.linear(x))

        class FuncLinear(torch.nn.Module):
            def __init__(self, has_relu=False, f_relu=False):
                super(FuncLinear, self).__init__()
                self.w = torch.randn(4, 30)
                self.b = torch.randn(4)
                if has_relu:
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()
                else:
                    self.relu = torch.nn.Identity()

            def forward(self, x):
                return self.relu(F.linear(x, self.w, self.b))

        data = [torch.rand((1, 30), dtype=torch.float)]
        options = itertools.product(
            [(ModuleLinear(has_relu=False), True),
             (FuncLinear(has_relu=False), False)],
            [QuantType.DYNAMIC, QuantType.STATIC, QuantType.QAT])
        quantized_ops = {
            # is_module
            True: {
                # quant_type:
                QuantType.DYNAMIC: ('call_module', nnqd.Linear),
                QuantType.STATIC: ('call_module', nnq.Linear),
                # note that we are checking the final result
                QuantType.QAT: ('call_module', nnq.Linear),
            },
            False: {
                # quant_type:
                QuantType.DYNAMIC: ('call_function', torch.ops.quantized.linear_dynamic),
                QuantType.STATIC: ('call_function', torch.ops.quantized.linear),
                QuantType.QAT: ('call_function', torch.ops.quantized.linear),
            }
        }
        for (model, is_module), quant_type in options:
            self.checkGraphModeFxOp(model, data, quantized_ops[is_module][quant_type], quant_type=quant_type)

        for f_relu, quant_type in itertools.product([True, False], [QuantType.STATIC, QuantType.QAT]):
            for model in [
                    (ModuleLinear(has_relu=True, f_relu=f_relu), ('call_module', nniq.LinearRelu))]:
                # TODO: (FuncLinear(has_relu=True, f_relu=f_relu), ('call_function', torch.ops.quantized.linear_relu))]:
                self.checkGraphModeFxOp(model, data, quantized_op, quant_type=quant_type)
