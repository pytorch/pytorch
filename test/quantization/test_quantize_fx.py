import torch
import torch.nn.functional as F

import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.intrinsic.quantized as nniq

from torch.quantization.fx import QuantType

# test utils
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skipIfNoFBGEMM,
)

import itertools

class TestQuantizeFx(QuantizationTestCase):
    """ Unit tests for functionalities
    """
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
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            self.checkGraphModeFxOp(M(), inputs, quantized_node, quant_type=quant_type)

class TestQuantizeFxOps(QuantizationTestCase):
    """Unit tests for individual ops
    """
    @skipIfNoFBGEMM
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

        data = (torch.rand((1, 30), dtype=torch.float),)
        options = itertools.product(
            [(ModuleLinear(has_relu=False), True)],
            # TODO: enable after raw `tensor` is supported in fx
            # (FuncLinear(has_relu=False), False)],
            self.all_quant_types)
        quantized_nodes = {
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
            self.checkGraphModeFxOp(model, data, quantized_nodes[is_module][quant_type], quant_type=quant_type)

        for f_relu, quant_type in itertools.product([True, False], [QuantType.STATIC, QuantType.QAT]):
            for model, quantized_node in [
                    (ModuleLinear(has_relu=True, f_relu=f_relu), ('call_module', nniq.LinearReLU))]:
                # TODO: support functional linear + relu fusion
                # (FuncLinear(has_relu=True, f_relu=f_relu), ('call_function', torch.ops.quantized.linear_relu))]:
                self.checkGraphModeFxOp(model, data, quantized_node, quant_type=quant_type)

    @skipIfNoFBGEMM
    def test_quantized_conv(self):
        conv_module = {1 : torch.nn.Conv1d, 2 : torch.nn.Conv2d, 3 : torch.nn.Conv3d}

        class Conv(torch.nn.Module):
            def __init__(self, dim):
                super(Conv, self).__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                return self.conv(x)

        options = itertools.product([1, 2, 3], self.static_quant_types)
        quantized_nodes = {
            # dim
            1: ('call_module', nnq.Conv1d),
            2: ('call_module', nnq.Conv2d),
            3: ('call_module', nnq.Conv3d),
        }
        for dim, quant_type in options:
            model = self.checkGraphModeFxOp(
                Conv(dim), self.img_data_dict[dim],
                quantized_nodes[dim], quant_type=quant_type)
