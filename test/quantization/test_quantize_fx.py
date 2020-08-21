import torch
import torch.nn.functional as F
import torch.nn as nn
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
import operator

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

    @skipIfNoFBGEMM
    def test_quantized_conv_relu(self):
        """tests for conv1d_relu/conv2d_relu/conv3d_relu"""
        conv_module = {1 : torch.nn.Conv1d, 2 : torch.nn.Conv2d, 3 : torch.nn.Conv3d}

        class ConvNdRelu(torch.nn.Module):
            def __init__(self, dim, inplace):
                super(ConvNdRelu, self).__init__()
                self.conv = conv_module[dim](3, 3, 3).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                return self.relu(self.conv(x))

        class ConvNdFunctionalRelu(torch.nn.Module):
            def __init__(self, dim):
                super(ConvNdFunctionalRelu, self).__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                return F.relu(self.conv(x))

        class ConvNdInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self, dim):
                super(ConvNdInplaceFunctionalRelu, self).__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                return F.relu(self.conv(x), True)

        options = itertools.product([1, 2, 3], self.static_quant_types)
        quantized_nodes = {
            # dim
            1: ('call_module', nniq.ConvReLU1d),
            2: ('call_module', nniq.ConvReLU2d),
            3: ('call_module', nniq.ConvReLU3d),
        }
        for dim, quant_type in options:
            for orig_m in [ConvNdRelu(dim, True),
                           ConvNdRelu(dim, False),
                           ConvNdFunctionalRelu(dim),
                           ConvNdInplaceFunctionalRelu(dim)]:
                conv_name = "conv{}d".format(dim)
                m = self.checkGraphModeFxOp(
                    orig_m, self.img_data_dict[dim],
                    quantized_nodes[dim], quant_type=quant_type)


    def _test_quantized_binary_op_impl(self, binary_op, ibinary_op, quantized_op):
        class Op(torch.nn.Module):
            def __init__(self, is_inplace, is_scalar):
                super(Op, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.is_scalar = is_scalar
                self.op = ibinary_op if is_inplace else binary_op

            def forward(self, x, y):
                x = self.conv1(x)
                y = 3 if self.is_scalar else self.conv2(y)
                x = self.op(x, y)
                return x

        # TODO: decide whether we want to quantize or not
        # in this case
        # class NonQuantizedOp(torch.nn.Module):
        #     def __init__(self, is_inplace, is_scalar):
        #         super(NonQuantizedOp, self).__init__()
        #         self.is_scalar = is_scalar
        #         self.op = ibinary_op if is_inplace else binary_op

        #     def forward(self, x, y):
        #         y = 3 if self.is_scalar else y
        #         x = self.op(x, y)
        #         return x

        data = (torch.randn(1, 2, 3, 3, dtype=torch.float),
                torch.randn(1, 2, 3, 3, dtype=torch.float))
        quantized_node = ('call_function', quantized_op)
        options = itertools.product([True, False], [True, False], self.static_quant_types)
        for is_inplace, is_scalar, quant_type in options:
            self.checkGraphModeFxOp(Op(is_inplace, is_scalar), data, quantized_node, quant_type=quant_type)

    def _test_quantized_binary_op_relu_impl(self, binary_op, ibinary_op, quantized_op):
        class OpRelu(torch.nn.Module):
            def __init__(self, is_inplace, is_functional_relu,
                         is_inplace_relu, is_scalar):
                super(OpRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.op = ibinary_op if is_inplace else binary_op
                self.is_functional_relu = is_functional_relu
                self.is_inplace_relu = is_inplace_relu
                self.is_scalar = is_scalar

                if self.is_functional_relu:
                    self.relu = F.relu
                else:
                    self.relu = torch.nn.ReLU(self.is_inplace_relu)

            def forward(self, x, y):
                x = self.conv1(x)
                y = 3 if self.is_scalar else self.conv2(y)
                x = self.op(x, y)
                x = self.relu(x, self.is_inplace_relu) if \
                    self.is_functional_relu else self.relu(x)
                return x

        data = (torch.rand((1, 2, 5, 5), dtype=torch.float),
                torch.rand((1, 2, 5, 5), dtype=torch.float))
        quantized_node = ('call_function', quantized_op)
        options = itertools.product(
            [True, False], [True, False], [True, False], [True, False], self.static_quant_types)
        for is_inplace_op, is_functional_relu, is_inplace_relu, is_scalar, quant_type in options:
            self.checkGraphModeFxOp(
                OpRelu(is_inplace_op, is_functional_relu, is_inplace_relu, is_scalar),
                data, quantized_node, quant_type=quant_type)

    @skipIfNoFBGEMM
    def test_quantized_binary_op(self):
        self._test_quantized_binary_op_impl(
            operator.add, operator.iadd, torch.ops.quantized.add)
        self._test_quantized_binary_op_impl(
            operator.mul, operator.imul, torch.ops.quantized.mul)

    @skipIfNoFBGEMM
    def test_quantized_binary_op_relu(self):
        self._test_quantized_binary_op_relu_impl(
            operator.add, operator.iadd, torch.ops.quantized.add_relu)
        self._test_quantized_binary_op_relu_impl(
            operator.mul, operator.imul, torch.ops.quantized.mul_relu)

    @skipIfNoFBGEMM
    def test_quantized_cat(self):
        """ quantization of the output of cat will be depend on the
        input of cat. we only quantize the output of cat when its inputs are quantized.
        """
        class QuantizedCat(torch.nn.Module):
            def __init__(self):
                super(QuantizedCat, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                return torch.cat([x, y], 1)

        # TODO: decide whether to quantize in this case
        # class NonQuantizedCat(torch.nn.Module):
        #     def __init__(self):
        #         super(NonQuantizedCat, self).__init__()

        #     def forward(self, x, y):
        #         return torch.cat([x, y], 1)

        data = (torch.randn(1, 2, 5, 5, dtype=torch.float),
                torch.randn(1, 2, 5, 5, dtype=torch.float))
        quantized_node = ('call_function', torch.ops.quantized.cat)
        for quant_type in self.static_quant_types:
            self.checkGraphModeFxOp(QuantizedCat(), data, quantized_node, quant_type=quant_type)


    @skipIfNoFBGEMM
    def test_qbatch_norm(self):
        bn_module = {
            # TODO: quantized batchnorm 1d module is missing
            # 1 : torch.nn.BatchNorm1d,
            2 : torch.nn.BatchNorm2d,
            3 : torch.nn.BatchNorm3d,
        }

        class M(torch.nn.Module):
            def __init__(self, dim):
                super(M, self).__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                return self.bn(x)

        options = itertools.product(self.static_quant_types, [2, 3])
        quantized_nodes = {
            # 1: ('call_module', nnq.BatchNorm1d),
            2: ('call_module', nnq.BatchNorm2d),
            3: ('call_module', nnq.BatchNorm3d),
        }
        for quant_type, dim in options:
            model = self.checkGraphModeFxOp(M(dim), self.img_data_dict[dim], quantized_nodes[dim], quant_type)

    @skipIfNoFBGEMM
    def test_qbatch_norm_relu(self):
        bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}

        class BNRelu(torch.nn.Module):
            def __init__(self, dim, inplace):
                super(BNRelu, self).__init__()
                self.bn = bn_module[dim](3).to(torch.float)
                self.relu = torch.nn.ReLU(inplace=inplace)

            def forward(self, x):
                return self.relu(self.bn(x))

        class BNFuncRelu(torch.nn.Module):
            def __init__(self, dim):
                super(BNFuncRelu, self).__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                return F.relu(self.bn(x), False)

        class BNFuncInplaceRelu(torch.nn.Module):
            def __init__(self, dim):
                super(BNFuncInplaceRelu, self).__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                return F.relu(self.bn(x), True)

        options = itertools.product(self.static_quant_types, [2, 3])
        quantized_nodes = {
            2: ('call_module', nniq.BNReLU2d),
            3: ('call_module', nniq.BNReLU3d),
        }
        for quant_type, dim in options:
            for instance in [BNRelu(dim, True), BNRelu(dim, False),
                             BNFuncRelu(dim), BNFuncInplaceRelu(dim)]:
                self.checkGraphModeFxOp(
                    instance, self.img_data_dict[dim], quantized_nodes[dim], quant_type)

    def _test_activation_impl(
            self, float_module, float_op, quantized_module, quantized_op):
        ''' Test for activation op(with inplace options), float_op can be
        torch op or functional op
        '''
        class M(torch.nn.Module):
            def __init__(self, is_module, inplace):
                super(M, self).__init__()
                self.is_module = is_module
                self.inplace = inplace
                if self.is_module:
                    self.op = float_module(self.inplace)
                else:
                    self.op = float_op

            def forward(self, input):
                if self.is_module:
                    return self.op(input)
                else:
                    return self.op(input, self.inplace)

        options = itertools.product([True, False], [True, False], self.static_quant_types)
        quantized_nodes = {
            # is_module
            True: ('call_module', quantized_module),
            False: ('call_function', quantized_op),
        }

        for is_module, is_inplace, quant_type in options:
            self.checkGraphModeFxOp(
                M(is_module, is_inplace), self.img_data_2d, quantized_nodes[is_module], quant_type)

    def test_hardswish(self):
        self._test_activation_impl(nn.Hardswish, F.hardswish, nnq.Hardswish, torch.ops.quantized.hardswish)

    def test_elu(self):
        self._test_activation_impl(nn.ELU, F.elu, nnq.ELU, torch.ops.quantized.elu)

    def _test_norm_impl(
            self, float_module, float_op, op_args, data, quantized_module, quantized_op):
        ''' Test for normalization op, float_op can be torch op or functional op,
        op_args is a list of positional argument for the module/op
        '''
        class M(torch.nn.Module):
            def __init__(self, is_module):
                super(M, self).__init__()
                self.is_module = is_module
                if self.is_module:
                    self.op = float_module(*op_args)
                else:
                    self.op = float_op

            def forward(self, input):
                if self.is_module:
                    return self.op(input)
                else:
                    args = [input] + op_args
                    return self.op(*args)

        options = itertools.product([True, False], self.static_quant_types)
        quantized_nodes = {
            # is_module
            True: ('call_module', quantized_module),
            False: ('call_function', quantized_op),
        }

        for is_module, quant_type in options:
            self.checkGraphModeFxOp(
                M(is_module), data, quantized_nodes[is_module], quant_type)

    def test_layer_norm(self):
        data = (torch.rand((1, 2, 5, 5), dtype=torch.float),)
        self._test_norm_impl(
            nn.LayerNorm, F.layer_norm, [[2, 5, 5]], data, nnq.LayerNorm, torch.ops.quantized.layer_norm)
