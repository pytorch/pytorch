from __future__ import absolute_import, division, print_function, unicode_literals

from test_pytorch_common import TestCase, run_tests, flatten, skipIfNoLapack

import torch
import torch.onnx
from torch.autograd import Variable, Function
from torch.nn import Module, functional
import torch.nn as nn

import itertools
import io
import inspect
import glob
import os
import shutil
import torch.testing._internal.common_utils as common


'''Usage: python test/onnx/test_operators.py [--no-onnx] [--produce-onnx-test-data]
          --no-onnx: no onnx python dependence
          --produce-onnx-test-data: generate onnx test data
'''

_onnx_test = False  # flag to produce onnx test cases.
_onnx_dep = True  # flag to import onnx package.


def export_to_pbtxt(model, inputs, *args, **kwargs):
    return torch.onnx.export_to_pretty_string(
        model, inputs, None, verbose=False, google_printer=True,
        *args, **kwargs)


def export_to_pb(model, inputs, *args, **kwargs):
    kwargs['operator_export_type'] = torch.onnx.OperatorExportTypes.ONNX
    f = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, inputs, f, *args, **kwargs)
    return f.getvalue()


class FuncModule(Module):
    def __init__(self, f, params=None):
        if params is None:
            params = ()
        super(FuncModule, self).__init__()
        self.f = f
        self.params = nn.ParameterList(list(params))

    def forward(self, *args):
        return self.f(*itertools.chain(args, self.params))


class TestOperators(TestCase):

    def assertONNX(self, f, args, params=None, **kwargs):
        if params is None:
            params = ()
        if isinstance(f, nn.Module):
            m = f
        else:
            m = FuncModule(f, params)
        m.eval()
        onnx_model_pbtxt = export_to_pbtxt(m, args, **kwargs)
        subname = kwargs.pop('subname', None)
        self.assertExpected(onnx_model_pbtxt, subname)
        if _onnx_dep:
            onnx_model_pb = export_to_pb(m, args, **kwargs)
            import onnx
            import onnx.checker
            import onnx.numpy_helper
            import test_onnx_common
            model_def = onnx.ModelProto.FromString(onnx_model_pb)
            onnx.checker.check_model(model_def)
            if _onnx_test:
                test_function = inspect.stack()[1][0].f_code.co_name
                test_name = test_function[0:4] + "_operator" + test_function[4:]
                output_dir = os.path.join(test_onnx_common.pytorch_operator_dir, test_name)
                # Assume:
                #     1) the old test should be delete before the test.
                #     2) only one assertONNX in each test, otherwise will override the data.
                assert not os.path.exists(output_dir), "{} should not exist!".format(output_dir)
                os.makedirs(output_dir)
                with open(os.path.join(output_dir, "model.onnx"), 'wb') as file:
                    file.write(model_def.SerializeToString())
                data_dir = os.path.join(output_dir, "test_data_set_0")
                os.makedirs(data_dir)
                if isinstance(args, Variable):
                    args = (args,)
                for index, var in enumerate(flatten(args)):
                    tensor = onnx.numpy_helper.from_array(var.data.numpy())
                    with open(os.path.join(data_dir, "input_{}.pb".format(index)), 'wb') as file:
                        file.write(tensor.SerializeToString())
                outputs = m(*args)
                if isinstance(outputs, Variable):
                    outputs = (outputs,)
                for index, var in enumerate(flatten(outputs)):
                    tensor = onnx.numpy_helper.from_array(var.data.numpy())
                    with open(os.path.join(data_dir, "output_{}.pb".format(index)), 'wb') as file:
                        file.write(tensor.SerializeToString())

    def assertONNXRaises(self, err, f, args, params=None, **kwargs):
        if params is None:
            params = ()
        if isinstance(f, nn.Module):
            m = f
        else:
            m = FuncModule(f, params)
        self.assertExpectedRaises(err, lambda: export_to_pbtxt(m, args, **kwargs))

    def assertONNXRaisesRegex(self, err, reg, f, args, params=None, **kwargs):
        if params is None:
            params = ()
        if isinstance(f, nn.Module):
            m = f
        else:
            m = FuncModule(f, params)
        with self.assertRaisesRegex(err, reg):
            export_to_pbtxt(m, args, **kwargs)

    def test_basic(self):
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)
        self.assertONNX(lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))), (x, y))

    def test_view(self):
        x = torch.tensor([0.0], requires_grad=True)
        self.assertONNX(lambda x: x.view(1, 1), x)

    def test_index(self):
        x = torch.tensor([[0.0]], requires_grad=True)
        self.assertONNX(lambda x: x[0], x)

    def test_type_as(self):
        x = torch.tensor([0.0], requires_grad=True)
        self.assertONNX(lambda x: x.type_as(x), x)

    def test_addconstant(self):
        x = torch.randn(2, 3, requires_grad=True).double()
        self.assertONNX(lambda x: x + 1, x)

    def test_add_broadcast(self):
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(3, requires_grad=True).double()
        self.assertONNX(lambda x, y: x + y, (x, y))

    def test_add_left_broadcast(self):
        x = torch.randn(3, requires_grad=True).double()
        y = torch.randn(2, 3, requires_grad=True).double()
        self.assertONNX(lambda x, y: x + y, (x, y))

    def test_add_size1_broadcast(self):
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(2, 1, requires_grad=True).double()
        self.assertONNX(lambda x, y: x + y, (x, y))

    def test_add_size1_right_broadcast(self):
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(3, requires_grad=True).double()
        self.assertONNX(lambda x, y: x + y, (x, y))

    def test_add_size1_singleton_broadcast(self):
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(1, 3, requires_grad=True).double()
        self.assertONNX(lambda x, y: x + y, (x, y))

    def test_rsub(self):
        x = torch.randn(2, 3, requires_grad=True).double()
        self.assertONNX(lambda x: 1 - x, (x,))

    def test_transpose(self):
        x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
        self.assertONNX(lambda x: x.transpose(0, 1).transpose(1, 0), x)

    def test_chunk(self):
        x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        self.assertONNX(lambda x: x.chunk(2), x)

    def test_split(self):
        x = torch.tensor([[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]])
        self.assertONNX(lambda x: torch.split(x, 2, 1), x)

    def test_split_with_sizes(self):
        x = torch.tensor([[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]])
        self.assertONNX(lambda x: torch.split(x, [2, 1, 3], 1), x)

    def test_concat2(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.assertONNX(lambda inputs: torch.cat(inputs, 1), ((x, y),))

    def test_mm(self):
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(torch.mm, (m1, m2))

    def test_addmm(self):
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(3, 4, requires_grad=True)
        m3 = torch.randn(4, requires_grad=True)
        self.assertONNX(lambda x, y, z: torch.addmm(torch.addmm(z, x, y), x, y), (m1, m2, m3))

    def test_permute2(self):
        x = torch.tensor([[[[[[0.0]]]]]], requires_grad=True)
        self.assertONNX(lambda x: x.permute(0, 1, 4, 2, 5, 3), x)

    def test_pad(self):
        x = torch.tensor([[[[0.0, 1.0, 1.0, 1.0], [2.0, 3.0, 7.0, 7.0]]]], requires_grad=True)
        self.assertONNX(nn.ReflectionPad2d((2, 3, 0, 1)), x)

    def test_params(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
        self.assertONNX(lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))), x, params=(y, ),
                        keep_initializers_as_inputs=True)

    def test_params_onnx_irv4(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
        self.assertONNX(lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))), x, params=(y, ),
                        keep_initializers_as_inputs=False)

    def test_symbolic_mismatch(self):
        class MyFun(Function):
            @staticmethod
            def symbolic(g, x):
                # The inside of this function should never be invoked, because
                # we will fail due to an argument mismatch first.
                raise AssertionError()

            @staticmethod
            def forward(ctx, x, y):
                return x + y

        x = torch.ones(2, 2)
        y = torch.ones(2, 2)
        # NB: Don't use expect test here, the type error wobbles depending
        # on Python version
        with self.assertRaisesRegex(TypeError, "occurred when translating MyFun"):
            export_to_pbtxt(FuncModule(MyFun().apply), (x, y))

    # TODO: Do an nn style test for these
    def test_batchnorm(self):
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        self.assertONNX(nn.BatchNorm2d(2), x, keep_initializers_as_inputs=True)

    def test_batchnorm_onnx_irv4(self):
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        self.assertONNX(nn.BatchNorm2d(2), x)

    def test_batchnorm_1d(self):
        x = torch.ones(2, 2, requires_grad=True)
        self.assertONNX(nn.BatchNorm1d(2), x, keep_initializers_as_inputs=True)

    def test_batchnorm_training(self):
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        self.assertONNX(nn.BatchNorm2d(2), x, training=True, keep_initializers_as_inputs=True)

    def test_conv(self):
        x = torch.ones(20, 16, 50, 40, requires_grad=True)
        self.assertONNX(nn.Conv2d(16, 13, 3, bias=False), x, keep_initializers_as_inputs=True)

    def test_conv_onnx_irv4(self):
        x = torch.ones(20, 16, 50, 40, requires_grad=True)
        self.assertONNX(nn.Conv2d(16, 13, 3, bias=False), x)

    def test_conv_onnx_irv4_opset8(self):
        # This test point checks that for opset 8 (or lower), even if
        # keep_initializers_as_inputs is set to False, it is ignored,
        # and initializers are listed as ONNX graph input, in accordance
        # with ONNX IR v3 semantics (which apply to opset version <= 8).
        x = torch.ones(1, 2, 5, 7, requires_grad=True)
        conv_node = nn.Conv2d(2, 4, 3, bias=False)
        conv_node.weight.data.fill_(1.0)
        self.assertONNX(conv_node, x, opset_version=8, keep_initializers_as_inputs=False)

    def test_conv_variable_length(self):
        x = torch.ones(5, 3, 6, 6, requires_grad=True)
        model = torch.nn.Conv2d(3, 2, 3)
        y = model(x)

        dynamic_axes = {'input_1': [0, 2, 3], 'output_1': {0: 'output_1_variable_dim_0', 1: 'output_1_variable_dim_1'}}
        model_proto_name = 'conv2d.onnx'
        torch.onnx.export(model, x, model_proto_name, verbose=True, input_names=["input_1"], output_names=["output_1"],
                          example_outputs=y, dynamic_axes=dynamic_axes)

        import onnx
        onnx_model = onnx.load(model_proto_name)
        onnx.checker.check_model(onnx_model)

        # Asserting the default dynamic axes names are generated when custom names are not provided
        assert(onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "input_1_dynamic_axes_1")
        assert(onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_param == "input_1_dynamic_axes_2")
        assert(onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_param == "input_1_dynamic_axes_3")

        # Asserting the custom names are applied when provided
        assert(onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "output_1_variable_dim_0")
        assert(onnx_model.graph.output[0].type.tensor_type.shape.dim[1].dim_param == "output_1_variable_dim_1")

    def test_convtranspose(self):
        x = torch.ones(2, 3, 4, 5, requires_grad=True)
        self.assertONNX(nn.ConvTranspose2d(3, 3, 3, stride=3, bias=False,
                                           padding=1, output_padding=2), x,
                        keep_initializers_as_inputs=True)

    def test_maxpool(self):
        x = torch.randn(20, 16, 50)
        self.assertONNX(nn.MaxPool1d(3, stride=2), x)

    def test_maxpool_dilations(self):
        x = torch.randn(20, 16, 50)
        self.assertONNX(nn.MaxPool1d(2, stride=1, dilation=2), x, opset_version=10)

    def test_avg_pool2d(self):
        x = torch.randn(20, 16, 50, 32)
        self.assertONNX(nn.AvgPool2d(3, stride=2), x)

    def test_maxpool_indices(self):
        x = torch.randn(20, 16, 50)
        self.assertONNX(nn.MaxPool1d(3, stride=2, return_indices=True), x)

    def test_at_op(self):
        x = torch.randn(3, 4)

        class MyFun(Function):

            @staticmethod
            def symbolic(g, x):
                return g.at("add", x, x)

            @staticmethod
            def forward(ctx, x):
                return x + x

        class MyModule(Module):
            def forward(self, x):
                return MyFun.apply(x)

        self.assertONNX(MyModule(), x)

    def test_clip(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.clamp(x, min=-0.5, max=0.5), x)

    def test_clip_min(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.clamp(min=-0.1), x)

    def test_clip_max(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.clamp(max=0.1), x)

    def test_hardtanh(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.nn.Hardtanh(-0.5, 0.5)(x), x)

    def test_full(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.full(x.shape, 2), x)

    def test_full_like(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.full_like(x, 2), x)

    def test_max(self):
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x, y: torch.max(x, y), (x, y))

    def test_min(self):
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x, y: torch.min(x, y), (x, y))

    def test_mean(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.mean(x), x)

    def test_reduced_mean(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.mean(x, dim=2), x)

    def test_reduced_mean_keepdim(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.mean(x, dim=(2, 3), keepdim=True), x)

    def test_mean_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNXRaisesRegex(RuntimeError, 'Couldn\'t export operator aten::mean',
                                   lambda x: torch.mean(x, dtype=torch.double), x)

    def test_reduced_mean_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNXRaisesRegex(RuntimeError, 'Couldn\'t export operator aten::mean',
                                   lambda x: torch.mean(x, dim=0, dtype=torch.double), x)

    def test_sum(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x), x)

    def test_sum_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNXRaisesRegex(RuntimeError, 'Couldn\'t export operator aten::sum',
                                   lambda x: torch.sum(x, dtype=torch.double), x)

    def test_reduced_sum_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNXRaisesRegex(RuntimeError, 'Couldn\'t export operator aten::sum',
                                   lambda x: torch.sum(x, dim=0, dtype=torch.double), x)

    def test_reduced_sum(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x, dim=(1, 2)), x)

    def test_reduced_sum_keepdim(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x, dim=2, keepdim=True), x)

    def test_prod(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.prod(x), x)

    def test_reduced_prod(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.prod(x, dim=2), x)

    def test_reduced_prod_keepdim(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.prod(x, dim=2, keepdim=True), x)

    def test_prod_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNXRaisesRegex(RuntimeError, 'Couldn\'t export operator aten::prod',
                                   lambda x: torch.prod(x, dtype=torch.double), x)

    def test_reduced_prod_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNXRaisesRegex(RuntimeError, 'Couldn\'t export operator aten::prod',
                                   lambda x: torch.prod(x, dim=0, dtype=torch.double), x)

    def test_sqrt(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sqrt(x), x)

    def test_rsqrt(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.rsqrt(x), x)

    def test_equal(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.assertONNX(lambda x, y: x == y, (x, y))

    def test_lt(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.assertONNX(lambda x, y: x < y, (x, y))

    def test_gt(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.assertONNX(lambda x, y: x > y, (x, y))

    def test_le(self):
        x = torch.randn(3, 4, requires_grad=False).int()
        y = torch.randn(3, 4, requires_grad=False).int()
        self.assertONNX(lambda x, y: x <= y, (x, y))

    def test_ge(self):
        x = torch.randn(3, 4, requires_grad=False).int()
        y = torch.randn(3, 4, requires_grad=False).int()
        self.assertONNX(lambda x, y: x >= y, (x, y))

    def test_exp(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.exp(), x)

    def test_sin(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.sin(), x)

    def test_cos(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.cos(), x)

    def test_tan(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.tan(), x)

    def test_asin(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.asin(), x)

    def test_acos(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.acos(), x)

    def test_slice(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x[:, 1:2], x)

    def test_slice_dynamic(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x[x.size(0):, x.size(1) - 3], x, opset_version=10)

    def test_sign(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.sign(), x)

    def test_narrow(self):
        x = torch.randn(3, 3, requires_grad=True)
        self.assertONNX(lambda x: torch.narrow(x, 0, 0, 2), x)

    def test_atan(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.atan(), x)

    def test_view_flatten(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.view(x.size()[0], x.numel() // x.size()[0]), x)

    def test_flatten(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.flatten(x), x)

    def test_flatten2D(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.flatten(x, 1), x)

    def test_isnan(self):
        x = torch.tensor([1, float('nan'), 2])
        self.assertONNX(lambda x: torch.isnan(x), x)

    def test_argmax(self):
        x = torch.randn(4, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.argmax(x, dim=1), x)

    def test_logsoftmax(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(nn.LogSoftmax(dim=3), x)

    def test_pow(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        y = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x, y: x.pow(y), (x, y))

    def test_elu(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(nn.ELU(), x)

    def test_selu(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(nn.SELU(), x)

    def test_repeat(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.repeat(1, 2, 3, 4), x)

    def test_repeat_dim_overflow(self):
        x = torch.randn(1, 2, requires_grad=True)
        self.assertONNX(lambda x: x.repeat(1, 2, 3, 4), x)

    def test_norm_p1(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.norm(p=1, dim=2), (x))

    def test_norm_p2(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.norm(p=2, dim=2), (x))

    def test_upsample_nearest_scale(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: nn.functional.interpolate(x, scale_factor=2.,
                        mode='nearest', recompute_scale_factor=False), x)

    def test_upsample_nearest_scale_default_scale_factor(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: nn.functional.interpolate(x, scale_factor=2.,
                        mode='nearest'), x)

    def test_upsample_nearest_size(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: nn.functional.interpolate(x, size=16, mode='nearest'), x)

    def test_unsqueeze(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.unsqueeze(len(x.shape)), x)

    def test_batchnorm_noaffine(self):
        x = torch.randn(128, 128, 1, 1, requires_grad=True)
        self.assertONNX(nn.BatchNorm2d(128, affine=False, momentum=0.3), x,
                        keep_initializers_as_inputs=True)

    def test_embedding_bags(self):
        emb_bag = nn.EmbeddingBag(10, 8)
        input = torch.tensor([1, 2, 3, 4]).long()
        offset = torch.tensor([0]).long()
        self.assertONNX(emb_bag, (input, offset), keep_initializers_as_inputs=True)

    def test_implicit_expand(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x + 1, x)

    def test_reduce_sum_negative_indices(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.sum(-1), x)

    def test_randn(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(lambda x: torch.randn(1, 2, 3, 4) + x, x)

    def test_rand(self):
        x = torch.rand(1, 2, 3, 4)
        self.assertONNX(lambda x: torch.rand(1, 2, 3, 4) + x, x)

    def test_rrelu(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(torch.nn.RReLU(), x)

    def test_prelu(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(torch.nn.PReLU(2), x, keep_initializers_as_inputs=True)

    def test_log_sigmoid(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(torch.nn.LogSigmoid(), x)

    def test_linear(self):
        x = torch.randn(3, 4)
        self.assertONNX(torch.nn.Linear(4, 5, bias=True), x,
                        keep_initializers_as_inputs=True)

    def test_empty_like(self):
        x = torch.randn(5, 8, requires_grad=True)
        self.assertONNX(lambda x: torch.empty_like(x), x)

    def test_empty_like_opset7(self):
        x = torch.randn(5, 8, requires_grad=True)
        self.assertONNX(lambda x: torch.empty_like(x), x, opset_version=7)

    def test_zeros_like(self):
        x = torch.randn(5, 8, requires_grad=True)
        self.assertONNX(lambda x: torch.zeros_like(x), x)

    def test_ones_like(self):
        x = torch.randn(6, 10, requires_grad=True)
        self.assertONNX(lambda x: torch.ones_like(x), x)

    def test_expand(self):
        x = torch.randn(6, 1, requires_grad=True)
        self.assertONNX(lambda x: x.expand(4, 6, 2), x)

    def test_ne(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.assertONNX(lambda x, y: torch.ne(x, y), (x, y))

    def test_reducemax(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(lambda x: torch.max(x), x)

    def test_reducemin(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(lambda x: torch.min(x), x)

    def test_erf(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(lambda x: x.erf(), x)

    def test_dropout(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.max(functional.dropout(x, training=False)), x)

    def test_nonzero(self):
        x = torch.tensor([[[2., 2.], [1., 0.]], [[0., 0.], [1., 1.]]], requires_grad=True)
        self.assertONNX(lambda x: torch.nonzero(x), x)

    def test_gather(self):
        data = torch.randn(3, 4, 3, requires_grad=True)
        index = torch.tensor([2, 0]).view(1, 2, 1).expand(3, 2, 3)
        self.assertONNX(lambda data, index: data.gather(1, index), (data, index))

    def test_gather_opset11(self):
        data = torch.randn(3, 4, 3, requires_grad=True)
        index = torch.tensor([2, 0]).view(1, 2, 1).expand(3, 2, 3)
        self.assertONNX(lambda data, index: data.gather(1, index), (data, index), opset_version=11)

    def test_scatter_add(self):
        data = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.assertONNX(lambda data, index: data.scatter_add(1, indices, values), (data, (indices, values)))

    def test_scatter_add_opset11(self):
        data = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.assertONNX(lambda data, index: data.scatter_add(1, indices, values), (data, (indices, values)), opset_version=11)

    def test_master_opset(self):
        x = torch.randn(2, 3).float()
        y = torch.randn(2, 3).float()
        self.assertONNX(lambda x, y: x + y, (x, y), opset_version=10)

    def test_std(self):
        x = torch.randn(2, 3, 4).float()
        self.assertONNX(lambda x: torch.std(x, dim=(0, 1), unbiased=True, keepdim=True), x)

    def test_cumsum(self):
        x = torch.randn(2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.cumsum(x, dim=1), x, opset_version=11)

    def test_retain_param_name_disabled(self):
        class MyModule(Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.fc1 = nn.Linear(4, 5, bias=False)
                self.fc1.weight.data.fill_(2.)
                self.fc2 = nn.Linear(5, 6, bias=False)
                self.fc2.weight.data.fill_(3.)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        x = torch.randn(3, 4).float()
        self.assertONNX(MyModule(), (x,), _retain_param_name=False,
                        keep_initializers_as_inputs=True)

    def test_c2_op(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, scores, bbox_deltas, im_info, anchors):
                a, b = torch.ops._caffe2.GenerateProposals(
                    (scores), (bbox_deltas), (im_info), (anchors),
                    2.0, 6000, 300, 0.7, 16, True, -90, 90, 1.0, True,
                )
                return a, b

        model = MyModel()
        A = 4
        H = 10
        W = 8
        img_count = 3
        scores = torch.ones(img_count, A, H, W, dtype=torch.float32)
        bbox_deltas = torch.linspace(0, 10, steps=img_count * 4 * A * H * W,
                                     dtype=torch.float32)
        bbox_deltas = bbox_deltas.view(img_count, 4 * A, H, W)
        im_info = torch.ones(img_count, 3, dtype=torch.float32)
        anchors = torch.ones(A, 4, dtype=torch.float32)
        inputs = (scores, bbox_deltas, im_info, anchors)
        self.assertONNX(model, inputs, custom_opsets={'org.pytorch._caffe2': 0})

    def test_dict(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in[list(x_in.keys())[0]], list(x_in.keys())[0])
                return x_out

        x = {torch.tensor(1.): torch.randn(1, 2, 3)}
        self.assertONNX(MyModel(), (x,))

    def test_dict_str(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in["test_key_in"], 2.)
                return x_out

        x = {"test_key_in": torch.randn(1, 2, 3)}
        self.assertONNX(MyModel(), (x,))

    def test_arange_dynamic(self):
        class TestModel(torch.nn.Module):
            def forward(self, input):
                return torch.arange(input.shape[0], input.shape[0] + 5, 0.5)

        input = torch.randn(5, 3, 2)
        self.assertONNX(TestModel(), input, opset_version=11)

    def test_bitshift(self):
        class BitshiftModel(torch.nn.Module):
            def forward(self, input, input2):
                return input >> 1, input2 >> 2
        input = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
        input2 = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        self.assertONNX(BitshiftModel(), (input, input2), opset_version=11)

    def test_layer_norm_aten(self):
        model = torch.nn.LayerNorm([10, 10])
        x = torch.randn(20, 5, 10, 10)
        self.assertONNX(model, x,
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    def test_pixel_shuffle(self):
        x = torch.randn(2, 8, 3, 4).float()
        self.assertONNX(lambda x: torch.pixel_shuffle(x, upscale_factor=2), x, opset_version=11)

    def test_frobenius_norm(self):
        x = torch.randn(2, 3, 4).float()
        self.assertONNX(lambda x: torch.norm(x, p="fro", dim=(0, 1), keepdim=True), x)

    def test_unfold(self):
        x = torch.randn(2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.unfold(dimension=2, size=2, step=2), x)

    def test_remainder(self):
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 1, 4)
        self.assertONNX(lambda x, y: torch.remainder(x, y), (x, y))

    def test_fmod(self):
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 1, 4)
        self.assertONNX(lambda x, y: torch.fmod(x, y), (x, y), opset_version=10)

    def test_gelu(self):
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        self.assertONNX(lambda x: torch.nn.functional.gelu(x), x)

    def test_unique(self):
        x = torch.randint(3, (2, 3, 4, 5)).float()
        self.assertONNX(lambda x: torch.unique(x, dim=0, sorted=True, return_inverse=False, return_counts=True), x,
                        opset_version=11)

    def test_meshgrid(self):
        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.ones(5, requires_grad=True)
        self.assertONNX(lambda x, y, z: torch.meshgrid(x, y, z), (x, y, z))

    def test_topk(self):
        x = torch.arange(1., 6., requires_grad=True)
        k = torch.tensor(3)
        self.assertONNX(lambda x, k: torch.topk(x, k), (x, k), opset_version=10)

    def test_topk_smallest_unsorted(self):
        x = torch.arange(1., 6., requires_grad=True)
        k = torch.tensor(3)
        self.assertONNX(lambda x, k: torch.topk(x, k, largest=False, sorted=False), (x, k), opset_version=11)

    def test_baddbmm(self):
        x = torch.randn(10, 3, 5)
        b1 = torch.randn(10, 3, 4)
        b2 = torch.randn(10, 4, 5)
        self.assertONNX(lambda x, b1, b2: torch.baddbmm(x, b1, b2), (x, b1, b2))

    def test_round(self):
        x = torch.tensor([0.9920, -1.0362, -1.5000, 2.5000], requires_grad=True)
        self.assertONNX(lambda x: torch.round(x), x, opset_version=11)

    def test_dim(self):
        x = torch.ones((2, 2), requires_grad=True)
        self.assertONNX(lambda x: torch.scalar_tensor(x.dim()), x)

    @skipIfNoLapack
    def test_det(self):
        x = torch.randn(2, 3, 5, 5, device=torch.device('cpu'))
        self.assertONNX(lambda x: torch.det(x), x, opset_version=11)


if __name__ == '__main__':
    no_onnx_dep_flag = '--no-onnx'
    _onnx_dep = no_onnx_dep_flag not in common.UNITTEST_ARGS
    if no_onnx_dep_flag in common.UNITTEST_ARGS:
        common.UNITTEST_ARGS.remove(no_onnx_dep_flag)
    onnx_test_flag = '--produce-onnx-test-data'
    _onnx_test = onnx_test_flag in common.UNITTEST_ARGS
    if onnx_test_flag in common.UNITTEST_ARGS:
        common.UNITTEST_ARGS.remove(onnx_test_flag)
    if _onnx_test:
        _onnx_dep = True
        import test_onnx_common
        for d in glob.glob(os.path.join(test_onnx_common.pytorch_operator_dir, "test_operator_*")):
            shutil.rmtree(d)
    run_tests()
