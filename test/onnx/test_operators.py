from test_pytorch_common import TestCase, run_tests, skipIfNoLapack, flatten

import torch
import torch.onnx
from torch.autograd import Variable, Function
from torch.nn import Module
import torch.nn as nn

import itertools
import io
import unittest
import inspect
import argparse
import glob
import os
import shutil
import sys
import common_utils as common

from test_pytorch_common import skipIfCI


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
        self.assertONNX(lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))), x, params=(y, ))

    def test_symbolic_mismatch(self):
        class MyFun(Function):
            @staticmethod
            def symbolic(g, x):
                # The inside of this function should never be invoked, because
                # we will fail due to an argument mismatch first.
                assert False

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
        self.assertONNX(nn.BatchNorm2d(2), x)

    def test_batchnorm_1d(self):
        x = torch.ones(2, 2, requires_grad=True)
        self.assertONNX(nn.BatchNorm1d(2), x)

    def test_batchnorm_training(self):
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        self.assertONNX(nn.BatchNorm2d(2), x, training=True)

    def test_conv(self):
        x = torch.ones(20, 16, 50, 40, requires_grad=True)
        self.assertONNX(nn.Conv2d(16, 13, 3, bias=False), x)

    def test_convtranspose(self):
        x = torch.ones(2, 3, 4, 5, requires_grad=True)
        self.assertONNX(nn.ConvTranspose2d(3, 3, 3, stride=3, bias=False,
                                           padding=1, output_padding=2), x)

    def test_maxpool(self):
        x = torch.randn(20, 16, 50)
        self.assertONNX(nn.MaxPool1d(3, stride=2), x)

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
        self.assertONNX(lambda x: torch.mean(x, dim=2, keepdim=True), x)

    def test_sum(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x), x)

    def test_reduced_sum(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x, dim=2), x)

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

    def test_sqrt(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sqrt(x), x)

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

    def test_atan(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.atan(), x)

    def test_flatten(self):
        # Flatten is a special case of Reshape when the output is a 2-D tensor.
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.view(x.size()[0], x.numel() // x.size()[0]), x)

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

    def test_norm(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.norm(p=2, dim=2), (x))

    def test_upsample(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: nn.functional.interpolate(x, scale_factor=2., mode='bilinear'), x)

    def test_unsqueeze(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.unsqueeze(len(x.shape)), x)

    def test_batchnorm_noaffine(self):
        x = torch.randn(128, 128, 1, 1, requires_grad=True)
        self.assertONNX(nn.BatchNorm2d(128, affine=False), x)

    def test_embedding_bags(self):
        emb_bag = nn.EmbeddingBag(10, 8)
        input = torch.tensor([1, 2, 3, 4]).long()
        offset = torch.tensor([0]).long()
        self.assertONNX(emb_bag, (input, offset))

    def test_implicit_expand(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x + 1, x)

    def test_reduce_sum_negative_indices(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.sum(-1), x)

    def test_randn(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(lambda x: torch.randn(1, 2, 3, 4) + x, x)

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
