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
import common


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
    def __init__(self, f, params=tuple()):
        super(FuncModule, self).__init__()
        self.f = f
        self.params = nn.ParameterList(list(params))

    def forward(self, *args):
        return self.f(*itertools.chain(args, self.params))


class TestOperators(TestCase):

    def assertONNX(self, f, args, params=tuple(), **kwargs):
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

    def assertONNXRaises(self, err, f, args, params=tuple(), **kwargs):
        if isinstance(f, nn.Module):
            m = f
        else:
            m = FuncModule(f, params)
        self.assertExpectedRaises(err, lambda: export_to_pbtxt(m, args, **kwargs))

    def assertONNXRaisesRegex(self, err, reg, f, args, params=tuple(), **kwargs):
        if isinstance(f, nn.Module):
            m = f
        else:
            m = FuncModule(f, params)
        with self.assertRaisesRegex(err, reg):
            export_to_pbtxt(m, args, **kwargs)

    def test_basic(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)
        self.assertONNX(lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))), (x, y))

    def test_view(self):
        x = Variable(torch.Tensor([0]), requires_grad=True)
        self.assertONNX(lambda x: x.view(1, 1), x)

    def test_index(self):
        x = Variable(torch.Tensor([[0]]), requires_grad=True)
        self.assertONNX(lambda x: x[0], x)

    def test_type_as(self):
        x = Variable(torch.Tensor([0]), requires_grad=True)
        self.assertONNX(lambda x: x.type_as(x), x)

    def test_addconstant(self):
        x = Variable(torch.DoubleTensor(2, 3), requires_grad=True)
        self.assertONNX(lambda x: x + 1, x)

    def test_add_broadcast(self):
        x = Variable(torch.DoubleTensor(2, 3), requires_grad=True)
        y = Variable(torch.DoubleTensor(3), requires_grad=True)
        self.assertONNX(lambda x, y: x + y, (x, y))

    def test_add_left_broadcast(self):
        x = Variable(torch.DoubleTensor(3), requires_grad=True)
        y = Variable(torch.DoubleTensor(2, 3), requires_grad=True)
        self.assertONNX(lambda x, y: x + y, (x, y))

    def test_add_size1_broadcast(self):
        x = Variable(torch.DoubleTensor(2, 3), requires_grad=True)
        y = Variable(torch.DoubleTensor(2, 1), requires_grad=True)
        self.assertONNX(lambda x, y: x + y, (x, y))

    def test_add_size1_right_broadcast(self):
        x = Variable(torch.DoubleTensor(2, 3), requires_grad=True)
        y = Variable(torch.DoubleTensor(3), requires_grad=True)
        self.assertONNX(lambda x, y: x + y, (x, y))

    def test_add_size1_singleton_broadcast(self):
        x = Variable(torch.DoubleTensor(2, 3), requires_grad=True)
        y = Variable(torch.DoubleTensor(1, 3), requires_grad=True)
        self.assertONNX(lambda x, y: x + y, (x, y))

    def test_rsub(self):
        x = Variable(torch.DoubleTensor(2, 3), requires_grad=True)
        self.assertONNX(lambda x: 1 - x, (x,))

    def test_transpose(self):
        x = Variable(torch.Tensor([[0, 1], [2, 3]]), requires_grad=True)
        self.assertONNX(lambda x: x.transpose(0, 1).transpose(1, 0), x)

    def test_chunk(self):
        x = Variable(torch.Tensor([0, 1, 2]), requires_grad=True)
        self.assertONNX(lambda x: x.chunk(2), x)

    def test_concat2(self):
        x = Variable(torch.randn(2, 3))
        y = Variable(torch.randn(2, 3))
        self.assertONNX(lambda inputs: torch.cat(inputs, 1), ((x, y),))

    def test_mm(self):
        m1 = Variable(torch.randn(2, 3), requires_grad=True)
        m2 = Variable(torch.randn(3, 4), requires_grad=True)
        self.assertONNX(torch.mm, (m1, m2))

    def test_addmm(self):
        m1 = Variable(torch.randn(2, 3), requires_grad=True)
        m2 = Variable(torch.randn(3, 4), requires_grad=True)
        m3 = Variable(torch.randn(4), requires_grad=True)
        self.assertONNX(lambda x, y, z: torch.addmm(torch.addmm(z, x, y), x, y), (m1, m2, m3))

    def test_permute2(self):
        x = Variable(torch.Tensor([[[[[[0]]]]]]), requires_grad=True)
        self.assertONNX(lambda x: x.permute(0, 1, 4, 2, 5, 3), x)

    def test_pad(self):
        x = Variable(torch.Tensor([[[[0, 1, 1, 1], [2, 3, 7, 7]]]]), requires_grad=True)
        self.assertONNX(nn.ReflectionPad2d((2, 3, 0, 1)), x)

    def test_params(self):
        x = Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
        y = nn.Parameter(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
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

        x = Variable(torch.randn(2, 2).fill_(1.0))
        y = Variable(torch.randn(2, 2).fill_(1.0))
        # NB: Don't use expect test here, the type error wobbles depending
        # on Python version
        with self.assertRaisesRegex(TypeError, "occurred when translating MyFun"):
            export_to_pbtxt(FuncModule(MyFun().apply), (x, y))

    # TODO: Do an nn style test for these
    def test_batchnorm(self):
        x = Variable(torch.randn(2, 2, 2, 2).fill_(1.0), requires_grad=True)
        self.assertONNX(nn.BatchNorm2d(2), x)

    def test_batchnorm_1d(self):
        x = Variable(torch.randn(2, 2).fill_(1.0), requires_grad=True)
        self.assertONNX(nn.BatchNorm1d(2), x)

    def test_batchnorm_training(self):
        x = Variable(torch.randn(2, 2, 2, 2).fill_(1.0), requires_grad=True)
        self.assertONNX(nn.BatchNorm2d(2), x, training=True)

    def test_conv(self):
        x = Variable(torch.randn(20, 16, 50, 40).fill_(1.0), requires_grad=True)
        self.assertONNX(nn.Conv2d(16, 13, 3, bias=False), x)

    def test_convtranspose(self):
        x = Variable(torch.randn(2, 3, 4, 5).fill_(1.0), requires_grad=True)
        self.assertONNX(nn.ConvTranspose2d(3, 3, 3, stride=3, bias=False,
                                           padding=1, output_padding=2), x)

    def test_maxpool(self):
        x = Variable(torch.randn(20, 16, 50))
        self.assertONNX(nn.MaxPool1d(3, stride=2), x)

    def test_at_op(self):
        x = Variable(torch.randn(3, 4))

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
        x = Variable(torch.randn(3, 4), requires_grad=True)
        self.assertONNX(lambda x: torch.clamp(x, min=-0.5, max=0.5), x)

    def test_clip_min(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: x.clamp(min=-0.1), x)

    def test_clip_max(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: x.clamp(max=0.1), x)

    def test_hardtanh(self):
        x = Variable(torch.randn(3, 4), requires_grad=True)
        self.assertONNX(lambda x: torch.nn.Hardtanh(-0.5, 0.5)(x), x)

    def test_max(self):
        x = Variable(torch.randn(3, 4), requires_grad=True)
        y = Variable(torch.randn(3, 4), requires_grad=True)
        self.assertONNX(lambda x, y: torch.max(x, y), (x, y))

    def test_min(self):
        x = Variable(torch.randn(3, 4), requires_grad=True)
        y = Variable(torch.randn(3, 4), requires_grad=True)
        self.assertONNX(lambda x, y: torch.min(x, y), (x, y))

    def test_mean(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: torch.mean(x), x)

    def test_reduced_mean(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: torch.mean(x, dim=2), x)

    def test_reduced_mean_keepdim(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: torch.mean(x, dim=2, keepdim=True), x)

    def test_sum(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x), x)

    def test_reduced_sum(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x, dim=2), x)

    def test_reduced_sum_keepdim(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x, dim=2, keepdim=True), x)

    def test_prod(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: torch.prod(x), x)

    def test_reduced_prod(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: torch.prod(x, dim=2), x)

    def test_reduced_prod_keepdim(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: torch.prod(x, dim=2, keepdim=True), x)

    def test_sqrt(self):
        x = Variable(torch.randn(3, 4), requires_grad=True)
        self.assertONNX(lambda x: torch.sqrt(x), x)

    def test_equal(self):
        x = Variable(torch.randn(3, 4).int(), requires_grad=False)
        y = Variable(torch.randn(3, 4).int(), requires_grad=False)
        self.assertONNX(lambda x, y: x == y, (x, y))

    def test_lt(self):
        x = Variable(torch.randn(3, 4).int(), requires_grad=False)
        y = Variable(torch.randn(3, 4).int(), requires_grad=False)
        self.assertONNX(lambda x, y: x < y, (x, y))

    def test_gt(self):
        x = Variable(torch.randn(3, 4).int(), requires_grad=False)
        y = Variable(torch.randn(3, 4).int(), requires_grad=False)
        self.assertONNX(lambda x, y: x > y, (x, y))

    def test_le(self):
        x = Variable(torch.randn(3, 4).int(), requires_grad=False)
        y = Variable(torch.randn(3, 4).int(), requires_grad=False)
        self.assertONNX(lambda x, y: x <= y, (x, y))

    def test_ge(self):
        x = Variable(torch.randn(3, 4).int(), requires_grad=False)
        y = Variable(torch.randn(3, 4).int(), requires_grad=False)
        self.assertONNX(lambda x, y: x >= y, (x, y))

    def test_exp(self):
        x = Variable(torch.randn(3, 4), requires_grad=True)
        self.assertONNX(lambda x: x.exp(), x)

    def test_sin(self):
        x = Variable(torch.randn(3, 4), requires_grad=True)
        self.assertONNX(lambda x: x.sin(), x)

    def test_cos(self):
        x = Variable(torch.randn(3, 4), requires_grad=True)
        self.assertONNX(lambda x: x.cos(), x)

    def test_tan(self):
        x = Variable(torch.randn(3, 4), requires_grad=True)
        self.assertONNX(lambda x: x.tan(), x)

    def test_asin(self):
        x = Variable(torch.rand(3, 4), requires_grad=True)
        self.assertONNX(lambda x: x.asin(), x)

    def test_acos(self):
        x = Variable(torch.rand(3, 4), requires_grad=True)
        self.assertONNX(lambda x: x.acos(), x)

    def test_atan(self):
        x = Variable(torch.randn(3, 4), requires_grad=True)
        self.assertONNX(lambda x: x.atan(), x)

    def test_flatten(self):
        # Flatten is a special case of Reshape when the output is a 2-D tensor.
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: x.view(x.size()[0], x.numel() // x.size()[0]), x)

    def test_logsoftmax(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(nn.LogSoftmax(dim=3), x)

    def test_pow(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        y = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x, y: x.pow(y), (x, y))

    def test_elu(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(nn.ELU(), x)

    def test_selu(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(nn.SELU(), x)

    def test_repeat(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: x.repeat(1, 2, 3, 4), x)

    def test_repeat_dim_overflow(self):
        x = Variable(torch.randn(1, 2), requires_grad=True)
        self.assertONNX(lambda x: x.repeat(1, 2, 3, 4), x)

    def test_norm(self):
        x = Variable(torch.randn(1, 2, 3, 4), requires_grad=True)
        self.assertONNX(lambda x: x.norm(dim=2), (x))

    def test_symbolic_override(self):
        """Lifted from fast-neural-style: custom implementation of instance norm
        to be mapped to ONNX operator"""

        class CustomInstanceNorm(torch.nn.Module):
            def __init__(self, dim, eps=1e-9):
                super(CustomInstanceNorm, self).__init__()
                self.scale = nn.Parameter(torch.FloatTensor(dim).uniform_())
                self.shift = nn.Parameter(torch.FloatTensor(dim).zero_())
                self.eps = eps

            def forward(self, x):
                return self._run_forward(x, self.scale, self.shift, eps=self.eps)

            @staticmethod
            @torch.onnx.symbolic_override(
                lambda g, x, scale, shift, eps: g.op(
                    'InstanceNormalization', x, scale, shift, epsilon_f=eps)
            )
            def _run_forward(x, scale, shift, eps):
                # since we hand-roll instance norm it doesn't perform well all in fp16
                n = x.size(2) * x.size(3)
                t = x.view(x.size(0), x.size(1), n)
                mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
                # Calculate the biased var. torch.var returns unbiased var
                var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((float(n) - 1) / float(n))
                scale_broadcast = scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
                scale_broadcast = scale_broadcast.expand_as(x)
                shift_broadcast = shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
                shift_broadcast = shift_broadcast.expand_as(x)
                out = (x - mean) / torch.sqrt(var + eps)
                out = out * scale_broadcast + shift_broadcast
                return out

        instnorm = CustomInstanceNorm(10)
        x = Variable(torch.randn(2, 10, 32, 32))
        self.assertONNX(instnorm, x)

    """
    def test_rnn(self):
        rnn = nn.RNN(30, 20, 2)
        input = Variable(torch.randn(10, 32, 30))
        output, hidden = rnn(input)
        self.assertONNX(rnn, input)
    """

    def test_symbolic_override_nested(self):
        def symb(g, x, y):
            assert isinstance(x, torch._C.Value)
            assert isinstance(y[0], torch._C.Value)
            assert isinstance(y[1], torch._C.Value)
            return g.op('Sum', x, y[0], y[1]), (
                g.op('Neg', x), g.op('Neg', y[0]))

        @torch.onnx.symbolic_override(symb)
        def foo(x, y):
            return x + y[0] + y[1], (-x, -y[0])

        class BigModule(torch.nn.Module):
            def forward(self, x, y):
                return foo(x, y)

        inp = (Variable(torch.FloatTensor([1])),
               (Variable(torch.FloatTensor([2])),
                Variable(torch.FloatTensor([3]))))
        BigModule()(*inp)
        self.assertONNX(BigModule(), inp)


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
