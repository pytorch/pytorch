# Owner(s): ["module: onnx"]

"""
Usage: python test/onnx/test_operators.py [--no-onnx] [--produce-onnx-test-data]
          --no-onnx: no onnx python dependency
          --produce-onnx-test-data: generate onnx test data
          --accept: accept onnx updates and overwrite models
"""
import glob
import inspect
import io
import itertools
import os
import shutil
import tempfile

# Full diff for expect files
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

from pytorch_test_common import (
    BATCH_SIZE,
    flatten,
    RNN_HIDDEN_SIZE,
    RNN_INPUT_SIZE,
    RNN_SEQUENCE_LENGTH,
)
from torch.autograd import Function, Variable
from torch.nn import functional, Module
from torch.onnx.symbolic_helper import (
    _get_tensor_dim_size,
    _get_tensor_sizes,
    parse_args,
)
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfCaffe2, skipIfNoLapack

unittest.TestCase.maxDiff = None

_onnx_test = False  # flag to produce onnx test cases.
_onnx_dep = True  # flag to import onnx package.


def export_to_pbtxt(model, inputs, *args, **kwargs):
    return torch.onnx.export_to_pretty_string(
        model, inputs, google_printer=True, *args, **kwargs
    )


def export_to_pb(model, inputs, *args, **kwargs):
    f = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, inputs, f, *args, **kwargs)
    return f.getvalue()


class FuncModule(Module):
    def __init__(self, f, params=None):
        if params is None:
            params = ()
        super().__init__()
        self.f = f
        self.params = nn.ParameterList(list(params))

    def forward(self, *args):
        return self.f(*itertools.chain(args, self.params))


class TestOperators(common_utils.TestCase):
    def assertONNX(self, f, args, params=None, **kwargs):
        if params is None:
            params = ()
        if isinstance(f, nn.Module):
            m = f
        else:
            m = FuncModule(f, params)
        m.eval()
        onnx_model_pbtxt = export_to_pbtxt(m, args, **kwargs)
        subname = kwargs.pop("subname", None)
        self.assertExpected(onnx_model_pbtxt, subname)
        if _onnx_dep:
            onnx_model_pb = export_to_pb(m, args, **kwargs)
            import onnx
            import onnx.checker
            import onnx.numpy_helper
            import onnx_test_common

            model_def = onnx.ModelProto.FromString(onnx_model_pb)
            onnx.checker.check_model(model_def)
            if _onnx_test:
                test_function = inspect.stack()[1][0].f_code.co_name
                test_name = test_function[0:4] + "_operator" + test_function[4:]
                output_dir = os.path.join(
                    onnx_test_common.pytorch_operator_dir, test_name
                )
                # Assume:
                #     1) the old test should be delete before the test.
                #     2) only one assertONNX in each test, otherwise will override the data.
                assert not os.path.exists(output_dir), "{} should not exist!".format(
                    output_dir
                )
                os.makedirs(output_dir)
                with open(os.path.join(output_dir, "model.onnx"), "wb") as file:
                    file.write(model_def.SerializeToString())
                data_dir = os.path.join(output_dir, "test_data_set_0")
                os.makedirs(data_dir)
                if isinstance(args, Variable):
                    args = (args,)
                for index, var in enumerate(flatten(args)):
                    tensor = onnx.numpy_helper.from_array(var.data.numpy())
                    with open(
                        os.path.join(data_dir, f"input_{index}.pb"), "wb"
                    ) as file:
                        file.write(tensor.SerializeToString())
                outputs = m(*args)
                if isinstance(outputs, Variable):
                    outputs = (outputs,)
                for index, var in enumerate(flatten(outputs)):
                    tensor = onnx.numpy_helper.from_array(var.data.numpy())
                    with open(
                        os.path.join(data_dir, f"output_{index}.pb"), "wb"
                    ) as file:
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

    def test_mul_bool(self):
        x = torch.tensor([True, False, True, False])
        y = torch.tensor([True, True, False, False])
        self.assertONNX(lambda x, y: torch.mul(x, y), (x, y))

    def test_mul_fp_bool(self):
        x = torch.tensor([9.4, 1.7, 3.6])
        y = torch.tensor([True, True, False])
        self.assertONNX(lambda x, y: torch.mul(x, y), (x, y))

    def test_transpose(self):
        x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
        self.assertONNX(lambda x: x.transpose(0, 1).transpose(1, 0), x)

    def test_chunk(self):
        x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        self.assertONNX(lambda x: x.chunk(2), x)

    def test_split(self):
        x = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]]
        )
        self.assertONNX(lambda x: torch.split(x, 2, 1), x)

    def test_split_with_sizes(self):
        x = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]]
        )
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
        self.assertONNX(
            lambda x, y, z: torch.addmm(torch.addmm(z, x, y), x, y), (m1, m2, m3)
        )

    def test_permute2(self):
        x = torch.tensor([[[[[[0.0]]]]]], requires_grad=True)
        self.assertONNX(lambda x: x.permute(0, 1, 4, 2, 5, 3), x)

    def test_pad(self):
        x = torch.tensor(
            [[[[0.0, 1.0, 1.0, 1.0], [2.0, 3.0, 7.0, 7.0]]]], requires_grad=True
        )
        self.assertONNX(nn.ReflectionPad2d((2, 3, 0, 1)), x)

    def test_params(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
        self.assertONNX(
            lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))),
            x,
            params=(y,),
            keep_initializers_as_inputs=True,
        )

    def test_params_onnx_irv4(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
        self.assertONNX(
            lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))),
            x,
            params=(y,),
            keep_initializers_as_inputs=False,
        )

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
        self.assertONNX(
            nn.BatchNorm2d(2),
            x,
            training=torch.onnx.TrainingMode.TRAINING,
            keep_initializers_as_inputs=True,
        )

    def test_conv(self):
        x = torch.ones(20, 16, 50, 40, requires_grad=True)
        self.assertONNX(
            nn.Conv2d(16, 13, 3, bias=False), x, keep_initializers_as_inputs=True
        )

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
        self.assertONNX(
            conv_node, x, opset_version=8, keep_initializers_as_inputs=False
        )

    def test_conv_variable_length(self):
        x = torch.ones(5, 3, 6, 6, requires_grad=True)
        model = torch.nn.Conv2d(3, 2, 3)

        dynamic_axes = {
            "input_1": [0, 2, 3],
            "output_1": {0: "output_1_variable_dim_0", 1: "output_1_variable_dim_1"},
        }
        model_proto_file = tempfile.NamedTemporaryFile()
        torch.onnx.export(
            model,
            x,
            model_proto_file.name,
            verbose=True,
            input_names=["input_1"],
            output_names=["output_1"],
            dynamic_axes=dynamic_axes,
        )

        import onnx

        onnx_model = onnx.load(model_proto_file.name)
        onnx.checker.check_model(onnx_model)

        # Asserting the default dynamic axes names are generated when custom names are not provided
        assert (
            onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param
            == "input_1_dynamic_axes_1"
        )
        assert (
            onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_param
            == "input_1_dynamic_axes_2"
        )
        assert (
            onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_param
            == "input_1_dynamic_axes_3"
        )

        # Asserting the custom names are applied when provided
        assert (
            onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param
            == "output_1_variable_dim_0"
        )
        assert (
            onnx_model.graph.output[0].type.tensor_type.shape.dim[1].dim_param
            == "output_1_variable_dim_1"
        )

    def test_convtranspose(self):
        x = torch.ones(2, 3, 4, 5, requires_grad=True)
        self.assertONNX(
            nn.ConvTranspose2d(
                3, 3, 3, stride=3, bias=False, padding=1, output_padding=2
            ),
            x,
            keep_initializers_as_inputs=True,
        )

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

    @skipIfCaffe2
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

        self.assertONNX(
            MyModule(),
            x,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )

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
        self.assertONNX(lambda x: torch.full(x.shape, 2.0), x)

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
        self.assertONNX(lambda x: torch.mean(x, dtype=torch.double), x)

    def test_reduced_mean_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.mean(x, dim=0, dtype=torch.double), x)

    def test_sum(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x), x)

    def test_sum_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x, dtype=torch.double), x)

    def test_reduced_sum_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x, dim=0, dtype=torch.double), x)

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
        self.assertONNX(lambda x: torch.prod(x, dtype=torch.double), x)

    def test_reduced_prod_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.prod(x, dim=0, dtype=torch.double), x)

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
        self.assertONNX(lambda x: x[x.size(0) :, x.size(1) - 3], x, opset_version=10)

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
        x = torch.tensor([1, float("nan"), 2])
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

    @unittest.skip("It started failing after #81761")
    # TODO(#83661): Fix and enable the test
    def test_norm_p1(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.norm(p=1, dim=2), (x))

    @unittest.skip("It started failing after #81761")
    # TODO(#83661): Fix and enable the test
    def test_norm_p2(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.norm(p=2, dim=2), (x))

    def test_upsample_nearest_scale(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: nn.functional.interpolate(
                x, scale_factor=2.0, mode="nearest", recompute_scale_factor=False
            ),
            x,
        )

    def test_upsample_nearest_scale_default_scale_factor(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: nn.functional.interpolate(x, scale_factor=2.0, mode="nearest"), x
        )

    def test_upsample_nearest_size(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: nn.functional.interpolate(x, size=16, mode="nearest"), x
        )

    def test_unsqueeze(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.unsqueeze(len(x.shape)), x)

    def test_batchnorm_noaffine(self):
        x = torch.randn(128, 128, 1, 1, requires_grad=True)
        self.assertONNX(
            nn.BatchNorm2d(128, affine=False, momentum=0.3),
            x,
            keep_initializers_as_inputs=True,
        )

    @skipIfCaffe2
    def test_embedding_bags(self):
        emb_bag = nn.EmbeddingBag(10, 8)
        input = torch.tensor([1, 2, 3, 4]).long()
        offset = torch.tensor([0]).long()
        self.assertONNX(
            emb_bag,
            (input, offset),
            keep_initializers_as_inputs=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )

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
        self.assertONNX(
            torch.nn.Linear(4, 5, bias=True), x, keep_initializers_as_inputs=True
        )

    def test_empty_like(self):
        x = torch.randn(5, 8, requires_grad=True)
        self.assertONNX(lambda x: torch.empty_like(x), x)

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

    def test_dropout_default(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(
                functional.dropout(
                    x,
                )
            ),
            x,
        )

    def test_dropout_training(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(functional.dropout(x)),
            x,
            training=torch.onnx.TrainingMode.TRAINING,
        )

    def test_dropout_opset12(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(functional.dropout(x, training=False)),
            x,
            opset_version=12,
        )

    def test_dropout_training_opset12(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(functional.dropout(x)),
            x,
            opset_version=12,
            training=torch.onnx.TrainingMode.TRAINING,
        )

    def test_nonzero(self):
        x = torch.tensor(
            [[[2.0, 2.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]], requires_grad=True
        )
        self.assertONNX(lambda x: torch.nonzero(x), x)

    def test_gather(self):
        data = torch.randn(3, 4, 3, requires_grad=True)
        index = torch.tensor([2, 0]).view(1, 2, 1).expand(3, 2, 3)
        self.assertONNX(lambda data, index: data.gather(1, index), (data, index))

    def test_gather_opset11(self):
        data = torch.randn(3, 4, 3, requires_grad=True)
        index = torch.tensor([2, 0]).view(1, 2, 1).expand(3, 2, 3)
        self.assertONNX(
            lambda data, index: data.gather(1, index), (data, index), opset_version=11
        )

    def test_scatter_add(self):
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.assertONNX(
            lambda data, index: data.scatter_add(1, indices, values),
            (data, (indices, values)),
        )

    def test_scatter_add_opset11(self):
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.assertONNX(
            lambda data, index: data.scatter_add(1, indices, values),
            (data, (indices, values)),
            opset_version=11,
        )

    def test_scatter_add_opset16(self):
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[0, 0], [1, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.assertONNX(
            lambda data, index: data.scatter_add(1, indices, values),
            (data, (indices, values)),
            opset_version=16,
        )

    def test_master_opset(self):
        x = torch.randn(2, 3).float()
        y = torch.randn(2, 3).float()
        self.assertONNX(lambda x, y: x + y, (x, y), opset_version=10)

    def test_std(self):
        x = torch.randn(2, 3, 4).float()
        self.assertONNX(
            lambda x: torch.std(x, dim=(0, 1), unbiased=True, keepdim=True), x
        )

    def test_cumsum(self):
        x = torch.randn(2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.cumsum(x, dim=1), x, opset_version=11)

    # Github Issue: https://github.com/pytorch/pytorch/issues/71095
    #    def test_c2_op(self):
    #        class MyModel(torch.nn.Module):
    #            def __init__(self):
    #                super(MyModel, self).__init__()
    #
    #            def forward(self, scores, bbox_deltas, im_info, anchors):
    #                a, b = torch.ops._caffe2.GenerateProposals(
    #                    (scores), (bbox_deltas), (im_info), (anchors),
    #                    2.0, 6000, 300, 0.7, 16, True, -90, 90, 1.0, True,
    #                )
    #                return a, b
    #
    #        model = MyModel()
    #        A = 4
    #        H = 10
    #        W = 8
    #        img_count = 3
    #        scores = torch.ones(img_count, A, H, W, dtype=torch.float32)
    #        bbox_deltas = torch.linspace(0, 10, steps=img_count * 4 * A * H * W,
    #                                     dtype=torch.float32)
    #        bbox_deltas = bbox_deltas.view(img_count, 4 * A, H, W)
    #        im_info = torch.ones(img_count, 3, dtype=torch.float32)
    #        anchors = torch.ones(A, 4, dtype=torch.float32)
    #        inputs = (scores, bbox_deltas, im_info, anchors)
    #        self.assertONNX(model, inputs, custom_opsets={"org.pytorch._caffe2": 0})

    def test_dict(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(
                    x_in[list(x_in.keys())[0]], list(x_in.keys())[0]
                )
                return x_out

        x = {torch.tensor(1.0): torch.randn(1, 2, 3)}
        self.assertONNX(MyModel(), (x, {}))

    def test_dict_str(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in["test_key_in"], 2.0)
                return x_out

        x = {"test_key_in": torch.randn(1, 2, 3)}
        self.assertONNX(MyModel(), (x, {}))

    def test_arange_dynamic(self):
        class TestModel(torch.nn.Module):
            def forward(self, input):
                return torch.arange(input.shape[0], input.shape[0] + 5, 0.5)

        input = torch.randn(5, 3, 2)
        self.assertONNX(TestModel(), input, opset_version=11)

    def test_bitshift(self):
        class BitshiftModel(torch.nn.Module):
            def forward(self, input):
                return input >> 1, input >> 2

        input = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        self.assertONNX(BitshiftModel(), input, opset_version=11)

    @skipIfCaffe2
    def test_layer_norm_aten(self):
        model = torch.nn.LayerNorm([10, 10])
        x = torch.randn(20, 5, 10, 10)
        self.assertONNX(
            model,
            x,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )

    def test_pixel_shuffle(self):
        x = torch.randn(2, 8, 3, 4).float()
        self.assertONNX(
            lambda x: torch.pixel_shuffle(x, upscale_factor=2), x, opset_version=11
        )

    @unittest.skip("It started failing after #81761")
    # TODO(#83661): Fix and enable the test
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
        self.assertONNX(
            lambda x: torch.unique(
                x, dim=0, sorted=True, return_inverse=False, return_counts=True
            ),
            x,
            opset_version=11,
        )

    def test_meshgrid(self):
        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.ones(5, requires_grad=True)
        self.assertONNX(lambda x, y, z: torch.meshgrid(x, y, z), (x, y, z))

    def test_topk(self):
        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        self.assertONNX(lambda x, k: torch.topk(x, k), (x, k), opset_version=10)

    def test_topk_smallest_unsorted(self):
        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        self.assertONNX(
            lambda x, k: torch.topk(x, k, largest=False, sorted=False),
            (x, k),
            opset_version=11,
        )

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
        x = torch.randn(2, 3, 5, 5, device=torch.device("cpu"))
        self.assertONNX(lambda x: torch.det(x), x, opset_version=11)
        self.assertONNX(lambda x: torch.linalg.det(x), x, opset_version=11)

    def test_softmaxcrossentropy(self):
        x = torch.randn(3, 5)
        y = torch.empty(3, dtype=torch.long).random_(5)
        self.assertONNX(torch.nn.CrossEntropyLoss(), (x, y), opset_version=12)

    def test_softmaxcrossentropy_ignore_index(self):
        x = torch.randn(3, 5)
        y = torch.empty(3, dtype=torch.long).random_(5)
        self.assertONNX(
            torch.nn.CrossEntropyLoss(ignore_index=1), (x, y), opset_version=12
        )

    def test_softmaxcrossentropy_weights(self):
        x = torch.randn(3, 5)
        y = torch.empty(3, dtype=torch.long).random_(5)
        self.assertONNX(
            torch.nn.CrossEntropyLoss(weight=torch.randn(5)), (x, y), opset_version=12
        )

    def test_softmaxcrossentropy_3d(self):
        x = torch.randn(3, 5, 2)
        y = torch.empty(3, 2, dtype=torch.long).random_(5)
        self.assertONNX(torch.nn.CrossEntropyLoss(), (x, y), opset_version=12)

    def test_softmaxcrossentropy_3d_none(self):
        x = torch.randn(3, 5, 2)
        y = torch.empty(3, 2, dtype=torch.long).random_(5)
        self.assertONNX(
            torch.nn.CrossEntropyLoss(reduction="none"), (x, y), opset_version=12
        )

    def test_softmaxcrossentropy_4d(self):
        x = torch.randn(3, 5, 2, 1)
        y = torch.empty(3, 2, 1, dtype=torch.long).random_(5)
        self.assertONNX(torch.nn.CrossEntropyLoss(), (x, y), opset_version=12)

    def test_lstm_none_sequence_lens(self):
        """Test symbolic shape inference for LSTM when the input sequence_lens = None."""
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)

        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = torch.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )

            def forward(self, x, h0, c0):
                a, b = self.rnn(x, (h0, c0))
                return torch.ones(b[0].shape)

        self.assertONNX(
            LSTMModel(),
            (input, h0, c0),
            input_names=["x", "y"],
            dynamic_axes={"x": {0: "batch"}},
            opset_version=12,
        )

    def test_dynamic_axes_add(self):
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(2, 1, requires_grad=True)
        self.assertONNX(
            lambda x, y: torch.add(x, y),
            (m1, m2),
            input_names=["input_1", "input_2"],
            dynamic_axes={"input_1": {1: "dim_1"}, "input_2": {1: "dim_2"}},
            opset_version=12,
        )

    def test_dynamic_axes_add_inputs_same_symbolic_shape(self):
        m1 = torch.randn(2, 3, requires_grad=True)
        self.assertONNX(
            lambda x: torch.add(x, x),
            (m1,),
            input_names=["input_1"],
            dynamic_axes={"input_1": {1: "dim_1"}},
            opset_version=12,
        )

    def test_dynamic_axes_matmul(self):
        m1 = torch.randn(2, 2, 4, requires_grad=True)
        m2 = torch.randn(2, 4, 3, requires_grad=True)
        self.assertONNX(
            lambda x, y: torch.matmul(x, y),
            (m1, m2),
            input_names=["input_1", "input_2"],
            dynamic_axes={"input_1": {1: "dim_0"}, "input_2": {2: "dim_1"}},
            opset_version=12,
        )

    def test_dynamic_axes_reduce_mean(self):
        m1 = torch.randn(2, 3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.mean(x, dim=1),
            (m1),
            input_names=["input"],
            dynamic_axes={"input": {1: "dim_1", 2: "dim_2"}},
            opset_version=12,
        )

    def test_dynamic_axes_unchange(self):
        """Test ProcessUnchangeNode in symbolic shape inference."""
        m1 = torch.randn(2, 3, requires_grad=True)
        self.assertONNX(
            lambda x: torch.softmax(x, dim=0),
            (m1,),
            input_names=["input"],
            dynamic_axes={"input": {1: "dim_1"}},
            opset_version=12,
        )

    def test_aten_embedding_1(self):
        _onnx_opset_version = 12

        @parse_args("v", "v", "i", "b", "b")
        def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
            custom_attributes_json = (
                "{"
                f'"padding_idx":{str(padding_idx)},'
                f'"scale_grad_by_freq":{str(scale_grad_by_freq).lower()},'
                f'"sparse":{str(sparse).lower()}'
                "}"
            )
            output = g.at(
                "embedding",
                weight,
                indices,
                custom_attributes_json_s=custom_attributes_json,
            )
            return output

        torch.onnx.register_custom_op_symbolic(
            "::embedding", embedding, _onnx_opset_version
        )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(4, 8)

            def forward(self, x, y):
                res = self.emb(x)
                res = res + y
                return torch.ones(res.shape[0])

        model = Model()
        x = torch.ones(32, dtype=torch.long)
        y = torch.randn(1, 8)
        self.assertONNX(model, (x, y), opset_version=_onnx_opset_version)

        torch.onnx.unregister_custom_op_symbolic("::embedding", _onnx_opset_version)

    # This is test_aten_embedding_1 with shape inference on custom symbolic aten::embedding.
    @skipIfCaffe2
    def test_aten_embedding_2(self):
        _onnx_opset_version = 12

        @parse_args("v", "v", "i", "b", "b")
        def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
            custom_attributes_json = (
                "{"
                f'"padding_idx":{str(padding_idx)},'
                f'"scale_grad_by_freq":{str(scale_grad_by_freq).lower()},'
                f'"sparse":{str(sparse).lower()}'
                "}"
            )
            output = g.at(
                "embedding",
                weight,
                indices,
                custom_attributes_json_s=custom_attributes_json,
            )

            # do shape inference and set it via setType
            indices_shape = _get_tensor_sizes(indices)
            if indices_shape is not None and hasattr(weight.type(), "with_sizes"):
                output_type = weight.type().with_sizes(
                    indices_shape + [_get_tensor_dim_size(weight, 1)]
                )
                output.setType(output_type)
            return output

        torch.onnx.register_custom_op_symbolic(
            "::embedding", embedding, _onnx_opset_version
        )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(4, 8)

            def forward(self, x, y):
                res = self.emb(x)
                res = res + y
                return torch.ones(res.shape[0])

        model = Model()
        x = torch.ones(32, dtype=torch.long)
        y = torch.randn(1, 8)
        self.assertONNX(
            model,
            (x, y),
            opset_version=_onnx_opset_version,
            input_names=["input_1", "input_2"],
            dynamic_axes={"input_1": {0: "dim_0"}, "input_2": {0: "dim_1", 1: "dim_2"}},
            keep_initializers_as_inputs=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )

        torch.onnx.unregister_custom_op_symbolic("::embedding", _onnx_opset_version)

    # Without shapeValueMap, the onnx graph looks like:
    # graph(%0 : Float(*, 1, 128, 1, strides=[128, 128, 1, 1], requires_grad=0, device=cpu)):
    #   %2 : Long(4, strides=[1], device=cpu) = onnx::Shape(%0)
    #   %4 : Long(device=cpu) = onnx::Constant[value={0}]()
    #   %5 : Long(device=cpu) = onnx::Gather[axis=0](%2, %4)
    #   %6 : Long(device=cpu) = onnx::Constant[value={1}]()
    #   %7 : Long(device=cpu) = onnx::Constant[value={2}]()
    #   %8 : Long(device=cpu) = onnx::Constant[value={-1}]()
    #   %9 : int[] = prim::ListConstruct(%5, %6, %7, %8)
    #   %10 : Float(*, *, *, *, strides=[128, 128, 64, 1], requires_grad=0, device=cpu) = onnx::Reshape(%0, %9)
    #   ...
    # With shapeValueMap, it becomes:
    #   ...
    #   %10 : Float(*, 1, 2, 64, strides=[128, 128, 64, 1], requires_grad=0, device=cpu) = onnx::Reshape(%0, %9)
    #   ...
    def test_shape_value_map(self):
        class RSoftMax(torch.nn.Module):
            def __init__(self, radix, cardinality):
                super().__init__()
                self.radix = radix
                self.cardinality = cardinality

            def forward(self, x):
                batch = x.size(0)
                x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
                x = F.softmax(x, dim=1)
                x = x.reshape(batch, -1)
                return x

        radix = 2
        cardinality = 1
        x = torch.randn(10, 1, 128, 1)
        self.assertONNX(
            RSoftMax(radix, cardinality),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": {0: "dim_0"}},
        )


if __name__ == "__main__":
    no_onnx_dep_flag = "--no-onnx"
    _onnx_dep = no_onnx_dep_flag not in common_utils.UNITTEST_ARGS
    if no_onnx_dep_flag in common_utils.UNITTEST_ARGS:
        common_utils.UNITTEST_ARGS.remove(no_onnx_dep_flag)
    onnx_test_flag = "--produce-onnx-test-data"
    _onnx_test = onnx_test_flag in common_utils.UNITTEST_ARGS
    if onnx_test_flag in common_utils.UNITTEST_ARGS:
        common_utils.UNITTEST_ARGS.remove(onnx_test_flag)
    if _onnx_test:
        _onnx_dep = True
        import onnx_test_common

        for d in glob.glob(
            os.path.join(onnx_test_common.pytorch_operator_dir, "test_operator_*")
        ):
            shutil.rmtree(d)
    common_utils.run_tests()
