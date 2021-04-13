#!/usr/bin/env python3
import os
import ctypes
import torch
from typing import Tuple
from torch.backends._nnapi.prepare import convert_model_to_nnapi
from torch.testing._internal.common_utils import TestCase, run_tests


def qpt(t, scale, zero_point, dtype=torch.quint8):
    t = torch.tensor(t)
    return torch.quantize_per_tensor(t, scale, zero_point, dtype)


def nhwc(t):
    t = t.clone().contiguous(memory_format=torch.channels_last)
    t.nnapi_nhwc = True
    return t


class TestNNAPI(TestCase):

    def setUp(self):
        # Avoid saturation in fbgemm
        torch.backends.quantized.engine = 'qnnpack'

        libneuralnetworks_path = os.environ.get("LIBNEURALNETWORKS_PATH")
        if libneuralnetworks_path:
            ctypes.cdll.LoadLibrary(libneuralnetworks_path)
            print("Will attempt to run NNAPI models.")
            self.can_run_nnapi = True
        else:
            self.can_run_nnapi = False

    def check(
        self,
        module,
        arg_or_args,
        *,
        trace_args=None,
        convert_args=None,
        atol_rtol=None,
        limit=None,
    ):
        with torch.no_grad():
            if isinstance(arg_or_args, torch.Tensor):
                args = [arg_or_args]
            else:
                args = arg_or_args
            module.eval()
            traced = torch.jit.trace(module, trace_args or args)
            nnapi_module = convert_model_to_nnapi(traced, convert_args or args)
            if not self.can_run_nnapi:
                # Only test that the model was converted successfully.
                return
            eager_output = module(*args)
            nnapi_output = nnapi_module(*args)
            kwargs = {}
            if atol_rtol is not None:
                kwargs["atol"] = atol_rtol[0]
                kwargs["rtol"] = atol_rtol[1]
            self.assertEqual(eager_output, nnapi_output, **kwargs)
            if limit is not None:
                mismatches = \
                    eager_output.int_repr().to(torch.int32) - \
                    nnapi_output.int_repr().to(torch.int32)
                if mismatches.count_nonzero() > limit:
                    # Too many mismatches.  Re-run the check with no tolerance
                    # to get a nice message.
                    self.assertEqual(eager_output, nnapi_output, atol=0, rtol=0)

    def float_and_quant_and_nhwc(self, inp_float, scale, zero_point):
        torch.manual_seed(29)
        inp_quant = qpt(inp_float, 0.03, 128)
        return [
            ("float", inp_float),
            ("float-nhwc", nhwc(inp_float)),
            ("quant", inp_quant),
            ("quant-nhwc", nhwc(inp_quant)),
        ]

    def test_prelu(self):
        arg = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
        single_a = torch.nn.PReLU()
        self.check(single_a, arg)
        multi_a = torch.nn.PReLU(4)
        with torch.no_grad():
            multi_a.weight.copy_(torch.tensor([.1, .2, .3, .4]))
        self.check(multi_a, nhwc(arg))

        # Test flexible size
        self.check(
            multi_a,
            arg,
            trace_args=[torch.zeros(1, 4, 3, 3)],
            convert_args=[nhwc(torch.zeros(1, 4, 0, 0))],
        )

    def test_quantize(self):
        self.check(
            torch.nn.quantized.Quantize(0.25, 2, torch.quint8),
            nhwc(torch.tensor([[[[1.0]], [[2.0]]]])))

    def test_dequantize(self):
        self.check(
            torch.nn.quantized.DeQuantize(),
            nhwc(qpt([[[[1.0]], [[2.0]]]], 0.25, 2)))

    def test_unsqueeze(self):
        class UnsqueezeModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, arg):
                return arg.unsqueeze(self.dim)

        self.check(UnsqueezeModule(-2), torch.randn(4, 2, 2))
        self.check(UnsqueezeModule(-1), torch.randn(4, 2, 2))
        self.check(UnsqueezeModule(0), torch.randn(4, 2, 2))
        self.check(UnsqueezeModule(1), torch.randn(4, 2, 2))
        self.check(UnsqueezeModule(2), torch.randn(4, 2, 2))

    def test_reshape(self):
        class ReshapeModule(torch.nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.shape = shape

            def forward(self, arg):
                return arg.reshape(self.shape)

        self.check(
            ReshapeModule((2, 4)),
            torch.randn(4, 2, 1, 1))

        self.check(
            ReshapeModule((8, -1)),
            nhwc(torch.randn(4, 2, 1, 1)))

        with self.assertRaisesRegex(Exception, "target size"):
            self.check(
                ReshapeModule((2, 4)),
                nhwc(torch.randn(4, 2, 1, 1)))

    def test_cat(self):
        class CatModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, t1, t2):
                return torch.cat([t1, t2], self.dim)

        self.check(
            CatModule(0),
            [
                torch.randn(1, 2, 3, 3),
                torch.randn(2, 2, 3, 3),
            ])

        self.check(
            CatModule(1),
            [
                torch.randn(1, 2, 3, 3),
                torch.randn(1, 4, 3, 3),
            ])

        self.check(
            CatModule(1),
            [
                nhwc(torch.randn(1, 2, 3, 3)),
                nhwc(torch.randn(1, 4, 3, 3)),
            ])

    def test_pointwise_unary(self):
        for op in ["relu", "sigmoid"]:
            with self.subTest(op):
                class UnaryModule(torch.nn.Module):
                    def forward(self, arg):
                        if op == "relu":
                            return torch.nn.functional.relu(arg)
                        if op == "sigmoid":
                            return torch.sigmoid(arg)
                        raise Exception("Bad op")
                self.check(UnaryModule(), torch.tensor([-1.0, 1.0]))

    def test_pointwise_binary(self):
        for op in ["add", "sub", "mul"]:
            with self.subTest(op):
                class BinaryModule(torch.nn.Module):
                    def forward(self, lhs, rhs):
                        if op == "add":
                            return lhs + rhs
                        if op == "sub":
                            return lhs - rhs
                        if op == "mul":
                            return lhs * rhs
                        raise Exception("Bad op")

                self.check(
                    BinaryModule(),
                    [
                        torch.tensor([1.0, 2.0]),
                        torch.tensor([3.0, 4.0]),
                    ])

                self.check(
                    BinaryModule(),
                    [
                        torch.tensor([[1.0, 2.0]]),
                        torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
                    ])

                with self.assertRaisesRegex(Exception, "Non-equal-rank broadcast"):
                    self.check(
                        BinaryModule(),
                        [
                            torch.tensor([1.0, 2.0]),
                            torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
                        ])

    def test_hardtanh(self):
        inp = torch.tensor([-2.0, -0.5, 0.5, 2.0, 7.0])
        self.check(torch.nn.Hardtanh(), inp)
        self.check(torch.nn.Hardtanh(0.0, 6.0), inp)
        with self.assertRaisesRegex(Exception, "hardtanh with args"):
            self.check(torch.nn.Hardtanh(0.0, 5.0), inp)

    def test_mean(self):
        class MeanModule(torch.nn.Module):
            def __init__(self, dim, keep=False):
                super().__init__()
                self.dim = dim
                self.keep = keep

            def forward(self, t):
                return torch.mean(t, dim=self.dim, keepdim=self.keep)

        self.check(MeanModule(0), torch.randn(2, 3))
        self.check(MeanModule(1), torch.randn(2, 3))
        self.check(MeanModule([2, 3]), torch.randn(2, 3, 6, 6))
        self.check(MeanModule([2, 3]), nhwc(torch.randn(2, 3, 6, 6)))
        self.check(MeanModule([-1, -2]), nhwc(torch.randn(2, 3, 6, 6)))
        self.check(MeanModule([-1, -2], keep=True), nhwc(torch.randn(2, 3, 6, 6)))

    def test_max_pool2d(self):
        for (name, inp) in self.float_and_quant_and_nhwc(torch.randn(2, 3, 12, 16), 0.3, 128):
            with self.subTest(name):
                self.check(torch.nn.MaxPool2d(2), inp)
                self.check(torch.nn.MaxPool2d((3, 4)), inp)
                self.check(torch.nn.MaxPool2d((3, 4), (1, 2)), inp)

    def test_adaptive_avg_pool2d(self):
        for (name, inp) in self.float_and_quant_and_nhwc(torch.randn(2, 3, 12, 16), 0.3, 128):
            with self.subTest(name):
                self.check(torch.nn.AdaptiveAvgPool2d((1, 1)), inp)
                with self.assertRaisesRegex(Exception, "with output size"):
                    self.check(torch.nn.AdaptiveAvgPool2d((2, 2)), inp)

    def test_upsample_nearest2d(self):
        for (name, inp) in self.float_and_quant_and_nhwc(torch.randn(2, 3, 12, 16), 0.3, 128):
            with self.subTest(name):
                self.check(torch.nn.UpsamplingNearest2d(size=(16, 20)), inp)
                self.check(torch.nn.UpsamplingNearest2d(size=(24, 32)), inp)
                self.check(torch.nn.UpsamplingNearest2d(size=(36, 48)), inp)
                self.check(torch.nn.UpsamplingNearest2d(scale_factor=(1.5, 1.5)), inp)
                self.check(torch.nn.UpsamplingNearest2d(scale_factor=(2.0, 2.0)), inp)
                self.check(torch.nn.UpsamplingNearest2d(scale_factor=(3.0, 3.0)), inp)

    def test_linear(self):
        torch.manual_seed(29)
        self.check(torch.nn.Linear(16, 32), torch.randn(2, 16))

    def test_conv2d(self):
        cases = [
            # in_ch, out_ch, kernel, stride, padding, groups, bias, input_dim,      name
            ( 4,     8,      (3, 3), 1,      0,       1,      1,    (2, 4, 16, 16), "3x3"),        # noqa
            ( 4,     8,      (3, 3), 1,      0,       1,      0,    (2, 4, 16, 16), "3x3nobias"),  # noqa
            ( 4,     16,     (3, 3), 1,      1,       1,      1,    (2, 4, 16, 16), "3x3p1"),      # noqa
            ( 8,     8,      (3, 3), 2,      0,       1,      1,    (2, 8, 16, 16), "3x3s2"),      # noqa
            ( 4,     8,      (5, 5), 1,      0,       1,      1,    (2, 4, 16, 16), "5x5"),        # noqa
            ( 4,     4,      (3, 3), 1,      0,       4,      1,    (2, 4, 16, 16), "3x3dw"),      # noqa
            ( 8,     4,      (1, 1), 1,      0,       1,      1,    (2, 8, 16, 16), "1x1"),        # noqa
        ]

        for kind in ["float", "float-nhwc", "quant", "quant-nhwc"]:
            for case in cases:
                in_ch, out_ch, kernel, stride, padding, groups, bias, input_dim, name = case
                with self.subTest("{}-{}".format(kind, name)):
                    inp = torch.randn(input_dim)
                    model = torch.nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=bool(bias))
                    output_size = model(inp).numel()
                    atol_rtol = None
                    limit = None
                    if "quant" in kind:
                        model = torch.nn.Sequential(model)
                        model.eval()
                        model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
                        model = torch.quantization.prepare(model)
                        model(inp)
                        model = torch.quantization.convert(model)
                        inp = qpt(inp, 1.0 / 16, 128)
                        # I've seen numerical differences between QNNPACK and NNAPI,
                        # but never more than 1 quantum, and never more than ~1% of
                        # the output in this test.
                        atol_rtol = (1, 0)
                        limit = output_size * 0.03
                    if "nhwc" in kind:
                        inp = nhwc(inp)

                    self.check(model, inp, atol_rtol=atol_rtol, limit=limit)

    def test_qadd(self):
        func = torch.nn.quantized.QFunctional()
        func.scale = 0.5
        func.zero_point = 120

        class AddMod(torch.nn.Module):
            def forward(self, lhs, rhs):
                return func.add(lhs, rhs)

        class AddReluMod(torch.nn.Module):
            def forward(self, lhs, rhs):
                return func.add_relu(lhs, rhs)

        for (name, mod) in [("add", AddMod), ("add_relu", AddReluMod)]:
            with self.subTest(name):
                self.check(
                    mod(),
                    [
                        qpt([1.0, 2.0], 0.25, 128),
                        qpt([3.0, 4.0], 0.25, 128),
                    ])
                # NOTE: NNAPI qadd supports broadcast, but PT does not.

    def test_qlinear(self):
        torch.manual_seed(29)
        weight = qpt(torch.randn(16, 32), 0.125, 0, torch.qint8)
        bias = torch.randn(16)
        mod = torch.nn.quantized.Linear(32, 16)
        mod.set_weight_bias(weight, bias)
        inp = qpt(torch.randn(2, 32), 0.05, 130, torch.quint8)
        self.check(mod, inp)

    def test_seblock_mul(self):
        class MulModel(torch.nn.Module):
            def forward(self, lhs, rhs):
                return lhs * rhs

        self.check(
            MulModel(),
            [
                nhwc(torch.randn(2, 3, 4, 4)),
                torch.randn(1, 3, 1, 1),
            ])

    def test_multi_output(self):
        class MultiModel(torch.nn.Module):
            def forward(self, lhs, rhs) -> Tuple[torch.Tensor, torch.Tensor]:
                the_sum = lhs + rhs
                the_diff = lhs - rhs
                return the_sum, the_diff

        self.check(MultiModel(), [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0])])


if __name__ == '__main__':
    run_tests()
