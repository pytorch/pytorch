import unittest

import torch
from torch.testing._internal.common_quantization import (
    skipIfNoFBGEMM,
    QuantizationTestCase,
)

import torch.quantization._quantize_dynamic_tracing as _quantize_dynamic_tracing


class TestAutoTracing(QuantizationTestCase):

    @skipIfNoFBGEMM
    def test_auto_tracing_conv_relu_add(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x1 = self.conv(x)
                x2 = self.relu(x)
                x3 = x1 + x2
                x4 = x3 + x3
                return x4

        model_fp32 = M()

        model_fp32.eval()

        # model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_fp32.qconfig = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
                                                        weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8))

        model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

        model_fp32_prepared = _quantize_dynamic_tracing.prepare(model_fp32_fused)
        # print(model_fp32_prepared)

        input_fp32 = torch.randn(1, 1, 2, 2)
        model_fp32_prepared(input_fp32)

        input_fp32 = torch.randn(1, 1, 2, 2)
        model_fp32_prepared(input_fp32)
        print(model_fp32_prepared)

        model_int8 = _quantize_dynamic_tracing.convert(model_fp32_prepared)
        print(model_int8)
        input_q = torch.quantize_per_tensor(input_fp32, 0.1, 0, torch.quint8)
        out = model_int8(input_q)
        print(out)

        input_q2 = torch.quantize_per_tensor(torch.randn(1, 1, 2, 2), 0.1, 0, torch.quint8)
        out = model_int8(input_q2)

        traced_model_int8 = torch.jit.trace(model_int8, (input_q,), check_trace=False)
        # confirm quantized::add op
        traced_out = traced_model_int8(input_q2)
        assert torch.all(traced_out.int_repr() == out.int_repr())

        rewritten = model_int8.rewrite()
        rewritten_out = rewritten(input_q2)
        assert torch.all(rewritten_out.int_repr() == out.int_repr())

        scripted_rewritten = torch.jit.script(rewritten)
        scripted_rewritten_out = scripted_rewritten(input_q2)
        assert torch.all(scripted_rewritten_out.int_repr() == out.int_repr())

        traced_rewritten = torch.jit.trace(rewritten, (input_q,), check_trace=False)
        traced_rewritten_out = traced_rewritten(input_q2)
        assert torch.all(traced_rewritten_out.int_repr() == out.int_repr())

    # TODO(future PR): enable this test
    @unittest.skip("this is currently broken")
    def test_control_flow(self):
        class Looper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.i2h = torch.nn.Linear(5, 5)
                self.h2h = torch.nn.Linear(5, 5)

            def forward(self, x):
                h = torch.zeros(x.shape[1:])
                for i in range(x.shape[0]):
                    i2h = self.i2h(x[0])
                    h2h = self.h2h(h)
                    h = i2h + h2h
                return h

        l = Looper().eval()
        x = torch.randn(10, 5, 5)

        l(x)

        l.qconfig = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
                                               weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8))

        l_prepared = _quantize_dynamic_tracing.prepare(l)

        l_prepared(torch.randn(7, 5, 5))
        l_prepared(torch.randn(13, 5, 5))

        self.assertNotEqual(len(l_prepared._arithmetic_observers.op_observers), 0)
