import unittest

import torch
import torch.nn.intrinsic as nni
from torch.testing._internal.common_quantization import (
    skipIfNoFBGEMM,
    QuantizationTestCase,
)

import torch.quantization._quantize_dynamic_tracing as _quantize_dynamic_tracing


class TestAutoTracing(QuantizationTestCase):

    def _test_auto_tracing(self, m, example_inputs):
        mp = _quantize_dynamic_tracing.prepare(m, example_inputs)
        mp(*example_inputs)
        print(mp)
        mq = _quantize_dynamic_tracing.convert(mp)
        print(mq)
        # verify it runs
        out_q = mq(*example_inputs)
        print(out_q)

        # verify torch.jit.trace works
        mq_jit_traced = torch.jit.trace(
            mq, example_inputs, check_trace=False)
        # print(mq_jit_traced.graph)
        traced_out = mq_jit_traced(*example_inputs)
        self.assertTrue(torch.allclose(traced_out, out_q))

        # verify torch.jit.script works
        rewritten = mq.rewrite_for_scripting()
        print(rewritten)
        rewritten_out = rewritten(*example_inputs)
        self.assertTrue(torch.allclose(rewritten_out, out_q))

        scripted_rewritten = torch.jit.script(rewritten)
        scripted_rewritten_out = scripted_rewritten(*example_inputs)
        self.assertTrue(torch.allclose(scripted_rewritten_out, out_q))

        traced_rewritten = torch.jit.trace(
            rewritten, example_inputs, check_trace=False)
        traced_rewritten_out = traced_rewritten(*example_inputs)
        self.assertTrue(torch.allclose(traced_rewritten_out, out_q))

    @skipIfNoFBGEMM
    def test_fusion(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                return x

        m = M().eval()
        m.qconfig = torch.quantization.default_qconfig
        mp = _quantize_dynamic_tracing.prepare(m, (torch.randn(1, 1, 1, 1),))
        # testing that the conv got fused
        self.assertTrue(isinstance(mp.conv, nni.ConvReLU2d))

    @skipIfNoFBGEMM
    def test_conv(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x1 = self.conv(x)
                return x1

        m = M().eval()
        m.qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(m, (torch.randn(1, 1, 2, 2),))

    @skipIfNoFBGEMM
    def test_conv_add(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x1 = self.conv(x)
                print(x)
                x2 = x1 + x
                return x2

        m = M().eval()
        m.qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(m, (torch.randn(1, 1, 2, 2),))

    @skipIfNoFBGEMM
    def test_conv_relu_add(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x1 = self.conv(x)
                x2 = self.relu(x)
                x3 = x1 + x
                return x3
                # x4 = x3 + x3
                # return x4

        model_fp32 = M().eval()

        # model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_fp32.qconfig = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
                                                        weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8))

        model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])
        # print(model_fp32_fused)
        # return
        self._test_auto_tracing(model_fp32_fused, (torch.randn(1, 1, 2, 2),))

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
