import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
from torch.testing._internal.common_quantization import (
    skipIfNoFBGEMM,
    skip_if_no_torchvision,
    QuantizationTestCase,
)
from torch.quantization import (
    ObserverBase,
    FakeQuantizeBase,
)
from torch.quantization.quantize_fx import (
    prepare_fx,
    convert_fx,
)

import torch.quantization._quantize_dynamic_tracing as _quantize_dynamic_tracing


class AutoTracingTestCase(QuantizationTestCase):
    def _test_auto_tracing(self, m, qconfig, example_inputs):
        m_copy = copy.deepcopy(m)

        m.qconfig = qconfig

        mp = _quantize_dynamic_tracing.prepare(m, example_inputs)
        out_p = mp(*example_inputs)
        print(mp)
        mq = _quantize_dynamic_tracing.convert(mp)
        print(mq)
        # verify it runs
        out_q = mq(*example_inputs)
        # print(out_q)

        # compare it against FX
        m_copy_p = prepare_fx(m_copy, {'': qconfig})
        out_m_copy_p = m_copy_p(*example_inputs)
        # print(m_copy_p)
        m_copy_q = convert_fx(m_copy_p)
        out_q_fx = m_copy_q(*example_inputs)
        self.assertTrue(torch.allclose(out_p, out_m_copy_p))
        self.assertTrue(torch.allclose(out_q, out_q_fx))

        # verify torch.jit.trace works
        mq_jit_traced = torch.jit.trace(
            mq, example_inputs, check_trace=False)
        # print(mq_jit_traced.graph)
        traced_out = mq_jit_traced(*example_inputs)
        self.assertTrue(torch.allclose(traced_out, out_q))

        # verify torch.jit.script works
        rewritten = mq.rewrite_for_scripting()
        rewritten_out = rewritten(*example_inputs)
        # print(rewritten)
        self.assertTrue(torch.allclose(rewritten_out, out_q))

        scripted_rewritten = torch.jit.script(rewritten)
        # print(scripted_rewritten.graph)
        scripted_rewritten_out = scripted_rewritten(*example_inputs)
        # print('scripted_rewritten_out', scripted_rewritten_out)
        self.assertTrue(torch.allclose(scripted_rewritten_out, out_q))

        traced_rewritten = torch.jit.trace(
            rewritten, example_inputs, check_trace=False)
        traced_rewritten_out = traced_rewritten(*example_inputs)
        self.assertTrue(torch.allclose(traced_rewritten_out, out_q))


class TestAutoTracing(AutoTracingTestCase):
    @skipIfNoFBGEMM
    def test_fusion(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.relu = torch.nn.ReLU()
                self.child = nn.Sequential(
                    nn.Conv2d(1, 1, 1),
                    nn.ReLU(),
                )

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                x = self.child(x)
                return x

        m = M().eval()
        m.qconfig = torch.quantization.default_qconfig
        mp = _quantize_dynamic_tracing.prepare(m, (torch.randn(1, 1, 1, 1),))
        self.assertTrue(isinstance(mp.conv, nni.ConvReLU2d))
        self.assertTrue(isinstance(mp.child[0], nni.ConvReLU2d))

    @skipIfNoFBGEMM
    def test_observers_not_touched_by_tracing(self):
        """
        Verifies that running dynamic tracing does not change any data
        stored in observers and fake quants.
        """
        m = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        m.qconfig = torch.quantization.default_qconfig
        mp = _quantize_dynamic_tracing.prepare(m, (torch.randn(1, 1, 1, 1),))
        for _, mod in mp.named_modules():
            if isinstance(mod, (ObserverBase, FakeQuantizeBase)):
                scale, zp = mod.calculate_qparams()
                # Assume that if scale is 1.0 and zp is 0, no calibration
                # has happened.
                self.assertTrue(torch.allclose(scale, torch.ones(1)))
                self.assertTrue(torch.equal(zp, torch.zeros(1, dtype=torch.long)))

    @skipIfNoFBGEMM
    def test_multiple_modules(self):
        m = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 1, 1)),
            nn.Sequential(nn.Conv2d(1, 1, 1)),
        ).eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(m, qconfig, (torch.randn(1, 1, 2, 2),))

    @skipIfNoFBGEMM
    def test_child_modules(self):
        m = nn.Sequential(nn.Sequential(nn.Conv2d(1, 1, 1))).eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(m, qconfig, (torch.randn(1, 1, 2, 2),))

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
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(m, qconfig, (torch.randn(1, 1, 2, 2),))

    @skipIfNoFBGEMM
    def test_conv_flatten_linear(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x):
                x1 = self.conv(x)
                # TODO(future PR): unbreak this
                # x1 = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                x1 = torch.nn.functional.adaptive_avg_pool2d(x1, (1, 1))
                x2 = torch.flatten(x1, 1)
                x3 = self.linear(x2)
                return x3

        m = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(m, qconfig, (torch.randn(1, 1, 1, 1),))

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
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(m, qconfig, (torch.randn(1, 1, 2, 2),))

    @skipIfNoFBGEMM
    def test_conv_relu_add(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x1 = self.conv(x)
                x2 = self.relu(x1)
                x3 = x1 + x
                return x3

        model_fp32 = M().eval()

        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 2, 2),))

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


class TestAutoTracingModels(AutoTracingTestCase):
    @skip_if_no_torchvision
    @skipIfNoFBGEMM
    def test_mobilenet_v2(self):
        import torchvision
        m = torchvision.models.__dict__['mobilenet_v2'](pretrained=False).eval().float()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(m, qconfig, (torch.randn(1, 3, 224, 224),))
