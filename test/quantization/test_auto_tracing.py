import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
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

def _allclose(a, b):
    if isinstance(a, tuple):
        assert isinstance(b, tuple)
        result = True
        for a_inner, b_inner in zip(a, b):
            result = result and torch.allclose(a_inner, b_inner)
        return result
    elif isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        return torch.allclose(a, b)
    raise AssertionError('unhandled type')


class AutoTracingTestCase(QuantizationTestCase):
    def _test_auto_tracing(
        self,
        m,
        qconfig,
        example_args,
        fuse_modules=True,
        do_fx_comparison=True,
        do_torchscript_checks=True,
    ):
        m_copy = copy.deepcopy(m)

        m.qconfig = qconfig

        mp = _quantize_dynamic_tracing.prepare(
            m, example_args, fuse_modules=fuse_modules)
        out_p = mp(*example_args)
        print(mp)
        mq = _quantize_dynamic_tracing.convert(mp)
        print(mq)
        # verify it runs
        out_q = mq(*example_args)
        # print(out_q)

        # compare it against FX
        if do_fx_comparison:
            m_copy_p = prepare_fx(m_copy, {'': qconfig})
            out_m_copy_p = m_copy_p(*example_args)
            # print(m_copy_p)
            m_copy_q = convert_fx(m_copy_p)
            # print(m_copy_q)
            out_q_fx = m_copy_q(*example_args)
            self.assertTrue(_allclose(out_p, out_m_copy_p))
            # print(out_q)
            # print(out_q_fx)
            self.assertTrue(_allclose(out_q, out_q_fx))

        if do_torchscript_checks:
            # verify torch.jit.trace works
            mq_jit_traced = torch.jit.trace(
                mq, example_args, check_trace=False)
            # print(mq_jit_traced.graph)
            traced_out = mq_jit_traced(*example_args)
            self.assertTrue(_allclose(traced_out, out_q))

            # verify torch.jit.script works
            rewritten = mq.rewrite_for_scripting()
            rewritten_out = rewritten(*example_args)
            # print(rewritten)
            self.assertTrue(_allclose(rewritten_out, out_q))

            scripted_rewritten = torch.jit.script(rewritten)
            # print(scripted_rewritten.graph)
            scripted_rewritten_out = scripted_rewritten(*example_args)
            # print('scripted_rewritten_out', scripted_rewritten_out)
            self.assertTrue(_allclose(scripted_rewritten_out, out_q))

            traced_rewritten = torch.jit.trace(
                rewritten, example_args, check_trace=False)
            traced_rewritten_out = traced_rewritten(*example_args)
            self.assertTrue(_allclose(traced_rewritten_out, out_q))


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
    def test_fusion2(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.bn = torch.nn.BatchNorm2d(1)
                # self.conv2 = torch.nn.Conv2d(1, 1, 1)
                self.relu = torch.nn.LeakyReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        m = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(m, qconfig, (torch.randn(1, 1, 2, 2),))

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
    def test_dropout_conv(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # this can be sometimes inplace
                x1 = self.dropout(x)
                x1 = self.conv(x)
                return x1

        m = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(m, qconfig, (torch.randn(1, 1, 2, 2),))

    # TODO(future PR): implement observer sharing to match FX
    @skipIfNoFBGEMM
    def test_cat(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x = torch.cat([x, x], dim=1)
                return x

        m = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(m, qconfig, (torch.randn(1, 1, 2, 2),))

        class M(torch.nn.Module):
            def forward(self, x):
                x = torch.cat((x, x), dim=1)
                return x

        m = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(m, qconfig, (torch.randn(1, 1, 2, 2),))

    # TODO: fix this test (iteration over the (1, 1) arg for arg_quant_infos)
    @unittest.skip('foo')
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
    def test_conv_scalar_add(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv(x)
                x = x + 1.0
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 2, 2),))


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

    @skipIfNoFBGEMM
    def test_gelu_linear(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = torch.nn.GELU()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x):
                x = self.linear(x)
                x = self.gelu(x)
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 1, 1),))

    @skipIfNoFBGEMM
    def test_dropout(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout()
                self.linear = torch.nn.Linear(1, 1)
                self.linear2 = torch.nn.Linear(1, 1)

            def forward(self, x):
                x = self.linear(x)
                x = self.dropout(x)
                x = self.linear2(x)
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 1, 1),))

    @skipIfNoFBGEMM
    def test_add(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x = x + x
                x = x + 1.0
                x = 1.0 + x
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 2, 2),))

    @skipIfNoFBGEMM
    def test_module_then_add(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x):
                x = self.linear(x)
                x = x + 1.0
                x = x + 1.0
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 1, 1),))

    @skipIfNoFBGEMM
    def test_sub(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x = x - x
                x = x - 1.0
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 2, 2),))

    @skipIfNoFBGEMM
    def test_mul(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x = x * x
                x = x * 1.0
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 2, 2),))

    @unittest.skip("TODO next")
    @skipIfNoFBGEMM
    def test_mul_int32(self):
        # TODO: make all the math functions work correctly for integer types
        class M(torch.nn.Module):
            def forward(self, x):
                x = x * x
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(
            model_fp32, qconfig, (torch.ones(1, 1, 2, 2, dtype=torch.int32),))

    @skipIfNoFBGEMM
    def test_div(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x = x / x
                x = x / 1.0
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 2, 2),))

    @skipIfNoFBGEMM
    def test_method(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x = x + x
                x = torch.relu(x)
                # x = x.relu()
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 2, 2),))

    @skipIfNoFBGEMM
    def test_add_linear(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, x):
                x = x + x
                x = self.linear(x)
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 1, 1),))

    @skipIfNoFBGEMM
    def test_module_created_during_forward(self):
        """Some BERT models have this pattern"""
        class M(torch.nn.Module):
            def forward(self, x):
                x = nn.Softmax(dim=-1)(x)
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(
            model_fp32, qconfig, (torch.randn(1, 1, 1, 1),),
            # This syntax is not supported by FX or TorchScript
            do_fx_comparison=False, do_torchscript_checks=False)

    @unittest.skip('TODO build this')
    @skipIfNoFBGEMM
    def test_module_input_types(self):
        class M(torch.nn.Module):
            def forward(self, x=None, y=None):
                print('x', x)
                print('y', y)
                assert x is not None and y is not None
                return (x, y)

        model_fp32 = M().eval()
        example_inputs = {'y': torch.randn(1), 'x': torch.randn(1)}
        print('example_inputs', example_inputs)
        import collections
        ExampleInputsTupleCtr = collections.namedtuple('ExampleInputs', example_inputs)
        example_inputs_tuple = ExampleInputsTupleCtr(**example_inputs)
        ms = torch.jit.trace(model_fp32, example_inputs_tuple)
        print(ms.graph)

        return
        qconfig = torch.quantization.default_qconfig

        # dict
        kwargs = {'x': torch.randn(1, 1, 2, 2)}
        self._test_auto_tracing(model_fp32, qconfig, (), kwargs)

    @skipIfNoFBGEMM
    def test_module_return_types(self):
        class M1(torch.nn.Module):
            def forward(self, x):
                return x, x

        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = M1()

            def forward(self, x):
                x1, x2 = self.m1(x)
                return x1

        model_fp32 = M2().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 2, 2),))

    # TODO fix this test
    @unittest.skip('foo')
    @skipIfNoFBGEMM
    def test_unsupported_ops(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x = F.tanhshrink(x)
                x = x + x
                x = F.tanhshrink(x)
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(model_fp32, qconfig, (torch.randn(1, 1, 2, 2),))

    @skipIfNoFBGEMM
    def test_unknown_op_after_quantized(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x = x + x
                std = x.std()
                return std

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(
            model_fp32, qconfig, (torch.randn(1, 1, 2, 2),),
            fuse_modules=False)

    @skipIfNoFBGEMM
    def test_embedding(self):
        # test subclass

        class EmbeddingSubclass(nn.Embedding):
            pass

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = EmbeddingSubclass(1, 1)

            def forward(self, x):
                x = self.embedding(x)
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_dynamic_qconfig
        self._test_auto_tracing(
            model_fp32, qconfig, (torch.LongTensor([[0]]),),
            fuse_modules=False)

        # test regular embedding
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1, 1)

            def forward(self, x):
                x = self.embedding(x)
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_dynamic_qconfig
        self._test_auto_tracing(
            model_fp32, qconfig, (torch.LongTensor([[0]]),),
            fuse_modules=False)

    @skipIfNoFBGEMM
    def test_inplace_add(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding1 = nn.Embedding(1, 1)
                self.embedding2 = nn.Embedding(1, 1)
                self.layernorm = nn.LayerNorm(1)

            def forward(self, x):
                x1 = self.embedding1(x)
                x1 += self.embedding2(x)
                x2 = self.layernorm(x1)
                return x

        model_fp32 = M().eval()
        qconfig = torch.quantization.default_qconfig
        self._test_auto_tracing(
            model_fp32, qconfig, (torch.LongTensor([[0]]),),
            fuse_modules=False)


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
