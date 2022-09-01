# Owner(s): ["module: mkldnn"]
import itertools
import unittest

import torch
from torch import nn

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase

from test_tensorexpr import warmup_and_run_forward

FUSION_GROUP = 'prim::TensorExprGroup'


@unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
class TestMkldnnFusion(JitTestCase):
    def assertFused(self, graph, fused_patterns):
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)

    def _check_model(self, m, x, trace=False, bf16=False):
        rtol = 1.6e-2 if bf16 else 1.3e-6
        atol = 1e-2 if bf16 else 1e-5
        previous_jit_autocast_pass = torch._C._jit_set_autocast_mode(False)

        old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)

        old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        torch._C._jit_override_can_fuse_on_cpu(True)

        old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)

        m.eval()
        with torch.no_grad(), torch.cpu.amp.autocast(enabled=bf16, cache_enabled=False):
            if trace:
                script = torch.jit.trace(m, x)
            else:
                script = torch.jit.script(m)
        script = torch.jit.freeze(script)

        with torch.no_grad(), torch.cpu.amp.autocast(enabled=bf16):
            y = warmup_and_run_forward(script, x)
            y = script(x)
            y_ref = m(x)

            graph = script.graph_for(*x)
            self.assertEqual(y, y_ref, rtol=rtol, atol=atol)

        torch._C._jit_set_autocast_mode(previous_jit_autocast_pass)
        torch._C._debug_set_fusion_group_inlining(old_fusion_inlining)
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuser_state)
        torch._C._jit_set_te_must_use_llvm_cpu(old_te_must_use_llvm_cpu)
        return graph

    def _eltwise_list(self):
        eltwise_list = [
            [torch.relu, 'aten::relu'],
            [torch.sigmoid, 'aten::sigmoid'],
            [torch.tanh, 'aten::tanh'],
            [nn.LeakyReLU(0.1, inplace=False), 'aten::leaky_relu'],
            [nn.Hardtanh(inplace=False), 'aten::hardtanh'],
            [nn.GELU(approximate="none"), 'aten::gelu'],
            [nn.GELU(approximate="tanh"), 'aten::gelu'],
        ]
        return eltwise_list

    def _clamp_modules(self):
        class MNoOpt(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super(MNoOpt, self).__init__()
                self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                x = self.conv(x)
                x = torch.clamp(x, min=-0.5, max=0.9)
                return x

        class MInf(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super(MInf, self).__init__()
                self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                x = self.conv(x)
                x = torch.clamp(x, min=0, max=float('inf'))
                return x

        class MNegInf(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super(MNegInf, self).__init__()
                self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                x = self.conv(x)
                x = torch.clamp(x, min=float('-inf'), max=0)
                return x

        class MOptMin(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super(MOptMin, self).__init__()
                self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                x = self.conv(x)
                x = torch.clamp(x, max=2)
                return x

        class MOptMax(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super(MOptMax, self).__init__()
                self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                x = self.conv(x)
                x = torch.clamp(x, min=0)
                return x

        return [MNoOpt, MInf, MNegInf, MOptMin, MOptMax]

    def test_single_conv(self):
        class M(nn.Module):
            def __init__(self, in_channels, out_channels, bias, **kwargs):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                res = self.conv(x)
                return res

        for memory_format, enabled in [
            [torch.contiguous_format, False],
            [torch.channels_last, True],
        ]:
            for trace in [True, False]:
                for bf16 in [True, False]:
                    # torch.jit.script doesn't work with autocast
                    if not trace and bf16:
                        continue
                    input_size = 224
                    batch_size = 1
                    kernel_size = 3
                    options = itertools.product([True, False], [1, 2], [1, 4])
                    for bias, dilation, groups in options:
                        iC = 3 * groups
                        oC = 10 * groups
                        m = M(iC,
                            oC,
                            bias,
                            kernel_size=(kernel_size, kernel_size),
                            stride=2,
                            padding=1,
                            dilation=dilation,
                            groups=groups).to(memory_format=memory_format)
                        x = torch.randn(batch_size, iC, input_size, input_size).to(memory_format=memory_format)
                        graph = self._check_model(m, x, trace, bf16)
                        conv_node_name = 'aten::_convolution' if trace else 'aten::conv2d'
                        if enabled:
                            self.assertFused(graph, [conv_node_name])
                            self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)
                        else:
                            self.assertGraphContains(graph, kind=conv_node_name)

    def test_conv_eltwise(self):
        class M(nn.Module):
            def __init__(self, eltwise_fn, in_channels, out_channels, bias, **kwargs):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
                self.eltwise = eltwise_fn

            def forward(self, x):
                x = self.conv(x)
                x = self.eltwise(x)
                return x

        for memory_format, enabled in [
            [torch.contiguous_format, False],
            [torch.channels_last, True],
        ]:
            for trace in [True, False]:
                for bf16 in [True, False]:
                    # torch.jit.script doesn't work with autocast
                    if not trace and bf16:
                        continue                    
                    for eltwise_fn, op_name in self._eltwise_list():
                        for bias in [True, False]:
                            for oC in [1, 10]:
                                m = M(eltwise_fn, 3, oC, bias, kernel_size=(3, 3)).to(memory_format=memory_format)
                                x = torch.randn(1, 3, 224, 224).to(memory_format=memory_format)

                                graph = self._check_model(m, x, trace, bf16)
                                conv_node_name = 'aten::_convolution' if trace else 'aten::conv2d'
                                if enabled:
                                    self.assertFused(graph, ['aten::conv2d', op_name])
                                    self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)
                                else:
                                    self.assertGraphContains(graph, kind=conv_node_name)

    def test_conv_clamp(self):
        modules = self._clamp_modules()
        op_name = 'aten::clamp'

        for memory_format, enabled in [
            [torch.contiguous_format, False],
            [torch.channels_last, True],
        ]:
            for M in modules:
                for bias in [True, False]:
                    m = M(nn.Conv2d, 3, 10, bias, kernel_size=(3, 3)).to(memory_format=memory_format)
                    x = torch.randn(1, 3, 224, 224).to(memory_format=memory_format)

                    graph = self._check_model(m, x)
                    if enabled:
                        self.assertFused(graph, ['aten::conv2d', op_name])
                        self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)
                    else:
                        self.assertGraphContains(graph, kind='aten::conv2d')


    def test_unsupported_conv(self):
        class M(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super(M, self).__init__()
                self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                res = self.conv(x)
                return res

        for module, dim, memory_format in [
            [nn.Conv3d, 3, torch.contiguous_format],
            [nn.Conv3d, 3, torch.channels_last_3d],
            [nn.ConvTranspose2d, 2, torch.contiguous_format],
            [nn.ConvTranspose2d, 2, torch.channels_last],
        ]:
            trace = True
            input_size = 224
            batch_size = 1
            kernel_size = 3
            groups = 2
            bias = True
            iC = 3 * groups
            oC = 10 * groups
            dilation = 2
            m = M(module,
                  iC,
                  oC,
                  bias,
                  kernel_size=kernel_size,
                  stride=2,
                  padding=1,
                  dilation=dilation,
                  groups=groups).to(memory_format=memory_format)
            input_sizes = [batch_size, iC, input_size, input_size]
            if dim == 3:
                input_sizes.append(input_size)
            x = torch.randn(input_sizes).to(memory_format=memory_format)
            graph = self._check_model(m, x, trace)
            self.assertGraphContains(graph, kind='aten::_convolution')


if __name__ == "__main__":
    run_tests()
