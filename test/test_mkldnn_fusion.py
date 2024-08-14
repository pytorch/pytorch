# Owner(s): ["module: mkldnn"]
import itertools
import unittest
from typing import NamedTuple, List

import torch
from torch import nn

from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo
from torch.testing._internal.jit_utils import JitTestCase

from test_tensorexpr import warmup_and_run_forward

FUSION_GROUP = 'prim::TensorExprGroup'

class PointwisePostOp(NamedTuple):
    attr : str
    pointwise_module : nn.Module
    scalars : List = []
    algorithm : str = ""

CONV_MODULES = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
CONV_TRANSPOSE_MODULES = {2: torch.nn.ConvTranspose2d}

@skipIfTorchDynamo("too slow")
@unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled")
class TestOnednnFusion(JitTestCase):
    def assertFused(self, graph, fused_patterns):
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)

    def _check_model(self, m, x, trace=False):
        old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)

        old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        torch._C._jit_override_can_fuse_on_cpu(True)

        old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)

        m.eval()
        with torch.no_grad():
            if trace:
                script = torch.jit.trace(m, x)
            else:
                script = torch.jit.script(m)
        script = torch.jit.freeze(script)

        with torch.no_grad():
            y = warmup_and_run_forward(script, x)
            y = script(x)
            y_ref = m(x)

            graph = script.graph_for(*x)
            self.assertEqual(y, y_ref)

        torch._C._debug_set_fusion_group_inlining(old_fusion_inlining)
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuser_state)
        torch._C._jit_set_te_must_use_llvm_cpu(old_te_must_use_llvm_cpu)
        return graph

    def test_single_conv(self):
        class M(nn.Module):
            def __init__(self, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                res = self.conv(x)
                return res

        for memory_format, enabled in [
            [torch.contiguous_format, False],
            [torch.channels_last, True],
        ]:
            for trace in [True, False]:
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
                    graph = self._check_model(m, x, trace)
                    conv_node_name = 'aten::_convolution' if trace else 'aten::conv2d'
                    if enabled:
                        self.assertFused(graph, [conv_node_name])
                        self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)
                    else:
                        self.assertGraphContains(graph, kind=conv_node_name)

    def test_conv_unary_fusion_nnc(self):
        class M(nn.Module):
            def __init__(self, unary_fn, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
                self.unary = unary_fn

            def forward(self, x):
                x = self.conv(x)
                x = self.unary(x)
                return x

        for memory_format, enabled in [
            [torch.contiguous_format, False],
            [torch.channels_last, True],
        ]:
            for unary_fn in [torch.relu]:
                for bias in [True, False]:
                    for oC in [1, 10]:
                        m = M(unary_fn, 3, oC, bias, kernel_size=(3, 3)).to(memory_format=memory_format)
                        x = torch.randn(1, 3, 224, 224).to(memory_format=memory_format)

                        graph = self._check_model(m, x)
                        if enabled:
                            self.assertFused(graph, ['aten::conv2d', 'aten::' + unary_fn.__name__])
                            self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)
                        else:
                            self.assertGraphContains(graph, kind='aten::conv2d')

    def test_unsupported_conv(self):
        class M(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super().__init__()
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

    def _unary_list(self):
        unary_list = {
            "relu": PointwisePostOp("relu", nn.ReLU()),
            "sigmoid": PointwisePostOp("sigmoid", nn.Sigmoid()),
            "tanh": PointwisePostOp("tanh", nn.Tanh()),
            "hardswish": PointwisePostOp("hardswish", nn.Hardswish()),
            "leaky_relu": PointwisePostOp("leaky_relu", nn.LeakyReLU(0.1, inplace=False), scalars=[0.1]),
            "hardtanh": PointwisePostOp("hardtanh", nn.Hardtanh(min_val=-0.5, max_val=4, inplace=False), scalars=[-0.5, 4]),
            "gelu_none": PointwisePostOp("gelu", nn.GELU(approximate="none"), algorithm="none"),
            "gelu_tanh": PointwisePostOp("gelu", nn.GELU(approximate="tanh"), algorithm="tanh"),
        }
        return unary_list

    def _binary_list(self):
        binary_list = {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
            "div": torch.div,
        }
        return binary_list

    def test_linear_unary_fusion_ops(self):
        class M(nn.Module):
            def __init__(self, unary_fn, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.unary = unary_fn

            def forward(self, x):
                x = self.linear(x)
                x = self.unary(x)
                return x

        for pointwise_info in self._unary_list().values():
            # Tensor with size = [1, 10] and stride = [0, 1] is contiguous tensor
            # but it's strides is not default contiguous strides.
            options = itertools.product([[[2, 3, 10], None], [[2, 10], None], [[1, 10], [0, 1]]], [True, False])
            for (input_shape, input_stride), bias in options:
                with torch.no_grad():
                    mod = M(pointwise_info.pointwise_module, input_shape[-1], 10, bias).eval()
                    v = torch.randn(input_shape)
                    if input_stride is not None:
                        v = v.as_strided(input_shape, input_stride)
                    ref = mod(v)
                    attr = pointwise_info.attr
                    scalars = pointwise_info.scalars
                    algorithm = pointwise_info.algorithm
                    fused = torch.ops.onednn._linear_pointwise(
                        v, mod.linear.weight, mod.linear.bias, attr, scalars, algorithm
                    )
                    self.assertEqual(ref, fused)


    def test_conv_unary_fusion_ops(self):
        class M(nn.Module):
            def __init__(self, unary_fn, dim, in_channels, out_channels, dilation, groups, bias, **kwargs):
                super().__init__()
                self.conv = CONV_MODULES[dim](in_channels, out_channels, dilation=dilation, groups=groups, bias=bias, **kwargs)
                self.unary = unary_fn

            def forward(self, x):
                x = self.conv(x)
                x = self.unary(x)
                return x

        input_shapes = {2: (112, 112), 3: (55, 55, 55)}
        for pointwise_info in self._unary_list().values():
            for dim in [2, 3]:
                channels_last = torch.channels_last if dim == 2 else torch.channels_last_3d
                options = itertools.product([True, False], [1, 2], [1, 4], [torch.contiguous_format, channels_last])
                for bias, dilation, groups, memory_format in options:
                    oC = 32 * groups
                    iC = 3 * groups
                    x_shape = (1, iC) + input_shapes[dim]
                    x = torch.randn(x_shape, dtype=torch.float32).to(memory_format=memory_format)
                    mod = M(pointwise_info.pointwise_module, dim, iC, oC, dilation, groups, bias, kernel_size=3)
                    mod = mod.to(memory_format=memory_format).eval()
                    with torch.no_grad():
                        ref = mod(x)
                        attr = pointwise_info.attr
                        scalars = pointwise_info.scalars
                        algorithm = pointwise_info.algorithm
                        fused = torch.ops.onednn._convolution_pointwise(
                            x, mod.conv.weight, mod.conv.bias, mod.conv.padding, mod.conv.stride, mod.conv.dilation,
                            mod.conv.groups, attr, scalars, algorithm
                        )
                    self.assertEqual(ref, fused)


    def test_conv_binary_fusion_ops(self):
        class M(nn.Module):
            def __init__(self, binary_fn, dim, in_channels, out_channels, dilation, groups, bias, **kwargs):
                super().__init__()
                self.conv = CONV_MODULES[dim](in_channels, out_channels, dilation=dilation, groups=groups, bias=bias, **kwargs)
                self.binary = binary_fn

            def forward(self, x, other):
                x = self.conv(x)
                x = self.binary(x, other)
                return x

        input_shapes = {2: (112, 112), 3: (22, 22, 22)}
        for pointwise_name, pointwise_fn in self._binary_list().items():
            for dim in [2, 3]:
                channels_last = torch.channels_last if dim == 2 else torch.channels_last_3d
                options = itertools.product([False, True], [True, False], [1, 2], [1, 4], [torch.contiguous_format, channels_last])
                for fuse_relu, bias, dilation, groups, memory_format in options:
                    oC = 32 * groups
                    iC = 3 * groups
                    x_shape = (1, iC) + input_shapes[dim]
                    x = torch.randn(x_shape, dtype=torch.float32).to(memory_format=memory_format)
                    mod = M(pointwise_fn, dim, iC, oC, dilation, groups, bias, kernel_size=3)
                    mod = mod.to(memory_format=memory_format).eval()
                    other = torch.randn_like(mod.conv(x))
                    with torch.no_grad():
                        ref = mod(x, other)
                        unary_attr = None
                        if fuse_relu:
                            ref.relu_()
                            unary_attr = "relu"
                        attr = pointwise_name
                        fused = torch.ops.onednn._convolution_pointwise(
                            x, other, mod.conv.weight, mod.conv.bias, mod.conv.padding, mod.conv.stride, mod.conv.dilation,
                            mod.conv.groups, attr, None, unary_attr, [], None
                        )
                        # for binary add, we support inplace version.
                        if attr == "add":
                            fused_inplace = torch.ops.onednn._convolution_pointwise_(
                                other, x, mod.conv.weight, mod.conv.bias, mod.conv.padding, mod.conv.stride, mod.conv.dilation,
                                mod.conv.groups, attr, None, unary_attr, [], None
                            )
                            self.assertEqual(ref, other)
                            self.assertEqual(ref, fused_inplace)

                        self.assertEqual(ref, fused, atol=5e-4, rtol=5e-4)


    def test_linear_binary_fusion_ops(self):
        class M(nn.Module):
            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.binary = binary_fn

            def forward(self, x, other):
                x = self.linear(x)
                x = self.binary(x, other)
                return x

        out_feature = 20
        for pointwise_name, pointwise_fn in self._binary_list().items():
            # Tensor with size = [1, 10] and stride = [0, 1] is contiguous tensor
            # but it's strides is not default contiguous strides.
            options = itertools.product([[[2, 3, 10], None], [[2, 10], None], [[1, 10], [0, 1]]], [True, False])
            for (input_shape, input_stride), bias in options:
                with torch.no_grad():
                    mod = M(pointwise_fn, input_shape[-1], out_feature, bias).eval()
                    v = torch.randn(input_shape)
                    if input_stride is not None:
                        v = v.as_strided(input_shape, input_stride)
                    other = torch.randn(input_shape[:-1] + [out_feature])
                    ref = mod(v, other)
                    attr = pointwise_name
                    fused = torch.ops.onednn._linear_pointwise(
                        v, other, mod.linear.weight, mod.linear.bias, attr
                    )
                    self.assertEqual(ref, fused)

    def test_conv_transpose_unary_fusion_ops(self):
        class M(nn.Module):
            def __init__(self, unary_fn, dim, in_channels, out_channels, kernel_size, **kwargs):
                super().__init__()
                self.conv_transpose = CONV_TRANSPOSE_MODULES[dim](in_channels, out_channels, kernel_size, **kwargs)
                self.unary = unary_fn

            def forward(self, x):
                x = self.conv_transpose(x)
                x = self.unary(x)
                return x

        input_shapes = {2: (28, 28)}
        kernel_size = 3
        for pointwise_info in self._unary_list().values():
            for dim in [2]:
                channels_last = torch.channels_last if dim == 2 else torch.channels_last_3d
                options = itertools.product([True, False], [1, 2], [1, 4], [torch.contiguous_format, channels_last], [False, True])
                for bias, dilation, groups, memory_format, prepack_weight in options:
                    oC = 32 * groups
                    iC = 3 * groups
                    x_shape = (1, iC) + input_shapes[dim]
                    x = torch.randn(x_shape, dtype=torch.float32).to(memory_format=memory_format)
                    mod = M(pointwise_info.pointwise_module, dim, iC, oC, kernel_size, dilation=dilation, groups=groups, bias=bias)
                    mod = mod.to(memory_format=memory_format).eval()
                    with torch.no_grad():
                        ref = mod(x)
                        attr = pointwise_info.attr
                        scalars = pointwise_info.scalars
                        algorithm = pointwise_info.algorithm

                        if prepack_weight:
                            packed_weight = torch.ops.onednn._reorder_convolution_transpose_weight(
                                mod.conv_transpose.weight,
                                mod.conv_transpose.padding,
                                mod.conv_transpose.output_padding,
                                mod.conv_transpose.stride,
                                mod.conv_transpose.dilation,
                                mod.conv_transpose.groups,
                                x.size())
                            mod.conv_transpose.weight = torch.nn.Parameter(
                                packed_weight,
                                requires_grad=mod.conv_transpose.weight.requires_grad,
                            )

                        fused = torch.ops.onednn._convolution_transpose_pointwise(
                            x,
                            mod.conv_transpose.weight,
                            mod.conv_transpose.bias,
                            mod.conv_transpose.padding,
                            mod.conv_transpose.output_padding,
                            mod.conv_transpose.stride,
                            mod.conv_transpose.dilation,
                            mod.conv_transpose.groups,
                            attr,
                            scalars,
                            algorithm)
                    self.assertEqual(ref, fused)

if __name__ == "__main__":
    run_tests()
