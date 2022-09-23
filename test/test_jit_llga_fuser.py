# Owner(s): ["module: mkldnn"]
import torch
import unittest
import itertools

import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_utils import run_tests, TEST_SCIPY, IS_WINDOWS, IS_MACOS

LLGA_FUSION_GROUP = 'prim::oneDNNFusionGroup'
LLGA_NOT_ENABLED = not torch._C.has_mkldnn or IS_WINDOWS or IS_MACOS


def warmup_forward(f, *args, profiling_count=2):
    for i in range(profiling_count):
        results = f(*args)

    return results


class JitLlgaTestCase(JitTestCase):
    def setUp(self):
        torch.jit.enable_onednn_fusion(True)

    def tearDown(self):
        torch.jit.enable_onednn_fusion(False)

    def checkTrace(self, m, x, *args, **kwargs):
        if isinstance(m, torch.nn.Module):
            m.eval()
        with torch.no_grad(), \
                torch._jit_internal._disable_emit_hooks():
            traced = torch.jit.trace(m, x)
            if isinstance(m, torch.nn.Module):
                traced = torch.jit.freeze(traced)
            warmup_forward(traced, *x)
            fwd_graph = traced.graph_for(*x)

            ref_o = m(*x)
            jit_o = traced(*x)
            self.assertEqual(jit_o, ref_o)
        return traced, fwd_graph

    def assertFused(self, graph, fused_patterns):
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)


try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
except RuntimeError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, 'no torchvision')

def get_eltwise_fn(name):
    if hasattr(torch, name):
        return getattr(torch, name)
    elif hasattr(F, name):
        return getattr(F, name)
    else:
        raise NameError('Eltwise function %s not found' % name)


@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
class TestOp(JitLlgaTestCase):
    def test_conv2d(self):
        for [spatial, in_channels, out_channels, kernel, padding, stride, dilation, g, bias] in itertools.product(
                [7, 8],
                [8, 15],
                [7, 16],
                [3, 4],
                [0, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [True, False]):

            m = nn.Conv2d(in_channels=in_channels * g,
                          out_channels=out_channels * g,
                          kernel_size=kernel,
                          padding=padding,
                          stride=stride,
                          dilation=dilation,
                          groups=g,
                          bias=bias)

            x = torch.rand(1, in_channels * g, spatial, spatial)
            _, graph = self.checkTrace(m, [x])
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    def test_bn2d(self):
        m = nn.BatchNorm2d(32).eval()
        x = torch.rand(1, 32, 28, 28)
        _, graph = self.checkTrace(m, [x])
        # single-op partition shouldn't be created for softmax
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    def test_eltwise(self):
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super(M, self).__init__()
                self.eltwise = eltwise_fn

            def forward(self, x):
                return self.eltwise(x)

        for eltwise in ['relu', 'gelu']:
            eltwise_fn = get_eltwise_fn(eltwise)
            m = M(eltwise_fn)
            x = torch.rand(1, 32, 28, 28)
            _, graph = self.checkTrace(m, [x])
            # single-op partition shouldn't be created.
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    def test_max_pool2d(self):
        for [spatial, kernel, padding, stride, dilation, ceil_mode] in itertools.product(
                [15, 16, 17, 18, 19],
                [4, 5],
                [0, 1, 2],
                [1, 2],  # [1, 2, 4], TODO: fix issue in pad calculation
                [1],     # [1, 2], TODO: backend support for dilation
                [True, False]):

            m = nn.MaxPool2d(kernel_size=kernel,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             ceil_mode=ceil_mode)

            x = torch.rand(1, 4, spatial, spatial)
            _, graph = self.checkTrace(m, [x])
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    def test_avg_pool2d(self):
        for [spatial, kernel, padding, stride, ceil_mode, count_include_pad] in itertools.product(
                [15, 16, 17, 18, 19],
                [4, 5],
                [0, 1, 2],
                [1, 2, 4],
                [False],  # TODO: oneDNN Graph does not fully support ceil_mode=True
                [True, False]):

            m = nn.AvgPool2d(kernel_size=kernel,
                             stride=stride,
                             padding=padding,
                             ceil_mode=ceil_mode,
                             count_include_pad=count_include_pad)

            x = torch.rand(1, 4, spatial, spatial)
            _, graph = self.checkTrace(m, [x])
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    def test_variable_kernel_avg_pool2d(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()

            def forward(self, x):
                x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0, count_include_pad=False)
                return x

        x = torch.randn(1, 1000, 1, 1)
        m = M()
        _, graph = self.checkTrace(m, [x])
        # kernel_size is not Constant, shouldn't have any LLGA_FUSION_GROUP
        # TODO: with shape specialization, should have 1 LLGA_FUSION_GROUP
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    def test_softmax(self):
        for dim in [-4, -3, -2, -1, 0, 1, 2, 3]:
            m = nn.Softmax(dim=dim)
            x = torch.rand(8, 12, 12, 12)
            _, graph = self.checkTrace(m, [x])
            # single-op partition shouldn't be created for softmax
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    def test_linear(self):
        for bias in [True, False]:
            x = torch.rand(32, 28)
            m = torch.nn.Linear(in_features=28, out_features=64, bias=bias)
            _, graph = self.checkTrace(m, [x])
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            self.assertFused(graph, ['aten::linear'])

    def _gen_binary_inputs(self, gen_permute=True):
        for xshape, yshape in [
            [[1, 32, 28, 28], [1, 32, 28, 28]],
            [[1, 32, 28, 28], [1, 1, 28, 28]],
            [[1, 32, 28, 28], [28]],
            [[1, 32, 28, 28], [1]],

        ]:
            yield torch.rand(xshape), torch.rand(yshape)
            if gen_permute and xshape != yshape:
                yield torch.rand(yshape), torch.rand(xshape)

    def test_add(self):
        def forward_add(x, y):
            return torch.add(x, y, alpha=2)

        for x, y in self._gen_binary_inputs():
            _, graph = self.checkTrace(forward_add, [x, y])
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    def test_add_scalar(self):
        def add_scalar(x):
            return 42 + x + 3.14

        x = torch.rand(32, 32)
        _, graph = self.checkTrace(add_scalar, [x])
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    def test_addmm(self):
        def addmm(x, y, z):
            # alpha and beta are 1, by default
            return torch.addmm(z, x, y)

        x = torch.rand(64, 32)
        y = torch.rand(32, 32)
        z = torch.rand(64, 32)
        _, graph = self.checkTrace(addmm, [x, y, z])
        # single-op partition should be created for matmul with bias.
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    def test_mul(self):
        def forward_mul(x, y):
            return torch.mul(x, y) * 3

        for x, y in self._gen_binary_inputs():
            _, graph = self.checkTrace(forward_mul, [x, y])
            # single-op partitions shouldn't be created
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    def test_identity_binary(self):
        def forward(x):
            return x * 1 + 0.0

        x = torch.rand(32)
        _, graph = self.checkTrace(forward, [x])
        self.assertFused(graph, ['aten::add', 'aten::mul'])

    def test_layer_norm(self):
        # TODO: support more normalized_shape
        m = torch.nn.LayerNorm(10)
        x = torch.randn(2, 5, 10, 10)
        _, graph = self.checkTrace(m, [x])
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    def test_cat(self):
        def cat_along_dim(d):
            def forward_cat(*inputs):
                return torch.cat(inputs, d)
            return forward_cat

        for xshape in [
            [8, 8, 8, 8],
            [64, 8, 32],
            [2048, 64],
        ]:
            for d in range(len(xshape)):
                x = torch.rand(xshape)
                _, graph = self.checkTrace(cat_along_dim(d), [x, x, x])
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    def test_typecheck(self):
        x = torch.rand(32, 28)
        m = torch.nn.Linear(in_features=28, out_features=64, bias=True)
        traced, graph = self.checkTrace(m, [x])
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::linear'])
        # change the shape of the input, we should enter fallback graph
        x = torch.rand(5, 28)
        self.assertEqual(m(x), traced(x))


@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
class TestFusionPattern(JitLlgaTestCase):
    def test_conv2d_eltwise(self):
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
                self.eltwise = eltwise_fn

            def forward(self, x):
                x = self.conv1(x)
                x = self.eltwise(x)
                x = self.conv2(x)
                x = self.eltwise(x)
                return x

        # for eltwise in ['relu', 'sigmoid', 'sqrt', 'abs', 'square', 'hardtanh']:
        for eltwise in ['relu']:
            for inplace in [True, False]:
                eltwise_fn_name = eltwise + '_' if inplace else eltwise
                eltwise_fn = get_eltwise_fn(eltwise_fn_name)

                m = M(eltwise_fn)
                x = torch.rand(1, 32, 28, 28)
                _, graph = self.checkTrace(m, [x])
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
                # test if relu_ is replace with relu by mutation removal pass
                self.assertFused(graph, ['aten::' + eltwise_fn_name])
                # test if relu is fused into the fusion group
                self.assertFused(graph, ['aten::' + eltwise])

    def test_conv2d_bn(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.bn1 = nn.BatchNorm2d(32)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                return x

        m = M().eval()
        x = torch.rand(1, 32, 28, 28)
        _, graph = self.checkTrace(m, [x])
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::_convolution', 'aten::batch_norm'])


    def test_conv2d_bn_relu(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.bn1 = nn.BatchNorm2d(32)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = F.relu(x)
                return x

        m = M().eval()
        x = torch.rand(1, 32, 28, 28)
        _, graph = self.checkTrace(m, [x])
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::_convolution', 'aten::batch_norm',
                                 'aten::relu'])

    def test_bn2d_eltwise(self):
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super(M, self).__init__()
                self.eltwise = eltwise_fn
                self.bn = nn.BatchNorm2d(32)

            def forward(self, x):
                x = self.bn(x)
                x = self.eltwise(x)
                return x

        for eltwise in ['relu']:
            eltwise_fn = get_eltwise_fn(eltwise)
            m = M(eltwise_fn).eval()
            x = torch.rand(1, 32, 28, 28)
            _, graph = self.checkTrace(m, [x])
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            self.assertFused(graph, ['aten::' + eltwise])

    def test_linear_eltwise(self):
        class M(nn.Module):
            def __init__(self, eltwise_fn, bias):
                super(M, self).__init__()
                self.linear = nn.Linear(28, 64, bias)
                self.eltwise = eltwise_fn

            def forward(self, x):
                x = self.linear(x)
                x = self.eltwise(x)
                return x

        for [has_bias, eltwise] in itertools.product(
                [True, False],
                ['relu', 'gelu', 'sigmoid', 'hardtanh', 'relu6', 'elu']):

            eltwise_fn = get_eltwise_fn(eltwise)
            m = M(eltwise_fn, has_bias)
            x = torch.rand(32, 28, requires_grad=False)
            _, graph = self.checkTrace(m, [x])
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            self.assertFused(graph, ['aten::' + eltwise])

    def test_conv2d_sum(self):
        class M(nn.Module):
            def __init__(self, bias=False):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=bias)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=bias)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu = nn.ReLU()
                self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=bias)
                self.bn3 = nn.BatchNorm2d(32)

            def forward(self, x, y):
                x = self.conv1(x)
                x = self.bn1(x)
                y = self.conv2(y)
                y = self.bn2(y)
                z = self.relu(x + y)
                z = self.conv3(z)
                z = self.bn3(z)
                return z

        for bias in [True, False]:
            m = M(bias).eval()
            x = torch.rand(1, 32, 16, 16, requires_grad=False)
            y = torch.rand(1, 32, 16, 16, requires_grad=False)
            _, graph = self.checkTrace(m, [x, y])
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)

    def test_wildcard(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.eltwise = nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                y = self.eltwise(x)
                return [x, y]

        # The pattern is as the following:
        #      conv
        #     |    \
        # eltwise   \
        #    |       \
        #  ListConstruct
        #
        # The output of conv is used by a wildcard op: ListConstruct.
        # Thus conv-eltwise cannot be selected into the same Partition.
        m = M()
        x = torch.rand(1, 32, 28, 28)
        _, graph = self.checkTrace(m, [x])
        # conv can exist in a single-op oneDNN Graph partition but not relu
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::_convolution'])

    def test_rewrap_tensor_input_to_pytorch(self):
        class M(nn.Module):
            def __init__(self, eltwise_fn, data_type):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True, dtype=data_type)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True, dtype=data_type)
                self.eltwise = eltwise_fn
                self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((5, 7))

            def forward(self, x, y):
                x = self.conv1(x)
                x = self.eltwise(x)
                x = self.conv2(x)
                x = self.eltwise(x)
                x = torch.add(x, y)
                x = self.adaptive_avg_pool_2d(x)
                return x

        eltwise_fn_name = 'relu'
        eltwise_fn = get_eltwise_fn(eltwise_fn_name)
        # Add bfloat16 later
        for data_type in [torch.float]:
            m = M(eltwise_fn, data_type)
            m = m.to(memory_format=torch.channels_last)
            x = torch.rand(1, 32, 28, 28, dtype=data_type).to(memory_format=torch.channels_last)
            y = torch.rand(1, 32, 28, 28, dtype=data_type).to(memory_format=torch.channels_last)
            # Simply test if the output is accurate
            # The output of the second partition is input to adaptive_avg_pool2d, which is
            # unsupported by LLGA, so it must be handled by PyTorch, which should receive
            # correct strides info of the channels-last tensor.
            graph, _ = self.checkTrace(m, [x, y])


@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
class TestEnableDisableLlgaFuser(JitTestCase):
    def setUp(self):
        super().setUp()
        self.is_enabled = torch._C._jit_set_llga_enabled(False)

    def tearDown(self):
        torch._C._jit_set_llga_enabled(self.is_enabled)
        super().tearDown()

    def test_context_manager(self):
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        with torch.jit.fuser('fuser3'):
            with torch.jit.fuser('fuser3'):

                def t1(x, y):
                    o = x + y
                    o = o + 2.0
                    return o
                t_jit = torch.jit.script(t1)
                t_jit(x, y)
                t_jit(x, y)
                self.assertGraphContains(t_jit.graph_for(x, y), LLGA_FUSION_GROUP)

            def t2(x, y):
                o = x + y
                o = o + 3.0
                return o
            t_jit_2 = torch.jit.script(t2)
            t_jit_2(x, y)
            t_jit_2(x, y)
            self.assertGraphContains(t_jit_2.graph_for(x, y), LLGA_FUSION_GROUP)

        def t3(x, y):
            o = x + y
            o = o + 4.0
            return o
        t_jit_3 = torch.jit.script(t3)
        t_jit_3(x, y)
        t_jit_3(x, y)
        self.assertGraphContainsExactly(t_jit_3.graph_for(x, y), LLGA_FUSION_GROUP, 0)


@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
class TestModel(JitLlgaTestCase):
    @skipIfNoTorchVision
    def _test_vision(self, model_name):
        m = getattr(torchvision.models, model_name)().eval()
        x = torch.rand(1, 3, 224, 224) / 10
        _, graph = self.checkTrace(m, [x])
        self.assertFused(graph, ['aten::_convolution', 'aten::batch_norm',
                                 'aten::relu', 'aten::linear',
                                 'aten::avg_pool2d', 'aten::max_pool2d'])


for model_name, enabled in [
    ['resnet50', True],
    ['resnext50_32x4d', True],
    ['resnext101_32x8d', True],
    ['densenet121', True],
    ['googlenet', TEST_SCIPY],
    ['mobilenet_v2', True],
    ['mnasnet1_0', True],
    ['squeezenet1_0', True],
    ['vgg16', True],
    ['alexnet', True],
    ['shufflenet_v2_x1_0', True],
    ['wide_resnet50_2', True],
]:
    def wrapper(mname):
        @unittest.skipIf(not enabled, 'Disabled')
        def test(self):
            return self._test_vision(mname)
        return test

    setattr(TestModel, 'test_vision_%s' % model_name, wrapper(model_name))

if __name__ == '__main__':
    run_tests()
