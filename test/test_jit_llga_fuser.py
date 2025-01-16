# Owner(s): ["module: mkldnn"]
import sys
import torch
import unittest
import itertools
import torch.nn as nn
from functools import wraps
from concurrent import futures
import torch.nn.functional as F
import torch.fx.experimental.optimization as optimization
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_utils import run_tests, TEST_SCIPY, IS_WINDOWS, IS_MACOS
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    dtypes
)

# We use this wrapper to run UTs of TorchVision models because of a memory-leak
# issue with JIT tracing that causes traced model objects to persist in the
# memory. Ref: https://github.com/pytorch/pytorch/issues/35600
# Memory requirement for running these UTs was thus increasing cumulatively, and
# invoked the Linux kernel OOM killer on linux.2xlarge PyTorch CI runners, which
# only have 16 GB RAM. Cumulatively, these UTs had been using more than 14 GB
# memory (as per psutils). So now we run each TorchVision model UTs in separate processes.
def separate_process(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with futures.ProcessPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            futures.wait([future])
    return wrapper

def is_avx512_supported():
    if sys.platform != 'linux':
        return False
    with open("/proc/cpuinfo", encoding="ascii") as f:
        lines = f.read()
    return "avx512" in lines

IS_AVX512_UNSUPPORTED = not is_avx512_supported()

LLGA_FUSION_GROUP = 'prim::oneDNNFusionGroup'
LLGA_NOT_ENABLED = not torch.backends.mkldnn.is_available() or IS_WINDOWS or IS_MACOS

def warmup_forward(f, *args, profiling_count=3):
    for _ in range(profiling_count):
        results = f(*args)

    return results

class JitLlgaTestCase(JitTestCase):

    def setUp(self):
        # PyTorch has divergent op support for AMP in JIT & eager modes
        # so we disable AMP for JIT & leverage eager-mode AMP.
        # Ref: https://github.com/pytorch/pytorch/issues/75956
        self.original_autocast_mode = torch._C._jit_set_autocast_mode(False)
        torch.jit.enable_onednn_fusion(True)

    def tearDown(self):
        torch.jit.enable_onednn_fusion(False)
        torch._C._jit_set_autocast_mode(self.original_autocast_mode)

    def checkTrace(self, m, x, dtype=torch.float32, *args, **kwargs):
        if isinstance(m, torch.nn.Module):
            m.eval()
        with torch.no_grad(), torch._jit_internal._disable_emit_hooks():
            if dtype == torch.bfloat16:
                # We rely upon eager-mode AMP support for BF16
                with torch.autocast(device_type="cpu", cache_enabled=False, dtype=torch.bfloat16):
                    traced = torch.jit.trace(m, x)
                    if isinstance(m, torch.nn.Module):
                        traced = torch.jit.freeze(traced)
                    warmup_forward(traced, *x)
                    ref_o = m(*x)
                    fwd_graph = traced.graph_for(*x)
            else:
                traced = torch.jit.trace(m, x)
                if isinstance(m, torch.nn.Module):
                    traced = torch.jit.freeze(traced)
                warmup_forward(traced, *x)
                ref_o = m(*x)
                fwd_graph = traced.graph_for(*x)

            jit_o = traced(*x)
            self.assertEqual(jit_o, ref_o)
            return traced, fwd_graph


    def assertFused(self, graph, fused_patterns):
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)

    def findFusionGroups(self, graph):
        result = []
        for n in graph.nodes():
            if n.kind() == LLGA_FUSION_GROUP:
                result.append(n.g('Subgraph'))
                continue
            for block in n.blocks():
                result += self.findFusionGroups(block)
        return result

    def checkPatterns(self, graph, patterns):
        fusion_groups = self.findFusionGroups(graph)
        assert len(fusion_groups) == len(patterns), "length of subgraphs not equal to length of given patterns"

        for i in range(len(fusion_groups)):
            for pattern in patterns[i]:
                self.assertGraphContains(fusion_groups[i], pattern)

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
    elif name == 'hardswish_':
        return torch.nn.Hardswish(inplace=True)
    else:
        raise NameError(f'Eltwise function {name} not found')


@unittest.skipIf(IS_AVX512_UNSUPPORTED, "This test fails for BF16 on machines without AVX512.")
@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
class TestOp(JitLlgaTestCase):
    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d(self, dtype):
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
            _, graph = self.checkTrace(m, [x], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_bn2d(self, dtype):
        m = nn.BatchNorm2d(32).eval()
        x = torch.rand(1, 32, 28, 28)
        _, graph = self.checkTrace(m, [x], dtype)
        # single-op partition shouldn't be created for softmax
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_eltwise(self, dtype):
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super().__init__()
                self.eltwise = eltwise_fn

            def forward(self, x):
                return self.eltwise(x)

        for eltwise in ['relu', 'gelu']:
            eltwise_fn = get_eltwise_fn(eltwise)
            m = M(eltwise_fn)
            x = torch.rand(1, 32, 28, 28)
            _, graph = self.checkTrace(m, [x], dtype)
            # single-op partition shouldn't be created.
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_max_pool2d(self, dtype):
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
            _, graph = self.checkTrace(m, [x], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_avg_pool2d(self, dtype):
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
            _, graph = self.checkTrace(m, [x], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_variable_kernel_avg_pool2d(self, dtype):
        class M(nn.Module):
            def forward(self, x):
                x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0, count_include_pad=False)
                return x

        x = torch.randn(1, 1000, 1, 1)
        m = M()
        _, graph = self.checkTrace(m, [x], dtype)
        # kernel_size is not Constant, shouldn't have any LLGA_FUSION_GROUP
        # TODO: with shape specialization, should have 1 LLGA_FUSION_GROUP
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_softmax(self, dtype):
        for dim in [-4, -3, -2, -1, 0, 1, 2, 3]:
            m = nn.Softmax(dim=dim)
            x = torch.rand(8, 12, 12, 12)
            _, graph = self.checkTrace(m, [x], dtype)
            # single-op partition shouldn't be created for softmax
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_linear(self, dtype):
        for bias in [True, False]:
            x = torch.rand(32, 28)
            m = torch.nn.Linear(in_features=28, out_features=64, bias=bias)
            _, graph = self.checkTrace(m, [x], dtype)
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

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_add(self, dtype):
        def forward_add(x, y):
            return torch.add(x, y, alpha=2)

        for x, y in self._gen_binary_inputs():
            _, graph = self.checkTrace(forward_add, [x, y], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_add_scalar(self, dtype):
        def add_scalar(x):
            return 42 + x + 3.14

        x = torch.rand(32, 32)
        _, graph = self.checkTrace(add_scalar, [x], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_addmm(self, dtype):
        # Just a sidenote - comparison of eager-mode & oneDNN Graph JIT outputs of
        # addmm (which entails matmul-bias-add fusion) might require higher tolerance
        # bounds for BF16. This is subject to change in the near future.
        def addmm(x, y, z):
            # alpha and beta are 1, by default
            return torch.addmm(z, x, y)

        x = torch.rand(64, 32)
        y = torch.rand(32, 32)
        z = torch.rand(64, 32)
        _, graph = self.checkTrace(addmm, [x, y, z], dtype)
        # single-op partition should be created for matmul with bias.
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_mul(self, dtype):
        def forward_mul(x, y):
            return torch.mul(x, y) * 3

        for x, y in self._gen_binary_inputs():
            _, graph = self.checkTrace(forward_mul, [x, y], dtype)
            # single-op partitions shouldn't be created
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_identity_binary(self, dtype):
        def forward(x):
            return x * 1 + 0.0

        x = torch.rand(32)
        _, graph = self.checkTrace(forward, [x], dtype)
        self.assertFused(graph, ['aten::add', 'aten::mul'])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_layer_norm(self, dtype):
        # TODO: support more normalized_shape
        m = torch.nn.LayerNorm(10)
        x = torch.randn(2, 5, 10, 10)
        _, graph = self.checkTrace(m, [x], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_cat(self, dtype):
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
                _, graph = self.checkTrace(cat_along_dim(d), [x, x, x], dtype)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_typecheck(self, dtype):
        x = torch.rand(32, 28, dtype=dtype)
        m = torch.nn.Linear(in_features=28, out_features=64, bias=True, dtype=dtype)
        traced, graph = self.checkTrace(m, [x], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::linear'])
        # change the shape of the input, we should enter fallback graph
        x = torch.rand(5, 28, dtype=dtype)
        self.assertEqual(m(x), traced(x))


@unittest.skipIf(IS_AVX512_UNSUPPORTED, "This test fails for BF16 on machines without AVX512.")
@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
class TestFusionPattern(JitLlgaTestCase):
    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_eltwise(self, dtype):
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
                self.eltwise = eltwise_fn

            def forward(self, x):
                x = self.conv1(x)
                x = self.eltwise(x)
                x = self.conv2(x)
                x = self.eltwise(x)
                return x

        for eltwise in ['relu', 'leaky_relu', 'sigmoid', 'square',
                        'abs', 'exp', 'hardswish', 'tanh', 'hardtanh']:
            for inplace in [True, False]:
                eltwise_fn_name = eltwise + '_' if inplace else eltwise
                eltwise_fn = get_eltwise_fn(eltwise_fn_name)

                m = M(eltwise_fn)
                x = torch.rand(1, 32, 28, 28)
                _, graph = self.checkTrace(m, [x], dtype=dtype)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
                # test if relu_ is replace with relu by mutation removal pass
                self.assertFused(graph, ['aten::' + eltwise_fn_name])
                # test if relu is fused into the fusion group
                self.assertFused(graph, ['aten::' + eltwise])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_silu(self, dtype):
        class M(nn.Module):
            def __init__(self, inplace):
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.eltwise = nn.SiLU(inplace=inplace)

            def forward(self, x):
                x = self.conv1(x)
                x = self.eltwise(x)
                x = self.conv2(x)
                return x
        for inplace in [False, True]:
            for memory_format in [torch.contiguous_format, torch.channels_last]:
                m = M(inplace)
                x = torch.rand(1, 32, 28, 28).to(memory_format=memory_format)

                _, graph = self.checkTrace(m, [x], dtype)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
                # oneDNN graph does not have silu OP. The bridge will convert silu to sigmoid - mul
                # Inplace op will become outplace op on the JIT graph
                patterns = [
                    ["aten::_convolution", 'aten::sigmoid', 'aten::mul'],
                    ["aten::_convolution"]
                ]
                silu_op = 'aten::silu_' if inplace else 'aten::silu'
                self.assertFused(graph, ['aten::_convolution', silu_op])
                self.checkPatterns(graph, patterns)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_ensure_tensor_is_rewrapped(self, dtype):
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv4 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.eltwise = eltwise_fn
                self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((5, 7))

            def forward(self, x, y):
                x = self.conv1(x)
                x = self.eltwise(x)
                x = self.conv2(x)
                x = self.eltwise(x)
                y = self.conv3(y)
                y = self.eltwise(y)
                y = self.conv4(y)
                y = self.eltwise(y)

                x = torch.add(x, y)
                x = self.adaptive_avg_pool_2d(x)
                return x

        eltwise_fn_name = 'relu'
        eltwise_fn = get_eltwise_fn(eltwise_fn_name)
        m = M(eltwise_fn)
        m = m.to(memory_format=torch.channels_last)
        x = torch.rand(1, 32, 28, 28).to(memory_format=torch.channels_last)
        y = torch.rand(1, 32, 28, 28).to(memory_format=torch.channels_last)
        # Simply test if the output is accurate
        # The output of the second partition is input to adaptive_avg_pool2d, which is
        # unsupported by LLGA. In resnext101 32x16d, we encountered an accuracy issue.
        _, graph = self.checkTrace(m, [x, y], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_clamp(self, dtype):
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv4 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv5 = nn.Conv2d(32, 32, 3, padding=1, bias=True)

            def forward(self, x):
                x = self.conv1(x)
                x = torch.clamp(x, min=float('-inf'))
                x = self.conv2(x)
                x = torch.clamp(x, min=-5)
                x = self.conv3(x)
                x = torch.clamp(x, min=0, max=float('inf'))
                x = self.conv4(x)
                x = torch.clamp(x, min=1, max=5)
                x = self.conv5(x)
                x = torch.clamp(x, max=2)
                return x

        for inplace in [False, True]:  # noqa: F841
            for memory_format in [torch.contiguous_format, torch.channels_last]:
                x = torch.rand(1, 32, 28, 28).to(memory_format=memory_format)
                m = M()
                _, graph = self.checkTrace(m, [x], dtype)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 5)
                self.assertFused(graph, ['aten::_convolution', "aten::clamp"])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_bn(self, dtype):
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.bn1 = nn.BatchNorm2d(32)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                return x

        m = M().eval()
        if dtype == torch.bfloat16:
            m = optimization.fuse(m)
        x = torch.rand(1, 32, 28, 28)
        _, graph = self.checkTrace(m, [x], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::_convolution', 'aten::batch_norm'])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_bn_relu(self, dtype):
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.bn1 = nn.BatchNorm2d(32)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = F.relu(x)
                return x

        m = M().eval()
        if dtype == torch.bfloat16:
            m = optimization.fuse(m)
        x = torch.rand(1, 32, 28, 28)
        _, graph = self.checkTrace(m, [x], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::_convolution', 'aten::batch_norm',
                                 'aten::relu'])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_bn2d_eltwise(self, dtype):
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super().__init__()
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
            _, graph = self.checkTrace(m, [x], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            self.assertFused(graph, ['aten::' + eltwise])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_linear_eltwise(self, dtype):
        class M(nn.Module):
            def __init__(self, eltwise_fn, bias):
                super().__init__()
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
            _, graph = self.checkTrace(m, [x], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            self.assertFused(graph, ['aten::' + eltwise])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_sum(self, dtype):
        class M(nn.Module):
            def __init__(self, bias=False):
                super().__init__()
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
            if dtype == torch.bfloat16:
                m = optimization.fuse(m)
            x = torch.rand(1, 32, 16, 16, requires_grad=False)
            y = torch.rand(1, 32, 16, 16, requires_grad=False)
            _, graph = self.checkTrace(m, [x, y], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_wildcard(self, dtype):
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
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
        _, graph = self.checkTrace(m, [x], dtype)
        # conv can exist in a single-op oneDNN Graph partition but not relu
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::_convolution'])

    @onlyCPU
    @dtypes(torch.int32)
    def test_wildcard_unsupported_dtype(self, dtype):
        class M(nn.Module):
            def forward(self, x):
                y = x // 2
                return y

        # In shufflenet_v2_x1_0, channels_per_groups is computed as:
        # channels_per_group = num_channels // groups
        # JIT IR converts groups to Long dtype, which is unsupported
        # by oneDNN Graph, viz. Long(requires_grad=0, device=cpu) = prim::Constant[value={2}]()
        # This test just ensures that the bridge code can handle
        # unsupported dtypes for inputs to ops unsupported
        # by oneDNN Graph. In this particular UT, aten::floor_divide
        # would be added as a wildcard in graph-construction stage.
        m = M()
        x = torch.tensor([32], dtype=dtype)
        _, graph = self.checkTrace(m, [x], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_rewrap_tensor_input_to_pytorch(self, dtype):
        class M(nn.Module):
            def __init__(self, eltwise_fn):
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
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
        m = M(eltwise_fn)
        m = m.to(memory_format=torch.channels_last)
        x = torch.rand(1, 32, 28, 28).to(memory_format=torch.channels_last)
        y = torch.rand(1, 32, 28, 28).to(memory_format=torch.channels_last)
        # Simply test if the output is accurate
        # The output of the second partition is input to adaptive_avg_pool2d, which is
        # unsupported by LLGA, so it must be handled by PyTorch, which should receive
        # correct strides info of the channels-last tensor.
        self.checkTrace(m, [x, y], dtype)

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
@unittest.skip("Enable when integration with dynamo aot_autograd is more stable")
class TestDynamoAOT(JitTestCase):
    def test_dynamo_aot_ts_onednn(self):
        class Seq(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.ReLU(),
                    nn.Linear(10, 10),
                    nn.ReLU(),
                )

            def forward(self, x):
                return self.layers(x)

        mod = Seq()

        import torch._dynamo
        aot_mod = torch.compile(mod, backend="aot_ts", fullgraph=True)

        for _ in range(10):
            with torch.jit.fuser("fuser3"):
                loss = aot_mod(torch.rand([10, 10])).sum()
                loss.backward()

        torch._dynamo.reset()


@unittest.skipIf(IS_AVX512_UNSUPPORTED, "This test fails for BF16 on machines without AVX512.")
@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
class TestModel(JitLlgaTestCase):
    @skipIfNoTorchVision
    def _test_vision(self, model_name, dtype):
        m = getattr(torchvision.models, model_name)().eval()
        if dtype == torch.bfloat16:
            m = optimization.fuse(m)
        x = torch.rand(1, 3, 224, 224) / 10
        _, graph = self.checkTrace(m, [x], dtype)
        self.assertFused(graph, ['aten::_convolution', 'aten::batch_norm',
                                 'aten::relu', 'aten::linear',
                                 'aten::avg_pool2d', 'aten::max_pool2d'])

for model_name, enabled in [
    ['resnet50', True],
    ['resnext50_32x4d', True],
    ['resnext101_32x8d', True],
    ['densenet121', True],
    ['densenet161', True],
    ['densenet169', True],
    ['densenet201', True],
    ['efficientnet_b0', True],
    ['efficientnet_b1', True],
    ['efficientnet_b2', True],
    ['efficientnet_b3', True],
    ['efficientnet_b4', True],
    ['efficientnet_b5', True],
    ['efficientnet_b6', True],
    ['efficientnet_b7', True],
    ['regnet_y_400mf', True],
    ['googlenet', TEST_SCIPY],
    ['mobilenet_v2', True],
    ['mobilenet_v3_large', True],
    ['mnasnet1_0', True],
    ['squeezenet1_0', True],
    ['vgg16', True],
    ['alexnet', True],
    ['shufflenet_v2_x1_0', True],
    ['wide_resnet50_2', True],
]:
    def _wrapper(mname, dtype):
        @unittest.skipIf(not enabled, 'Disabled')
        @separate_process
        def test(self, dtype=dtype):
            return self._test_vision(mname, dtype)
        return test

    for dtype in [torch.bfloat16, torch.float32]:
        setattr(TestModel, 'test_vision_{}_{}'.format(model_name, str(dtype).split("torch.")[1]), _wrapper(model_name, dtype))


instantiate_device_type_tests(TestFusionPattern, globals())
instantiate_device_type_tests(TestOp, globals())

if __name__ == '__main__':
    run_tests()
