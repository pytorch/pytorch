# Owner(s): ["module: fx"]

import copy
import sys
import logging
from typing import List, Tuple

import torch
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.backends.nvfuser import NvFuserBackend

from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase, NoTest
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCUDAIfRocm,
    dtypes,
)

if not TEST_CUDA:
    print('CUDA not available, skipping tests', file=sys.stderr)
    TestCase = NoTest  # noqa: F811

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class HF_T5_Partial(torch.nn.Module):

    def inputs_meta(self):
        return [
            (torch.Size([512, 512]), torch.float32),
            (torch.Size([512, 512]), torch.float32),
            (torch.Size([512, 512]), torch.float32),
            (torch.Size([512, 512]), torch.float32),
            (torch.Size([512]), torch.float32),
            (torch.Size([2048, 512]), torch.float32),
            (torch.Size([512, 2048]), torch.float32),
            (torch.Size([512]), torch.float32),
            (torch.Size([8, 1024, 512]), torch.float32),
            (torch.Size([8, 8, 1024, 1024]), torch.float32),
        ]

    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5,
                primals_6, primals_7, primals_8, primals_9, primals_10):
        pow_1 = torch.ops.aten.pow(primals_9, 2)
        mean = torch.ops.aten.mean(pow_1, [-1], True)
        add = torch.ops.aten.add(mean, 1e-06)
        rsqrt = torch.ops.aten.rsqrt(add)
        mul = torch.ops.aten.mul(primals_9, rsqrt)
        mul_1 = torch.ops.aten.mul(primals_5, mul)
        t = torch.ops.aten.t(primals_3)
        view = torch.ops.aten.view(mul_1, [8192, 512])
        mm = torch.ops.aten.mm(view, t)
        _unsafe_view = torch.ops.aten._unsafe_view(mm, [8, 1024, 512])
        view_1 = torch.ops.aten.view(_unsafe_view, [8, -1, 8, 64])
        transpose = torch.ops.aten.transpose(view_1, 1, 2)
        t_1 = torch.ops.aten.t(primals_1)
        view_2 = torch.ops.aten.view(mul_1, [8192, 512])
        mm_1 = torch.ops.aten.mm(view_2, t_1)
        _unsafe_view_1 = torch.ops.aten._unsafe_view(mm_1, [8, 1024, 512])
        view_3 = torch.ops.aten.view(_unsafe_view_1, [8, -1, 8, 64])
        transpose_1 = torch.ops.aten.transpose(view_3, 1, 2)
        t_2 = torch.ops.aten.t(primals_4)
        view_4 = torch.ops.aten.view(mul_1, [8192, 512])
        mm_2 = torch.ops.aten.mm(view_4, t_2)
        _unsafe_view_2 = torch.ops.aten._unsafe_view(mm_2, [8, 1024, 512])
        view_5 = torch.ops.aten.view(_unsafe_view_2, [8, -1, 8, 64])
        transpose_2 = torch.ops.aten.transpose(view_5, 1, 2)
        transpose_3 = torch.ops.aten.transpose(transpose_1, 3, 2)
        expand = torch.ops.aten.expand(transpose, [8, 8, 1024, 64])
        clone = torch.ops.aten.clone(expand, memory_format=torch.contiguous_format)
        _unsafe_view_3 = torch.ops.aten._unsafe_view(clone, [64, 1024, 64])
        expand_1 = torch.ops.aten.expand(transpose_3, [8, 8, 64, 1024])
        clone_1 = torch.ops.aten.clone(expand_1, memory_format=torch.contiguous_format)
        _unsafe_view_4 = torch.ops.aten._unsafe_view(clone_1, [64, 64, 1024])
        bmm = torch.ops.aten.bmm(_unsafe_view_3, _unsafe_view_4)
        _unsafe_view_5 = torch.ops.aten._unsafe_view(bmm, [8, 8, 1024, 1024])
        add_ = torch.ops.aten.add_(_unsafe_view_5, primals_10)
        _softmax = torch.ops.aten._softmax(add_, -1, False)
        expand_2 = torch.ops.aten.expand(_softmax, [8, 8, 1024, 1024])
        view_6 = torch.ops.aten.view(expand_2, [64, 1024, 1024])
        expand_3 = torch.ops.aten.expand(transpose_2, [8, 8, 1024, 64])
        clone_2 = torch.ops.aten.clone(expand_3, memory_format=torch.contiguous_format)
        _unsafe_view_6 = torch.ops.aten._unsafe_view(clone_2, [64, 1024, 64])
        bmm_1 = torch.ops.aten.bmm(view_6, _unsafe_view_6)
        _unsafe_view_7 = torch.ops.aten._unsafe_view(bmm_1, [8, 8, 1024, 64])
        transpose_4 = torch.ops.aten.transpose(_unsafe_view_7, 1, 2)
        clone_3 = torch.ops.aten.clone(transpose_4, memory_format=torch.contiguous_format)
        view_7 = torch.ops.aten.view(clone_3, [8, -1, 512])
        t_3 = torch.ops.aten.t(primals_2)
        view_8 = torch.ops.aten.view(view_7, [8192, 512])
        mm_3 = torch.ops.aten.mm(view_8, t_3)
        _unsafe_view_8 = torch.ops.aten._unsafe_view(mm_3, [8, 1024, 512])
        add_1 = torch.ops.aten.add(primals_9, _unsafe_view_8)
        pow_2 = torch.ops.aten.pow(add_1, 2)
        mean_1 = torch.ops.aten.mean(pow_2, [-1], True)
        add_2 = torch.ops.aten.add(mean_1, 1e-06)
        rsqrt_1 = torch.ops.aten.rsqrt(add_2)
        mul_2 = torch.ops.aten.mul(add_1, rsqrt_1)
        mul_3 = torch.ops.aten.mul(primals_8, mul_2)
        t_4 = torch.ops.aten.t(primals_6)
        view_9 = torch.ops.aten.view(mul_3, [8192, 512])
        mm_4 = torch.ops.aten.mm(view_9, t_4)
        _unsafe_view_9 = torch.ops.aten._unsafe_view(mm_4, [8, 1024, 2048])
        relu = torch.ops.aten.relu(_unsafe_view_9)
        t_5 = torch.ops.aten.t(primals_7)
        view_10 = torch.ops.aten.view(relu, [8192, 2048])
        mm_5 = torch.ops.aten.mm(view_10, t_5)
        _unsafe_view_10 = torch.ops.aten._unsafe_view(mm_5, [8, 1024, 512])
        add_3 = torch.ops.aten.add(add_1, _unsafe_view_10)
        return [add_3, rsqrt, _unsafe_view_3, t_3, _softmax, view_6, mul_2, t, view_9, t_1, primals_5, add_1,
                _unsafe_view_4, view_2, view_10, t_5, t_2, primals_8, view_4, view_8, rsqrt_1, primals_9, t_4,
                mul, _unsafe_view_6, relu, view]


class TestFxNvFuserBackend(TestCase):

    def _generate_random_inputs(self, device, inputs_meta: List[Tuple[torch.Size, torch.dtype]]):
        inputs = []
        for meta in inputs_meta:
            shape, dtype = meta

            if dtype in {torch.int, torch.int32, torch.int64, torch.bool, torch.int, torch.uint8}:
                input = torch.randint(0, 1, shape, dtype=dtype, device=device)
            else:
                input = torch.rand(shape, dtype=dtype, device=device)

            inputs.append(input)

        return inputs


    @skipCUDAIfRocm
    @dtypes(torch.float32)
    def test_nvfuser_call_module_backend(self, device, dtype):

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.bn = torch.nn.BatchNorm2d(3)
                self.relu = torch.nn.ReLU()

            def forward(self, inp):
                o = self.bn(inp)
                o = self.relu(o)
                return o

        inp = torch.randn(2, 3, 4, 5).to(dtype=dtype, device=device)
        m = Model().to(dtype=dtype, device=device)

        # note that the traced module here contains only `call_module` node,
        # which isn't fused by nvfuser backend. But `nvfuser.compile` should run without error
        traced = symbolic_trace(m)

        nvfuser = NvFuserBackend()
        compiled_module = nvfuser.compile(traced)

        eager_result = m(inp)
        nvfuser_result = compiled_module(inp)

        torch.testing.assert_close(eager_result, nvfuser_result, rtol=1e-5, atol=1e-5)


    @skipCUDAIfRocm
    @dtypes(torch.float32)
    def test_nvfuser_backend(self, device, dtype):
        m = HF_T5_Partial()
        m.to(device)

        traced = symbolic_trace(m)

        nvfuser = NvFuserBackend()
        compiled_module = nvfuser.compile(traced)

        inputs = self._generate_random_inputs(device, m.inputs_meta())

        eager_result = m(*inputs)
        nvfuser_result = compiled_module(*inputs)

        torch.testing.assert_close(eager_result, nvfuser_result, rtol=1e-5, atol=1e-5)

    @skipCUDAIfRocm
    @dtypes(torch.float32)
    def test_aten_square(self, device, dtype):

        def fn(x):
            square = torch.square(x)
            a = square + 1
            b = a + 1
            return b

        inputs = torch.randn(4, device=device)
        traced = make_fx(fn)(inputs)

        nvfuser = NvFuserBackend()
        compiled_module = nvfuser.compile(copy.deepcopy(traced))

        for node in compiled_module.graph.nodes:
            if node.op == "call_function":
                assert "fused" in str(node.target), "the entire function should be fused into a single fusion group"

        eager_result = traced(inputs)
        nvfuser_result = compiled_module(inputs)
        torch.testing.assert_close(eager_result, nvfuser_result, rtol=1e-5, atol=1e-5)

    @skipCUDAIfRocm
    @dtypes(torch.float32)
    def test_aten_leakyrelu(self, device, dtype):

        def fn(x):
            square = torch.ops.aten.leaky_relu(x, 0.1)
            a = square + 1
            b = a + 1
            return b

        inputs = torch.randn(4, device=device)
        traced = make_fx(fn)(inputs)

        nvfuser = NvFuserBackend()
        compiled_module = nvfuser.compile(copy.deepcopy(traced))

        for node in compiled_module.graph.nodes:
            if node.op == "call_function":
                assert "fused" in str(node.target), "the entire function should be fused into a single fusion group"

        eager_result = traced(inputs)
        nvfuser_result = compiled_module(inputs)
        torch.testing.assert_close(eager_result, nvfuser_result, rtol=1e-5, atol=1e-5)

    @skipCUDAIfRocm
    @dtypes(torch.float32)
    def test_aten_where(self, device, dtype):

        def fn(x):
            where = torch.ops.aten.where(x < 0, -x, x)
            a = where + 1
            b = a + 1
            return b

        inputs = torch.randn(4, device=device)
        traced = make_fx(fn)(inputs)

        nvfuser = NvFuserBackend()
        compiled_module = nvfuser.compile(copy.deepcopy(traced))

        for node in compiled_module.graph.nodes:
            if node.op == "call_function":
                assert "fused" in str(node.target), "the entire function should be fused into a single fusion group"

        eager_result = traced(inputs)
        nvfuser_result = compiled_module(inputs)
        torch.testing.assert_close(eager_result, nvfuser_result, rtol=1e-5, atol=1e-5)

instantiate_device_type_tests(TestFxNvFuserBackend, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
