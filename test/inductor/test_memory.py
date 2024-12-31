# Owner(s): ["module: inductor"]
import unittest
from unittest import mock

import torch
from torch._C import FileCheck
from torch._dynamo.utils import same
from torch._inductor import config, memory
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class Foo(torch.nn.Module):
    """
    The default compiled graph is
    graph():
        ...
        %op0 : [num_users=2] = call_function[...](args = (%primals_2, %primals_1), ...)
        %op1 : [num_users=2] = call_function[...](args = (%primals_2, %primals_3), ...)
        %op2 : [num_users=1] = call_function[...](args = (%op0, %primals_4), ...)
        %op3 : [num_users=1] = call_function[...](args = (%op1, %primals_5), ...)
        %op4 : [num_users=1] = call_function[...](args = (%op2,), ...)
        %op5 : [num_users=1] = call_function[...](args = (%op3,), ...)
        %op6_op7 : [num_users=1] = call_function[...](args = (%op5, %op4), ...)
    """

    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.ones(1, 10))
        self.w2 = torch.nn.Parameter(torch.ones(1, 1))
        self.w3 = torch.nn.Parameter(torch.ones(10, 1))
        self.w4 = torch.nn.Parameter(torch.ones(1, 10))

    def forward(self, x):
        t1 = torch.matmul(x, self.w1)
        t2 = torch.matmul(x, self.w2)
        t3 = torch.matmul(t1, self.w3)
        t4 = torch.matmul(t2, self.w4)
        return t3.sum() + t4.sum()


# The tests in this class uses very small tensors. The default
# score_fusion_memory threshold will cause different fusion decisions and
# generate a different wrapper. Override the threshold to make these tests
# happy.
@config.patch("score_fusion_memory_threshold", 1)
class TestOperatorReorderForPeakMemory(TestCase):
    def setUp(self):
        super().setUp()

        self.model = Foo().to(GPU_TYPE)
        self.inputs = torch.ones((2048, 1), device=GPU_TYPE)
        self.orig_reorder_method = memory.reorder_for_peak_memory

    @mock.patch.object(config, "reorder_for_peak_memory", True)
    def test_reorder_peak_memory(self):
        outp_corr = self.model(self.inputs)
        compiled_model = torch.compile(self.model)
        code = run_and_get_triton_code(compiled_model, self.inputs)
        (
            FileCheck()
            .check("def call(args):")
            .check("buf1 = ")
            .check("buf0 = ")
            .check("buf2 = ")
            .check("buf4 = ")
            .check("buf3 = ")
            .check("buf5 = ")
            .check("buf7 = ")
            .run(code)
        )
        # check for correctness
        outp = compiled_model(self.inputs)
        self.assertTrue(same(outp, outp_corr))

    @mock.patch.object(config, "reorder_for_peak_memory", True)
    def test_reorder_peak_memory_lpmf(self):
        outp_corr = self.model(self.inputs)

        def reorder_with_only_lpmf(
            nodes,
            name_to_buf,
            name_to_fused_node,
            graph_inputs,
            graph_outputs,
            methods=None,
        ):
            return self.orig_reorder_method(
                nodes,
                name_to_buf,
                name_to_fused_node,
                graph_inputs,
                graph_outputs,
                methods=[memory.topological_sort_lpmf],
            )

        with mock.patch.object(
            memory, "reorder_for_peak_memory", reorder_with_only_lpmf
        ):
            compiled_model = torch.compile(self.model)

            code = run_and_get_triton_code(compiled_model, self.inputs)
            (
                FileCheck()
                .check("def call(args):")
                .check("buf1 = ")
                .check("buf0 = ")
                .check("buf2 = ")
                .check("buf4 = ")
                .check("buf3 = ")
                .check("buf5 = ")
                .check("buf7 = ")
                .run(code)
            )
            # check for correctness
            outp = compiled_model(self.inputs)
            self.assertTrue(same(outp, outp_corr))

    @mock.patch.object(config, "reorder_for_peak_memory", True)
    def test_reorder_peak_memory_bfs(self):
        outp_corr = self.model(self.inputs)

        def reorder_with_only_bfs(
            nodes,
            name_to_buf,
            name_to_fused_node,
            graph_inputs,
            graph_outputs,
            methods=None,
        ):
            return self.orig_reorder_method(
                nodes,
                name_to_buf,
                name_to_fused_node,
                graph_inputs,
                graph_outputs,
                methods=[memory.topological_sort_bfs],
            )

        with mock.patch.object(
            memory, "reorder_for_peak_memory", reorder_with_only_bfs
        ):
            compiled_model = torch.compile(self.model)

            code = run_and_get_triton_code(compiled_model, self.inputs)
            (
                FileCheck()
                .check("def call(args):")
                .check("buf0 = ")
                .check("buf1 = ")
                .check("buf2 = ")
                .check("buf3 = ")
                .check("buf4 = ")
                .check("buf5 = ")
                .check("buf7 = ")
                .run(code)
            )
            # check for correctness
            outp = compiled_model(self.inputs)
            self.assertTrue(same(outp, outp_corr))

    @mock.patch.object(config, "reorder_for_peak_memory", True)
    def test_reorder_peak_memory_dfs(self):
        outp_corr = self.model(self.inputs)

        def reorder_with_only_dfs(
            nodes,
            name_to_buf,
            name_to_fused_node,
            graph_inputs,
            graph_outputs,
            methods=None,
        ):
            return self.orig_reorder_method(
                nodes,
                name_to_buf,
                name_to_fused_node,
                graph_inputs,
                graph_outputs,
                methods=[memory.topological_sort_dfs],
            )

        with mock.patch.object(
            memory, "reorder_for_peak_memory", reorder_with_only_dfs
        ):
            compiled_model = torch.compile(self.model)

            code = run_and_get_triton_code(compiled_model, self.inputs)
            (
                FileCheck()
                .check("def call(args):")
                .check("buf0 = ")
                .check("buf2 = ")
                .check("buf4 = ")
                .check("buf1 = ")
                .check("buf3 = ")
                .check("buf5 = ")
                .check("buf7 = ")
                .run(code)
            )
            # check for correctness
            outp = compiled_model(self.inputs)
            self.assertTrue(same(outp, outp_corr))

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties().total_memory < int(1e10),
        "Need 10GB memory to be safe to run the test",
    )
    def test_fusing_reductions_increase_peak_memory(self):
        @torch.compile
        def f(a, b, c):
            return (a @ c).sum(dim=-1) + (b @ c).sum(dim=-1)

        a = torch.randn(1024 * 32, 16, device=GPU_TYPE)
        b = torch.randn(1024 * 32, 16, device=GPU_TYPE)
        c = torch.randn(16, 1024 * 32, device=GPU_TYPE)
        torch.cuda.reset_peak_memory_stats()
        f(a, b, c)
        peak_mem = torch.cuda.max_memory_allocated()

        expected_bound = a.size(0) * c.size(1) * a.dtype.itemsize * 2
        self.assertLess(peak_mem, expected_bound)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
