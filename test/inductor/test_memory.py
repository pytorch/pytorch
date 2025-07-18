# Owner(s): ["module: inductor"]
import unittest
from unittest import mock

import torch
from torch._C import FileCheck
from torch._dynamo.utils import same
from torch._inductor import config, memory
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.common_utils import serialTest
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


try:
    import triton
    from triton import language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


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

    @mock.patch.object(config, "allow_buffer_reuse", False)
    @unittest.skipUnless(TRITON_AVAILABLE, "Triton is not available")
    def test_mutation_size_propogation(self):
        """
        This tests correct size propogation in the case of mutations.
        In this example, buf1 is a mutation of buf0; we should have:
        * buf0: has size_alloc 2048 and size_free 0;
        * buf1: has size_alloc 0 and size_free 2048.
        This is because
        - when buf1 is created, no additional memory is used; and
        - the 2048 bytes of memory can only be released when buf1 is freed.
        Similar arguments for buf2 and buf3, buf4 and buf5, etc.
        """

        # using triton custom kernel to creat small example with mutations
        @triton.jit
        def convert_to_bf16_kernel(
            input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(input_ptr + offsets, mask=mask)
            x_bf16 = x.to(tl.bfloat16)
            tl.store(output_ptr + offsets, x_bf16, mask=mask)

        def convert_to_bf16(x):
            output = torch.empty_like(x, dtype=torch.bfloat16)
            n_elements = x.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            convert_to_bf16_kernel[grid](
                x.flatten(), output.flatten(), n_elements, BLOCK_SIZE
            )
            return output.view(x.shape)

        # create a custom function to record the buffer size information
        buffer_info = {}
        og_method = memory.assign_memory_planning_info_for_scheduler_buffers

        def assign_memory_planning_info_for_scheduler_buffers_with_records(
            nodes, name_to_buf
        ):
            og_method(nodes, name_to_buf)
            for buf_name, buf in name_to_buf.items():
                buffer_info[buf_name] = (
                    buf.mpi_buffer.size_alloc,
                    buf.mpi_buffer.size_free,
                )

        # test example and checks
        def f(a, p):
            for e in a:
                e = convert_to_bf16(e)
                p = p @ e
            return p

        a = [torch.randn(32, 32, device=GPU_TYPE) for _ in range(4)]
        p = torch.ones(a[0].size(), dtype=torch.bfloat16, device=GPU_TYPE)

        with mock.patch.object(
            memory,
            "assign_memory_planning_info_for_scheduler_buffers",
            assign_memory_planning_info_for_scheduler_buffers_with_records,
        ):
            f_compiled = torch.compile(f)
            f_compiled(a, p)
            for buf_name in ["buf0", "buf2", "buf4", "buf6"]:
                self.assertEqual(buffer_info[buf_name], (2048, 0))

            for buf_name in ["buf1", "buf3", "buf5", "buf7"]:
                self.assertEqual(buffer_info[buf_name], (0, 2048))

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

    @serialTest()
    def test_fusion_acc_large_reads(self):
        def f(x, y, z):
            res = torch.zeros_like(x[0])
            for i in range(4):
                temp = torch.matmul(x, y) + z
                res = res + temp
            return res

        N = 128
        x = torch.rand(N, N, dtype=torch.float32, device=GPU_TYPE)
        y = torch.rand(N, N, dtype=torch.float32, device=GPU_TYPE)
        z = torch.rand(N, N, dtype=torch.float32, device=GPU_TYPE)

        # CASE 1: no restriction on the amount of accumulation
        with config.patch({"realize_acc_reads_size_threshold": float("inf")}):
            f_compiled = torch.compile(f)
            code = run_and_get_triton_code(f_compiled, x, y, z)
            (
                FileCheck()
                .check("triton_poi_fused_add_0.run(buf4, arg2_1, buf1, buf2, buf3")
                .run(code)
            )

        # CASE 2: for tensors with the same size as x (which is 4 * N**2 bytes)
        # at most 12 / 4 = 3 reads can be accumulated during fusion
        with config.patch({"realize_acc_reads_size_threshold": 12 * N**2}):
            f_compiled = torch.compile(f)
            code = run_and_get_triton_code(f_compiled, x, y, z)
            (
                FileCheck()
                .check("triton_poi_fused_add_0.run(buf3, arg2_1, buf1, buf2,")
                .check("triton_poi_fused_add_1.run(buf5, buf4, arg2_1,")
                .run(code)
            )

        # CASE 3: no such fusion allowed
        with config.patch({"realize_acc_reads_size_threshold": N**2}):
            f_compiled = torch.compile(f)
            code = run_and_get_triton_code(f_compiled, x, y, z)
            (
                FileCheck()
                .check("triton_poi_fused_add_0.run(buf1, arg2_1,")
                .check("triton_poi_fused_add_0.run(buf3, arg2_1,")
                .check("triton_poi_fused_add_0.run(buf4, buf3,")
                .check("triton_poi_fused_add_0.run(buf6, arg2_1,")
                .check("triton_poi_fused_add_0.run(buf7, buf6,")
                .check("triton_poi_fused_add_0.run(buf9, arg2_1,")
                .check("triton_poi_fused_add_0.run(buf10, buf9,")
                .run(code)
            )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
