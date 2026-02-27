# Owner(s): ["module: inductor"]
import unittest
from unittest import mock

import torch
from torch._C import FileCheck
from torch._dynamo.utils import same
from torch._inductor import config, memory
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.common_utils import serialTest, skipIfXpu
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
        M = 4096 if torch.version.hip is not None else 2048
        self.inputs = torch.ones((M, 1), device=GPU_TYPE)
        self.orig_reorder_method = memory.reorder_for_peak_memory

    @mock.patch.object(config, "reorder_for_peak_memory", True)
    def test_reorder_peak_memory(self):
        outp_corr = self.model(self.inputs)
        compiled_model = torch.compile(self.model)
        code = run_and_get_triton_code(compiled_model, self.inputs)

        call_str = (
            "def call(self, args):"
            if torch._inductor.config.graph_partition
            else "def call(args):"
        )

        (
            FileCheck()
            .check(call_str)
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

        call_str = (
            "def call(self, args):"
            if torch._inductor.config.graph_partition
            else "def call(args):"
        )

        with mock.patch.object(
            memory, "reorder_for_peak_memory", reorder_with_only_lpmf
        ):
            compiled_model = torch.compile(self.model)

            code = run_and_get_triton_code(compiled_model, self.inputs)
            (
                FileCheck()
                .check(call_str)
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

        call_str = (
            "def call(self, args):"
            if torch._inductor.config.graph_partition
            else "def call(args):"
        )

        with mock.patch.object(
            memory, "reorder_for_peak_memory", reorder_with_only_bfs
        ):
            compiled_model = torch.compile(self.model)

            code = run_and_get_triton_code(compiled_model, self.inputs)

            (
                FileCheck()
                .check(call_str)
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

        call_str = (
            "def call(self, args):"
            if torch._inductor.config.graph_partition
            else "def call(args):"
        )

        with mock.patch.object(
            memory, "reorder_for_peak_memory", reorder_with_only_dfs
        ):
            compiled_model = torch.compile(self.model)

            code = run_and_get_triton_code(compiled_model, self.inputs)
            (
                FileCheck()
                .check(call_str)
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

    @skipIfXpu(msg="Blocked by https://github.com/pytorch/pytorch/issues/170049")
    @mock.patch.object(config, "allow_buffer_reuse", False)
    @unittest.skipUnless(TRITON_AVAILABLE, "Triton is not available")
    @config.patch("test_configs.track_memory_lifecycle", "assert")
    def test_mutation_size_propagation(self):
        """
        This tests correct size propagation in the case of mutations.
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
                    buf.mpi_buffer.succ_nodes,
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

            pre_mutation = ["buf0", "buf2", "buf4", "buf6"]
            post_mutation = ["buf1", "buf3", "buf5", "buf7"]

            for pre, post in zip(pre_mutation, post_mutation):
                self.assertEqual(buffer_info[pre][0:2], (2048, 2048))
                self.assertEqual(buffer_info[post][0:2], (0, 0))
                # succ nodes should be forwarded to pre mutation buffer
                self.assertTrue(buffer_info[post][2] <= buffer_info[pre][2])

    def test_fusing_reductions_increase_peak_memory(self):
        @torch.compile
        def f(a, b, c):
            return (a @ c).sum(dim=-1) + (b @ c).sum(dim=-1)

        a = torch.randn(1024 * 32, 16, device=GPU_TYPE)
        b = torch.randn(1024 * 32, 16, device=GPU_TYPE)
        c = torch.randn(16, 1024 * 32, device=GPU_TYPE)
        torch.get_device_module(GPU_TYPE).reset_peak_memory_stats()
        f(a, b, c)
        peak_mem = torch.get_device_module(GPU_TYPE).max_memory_allocated()

        expected_bound = a.size(0) * c.size(1) * a.dtype.itemsize * 2
        self.assertLess(peak_mem, expected_bound)

    @serialTest()
    def test_fusion_acc_large_reads(self):
        def f(x, y, z):
            res = torch.zeros_like(x[0])
            for _ in range(4):
                temp = torch.matmul(x, y) + z
                res = res + temp
            return res

        N = 128
        x = torch.rand(N, N, dtype=torch.float32, device=GPU_TYPE)
        y = torch.rand(N, N, dtype=torch.float32, device=GPU_TYPE)
        z = torch.rand(N, N, dtype=torch.float32, device=GPU_TYPE)

        from torch._inductor.choices import InductorChoices
        from torch._inductor.scheduler import BaseSchedulerNode, Scheduler

        class CustomInductorChoices(InductorChoices):
            @staticmethod
            def can_fuse(
                scheduler: Scheduler,
                node1: BaseSchedulerNode,
                node2: BaseSchedulerNode,
                shared_data_score: int,
            ) -> bool:
                can_fuse_default = InductorChoices.can_fuse(
                    scheduler, node1, node2, shared_data_score
                )
                if (not can_fuse_default) or (
                    not config.realize_acc_reads_size_threshold
                ):
                    return can_fuse_default

                all_reads = (node1.read_writes.reads | node2.read_writes.reads) - (
                    node1.read_writes.writes | node2.read_writes.writes
                )
                size_of_reads = [scheduler.dep_size_hint(dep) for dep in all_reads]
                return sum(size_of_reads) < config.realize_acc_reads_size_threshold

        torch._inductor.virtualized.V.set_choices_handler(CustomInductorChoices())

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
                .check("triton_poi_fused_add_0.run(buf2, arg2_1, buf1,")
                .check("triton_poi_fused_add_1.run(buf4, buf3, arg2_1")
                .check("triton_poi_fused_add_1.run(buf6, buf5, arg2_1,")
                .run(code)
            )

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton is not available")
    def test_multiple_mutations_of_buf(self):
        @torch.compile()
        def foo(inp, inp2):
            inp = inp @ inp
            inp = inp.view(2, -1, 256)
            x = inp[0]
            y = inp[1]
            x, y = torch._foreach_add([x, y], 1.0)
            out = x.sum()
            out2 = y.sum(dim=-1)

            return out, out2, inp2 @ inp2

        inp = torch.rand([256, 256], device=GPU_TYPE)
        inp2 = torch.rand([256, 256], device=GPU_TYPE)

        def replace_foreach(gm):
            nodes = gm.find_nodes(
                op="call_function", target=torch.ops.aten._foreach_add.Scalar
            )
            if len(nodes) != 1:
                raise AssertionError
            node = nodes[0]
            nodes[0].target = torch.ops.aten._foreach_add_.Scalar
            for inp, out in zip(node.args[0], list(node.users.keys())):
                out.replace_all_uses_with(inp)
                gm.erase_node(out)

        with torch._inductor.config.patch(
            {
                "post_grad_custom_post_pass": replace_foreach,
                "test_configs.track_memory_lifecycle": "assert",
                "allow_buffer_reuse": False,
                # make sure the mm is at the end so
                # the earlier deallocation is not at the last step,
                # which doesn't distinguish between returned tensors
                # and which tensors are deallocated immediately prior
                "reorder_for_peak_memory": False,
            }
        ):
            code = run_and_get_triton_code(foo, inp, inp2)
            FileCheck().check("allocated=['buf0']").run(code)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
