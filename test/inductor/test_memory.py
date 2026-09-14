# Owner(s): ["module: inductor"]
import contextlib
import re
import unittest
from unittest import mock

import torch
from torch._C import FileCheck
from torch._dynamo.utils import same
from torch._higher_order_ops.effects import _EffectType
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
        self.w1 = torch.nn.Parameter(torch.ones(2, 10))
        self.w2 = torch.nn.Parameter(torch.ones(2, 2))
        self.w3 = torch.nn.Parameter(torch.ones(10, 1))
        self.w4 = torch.nn.Parameter(torch.ones(2, 10))

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
        self.inputs = torch.ones((M, 2), device=GPU_TYPE)
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
        # Keep the add as pointwise so this test continues to exercise scheduler
        # fusion choices instead of addmm fusion.
        z = torch.rand(1, N, dtype=torch.float32, device=GPU_TYPE).expand(N, N)

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

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton is not available")
    def test_torch_cond_ordering_consistency(self):
        small_sz, large_sz = 256, 1024

        class MultiCondModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("large_buffer", torch.zeros(large_sz))
                self.register_buffer("small_buffer1", torch.zeros(small_sz))
                self.register_buffer("small_buffer2", torch.zeros(small_sz))
                self.register_buffer("counter", torch.tensor(0, dtype=torch.long))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                condition = self.counter % 2 == 0

                def true_fn_large(buf):
                    return buf.clone() * 2.0

                def false_fn_large(buf):
                    return buf.clone()

                def true_fn_small(buf):
                    return buf.clone() * 2.0

                def false_fn_small(buf):
                    return buf.clone()

                result_large = torch.cond(
                    condition,
                    lambda: true_fn_large(self.large_buffer),
                    lambda: false_fn_large(self.large_buffer),
                )
                result_small1 = torch.cond(
                    condition,
                    lambda: true_fn_small(self.small_buffer1),
                    lambda: false_fn_small(self.small_buffer1),
                )
                result_small2 = torch.cond(
                    condition,
                    lambda: true_fn_small(self.small_buffer2),
                    lambda: false_fn_small(self.small_buffer2),
                )
                return (
                    x + result_large.sum() + result_small1.sum() + result_small2.sum()
                )

        def extract_cond_order(code: str) -> list[tuple[str, int]]:
            """
            Extract the order of torch.cond operations from generated code.
            Returns list of (cond_name, buffer_size) tuples in execution order.
            """
            import re

            cond_order = []
            # Look for patterns like "cond" or "cond_1" in the generated code
            # along with their buffer sizes
            lines = code.split("\n")
            for i, line in enumerate(lines):
                # Match true_graph buffer allocations which indicate cond execution
                match = re.search(r"true_graph_(\d+)_buf0\s*=.*\((\d+),", line)
                if match:
                    cond_idx = int(match.group(1))
                    buf_size = int(match.group(2))
                    cond_order.append((f"cond_{cond_idx}", buf_size))
            return cond_order

        model = MultiCondModel().to(GPU_TYPE)
        x = torch.randn(10, device=GPU_TYPE)

        # Compile with base settings (no reordering)
        torch._dynamo.reset()
        with config.patch({"reorder_for_peak_memory": False}):
            compiled_base = torch.compile(model)
            code_base = run_and_get_triton_code(compiled_base, x)

        base_order = extract_cond_order(code_base)

        # Compile with reorder_for_peak_memory=True
        torch._dynamo.reset()
        with config.patch({"reorder_for_peak_memory": True}):
            compiled_peak_mem = torch.compile(model)
            code_peak_mem = run_and_get_triton_code(compiled_peak_mem, x)

        peak_mem_order = extract_cond_order(code_peak_mem)

        if base_order and peak_mem_order:
            self.assertEqual(
                base_order,
                peak_mem_order,
                msg=(
                    lambda msg: f"{msg}\ntorch.cond operations were reordered by reorder_for_peak_memory!\n"
                    f"Base order: {base_order}\n"
                    f"Peak memory order: {peak_mem_order}\n"
                    f"This can cause NCCL hangs when torch.cond contains collective operations "
                    f"because different ranks may execute collectives in different orders."
                ),
            )


# Every knob here decides whether a free is emitted for an intermediate buffer,
# or how that free is spelled, so pin them instead of inheriting whatever the
# shard happens to run with.
@config.patch(
    {
        # cpp_wrapper spells a free "buf.reset();" rather than "del buf", and CI
        # has TORCHINDUCTOR_CPP_WRAPPER=1 shards.
        "cpp_wrapper": False,
        # A reused buffer is freed as part of its reuse line ("buf1 = buf0; del
        # buf0"); with reuse off every free is its own "del buf" line.
        "allow_buffer_reuse": False,
        # Pooled allocation frees several buffers on a single "del a, b" line.
        "memory_planning": False,
        "memory_pool": "none",
        # Reordering rewrites the very schedule whose frees are asserted below.
        "reorder_for_peak_memory": False,
    }
)
class TestEffectfulOpMemory(TestCase):
    """
    An ORDERED effectful op does not extend the lifetime of its tensor inputs, so
    those buffers must still be freed once the op has run. Adding them to
    ``never_reuse_buffers`` makes ``codegen_free`` skip the free entirely -- it
    early-returns for any buffer ``can_reuse`` rejects -- and peak memory then
    grows with the number of effectful ops in the graph.
    """

    # Enough steps that a leak is unambiguous: with the frees dropped, the graph
    # holds this many intermediates at once instead of one.
    NUM_EFFECTFUL_OPS = 8

    # 8M elements, i.e. 32 MiB at the float32 the tests use, so that one leaked
    # intermediate is far larger than any allocator noise in the peak-memory test
    # below. The bound there is computed from the dtype, not from this constant.
    INPUT_NUMEL = 1024 * 1024 * 8

    @staticmethod
    @contextlib.contextmanager
    def _observe_op():
        """An ORDERED effectful op that ignores its input."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define("mylib::observe", "(Tensor x) -> ()", lib=lib)
            lib.impl("observe", lambda x: None, "CompositeExplicitAutograd")
            torch.library._register_effectful_op(
                "mylib::observe", _EffectType.ORDERED, lib=lib
            )
            yield

    @staticmethod
    def _make_fn(num_effectful_ops):
        def fn(x):
            # Seed the chain with a reduction rather than a constant: `x + 0` is
            # folded away, which would hand the first op the graph input itself
            # instead of an intermediate.
            total = x.mean()
            for _ in range(num_effectful_ops):
                # Each step depends on a reduction over the previous one, so only
                # one of them can be live at a time. Independent steps would be
                # fused into a single multi-output kernel and be co-resident
                # regardless of how they are freed.
                step = x + total
                torch.ops.mylib.observe(step)
                total = total + step.mean()
            return total

        return fn

    def _compile_serial_chain(self, x):
        """
        Compile and run the chain on ``x``, returning ``(compiled, code,
        observed_buffers)`` where ``observed_buffers`` are the intermediate
        buffers the effectful op is called on, in program order.

        Also asserts the shape the callers' assertions depend on: one effectful op
        per step, each called on its own intermediate buffer. An op handed a graph
        input exercises nothing -- ``codegen_free`` writes an unconditional
        ``FreeLine`` for an ``InputBuffer`` and returns before it ever consults
        ``never_reuse_buffers`` -- so without this the callers could be asserting
        against a graph that no longer reproduces the regression at all. It is
        not a fusion detector: a multi-output fusion would still emit one
        distinct buffer per step and satisfy all three checks.
        """
        torch._dynamo.reset()
        compiled = torch.compile(self._make_fn(self.NUM_EFFECTFUL_OPS), fullgraph=True)
        code = run_and_get_triton_code(compiled, x)

        observed = re.findall(r"observe\.default\((\w+)\)", code)
        self.assertEqual(
            len(observed),
            self.NUM_EFFECTFUL_OPS,
            f"expected one effectful op per step, got {observed}\n\n{code}",
        )
        # Each op must be called on an intermediate buffer. A graph input would
        # not exercise the regression at all: codegen_free writes an
        # unconditional FreeLine for an InputBuffer and returns before it ever
        # consults never_reuse_buffers.
        self.assertEqual(
            [b for b in observed if not re.fullmatch(r"buf\d+", b)],
            [],
            f"effectful op called on something other than an intermediate "
            f"buffer: {observed}\n\n{code}",
        )
        self.assertEqual(
            len(set(observed)),
            self.NUM_EFFECTFUL_OPS,
            f"the steps did not stay separate buffers: {observed}\n\n{code}",
        )
        return compiled, code, observed

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    def test_effectful_op_inputs_are_freed(self):
        # Assert the free directly in the generated wrapper. That is the property
        # which regressed, and unlike a memory measurement it cannot be perturbed
        # by anything else running in the process.
        with self._observe_op():
            x = torch.ones(self.INPUT_NUMEL, device=GPU_TYPE)
            _, code, observed = self._compile_serial_chain(x)

        for buf in observed:
            # Match the trailing newline so that "del buf1" is not satisfied by a
            # "del buf12" line.
            FileCheck().check(f"observe.default({buf})").check(f"del {buf}\n").run(code)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @serialTest()
    # This test does not read the generated code for frees, so unlike the codegen
    # test it does not need them spelled any particular way -- what it needs is
    # the schedule production actually runs. So restore every class pin that sits
    # off its default and the measurement is taken at stock config; the two that
    # remain (cpp_wrapper, memory_planning) already are the defaults and are
    # pinned only so a shard cannot move them from the environment. cudagraphs is
    # pinned for the same reason but is specific to this test: cudagraph trees
    # allocate from a private pool, which perturbs the measurement rather than
    # the codegen.
    @config.patch(
        {
            "allow_buffer_reuse": True,
            "reorder_for_peak_memory": True,
            "memory_pool": "intermediates",
            "triton.cudagraphs": False,
        }
    )
    def test_effectful_op_peak_memory_does_not_scale(self):
        # The user-visible symptom of the missing frees (the regression showed up
        # as training OOMs), asserted end to end.
        with self._observe_op():
            x = torch.ones(self.INPUT_NUMEL, device=GPU_TYPE)
            compiled, _, _ = self._compile_serial_chain(x)

            device_module = torch.get_device_module(GPU_TYPE)
            device_module.synchronize()
            device_module.reset_peak_memory_stats()
            allocated_before = device_module.memory_allocated()
            compiled(x)
            device_module.synchronize()
            peak_mem = device_module.max_memory_allocated() - allocated_before

        # Only one `x + total` is live at a time and everything else in the graph
        # is a scalar, so the call holds one intermediate however many effectful
        # ops there are. Allow two for slack; a dropped free costs
        # NUM_EFFECTFUL_OPS of them.
        intermediate = self.INPUT_NUMEL * x.dtype.itemsize
        self.assertLess(
            peak_mem,
            2 * intermediate,
            f"peak memory scaled with the effectful op count: {peak_mem} bytes "
            f"for {self.NUM_EFFECTFUL_OPS} ops, expected roughly one "
            f"{intermediate}-byte intermediate",
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
