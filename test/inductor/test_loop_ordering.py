# Owner(s): ["module: inductor"]

import contextlib
import os
import unittest

import numpy as np
import sympy

import torch
from torch import nn
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config as inductor_config, ir, metrics
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.test_operators import realize
from torch._inductor.utils import run_and_get_code, sympy_index_symbol
from torch._inductor.virtualized import ops, V
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map
from torch.utils._sympy.functions import FloorDiv, ModularIndexing


# set so that metrics appear
torch._logging.set_logs(inductor_metrics=True)
DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


if HAS_GPU:
    torch.set_default_device(GPU_TYPE)


class MockScheduler:
    available_buffer_names = ()

    @staticmethod
    def get_backend(cls, *args):
        return TritonScheduling(cls)

    def can_buffer_be_removed_through_fusion(self, *args, **kwargs):
        return False


class MockSchedulerTest(TestCase):
    _exit_stack = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        gm = torch.fx.symbolic_trace(lambda: 0)
        graph = GraphLowering(gm)
        graph.scheduler = MockScheduler
        cls._exit_stack = contextlib.ExitStack()
        cls._exit_stack.enter_context(V.set_graph_handler(graph))

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._exit_stack.close()


@inductor_config.patch(loop_ordering_after_fusion=True)
class ImplDetailTest(MockSchedulerTest):
    @staticmethod
    def _get_snode_body_sym_prefix(snode):
        body = snode._body
        prefix = ""

        for var in body.var_ranges:
            prefix = str(var)[0]
            break

        assert prefix
        return prefix

    @staticmethod
    def _create_computed_buffer_ax2(sizes=(32, 64), strides=None):
        """
        Create a ComputedBuffer for 'a x 2'
        """
        if strides is None:
            strides = ir.FlexibleLayout.contiguous_strides(sizes)

        box_a = ir.TensorBox.create(
            ir.Buffer(
                name="a",
                layout=ir.FixedLayout(
                    torch.device(GPU_TYPE),
                    dtype=torch.float32,
                    size=sizes,
                    stride=strides,
                ),
            )
        )
        box_a_loader = box_a.make_loader()

        def inner_fn(index):
            return box_a_loader(index) * 2

        buf = ir.Pointwise.create(
            device=box_a.get_device(),
            dtype=box_a.get_dtype(),
            inner_fn=inner_fn,
            ranges=box_a.get_size(),
        )
        buf.realize()
        computed_buf = buf.data.data
        computed_buf.decide_layout()
        return computed_buf

    def test_reorder_twice(self):
        """
        This may happen in practice if we pick a order when fusing A and B.
        Then we pick another order for AB when we fusion C into it.

        E.g. happens for BertForMaskedLM.
        """

        buf = self._create_computed_buffer_ax2()
        snode = SchedulerNode(V.graph.scheduler, buf)
        snode.apply_new_loop_order([1, 0])
        prefix1 = self._get_snode_body_sym_prefix(snode)
        self.assertTrue(prefix1 == "p")
        snode.apply_new_loop_order([1, 0])
        prefix2 = self._get_snode_body_sym_prefix(snode)
        self.assertTrue(prefix2 == "p")

    def test_reorder_and_merge_loops(self):
        sizes = (1024, 2048)
        strides = (1, 1024)
        buf = self._create_computed_buffer_ax2(sizes, strides)
        old_sizes, old_body = buf.simplify_and_reorder()

        # Make sure loop reordering happens here
        self.assertTrue(tuple(old_sizes[0]) == tuple(reversed(sizes)), f"{old_sizes=}")
        new_body = old_body.merge_loops()
        new_sizes = new_body.sizes
        self.assertTrue(tuple(new_sizes[0]) == (np.prod(sizes),), f"{new_sizes=}")

    def test_merge_loops_invalidate_pw_dep_cache(self):
        sizes = (1024, 2048)
        strides = (2048, 1)
        buf = self._create_computed_buffer_ax2(sizes, strides)

        snode = SchedulerNode(V.graph.scheduler, buf)
        old_var_ranges = snode.pointwise_read_writes().var_ranges
        self.assertTrue(len(old_var_ranges) == 2)  # 2 dimension not merged
        snode.merge_loops()
        new_var_ranges = snode.pointwise_read_writes().var_ranges

        # we cache pointwise_read_writes result on a scheduler node
        # make sure new_var_ranges is refreshed by invalidating the cache.
        self.assertTrue(len(new_var_ranges) == 1)  # 2 dimensions get merged

    def test_reorder_modular_indexing(self):
        """
        There was a bug that we wrongly map i0 to the dimension with size 49
        when reordering the loop and cause ModularIndexing get optimized away
        as an no-op.
        """

        def _create_computed_buffer():
            def inner_fn(index):
                i0, _, i2, i3 = index
                return ops.load(
                    "primal", i3 + 49 * i2 + 2401 * ModularIndexing(i0, 1, 64)
                )

            buf = ir.Pointwise.create(
                device=torch.device(GPU_TYPE),
                dtype=torch.float32,
                inner_fn=inner_fn,
                ranges=[128, 4, 49, 49],
            )
            buf.realize()
            cbuf = buf.data.data
            cbuf.decide_layout()
            return cbuf

        buf = _create_computed_buffer()
        _, body = buf.simplify_and_reorder()
        new_body = body.reorder_iter_loops([1, 2, 3, 0])

        z0, z1, z2, z3 = (sympy_index_symbol(f"p{i}") for i in range(4))
        self.assertEqual(body.var_ranges, {z0: 128, z1: 4, z2: 49, z3: 49})
        self.assertEqual(
            body.indexing_exprs["index0"],
            z3 + 49 * z2 + 2401 * ModularIndexing(z0, 1, 64),
        )
        self.assertEqual(new_body.var_ranges, {z0: 4, z1: 49, z2: 49, z3: 128})
        self.assertEqual(
            new_body.indexing_exprs["index0"],
            z2 + 49 * z1 + 2401 * ModularIndexing(z3, 1, 64),
        )


@inductor_config.patch(
    {
        "benchmark_kernel": True,
        "loop_ordering_after_fusion": True,
        "triton.unique_kernel_names": True,
    }
)
class LoopOrderingTest(TestCase):
    device = GPU_TYPE

    def do_acc_test(self, f, *args, cast_fp8=True):
        expect = f(*args)
        actual = torch.compile(f)(*args)

        if cast_fp8:

            def _cast(x):
                if isinstance(x, torch.Tensor) and x.dtype in (
                    torch.float8_e5m2,
                    torch.float8_e4m3fn,
                ):
                    return x.to(torch.float32)
                return x

            # Wordaround the issue that call allclose on fp8 tensor triggers error
            #   RuntimeError: "mul_cuda" not implemented for 'Float8_e4m3fn'
            expect = tree_map(_cast, expect)
            actual = tree_map(_cast, actual)
        self.assertTrue(same(expect, actual, tol=1e-3))

    def setUp(self):
        super().setUp()
        metrics.reset()

    def test_for_reordering_reindex(self):
        """
        ComputedBuffer.iter_reoredering_reindex can cause some fusion
        opportunitiies being skipped.

        In this test case, Inductor generates 2 triton kernels before.
        By removing ComputedBuffer.iter_reoredering_reindex, we can fuse those
        two kernels into a single one.
        """

        def f(x, y):
            """
            Add a matmul since inductor may force layout for output.
            """
            return (x.sum(dim=-1) + 1) @ y

        A, B = 20, 30
        # Make the first 2 dimension not able to merge on purpose so that
        # ComputedBuffer.iter_reoredering_reindex will be updated.
        x = rand_strided([A, A, B], [B, B * A + 300, 1], device=GPU_TYPE)
        y = torch.randn(A, A)

        self.do_acc_test(f, x, y)
        self.assertEqual(1, metrics.generated_kernel_count)
        expected_num_bytes = 0
        expected_num_bytes += A * A * B + A * A  # for the fused reduction
        expected_num_bytes += A * A * 3  # for matmul
        expected_num_bytes *= x.itemsize
        self.assertEqual(expected_num_bytes, metrics.num_bytes_accessed)

    def test_apbt_realize(self):
        M = 1024
        N = 2048

        def f(x, y):
            """
            There will be 2 kernels being generated without loop ordering after fusion:
              https://gist.github.com/shunting314/44df83f71de2c110232c50ac6638ed69
            """
            x = realize(x * 2)
            y = realize(y * 3)
            return x + y

        x = torch.randn(M, N)
        y = torch.randn(N, M).t()

        self.do_acc_test(f, x, y)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_sum_and_t(self):
        N = 1024

        def f(x):
            return x.sum(dim=-1), x.t().contiguous()

        x = torch.randn(N, N * 2)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_pw_outer_red(self):
        def f(x):
            x = realize(x + 1)
            return x.sum(dim=[0, 1])

        # make the first 2 dimension small so we don't split the reduction
        x = torch.randn(2, 4, 512)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_pw_outer_red_2(self):
        """
        The pointwise kernel is a fused kernel
        """

        def f(x):
            x = realize(x + 1)
            x = realize(x - 2)
            x = realize(x * 3)
            return x.sum(dim=[0, 1])

        # make the first 2 dimension small so we don't split the reduction
        x = torch.randn(2, 4, 512)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    @inductor_config.patch(split_reductions=False)
    def test_different_reduction_order(self):
        """
        We should not reorder loops in this case. Since reordering loops does
        not help!
        """

        def f(x):
            return x.sum(dim=0), x.sum(dim=1)

        x = torch.randn(1024, 2048)
        self.do_acc_test(f, x)
        self.assertEqual(2, metrics.generated_kernel_count)
        self.assertEqual(0, metrics.num_loop_reordering)

    def test_keep_fake_dep(self):
        """
        In this model, there are fake dependencies (StarDep) between Scatter
        and a following mutation kernel that computes the gradients of
        the embedding tables.

        When we do loop reordering for the mutation kernel, we re-analyze
        the node's dependencies. But the analysis result does not contains
        those fake dependencies. Have to add them back manually.
        """
        V = 2048
        hidden_size = 64
        max_seqlen = 512
        batch_size = 8

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.word_embeddings = nn.Embedding(V, hidden_size)
                self.position_embeddings = nn.Embedding(max_seqlen, hidden_size)
                self.layer_norm = nn.LayerNorm(hidden_size)

            def forward(self, input_ids, labels, position_ids):
                emb = self.word_embeddings(input_ids) + self.position_embeddings(
                    position_ids
                )
                return self.layer_norm(emb)

        m = Model()

        @torch.compile
        def f(*args):
            m(*args).sum().backward()

        input_ids = torch.randint(0, V, (batch_size, max_seqlen))
        labels = torch.randint(0, V, (batch_size, max_seqlen))
        position_ids = torch.arange(max_seqlen)[None, :]
        # Make sure this line does not raise exceptions. If we miss
        # fake dependencies after loop reordering, we may get exception that
        # some buffer is used before being defined.
        f(input_ids, labels, position_ids)

    def test_different_broadcast_shapes(self):
        def f(x, y, c):
            return x + c, y + c

        x = torch.randn(4, 256, 1024)
        y = torch.randn(2, 512, 1024)
        c = torch.randn(1024)
        self.do_acc_test(f, x, y, c)

        # The two kernels are not fused due to c is broadcasted
        self.assertEqual(2, metrics.generated_kernel_count)

    def test_view(self):
        """
        Passing this test relies that we compare normalized MemoryDep.
        Normlaization here means merging contiguous loops.

        To make loop reordering work, we don't merge loops when creating
        SchedulerNode. Thus we need explicitly normalize MemoryDep when
        we check if two MemeoryDep matches.
        """

        def f(x):
            y = x.sin()
            x = realize(x.view(10, 10))
            return x, y

        x = torch.randn(100)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "FP8 requires H100+ and MI300+")
    def test_fp8_cast_and_t(self):
        """
        This test repros the not able to fuses issue in
        https://github.com/pytorch/pytorch/issues/130015
        for fp8 cast and transpose
        """

        def f(x, scale):
            x = x * scale
            x = x.clamp(-1 * E4M3_MAX_POS, E4M3_MAX_POS)
            x = x.to(torch.float8_e4m3fn)
            x_t = x.t().contiguous().t()
            return x, x_t

        x = torch.randn(4096, 4096, dtype=torch.bfloat16)
        scale = torch.Tensor([10.0]).to(GPU_TYPE)
        E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max

        self.do_acc_test(f, x, scale)
        self.assertEqual(1, metrics.generated_kernel_count)

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "FP8 requires H100+ and MI300+")
    def test_fp8_pattern_2(self):
        """
        This test repros the fp8 fusion relation issue here:
            https://github.com/pytorch/pytorch/issues/133242
        """
        ref_dtype = torch.bfloat16
        M, K = 4096, 4096

        input_tensor = torch.randn(
            M, K, device=GPU_TYPE, dtype=ref_dtype, requires_grad=False
        )
        scale = torch.Tensor([10.0]).to(GPU_TYPE)

        E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max

        def test_pattern2(tensor_x_inp, scale_x):
            tensor_x = tensor_x_inp * scale_x
            tensor_x = tensor_x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
            tensor_fp8 = tensor_x.to(torch.float8_e4m3fn)

            tensor_x_t = (tensor_x_inp * scale_x).t()
            tensor_x_t = tensor_x_t.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
            tensor_fp8_t = tensor_x_t.to(torch.float8_e4m3fn)

            tensor_fp8_t = tensor_fp8_t.contiguous().t()

            return (tensor_fp8, tensor_fp8_t)

        test_pattern = torch.compile(test_pattern2)
        tensor_fp8, tensor_fp8_t = test_pattern(input_tensor, scale)

        self.assertEqual(1, metrics.generated_kernel_count)

        expected_numbytes = scale.nbytes  # scalar
        expected_numbytes += input_tensor.nbytes  # input
        expected_numbytes += tensor_fp8.nbytes + tensor_fp8_t.nbytes  # output
        self.assertEqual(expected_numbytes, metrics.num_bytes_accessed)

    # Disable split reduction to make it easier to calculate the expected
    # number of bytes accessed. In this case, split reduction does not
    # help perf much.
    @inductor_config.patch(split_reductions=False)
    def test_fuse_reduction_with_tiled_pw(self):
        def f(x):
            y = torch.sum(torch.sum(x, dim=-1))

            z = x / 10.0
            z_t = z.t().contiguous().t()
            return y, z, z_t

        # use this input sizes to test for perf
        if DO_PERF_TEST:
            M, N = 1024 * 32, 1024 * 8
        else:
            M, N = 200, 100
        x = torch.randn(M, N, device=GPU_TYPE)
        actual = f(x)
        opt_f = torch.compile(f)
        expected = opt_f(x)
        self.assertTrue(same(actual, expected, tol=1e-3))

        # We should fuse the first sum with the two pointwise.
        # Overall we read x once for all these three kernels and write
        # out 2 buffers with the same size as x.
        # This should be sort of 'optimal' for this workload.
        expected_numbytes = x.nbytes * 3

        # A small amount of extra memory access for:
        # - store output for the first reduction
        # - load input for the second redution
        # - store output for the second reduction
        expected_numbytes += (M * 2 + 1) * x.itemsize

        print(expected_numbytes)
        self.assertEqual(expected_numbytes, metrics.num_bytes_accessed)

        if DO_PERF_TEST:
            from triton.testing import do_bench

            ms = do_bench(lambda: opt_f(x))
            print(f"{ms=:.3f}")


@inductor_config.patch(
    {
        "triton.unique_kernel_names": True,
        "loop_ordering_after_fusion": True,
        "triton.max_tiles": 3,
        "triton.coalesce_tiling_analysis": True,
    }
)
@instantiate_parametrized_tests
class MemoryCoalescingTest(MockSchedulerTest):
    """Tests for memory coalescing analysis with specific tensor sizes."""

    device = GPU_TYPE
    _exit_stack = None

    def setUp(self):
        super().setUp()
        metrics.reset()

    def _create_buffer(self, name, sizes):
        """Create a buffer with specified sizes"""

        strides = ir.FlexibleLayout.contiguous_strides(sizes)

        box = ir.TensorBox.create(
            ir.Buffer(
                name=name,
                layout=ir.FixedLayout(
                    torch.device(self.device),
                    dtype=torch.float32,
                    size=sizes,
                    stride=strides,
                ),
            )
        )
        box_loader = box.make_loader()

        def inner_fn(index):
            return box_loader(index) * 2

        buf = ir.Pointwise.create(
            device=box.get_device(),
            dtype=box.get_dtype(),
            inner_fn=inner_fn,
            ranges=box.get_size(),
        )
        buf.realize()
        computed_buf = buf.data.data
        computed_buf.decide_layout()

        return computed_buf

    def _create_scheduler_node(self, buf):
        s = SchedulerNode(V.graph.scheduler, buf)
        s.min_order = 0
        s.max_order = 100
        return s

    @parametrize(
        "inps",
        (
            ((128, 384, 196), (768, 64, 196), (128, 6, 64, 196)),
            ((64,), (16, 4), (16, 4)),
            ((5, 6), (3, 10), (30,)),
            ((5, 6, 20), (3, 10, 20), (30, 20)),
        ),
    )
    def test_inferred_splits(self, inps):
        """
        Test memory coalescing analysis with the specified tensor sizes.
        Using direct SchedulerNode creation with sizes (128, 384, 196) and (768, 64, 196).
        """

        s1, s2, expected_size = inps

        # Create buffers with the specified sizes
        buf1 = self._create_buffer("buffer1", s1)
        buf2 = self._create_buffer("buffer2", s2)

        # Create scheduler nodes
        snode1 = self._create_scheduler_node(buf1)
        snode2 = self._create_scheduler_node(buf2)

        # Create a fused node
        fused_node = torch._inductor.scheduler.FusedSchedulerNode.fuse(snode1, snode2)

        from torch._inductor import tiling_utils

        fused_norm_read_writes = tiling_utils.extract_normalized_read_writes(fused_node)

        var_ranges = fused_norm_read_writes.var_ranges
        self.assertEqual(list(var_ranges.values()), list(expected_size))

    def test_remapped_reads(self):
        from torch._inductor import tiling_utils

        def fn(nodes):
            assert len(nodes) == 1
            fused_norm_read_writes = tiling_utils.extract_normalized_read_writes(
                nodes[0]
            )

            self.assertTrue(len(fused_norm_read_writes.var_ranges) == 2)

            # both reads remapped correctly
            FileCheck().check("4*n0 + n1").run(
                repr(fused_norm_read_writes.reads.keys())
            )
            FileCheck().check("n0 + 4*n1").run(
                repr(fused_norm_read_writes.reads.keys())
            )

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn):

            @torch.compile()
            def foo(x, y):
                return x + y

            foo(
                torch.rand([4, 4], device=GPU_TYPE),
                torch.rand([4, 4], device=GPU_TYPE).T,
            )

    def test_remapped_reads_split(self):
        from torch._inductor import tiling_utils

        def fn(nodes):
            self.assertTrue(len(nodes) == 1)
            fused_norm_read_writes = tiling_utils.extract_normalized_read_writes(
                nodes[0]
            )

            inp_node_reads = nodes[0].get_nodes()[1]._body.get_read_exprs()
            node_ranges = nodes[0].get_nodes()[1]._body.var_ranges
            self.assertTrue(len(node_ranges) == 1)
            self.assertTrue(next(iter(node_ranges.values())) == 36)
            var = next(iter(node_ranges.keys()))

            r = FloorDiv(var, 6) + 6 * ModularIndexing(var, 1, 6)
            self.assertTrue(r in inp_node_reads)

            # mapped reads
            self.assertTrue(list(fused_norm_read_writes.var_ranges.values()) == [6, 6])
            n0, n1 = list(fused_norm_read_writes.var_ranges.keys())

            # translation of above is n0 + 6 * n1
            self.assertTrue((n0 + 6 * n1) in fused_norm_read_writes.reads.keys())

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn):

            @torch.compile()
            def foo(x, y):
                return (
                    x + y
                ).contiguous().flatten() + torch.ops._inductor_test.realize(
                    (y.T + 1).flatten()
                )

            foo(
                torch.rand([6, 6], device=GPU_TYPE),
                torch.rand([6, 6], device=GPU_TYPE).T,
            )

    def test_reduction_pointwise(self):
        # test one pw var, one red var
        from torch._inductor import tiling_utils

        def fn(nodes):
            self.assertTrue(len(nodes) == 1)
            fused_rw = tiling_utils.extract_normalized_read_writes(nodes[0])

            i_vars, r_vars = fused_rw.index_vars, fused_rw.reduce_vars
            self.assertTrue(len(i_vars) == 1)
            self.assertTrue(len(r_vars) == 1)

            # single write to index var
            self.assertTrue(
                fused_rw.index_vars[0] == next(iter(fused_rw.writes.keys()))
            )

            # the write to the fused intermediary node should be removed
            self.assertTrue(len(fused_rw.writes) == 1)

            # single read
            self.assertTrue(len(fused_rw.reads) == 1)
            # that is applied to two bufs
            self.assertTrue(len(next(iter(fused_rw.reads.values()))) == 2)

            # and the read should be in terms of the index + reduce var,
            # even though node is pointwise
            self.assertTrue(256 * i_vars[0] + r_vars[0] in fused_rw.reads)

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn), torch.no_grad():

            @torch.compile()
            def foo(x, y):
                out = torch.ops._inductor_test.realize(x + y)
                return out.sum(dim=1)

            foo(
                torch.rand(256, 256, device=GPU_TYPE),
                torch.rand(256, 256, device=GPU_TYPE),
            )

    def test_reduction_no_pointwise(self):
        # test one pw var, one red var
        from torch._inductor import tiling_utils

        def fn(nodes):
            self.assertTrue(len(nodes) == 1)
            fused_rw = tiling_utils.extract_normalized_read_writes(nodes[0])

            i_vars, r_vars = fused_rw.index_vars, fused_rw.reduce_vars
            self.assertTrue(len(i_vars) == 0)
            self.assertTrue(len(r_vars) == 1)

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn), torch.no_grad():

            @torch.compile()
            def foo(x):
                return x.sum()

            foo(torch.rand(1024, device=GPU_TYPE))

    def test_coalescing(self):
        from torch._inductor import tiling_utils

        # Define symbolic variables
        i, j, n, m = sympy.symbols("i j n m", integer=True)

        # Test cases: (expression, var_ranges, expected_result)
        test_cases = [
            # Simple direct case
            (i + j * 5, {i: 10, j: 8}, i),
            # Floor division case
            (i + FloorDiv(j, 2), {i: 4, j: 8}, i),
            # Modular indexing
            (i * 10 + ModularIndexing(j, 1, 3), {i: 5, j: 10}, j),
            # Case with no coalescing variable
            (i * 2 + j * 3, {i: 8, j: 5}, None),
            # Division case
            (i / 2, {i: 10}, None),
            # More complex floor division
            (j + FloorDiv(i, 3), {i: 6, j: 12}, j),
            # Addition inside modular indexing
            (ModularIndexing(i + 3, 1, 6), {i: 8, j: 12}, i),
        ]

        for expr, var_ranges, expected in test_cases:
            # Test the function
            result = tiling_utils.find_coalesced_var(expr, var_ranges)
            self.assertEqual(result, expected)

    @parametrize("downcast_transposed_v", (False, True))
    def test_tiled_coalesce_analysis(self, downcast_transposed_v):
        # test one pw var, one red var
        from torch._inductor import tiling_utils

        def fn(nodes):
            self.assertTrue(len(nodes) == 1)

            coalesce_analysis = tiling_utils.analyze_memory_coalescing(nodes[0])

            i_vars = coalesce_analysis.norm_read_writes.index_vars

            # because output is contiguous, second dimension should
            # coalesce twice as many bytes as first dimension
            # if not downcasted
            # if downcasted, should be equal, bc larger dtype size
            # we also weight writes x 2
            cont_reads = coalesce_analysis.coalesced_by_var[i_vars[1]]
            t_reads = coalesce_analysis.coalesced_by_var[i_vars[0]]

            if not downcast_transposed_v:
                self.assertEqual(cont_reads, t_reads * 3)
            else:
                self.assertEqual(cont_reads, t_reads * 1.5)

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn), torch.no_grad():

            @torch.compile()
            def foo(x, y):
                return x + y.to(x.dtype)

            y_dtype = torch.float if not downcast_transposed_v else torch.float64
            foo(
                torch.rand(256, 256, device=GPU_TYPE),
                torch.rand(256, 256, device=GPU_TYPE, dtype=y_dtype).T,
            )

    def test_solve_for_zero(self):
        from torch._inductor import tiling_utils

        x, y = sympy.symbols("x y", integer=True)
        # Test cases: (expression, expected_result)
        test_cases = [
            # Simple linear expressions
            (x + 5, (-5)),
            (2 * x - 10, (5)),
            # Constant expressions (should return None)
            (sympy.Integer(7), None),
            (sympy.Integer(0), None),
            # FloorDiv cases (should return None per function)
            (FloorDiv(x, 2), None),
            (FloorDiv(x, 2) + 5, None),
            # ModularIndexing cases
            (ModularIndexing(x, 1, 5), (5)),
            (ModularIndexing(x, 1, 3), (3)),
            # Expressions with no constant solution
            (x**2 + 1, None),  # No real solution
        ]
        for expr, expected in test_cases:
            result = tiling_utils.solve_for_zero(expr)
            self.assertEqual(result, expected)

    def test_solve_for_tiling(self):
        from torch._inductor import tiling_utils

        x = sympy.Symbol("x", integer=True)

        test_cases = [
            # Simple linear cases that coalesce
            (3 * x, None),
            # # # # Expression with no free symbols
            # (sympy.Integer(5), None),
            (x / 3, 3),
            (FloorDiv(x * 2, 6), 3),
            # # ModularIndexing expressions
            (ModularIndexing(FloorDiv(x, 4), 1, 64), 4),
            (x + ModularIndexing(x, 1, 5), None),
            (x**2, None),  # Non-linear, diff is not constant
            (4096 * (ModularIndexing(32 * x, 1, 2048)) + FloorDiv(x, 64), 64),
            (4096 * (ModularIndexing(x, 1, 2048)) + FloorDiv(x, 2048), 2048),
        ]

        for expr, expected in test_cases:
            result = tiling_utils.solve_for_tiling(expr)
            self.assertEqual(result, expected)

    def test_induced_fused_tiling(self):
        from torch._inductor import tiling_utils

        def fn(nodes):
            self.assertTrue(len(nodes) == 1)

            coalesce_analysis = tiling_utils.analyze_memory_coalescing(nodes[0])
            self.assertEqual(coalesce_analysis.suggested_split.tiling_factor, 64)
            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn), torch.no_grad():

            def forward(permute):
                clone = torch.ops.aten.clone.default(
                    permute, memory_format=torch.contiguous_format
                )
                view_2 = torch.ops.aten.view.default(clone, [-1, 32])
                amax_1 = torch.ops.aten.amax.default(view_2, [1])
                return amax_1

            XDIM = 2048
            YDIM = 4096

            arg0_1 = torch.randn([XDIM, YDIM], device=GPU_TYPE, dtype=torch.bfloat16)
            permute = torch.ops.aten.permute.default(arg0_1, [1, 0])

            out, code = run_and_get_code(torch.compile(forward), (permute))

            self.assertEqual(out, forward(permute))
            FileCheck().check("YBLOCK").check("XBLOCK").run(code[0])


layouts = ("cont", "NHWC", "T")


@inductor_config.patch(
    {
        "triton.unique_kernel_names": True,
        "loop_ordering_after_fusion": True,
        "triton.coalesce_tiling_analysis": True,
    }
)
@instantiate_parametrized_tests
class TestTiling(TestCase):
    def T(self, layout: str):
        SIZE_A = 128
        SIZE_B = 256
        SIZE_C = 512

        if layout == "cont":
            return torch.rand(SIZE_A, SIZE_B, SIZE_C, device=GPU_TYPE).unsqueeze(0)
        elif layout == "T":
            return (
                torch.rand(SIZE_A, SIZE_B, SIZE_C, device=GPU_TYPE)
                .transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .unsqueeze(0)
            )
        else:
            assert layout == "NHWC"
            return torch.rand([1, SIZE_A, SIZE_B, SIZE_C], device=GPU_TYPE).to(
                memory_format=torch.channels_last
            )

    @parametrize("a", layouts)
    @parametrize("b", layouts)
    def test_pointwise(self, a, b):
        def foo(x, y):
            return x + y

        x, y = self.T(a), self.T(b)
        res, code = run_and_get_code(torch.compile(foo), x, y)

        if a != b:
            FileCheck().check("ynumel").run(code[0])
        else:
            FileCheck().check_not("ynumel").run(code[0])

        self.assertEqual(res, foo(x, y))

    def test_tiled_reduction(self):
        def f(a, b):
            return (a * b).sum(dim=-1)

        N = 512
        inps = (
            torch.randn(N, N, N, device=GPU_TYPE).permute(2, 1, 0),
            torch.randn(N, N, N, device=GPU_TYPE).permute(1, 2, 0),
        )
        f_c = torch.compile(f)
        out, code = run_and_get_code(f_c, *inps)

        FileCheck().check_dag("xnumel = 512").check_dag("ynumel = 512").check_dag(
            "rnumel"
        ).run(code[0])
        self.assertEqual(out, f(*inps), atol=0.001, rtol=0.04)

    def test_3d_pointwise(self):
        inps = (self.T("cont"), self.T("T"), self.T("NHWC"))

        def f(x, y, z):
            return x + y + z

        f_c = torch.compile(f)
        out, code = run_and_get_code(f_c, *inps)

        FileCheck().check_dag("znumel").check_dag("ynumel").check_dag("xnumel").run(
            code[0]
        )
        self.assertEqual(out, f(*inps))

    def test_cat(self):
        # test unwrapping Identity

        def f(x, y):
            return torch.cat((x, y)) + 1

        x = self.T("cont")
        y = self.T("T")

        inps = (x, y)

        f_c = torch.compile(f)
        out, code = run_and_get_code(f_c, *inps)
        FileCheck().check_dag("ynumel").check_dag("xnumel").run(code[0])
        self.assertEqual(out, f(*inps))

    def test_penalized_small_dim(self):
        x = torch.rand([2000, 1], device=GPU_TYPE)
        y = torch.rand([4, 1], device=GPU_TYPE).T

        # don't tile when it doesn't affect total coalesced mem accesses much
        def f(x, y):
            return x + y

        inps = (x, y)

        f_c = torch.compile(f)
        out, code = run_and_get_code(f_c, *inps)
        FileCheck().check_not("ynumel").check_dag("xnumel").run(code[0])
        self.assertEqual(out, f(*inps))

    def test_mutation_deps(self):
        def f(x):
            return x.add_(1)

        x = self.T("cont")

        from torch._inductor import tiling_utils

        def fn(nodes):
            self.assertTrue(len(nodes) == 1)

            coalesce_analysis = tiling_utils.analyze_memory_coalescing(nodes[0])
            assert coalesce_analysis is not None

            reads = coalesce_analysis.norm_read_writes.reads
            writes = coalesce_analysis.norm_read_writes.writes

            self.assertTrue(len(reads) == 1 and len(writes) == 1)
            self.assertEqual(
                list(coalesce_analysis.norm_read_writes.reads.values()),
                [OrderedSet(("arg0_1",))],
            )
            self.assertEqual(
                list(coalesce_analysis.norm_read_writes.writes.values()),
                [OrderedSet(("buf1",))],
            )

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn), torch.no_grad():
            torch.compile(f)(x)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
