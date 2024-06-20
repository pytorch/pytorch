# Owner(s): ["module: dynamo"]
import functools
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case

# for some reason importing functional collectives after dynamo breaks collectives handling!
import torch.distributed._functional_collectives as _functional_collectives
from torch._C import FileCheck
from torch._dynamo.testing import CompileCounter
from torch._dynamo.utils import same
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch._inductor.utils import run_and_get_triton_code
from torch.distributed.distributed_c10d import GroupMember
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    DynamoDistributedMultiProcTestCase,
    DynamoDistributedSingleProcTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
)
from torch.utils._triton import has_triton


def _tolist_with_constrain_as_size(tensor):
    lst = tensor.tolist()
    for elem in lst:
        torch._check_is_size(elem)
    return lst


@requires_nccl()
class TestCollectivesMultiProc(DynamoDistributedMultiProcTestCase):
    """
    Run correctness checks in multi-proc runner, mark with minimum # GPUs to run under
    """

    def get_world_trs(self):
        return {
            "tag": "",
            "ranks": list(range(self.world_size)),
            "group_size": self.world_size,
        }

    @property
    def world_size(self) -> int:
        # hack: no matter whether we have 2 or 3 or 4 gpus, just run on 2
        # works around issue with skipif<2 and workers with unpredictable #s gpu
        return 2

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_broadcast_inductor(self):
        """
        Testing if broadcast works correctly when using inductor
        """

        def example(tensor, src, *, tag, ranks, group_size):
            res = torch.ops.c10d_functional.broadcast(
                tensor, src, tag, ranks, group_size
            )
            res = torch.ops.c10d_functional.wait_tensor(res)
            return res

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            example = functools.partial(
                example,
                **self.get_world_trs(),
            )
            t = torch.randn(4, 4, device="cuda")
            inputs = (t if self.rank == 0 else torch.zeros(4, 4, device="cuda"), 0)
            eager_out = example(*inputs)
            self.assertTrue(same(t, eager_out))

            compiled_func = compile(example, inputs)
            compiled_out = compiled_func(*inputs)
            self.assertTrue(same(eager_out, compiled_out))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_allreduce_inductor(self):
        """
        This is matmul/cat/allreduce is a pattern we aim to optimize.
        """

        def matmul_cat_col(a, b, c, d, e, f, *, tag, ranks, group_size):
            x = torch.matmul(a, b)
            y = torch.matmul(c, d)
            z = torch.cat((x, y))
            ar = torch.ops.c10d_functional.all_reduce(z, "sum", tag, ranks, group_size)
            g = torch.matmul(e, f)
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            out = torch.add(ar, g.repeat(2, 1))
            return (out,)

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            matmul_cat_col = functools.partial(
                matmul_cat_col,
                **self.get_world_trs(),
            )
            inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 6

            eager_out = matmul_cat_col(*inputs)
            compiled_matmul_cat_col = compile(matmul_cat_col, inputs)
            inductor_out = compiled_matmul_cat_col(*inputs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_allreduce_inductor_cudagraph_trees(self):
        """
        Tests whether cudagraph trees support all_reduce from nccl
        """
        import torch.distributed as dist

        # dist.all_reduce is an inplace op in eager mode but a functionanlized op in compiled mode.
        # so we define eager_func and func separately for the same semantic.
        def eager_func(x):
            y = x * x
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
            x = torch.nn.functional.silu(x)
            return x * y

        def func(x):
            y = x * x
            y = dist.all_reduce(y, op=dist.ReduceOp.SUM)
            x = torch.nn.functional.silu(x)
            return x * y

        options = {
            "triton.cudagraphs": True,
            "triton.cudagraph_trees": True,
        }

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            compiled_func = torch.compile(
                func, backend="inductor", fullgraph=True, options=options, dynamic=None
            )

            for nelem in [1024, 2048, 4096]:
                x = torch.randn(nelem, device="cuda", dtype=torch.bfloat16)
                golden_out = eager_func(x)

                for _ in range(3):
                    compiled_out = compiled_func(x)
                    self.assertEqual(golden_out, compiled_out)

    def test_c10d_functional_tagged_pt2_compliant(self):
        op = torch.ops._c10d_functional.all_reduce.default
        self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)
        op = torch.ops.c10d_functional.all_reduce.default
        self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_eager_allreduce_inductor_wait(self):
        def eager_func(a, b, c, d, *, tag, ranks, group_size):
            x = torch.matmul(a, b)
            y = torch.matmul(c, d)
            z = torch.cat((x, y))
            ar = torch.ops.c10d_functional.all_reduce(z, "sum", tag, ranks, group_size)
            return ar

        def inductor_func(ar, e, f):
            g = torch.matmul(e, f)
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            out = torch.add(ar, g.repeat(2, 1))
            return (out,)

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            eager_func = functools.partial(
                eager_func,
                **self.get_world_trs(),
            )
            eager_inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 4
            inductor_inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 2

            eager_out = inductor_func(eager_func(*eager_inputs), *inductor_inputs)
            compiled_inductor_func = compile(
                inductor_func, [eager_func(*eager_inputs)] + list(inductor_inputs)
            )
            inductor_out = compiled_inductor_func(
                eager_func(*eager_inputs), *inductor_inputs
            )
            print(f"eager_out, {eager_out}")
            print(f"inductor_out, {inductor_out}")
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_inductor_allreduce_eager_wait(self):
        def inductor_func(a, b, c, d, *, tag, ranks, group_size):
            x = torch.matmul(a, b)
            y = torch.matmul(c, d)
            z = torch.cat((x, y))
            ar = torch.ops.c10d_functional.all_reduce(z, "sum", tag, ranks, group_size)
            return ar

        def eager_func(ar, e, f):
            g = torch.matmul(e, f)
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            out = torch.add(ar, g.repeat(2, 1))
            return (out,)

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inductor_func = functools.partial(
                inductor_func,
                **self.get_world_trs(),
            )
            inductor_inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 4
            eager_inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 2

            eager_out = eager_func(inductor_func(*inductor_inputs), *eager_inputs)
            compiled_inductor_func = compile(inductor_func, inductor_inputs)
            inductor_out = eager_func(
                compiled_inductor_func(*inductor_inputs), *eager_inputs
            )
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_allreduce_input_buffer_reuse(self):
        def func(a, *, tag, ranks, group_size):
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            c = torch.relu(a)
            d = torch.matmul(c, c)
            e = d + ar
            return (e,)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = torch.ones(4, 4, device="cuda") + self.rank
            compiled = torch.compile(func)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_permute_tensor(self):
        def func(tensor, src_dst_pairs, *, tag, ranks, group_size):
            return _functional_collectives.permute_tensor(
                tensor, src_dst_pairs, ranks, tag
            )

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = (
                # rank0: [0., 1.], rank1: [2., 3.]
                torch.arange(2, dtype=torch.float32, device="cuda") + 2 * self.rank,
                [1, 0],
            )
            compiled = torch.compile(func)
            out = compiled(*inputs, **self.get_world_trs())
            correct = func(*inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

            # rank0: [2., 3.], rank1: [0., 1.]
            expected = torch.arange(2, dtype=torch.float32, device="cuda") + 2 * (
                (self.rank - 1 + self.world_size) % self.world_size
            )
            self.assertEqual(out, expected)
            self.assertEqual(correct, expected)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_allgather_output_buffer_reuse(self):
        class Model(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.emb = torch.nn.Embedding(4, 4)

            def forward(self, x, world_size, tag, ranks, group_size):
                y = self.emb(x)
                last_dim = y.dim() - 1
                res = _functional_collectives.all_gather_tensor(y, 0, ranks, tag)
                out = torch.cat(torch.chunk(res, world_size, dim=0), dim=last_dim)
                return out

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model = Model().cuda()
            model_compiled = torch.compile(model)
            inp = torch.tensor([[2, 1, 3, 0]], dtype=torch.long, device="cuda")
            out = model_compiled(inp, self.world_size, **self.get_world_trs())
            correct = model(inp, self.world_size, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_allgather_contiguous_input(self):
        class Model(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.emb = torch.nn.Embedding(4, 4)

            def forward(self, x, world_size, tag, ranks, group_size):
                y = self.emb(x)
                last_dim = y.dim() - 1
                y = y.transpose_(0, last_dim).contiguous()
                res = _functional_collectives.all_gather_tensor(y, 0, ranks, tag)
                out = y.transpose_(0, last_dim).contiguous()
                return out

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model = Model().cuda()
            model_compiled = torch.compile(model)
            inp = torch.tensor([[2, 1, 3, 0]], dtype=torch.long, device="cuda")
            out = model_compiled(inp, self.world_size, **self.get_world_trs())
            correct = model(inp, self.world_size, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_allgather_into_tensor_inductor(self):
        """
        This is matmul/cat/allreduce is a pattern we aim to optimize.
        """

        def example(a, b, *, tag, ranks, group_size):
            c = torch.matmul(a, b)
            ag = torch.ops.c10d_functional.all_gather_into_tensor(
                c, tag, ranks, group_size
            )
            ag = torch.ops.c10d_functional.wait_tensor(ag)
            return (ag,)

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            example = functools.partial(
                example,
                **self.get_world_trs(),
            )
            inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 2

            eager_out = example(*inputs)
            compiled_matmul_cat_col = compile(example, inputs)
            inductor_out = compiled_matmul_cat_col(*inputs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_reduce_scatter_tensor_inductor(self):
        def example(a, b, *, tag, ranks, group_size):
            c = torch.matmul(a, b)
            ag = torch.ops.c10d_functional.reduce_scatter_tensor(
                c, "sum", tag, ranks, group_size
            )
            ag = torch.ops.c10d_functional.wait_tensor(ag)
            return (ag,)

        def compile(func, example_inputs):
            graph = make_fx(func)(*example_inputs)
            return inductor_compile_fx(graph, example_inputs)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            example = functools.partial(
                example,
                **self.get_world_trs(),
            )
            inputs = (torch.ones(4, 4, device="cuda") + self.rank,) * 2

            eager_out = example(*inputs)
            compiled_fn = compile(example, inputs)
            inductor_out = compiled_fn(*inputs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_all_to_all_single_inductor(self):
        def example(
            inp,
            input_split_sizes_tensor,
            output_split_sizes_tensor,
            *,
            tag,
            ranks,
            group_size,
        ):
            input_split_sizes = _tolist_with_constrain_as_size(input_split_sizes_tensor)
            output_split_sizes = _tolist_with_constrain_as_size(
                output_split_sizes_tensor
            )
            a2a = torch.ops.c10d_functional.all_to_all_single(
                inp,
                output_split_sizes,
                input_split_sizes,
                tag,
                ranks,
                group_size,
            )
            a2a = torch.ops.c10d_functional.wait_tensor(a2a)
            out = a2a / a2a.sum(dim=0)
            return out

        with _dynamo_dist_per_rank_init(
            self.rank, self.world_size
        ), torch._dynamo.config.patch(
            dynamic_shapes=True,
            capture_dynamic_output_shape_ops=True,
            capture_scalar_outputs=True,
        ):
            row = self.world_size * (self.rank + 1) * (self.world_size + 1) / 2
            input_split_sizes_tensor = torch.tensor(
                [(i + 1) * (self.rank + 1) for i in range(self.world_size)],
                dtype=torch.int64,
            )
            output_split_sizes_tensor = torch.tensor(
                [(i + 1) * (self.rank + 1) for i in range(self.world_size)],
                dtype=torch.int64,
            )
            inputs = (
                torch.ones(int(row), 5, device="cuda") * (self.rank + 1),
                input_split_sizes_tensor,
                output_split_sizes_tensor,
            )
            trs = self.get_world_trs()

            compiled_fn = torch.compile(example, fullgraph=True, dynamic=True)
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            (
                FileCheck()
                .check_regex(
                    "torch.ops._c10d_functional.all_to_all_single.default\\("
                    "arg\\d+_\\d+, "
                    "\\[u\\d+, u\\d+\\], "
                    "\\[u\\d+, u\\d+\\]"
                )
                .run(code)
            )

            eager_out = example(*inputs, **trs)
            inductor_out = compiled_fn(*inputs, **trs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_all_to_all_single_inductor_split_sizes_none(self):
        def example(inp, *, tag, ranks, group_size):
            a2a = torch.ops.c10d_functional.all_to_all_single(
                inp,
                None,
                None,
                tag,
                ranks,
                group_size,
            )
            a2a = torch.ops.c10d_functional.wait_tensor(a2a)
            out = a2a / a2a.sum(dim=0)
            return out

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = (
                torch.ones(self.world_size, self.world_size, device="cuda")
                * (self.rank + 1),
            )
            trs = self.get_world_trs()

            compiled_fn = torch.compile(example, fullgraph=True, dynamic=True)
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            (
                FileCheck()
                .check_regex(
                    "torch.ops._c10d_functional.all_to_all_single.default\\("
                    "arg\\d+_\\d+, "
                    "\\[\\(s\\d+ // \\d\\), \\(s\\d+ // \\d\\)\\], "
                    "\\[\\(s\\d+ // \\d\\), \\(s\\d+ // \\d\\)\\]"
                )
                .run(code)
            )

            eager_out = example(*inputs, **trs)
            inductor_out = compiled_fn(*inputs, **trs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))


@instantiate_parametrized_tests
@requires_nccl()
@requires_cuda
class TestCollectivesInductor(DynamoDistributedSingleProcTestCase):
    """
    Prefer single-proc test runner for basic tests as it is easier to work with.
    """

    def get_world_trs(self, world_size=1):
        return {
            "tag": "",
            "ranks": list(range(world_size)),
            "group_size": world_size,
        }

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(debug=True)
    def test_inductor_single_op(self):
        def func(inp, *, tag, ranks, group_size):
            ar = torch.ops.c10d_functional.all_reduce(
                inp, "sum", tag, ranks, group_size
            )
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            return ar

        inputs = torch.ones(4, 4, device="cuda")

        compiled = torch.compile(func)
        out = compiled(inputs, **self.get_world_trs())
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        # NOTE: Make sure we are not unneccessarily copying the outputs of
        # wait_tensors before they are returned from the graph.
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check(".run(arg0_1, buf0, 16")
            .check("buf1 = torch.ops._c10d_functional.all_reduce_.default(buf0")
            .check("buf3 = torch.ops._c10d_functional.wait_tensor.default(buf0")
            .check("return (buf0")
            .run(code)
        )
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(debug=True)
    def test_inductor_steal_buffer(self):
        """
        it's ok and optimal if inductor allreduce mutates the buffer of an intermediate
        that isn't going to be used again
        """

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            ar = torch.ops.c10d_functional.all_reduce(x, "sum", tag, ranks, group_size)
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            # ensure other is not incorrectly aliasing ar's buffer
            other = torch.ones_like(inp) + 22
            return ar, other

        inputs = torch.ones(4, 4, device="cuda")

        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check(".run(arg0_1, buf0")
            .check("buf1 = torch.ops._c10d_functional.all_reduce_.default(buf0")
            .check("buf3 = torch.ops._c10d_functional.wait_tensor.default(buf0")
            .check("buf5 = empty_strided")
            .check(".run(buf5, 16")
            .check("return (buf0, buf5")
            .run(code)
        )
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch({"debug": True, "triton.descriptive_names": False})
    def test_inductor_doesnt_mutate_shared(self):
        """
        make sure that an intermediate that's going to be reuse isn't mutated unless copied
        """

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            ar = torch.ops.c10d_functional.all_reduce(x, "sum", tag, ranks, group_size)
            y = x + 2
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            # ensure other is not incorrectly aliasing ar's buffer
            other = torch.ones_like(inp) + 22
            return ar, y, other

        inputs = torch.ones(4, 4, device="cuda")

        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        # NOTE: Make sure we are not unneccessarily copying the outputs of
        # wait_tensors before they are returned from the graph.
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check("buf5 = empty_strided")
            .check(".run(arg0_1, buf0, buf5, 16")
            .check("buf1 = torch.ops._c10d_functional.all_reduce_.default(buf0")
            .check("buf3 = torch.ops._c10d_functional.wait_tensor.default(buf0")
            .check("buf6 = empty_strided")
            .check(".run(buf6, 16")
            .check("return (buf0, buf5, buf6")
            .run(code)
        )
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    def test_dynamo_trace_allreduce(self):
        def func(inp):
            ar = _functional_collectives.all_reduce(inp, "sum", "0")
            return ar

        inputs = torch.ones(4, 4, device="cuda")
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        out = compiled(inputs)
        correct = func(inputs)
        self.assertEqual(counter.frame_count, 1)

        # should test more precisely, but the 2 is supposed to be (all_reduce, wait)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    def test_dynamo_trace_all_gather_tensor(self):
        def func(inp):
            ar = _functional_collectives.all_gather_tensor(inp, 0, "0")
            return ar

        inputs = torch.ones(4, 4, device="cuda")
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        out = compiled(inputs)
        correct = func(inputs)
        self.assertEqual(counter.frame_count, 1)

        # should test more precisely, but the 2 is supposed to be (all_gather, wait)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    def test_dynamo_trace_all_gather_tensor_pg(self):
        def func(inp, *, pg):
            ar = _functional_collectives.all_gather_tensor(inp, 0, pg)
            return ar

        inputs = torch.ones(4, 4, device=self.device)
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        out = compiled(inputs, pg=GroupMember.WORLD)
        correct = func(inputs, pg=GroupMember.WORLD)
        self.assertEqual(counter.frame_count, 1)

        # should test more precisely, but the 2 is supposed to be (all_gather, wait)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    def test_dynamo_rewrite_dist_all_gather(self):
        def func(inp, out, *, pg):
            torch.distributed.all_gather_into_tensor(
                out,
                inp,
                pg,
            )

        local_size = [4, 4]
        # single-proc test
        global_size = local_size

        inputs = torch.ones(local_size, device=self.device)
        outputs = torch.empty(global_size, device=self.device)
        correct_outputs = torch.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1

        # should test more precisely, but the 3 is supposed to be (all_gather, wait, copy_)
        assert counter.op_count == 3
        assert same(outputs, correct_outputs)

    def test_dynamo_rewrite_dist_all_gather_list(self):
        def func(inp, out, *, pg):
            torch.distributed.all_gather(
                out,
                inp,
                pg,
            )

        local_size = [4, 4]
        # single-proc test
        global_size = local_size

        inputs = torch.ones(local_size, device=self.device)
        outputs = [torch.empty(global_size, device=self.device)]
        correct_outputs = [torch.empty(global_size, device=self.device)]
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1
        assert same(outputs, correct_outputs)

    def test_dynamo_rewrite_dist_all_gather_args_match(self):
        # Duplicated most of the structure from test_dynamo_rewrite_dist_all_gather
        # except uses kwargs to ensure rewrite has matching arg names
        def func(inp, out, *, pg):
            torch.distributed.all_gather_into_tensor(
                output_tensor=out,
                input_tensor=inp,
                group=pg,
                async_op=False,
            )

        local_size = [4, 4]
        # single-proc test
        global_size = local_size

        inputs = torch.ones(local_size, device=self.device)
        outputs = torch.empty(global_size, device=self.device)
        correct_outputs = torch.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1

        # should test more precisely, but the 3 is supposed to be (all_gather, wait, copy_)
        assert counter.op_count == 3
        assert same(outputs, correct_outputs)

    def test_dynamo_rewrite_dist_reduce_scatter(self):
        def func(inp, out, *, pg):
            torch.distributed.reduce_scatter_tensor(
                out,
                inp,
                group=pg,
            )

        local_size = [4, 4]
        # single-proc test
        global_size = local_size

        inputs = torch.ones(local_size, device=self.device)
        outputs = torch.empty(global_size, device=self.device)
        correct_outputs = torch.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1

        # should test more precisely, but the 3 is supposed to be (reduce_scatter, wait, copy_)
        assert counter.op_count == 3
        assert same(outputs, correct_outputs)

    @parametrize(
        "pg_mode",
        [
            "positional",
            "positional_none",
            "kwargs",
            "kwargs_none",
            "unspecified",
        ],
    )
    def test_dynamo_rewrite_dist_allreduce(self, pg_mode):
        def func(tensor, *args, **kwargs):
            torch.distributed.all_reduce(
                tensor,
                *args,
                **kwargs,
            )

        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)

        args = []
        kwargs = {}

        if pg_mode == "positional":
            args.append(torch.distributed.ReduceOp.MAX)
            args.append(GroupMember.WORLD)
        elif pg_mode == "positional_none":
            args.append(torch.distributed.ReduceOp.MAX)
            args.append(None)
        elif pg_mode == "kwargs":
            kwargs["group"] = GroupMember.WORLD
        elif pg_mode == "kwargs_none":
            kwargs["group"] = None
        else:
            assert pg_mode == "unspecified"

        inputs_compiled = torch.ones(2, device=self.device)
        inputs_eager = torch.ones(2, device=self.device)

        compiled(inputs_compiled, *args, **kwargs)
        func(inputs_eager, *args, **kwargs)

        assert counter.frame_count == 1
        # should test more precisely, but the 3 is supposed to be (all_reduce, wait, copy_)
        assert counter.op_count == 3
        assert same(inputs_compiled, inputs_eager)

    def test_dynamo_rewrite_dist_all_to_all_single(self):
        def func(output, input, pg):
            torch.distributed.all_to_all_single(output, input, group=pg)

        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)

        input_compiled = torch.ones(2, device=self.device)
        input_eager = torch.ones(2, device=self.device)
        output_compiled = torch.empty(2, device=self.device)
        output_eager = torch.empty(2, device=self.device)

        compiled(output_compiled, input_compiled, GroupMember.WORLD)
        func(output_eager, input_eager, GroupMember.WORLD)

        assert counter.frame_count == 1
        assert same(output_compiled, output_eager)

    @parametrize(
        "reduce_op",
        [
            torch.distributed.ReduceOp.SUM,
            torch.distributed.ReduceOp.AVG,
            torch.distributed.ReduceOp.PRODUCT,
            torch.distributed.ReduceOp.MIN,
            torch.distributed.ReduceOp.MAX,
        ],
    )
    def test_dynamo_rewrite_dist_allreduce_reduce_op(self, reduce_op):
        from torch.distributed._functional_collectives import REDUCE_OP_TO_STR

        def verify_rewrite(gm, _):
            ar_nodes = []
            for node in gm.graph.nodes:
                if node.target in [
                    torch.ops.c10d_functional.all_reduce,
                    torch.ops._c10d_functional.all_reduce,
                ]:
                    ar_nodes.append(node)
            self.assertEqual(len(ar_nodes), 1)
            reduce_op_str = ar_nodes[0].args[1]
            self.assertEqual(REDUCE_OP_TO_STR[reduce_op], reduce_op_str)
            return gm

        compiled = torch.compile(
            torch.distributed.all_reduce,
            backend=verify_rewrite,
            fullgraph=True,
        )
        inputs = (
            torch.ones(2, device=self.device),
            reduce_op,
            GroupMember.WORLD,
        )
        compiled(*inputs)

    @parametrize(
        "source",
        [
            "GroupMember.WORLD",
            "group.WORLD",
            "_get_default_group",
        ],
    )
    def test_dynamo_get_world_group(self, source):
        def func(tensor):
            if source == "GroupMember.WORLD":
                group = torch.distributed.GroupMember.WORLD
            elif source == "group.WORLD":
                group = torch.distributed.group.WORLD
            else:
                assert source == "_get_default_group"
                group = torch.distributed.distributed_c10d._get_default_group()

            torch.distributed.all_reduce(
                tensor,
                group=group,
            )

        def verify(gm, _):
            ar_nodes = []
            for node in gm.graph.nodes:
                if node.target in [
                    torch.ops.c10d_functional.all_reduce,
                    torch.ops._c10d_functional.all_reduce,
                ]:
                    ar_nodes.append(node)
            self.assertEqual(len(ar_nodes), 1)
            return gm

        compiled = torch.compile(func, backend=verify, fullgraph=True)
        input = torch.ones(2, device=self.device)
        compiled(input)

    def test_dynamo_support_collective_op_with_async_op_False(self):
        def func(inp, out, *, pg):
            # user explicitly set the attribute `async_op` to False,
            # there should be no graph break
            torch.distributed.reduce_scatter_tensor(out, inp, group=pg, async_op=False)

        local_size = [4, 4]
        # single-proc test
        global_size = local_size

        inputs = torch.ones(local_size, device=self.device)
        outputs = torch.empty(global_size, device=self.device)
        correct_outputs = torch.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1
        assert counter.op_count == 3
        assert same(outputs, correct_outputs)

    def test_dynamo_graphbreaks_unsupported_async_op(self):
        def func(inp, out, *, pg):
            work = torch.distributed.reduce_scatter_tensor(
                out, inp, group=pg, async_op=True
            )
            work.wait()

        local_size = [4, 4]
        # single-proc test
        global_size = local_size

        inputs = torch.ones(local_size, device=self.device)
        outputs = torch.empty(global_size, device=self.device)
        correct_outputs = torch.empty(global_size, device=self.device)
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        compiled(inputs, outputs, pg=GroupMember.WORLD)
        func(inputs, correct_outputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 0
        assert counter.op_count == 0
        assert same(outputs, correct_outputs)

    def test_dynamo_pg_var(self):
        def func(inp, *, pg):
            x = pg.rank() + 1 % pg.size()
            return inp + x

        local_size = [4, 4]
        inputs = torch.ones(local_size, device=self.device)
        correct_outputs = torch.empty(local_size, device=self.device)
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter, fullgraph=True)
        outputs = compiled(inputs, pg=GroupMember.WORLD)
        correct_outputs = func(inputs, pg=GroupMember.WORLD)
        assert counter.frame_count == 1
        assert counter.op_count == 1
        assert same(outputs, correct_outputs)

    def test_dynamo_trace_reduce_scatter_tensor(self):
        def func(inp):
            ar = _functional_collectives.reduce_scatter_tensor(inp, "sum", 0, "0")
            return ar

        inputs = torch.ones(4, 4, device="cuda")
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        out = compiled(inputs)
        correct = func(inputs)
        self.assertEqual(counter.frame_count, 1)

        # should test more precisely, but the 2 is supposed to be (reduce_scatter, wait)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    def test_dynamo_trace_allgather_coalesced(self):
        def func(inp, *, tag, ranks, group_size):
            ar = torch.ops.c10d_functional.all_gather_into_tensor_coalesced(
                inp, tag, ranks, group_size
            )
            return ar

        inputs = [torch.ones(4, 4, device="cuda"), torch.ones(6, 6, device="cuda")]
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert counter.frame_count == 1
        assert counter.op_count == 3  # It generates 2 getattr to unpack the array
        assert same(out, correct)

    def test_backwards(self):
        """
        It's probably not that common to need backwards support for collectives.

        However, I wanted to at least see if it was possible to support it as a design goal.
        """

        def func(inp):
            ar = _functional_collectives.all_reduce(inp, "sum", "0")
            return ar

        input = torch.ones(4, 4, device="cuda", requires_grad=True)
        # TODO implement backwards
        with self.assertRaisesRegex(
            RuntimeError,
            "element 0 of tensors does not require grad and does not have a grad_fn",
        ):
            compiled = torch.compile(
                func, backend="aot_eager"
            )  # inductor bug with single-op allreduce graph
            out = compiled(input)
            out.sum().backward()

            correct_input = input.clone().detach().requires_grad_()
            correct = func(correct_input)
            correct.sum().backward()
            self.assertTrue(same(out, correct))
            self.assertTrue(same(input.grad, correct_input.grad))

    def test_meta(self):
        x = torch.rand((2, 3, 4), device="meta")
        out = torch.ops.c10d_functional.all_reduce(x, "sum", **self.get_world_trs())
        self.assertEqual(x.size(), out.size())

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch({"debug": True, "triton.descriptive_names": False})
    def test_inductor_all_gather_coalesced(self):
        """
        make sure that an intermediate that's going to be reuse isn't mutated unless copied
        """

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            tensor_list = torch.ops.c10d_functional.all_gather_into_tensor_coalesced(
                [x, inp], tag, ranks, group_size
            )
            y = x + 2
            ar0 = torch.ops.c10d_functional.wait_tensor(tensor_list[0])
            ar1 = torch.ops.c10d_functional.wait_tensor(tensor_list[1])
            # ensure other is not incorrectly aliasing ar's buffer
            other = torch.ones_like(inp) + 22
            return ar0, y, other, ar1

        inputs = torch.ones(4, 4, device="cuda")

        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        # NOTE: Make sure we are not unneccessarily copying the outputs of
        # wait_tensors before they are returned from the graph.
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check("buf6 = empty_strided")
            .check(".run(arg0_1, buf0, buf6, 16")
            .check(
                "buf1 = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default([buf0, arg0_1]"
            )
            .check("buf2 = buf1[0]")
            .check("buf3 = buf1[1]")
            .check("buf4 = torch.ops._c10d_functional.wait_tensor.default(buf2")
            .check("buf7 = buf0; del buf0  # reuse")
            .check(".run(buf7, 16")
            .check("buf8 = torch.ops._c10d_functional.wait_tensor.default(buf3")
            .check("return (buf2, buf6, buf7, buf3")
            .run(code)
        )
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert same(out, correct), f"{out} va {correct}"

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch({"debug": True, "triton.descriptive_names": False})
    def test_inductor_reduce_scatter_coalesced(self):
        """
        make sure that an intermediate that's going to be reuse isn't mutated unless copied
        """

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            tensor_list = torch.ops.c10d_functional.reduce_scatter_tensor_coalesced(
                [x, inp], "sum", tag, ranks, group_size
            )
            y = x + 2
            ar0 = torch.ops.c10d_functional.wait_tensor(tensor_list[0])
            ar1 = torch.ops.c10d_functional.wait_tensor(tensor_list[1])
            # ensure other is not incorrectly aliasing ar's buffer
            other = torch.ones_like(inp) + 22
            return ar0, y, other, ar1

        inputs = torch.ones(4, 4, device="cuda")

        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        # NOTE: The first return value should be the output of the first wait_tensor.
        # We want to make sure no unneccessary copy is made.
        (
            FileCheck()
            .check("buf0 = empty_strided")
            .check("buf6 = empty_strided")
            .check(".run(arg0_1, buf0, buf6, 16")
            .check(
                "buf1 = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced.default([buf0, arg0_1]"
            )
            .check("buf2 = buf1[0]")
            .check("buf3 = buf1[1]")
            .check("buf4 = torch.ops._c10d_functional.wait_tensor.default(buf2")
            .check("buf7 = buf0; del buf0  # reuse")
            .check(".run(buf7, 16")
            .check("buf8 = torch.ops._c10d_functional.wait_tensor.default(buf3")
            .check("return (buf2, buf6, buf7, buf3")
            .run(code)
        )
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert same(out, correct), f"{out} va {correct}"


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
