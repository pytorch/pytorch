# Owner(s): ["module: dynamo"]
import functools
import unittest
from unittest.mock import patch
import torch
from torch._C import FileCheck
# for some reason importing functional collectives after dynamo breaks collectives handling!
import torch.distributed._functional_collectives as _functional_collectives
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.utils import same
from torch._dynamo.testing import CompileCounter
from torch.distributed.distributed_c10d import GroupMember
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import (
    DynamoDistributedSingleProcTestCase,
    DynamoDistributedMultiProcTestCase,
    _dynamo_dist_per_rank_init,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch.utils._triton import has_triton
from torch._inductor.utils import run_and_get_triton_code
import torch._dynamo.logging

def _tolist_with_constrain_as_size(tensor):
    lst = tensor.tolist()
    for elem in lst:
        torch._constrain_as_size(elem)
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
            return (out, )

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
            return (out, )

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
            compiled_inductor_func = compile(inductor_func, [eager_func(*eager_inputs)] + list(inductor_inputs))
            inductor_out = compiled_inductor_func(eager_func(*eager_inputs), *inductor_inputs)
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
            return (out, )

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
            inductor_out = eager_func(compiled_inductor_func(*inductor_inputs), *eager_inputs)
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
            ag = torch.ops.c10d_functional.all_gather_into_tensor(c, tag, ranks, group_size)
            ag = torch.ops.c10d_functional.wait_tensor(ag)
            return (ag, )

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
        def example(inp, input_split_sizes_tensor, output_split_sizes_tensor, *, tag, ranks, group_size):
            input_split_sizes = _tolist_with_constrain_as_size(input_split_sizes_tensor)
            output_split_sizes = _tolist_with_constrain_as_size(output_split_sizes_tensor)
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

        with _dynamo_dist_per_rank_init(self.rank, self.world_size), torch._dynamo.config.patch(
            dynamic_shapes=True,
            capture_dynamic_output_shape_ops=True,
            capture_scalar_outputs=True,
        ):
            row = self.world_size * (self.rank + 1) * (self.world_size + 1) / 2
            input_split_sizes_tensor = torch.tensor([(i + 1) * (self.rank + 1) for i in range(self.world_size)], dtype=torch.int64)
            output_split_sizes_tensor = torch.tensor([(i + 1) * (self.rank + 1) for i in range(self.world_size)], dtype=torch.int64)
            inputs = (
                torch.ones(int(row), 5, device="cuda") * (self.rank + 1),
                input_split_sizes_tensor,
                output_split_sizes_tensor,
            )
            trs = self.get_world_trs()

            compiled_fn = torch.compile(example, fullgraph=True, dynamic=True)
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            FileCheck() \
                .check("all_to_all_single") \
                .run(code)

            eager_out = example(*inputs, **trs)
            inductor_out = compiled_fn(*inputs, **trs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_all_to_all_single_inductor_output_split_sizes_none(self):
        def example(inp, input_split_sizes_tensor, *, tag, ranks, group_size):
            input_split_sizes = _tolist_with_constrain_as_size(input_split_sizes_tensor)
            a2a = torch.ops.c10d_functional.all_to_all_single(
                inp,
                None,
                input_split_sizes,
                tag,
                ranks,
                group_size,
            )
            a2a = torch.ops.c10d_functional.wait_tensor(a2a)
            out = a2a / a2a.sum(dim=0)
            return out

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            input_split_sizes_tensor = torch.tensor([1] * self.world_size, dtype=torch.int64)
            inputs = (
                torch.ones(self.world_size, self.world_size, device="cuda") * (self.rank + 1),
                input_split_sizes_tensor,
            )
            trs = self.get_world_trs()

            compiled_fn = torch.compile(example, fullgraph=True, dynamic=True)
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            FileCheck() \
                .check("all_to_all_single") \
                .run(code)

            eager_out = example(*inputs, **trs)
            inductor_out = compiled_fn(*inputs, **trs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_all_to_all_single_inductor_input_split_sizes_none(self):
        def example(inp, output_split_sizes_tensor, *, tag, ranks, group_size):
            output_split_sizes = _tolist_with_constrain_as_size(output_split_sizes_tensor)
            a2a = torch.ops.c10d_functional.all_to_all_single(
                inp,
                output_split_sizes,
                None,
                tag,
                ranks,
                group_size,
            )
            a2a = torch.ops.c10d_functional.wait_tensor(a2a)
            out = a2a / a2a.sum(dim=0)
            return out

        with _dynamo_dist_per_rank_init(self.rank, self.world_size), torch._dynamo.config.patch(
            dynamic_shapes=True,
            capture_dynamic_output_shape_ops=True,
            capture_scalar_outputs=True,
        ):
            output_split_sizes_tensor = torch.tensor([1] * self.world_size, dtype=torch.int64)
            inputs = (
                torch.ones(self.world_size, self.world_size, device="cuda") * (self.rank + 1),
                output_split_sizes_tensor,
            )
            trs = self.get_world_trs()

            compiled_fn = torch.compile(example, fullgraph=True, dynamic=True)
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            FileCheck() \
                .check("all_to_all_single") \
                .run(code)

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
            inputs = (torch.ones(self.world_size, self.world_size, device="cuda") * (self.rank + 1),)
            trs = self.get_world_trs()

            compiled_fn = torch.compile(example, fullgraph=True, dynamic=True)
            code = run_and_get_triton_code(compiled_fn, *inputs, **trs)
            FileCheck() \
                .check("all_to_all_single") \
                .run(code)

            eager_out = example(*inputs, **trs)
            inductor_out = compiled_fn(*inputs, **trs)
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))

    def test_implicit_scheduler(self):
        pass


class AsyncTensor(torch.Tensor):
    def __new__(cls, fake_tensor):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            fake_tensor.size(),
            dtype=fake_tensor.dtype,
            device=fake_tensor.device,
            layout=fake_tensor.layout,
            requires_grad=fake_tensor.requires_grad,
        )
        r._materialized_tensor = None
        r._handle = None
        return r

    # NOTE: Any non-PyTorch reads or mutations in eager region will need to access one of these APIs: `.data_ptr` / `.storage` / `.data`.
    # We materialize the tensor before executing those calls, so that non-PyTorch reads or mutations in eager region still work normally.

    def data_ptr(self):
        AsyncTensor.wait_until_materialized([self])
        return self._materialized_tensor.data_ptr

    def storage(self):
        AsyncTensor.wait_until_materialized([self])
        return self._materialized_tensor.storage

    # TODO: implement `.data = X`
    @property
    def data(self):
        AsyncTensor.wait_until_materialized([self])
        return self._materialized_tensor.data

    def __repr__(self):
        # NOTE: `print(tensor)` goes through this
        AsyncTensor.wait_until_materialized([self])
        return self._materialized_tensor.__repr__()

    def handle(self):
        handle = self._handle()
        assert handle is not None
        return handle

    def set_handle(self, handle):
        self._handle = weakref.ref(handle)

    # NOTE: Any PyTorch reads or mutations in eager region will go through __torch_dispatch__, so we materialize the tensor here.
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # TODO: handle tuple / list / etc.
        # TODO: do the same for kwargs
        AsyncTensor.wait_until_materialized(args)
        return func(*args, **kwargs)

    @staticmethod
    def check_materialized(async_tensors):
        return all(
            (isinstance(t, AsyncTensor) and t._materialized_tensor is not None)
            or (isinstance(t, torch.Tensor))
            for t in async_tensors
        )

    @staticmethod
    def wait_until_materialized(scheduler, async_tensors):
        for async_tensor in async_tensors:
            if not AsyncTensor.check_materialized([async_tensor]):
                # NOTE: recursively schedule the deps first
                AsyncTensor.wait_until_materialized(scheduler, [async_tensor.handle().args])
                scheduler.schedule_immediately(async_tensor.handle())


class AsyncFuncHandle:
    def __init__(self, compiled_fn, fx_graph, args, outs_async):
        self.cuda_event = torch.cuda.Event()
        self.compiled_fn: Callable = compiled_fn
        self.fx_graph: torch.fx.Graph = fx_graph
        # Dependency graph is built implicitly as we run the program
        self.args = args
        self.outs_async = outs_async
        self.outs = None
        self.is_scheduled = False

    def schedule(self):
        # make sure to schedule only once
        if self.is_scheduled:
            return
        AsyncTensor.wait_until_materialized(self.args)
        self.outs = self.compiled_fn(self.args)
        self.cuda_event.record()
        self.is_scheduled = True

    def wait_for_completion(self):
        self.cuda_event.synchronize()

    def is_completed(self):
        return self.cuda_event.query()


# TODO: how to implement segment-boundary graph splitting in Dynamo?
# - we should figure out how to detect segment-boundary in Dynamo. Before that, how to know when we are entering / exiting a Python function.
#   - Yanbo told me that check_verbose() in skipfiles.py will be used on all Python functions.
#   - for module methods called by forward, File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/symbolic_convert.py", line 2270, in check_inlineable
#     `result = skipfiles.check_verbose(func, allow_torch=True)` will cover them
#   - for module forward method, this will cover it:
#     File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/eval_frame.py", line 184, in _initialize
#     `if isinstance(self._orig_mod.forward, types.MethodType) and skipfiles.check(`
#     or, File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/eval_frame.py", line 531, in catch_errors
#     `skipfiles.check(frame.f_code)`
#   - How to make the schedule exhaustive, while provide an elegant way to say "everything else"? How to do this both for forward methods and backward weight grad compute / allreduce / opt update?
- forward methods (do it via module hooks?)
- weight grad compute (within compiled autograd - maybe pass a schedule into compiled autograd API?)
- allreduce (via acc grad hook)
- opt update (via another hook)
#   - Can we start by creating a set of APIs that work for eager mode, and then trace through it?
#     - module method hook (swap out module method with the scheduler maybe_run) + some weight_grad_hook (can we have it? how?)?

# TODO: how to implement get_fx_graphs(segment)?


class AsyncScheduler:
    # NOTE: singleton of this class is a global object, it exists during the entire training process
    def __init__(self, schedule: List[Any]):
        """
        NOTE:
        - handles in work_list live across iterations, but
        """
        # Lives only within the same loop-slice (i.e. we schedule all remaining graphs at end of each iteration)
        self._fx_graph_to_handle_map = {}
        self._schedule = schedule
        self._next_segment_index = 0

    # NOTE: this function runs in compile_and_call_fx_graph(), after we obtain the compiled_fn from Inductor
    def record_compiled_fn(self, fx_graph, compiled_fn):
        cur_handle = AsyncFuncHandle(compiled_fn, fx_graph, args=None, outs_async=None)
        self._fx_graph_to_handle_map[fx_graph] = cur_handle

    # NOTE: this function substitutes `compile_fn` in Dynamo output_graph.py->compile_and_call_fx_graph(), and is called upon entry of each Inductor graph at runtime
    # `compile_fn = functools.partial(scheduler.maybe_run, fx_graph, compiled_fn)`
    def maybe_run(self, fx_graph, compiled_fn, args):
        # Create the handle and the async tensors
        with FakeTensorMode():
            outs_fake = fx_graph(args)
        # NOTE: important to make sure the same async tensor is used in downstream user code
        # as well as materialized when handle is finally run.
        outs_async = [AsyncTensor(out_fake, handle=None) for out_fake in outs_fake]
        cur_handle = self._fx_graph_to_handle_map[fx_graph]
        cur_handle.args = args
        cur_handle.outs_async = outs_async
        for out_async in outs_async:
            out_async.set_handle(cur_handle)

        # First, schedule all FX graphs from all segments that are before the incoming graph in the schedule.
        all_preceding_graph_handles = []
        reached_current_segment = False
        while self._next_segment_index < len(self._schedule):
            segment = self._schedule[self._next_segment_index]
            for g in get_fx_graphs(segment):
                if g == fx_graph:
                    reached_current_segment = True
                    break
                all_preceding_graph_handles.append(
                    self._fx_graph_to_handle_map.get(g, None)
                )
            self._next_segment_index += 1
            if reached_current_segment:
                break

        all_preceding_graph_handles_are_scheduled = True
        for handle in all_preceding_graph_handles:
            if handle is not None:
                handle.schedule()
            else:
                # Some preceding graph is not recorded yet
                all_preceding_graph_handles_are_scheduled = False
                break

        # Then, if all preceding FX graph handles are scheduled, then we schedule the incoming graph; otherwise, we don’t schedule the incoming graph.
        if all_preceding_graph_handles_are_scheduled:
            cur_handle.schedule()

        # TODO: at end of loop slice, we need to schedule all remaining unscheduled handles

        return outs_async

    def schedule_immediately(self, handle):
        handle.schedule()

    def schedule_remaining_graphs(self):
        # TODO: schedule all remaining graphs at end of each iteration
        pass



@requires_nccl()
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
    def test_inductor_single_op(self):
        torch._inductor.config.debug = True

        def func(inp, *, tag, ranks, group_size):
            ar = torch.ops.c10d_functional.all_reduce(inp, "sum", tag, ranks, group_size)
            ar = torch.ops.c10d_functional.wait_tensor(ar)
            return ar

        inputs = torch.ones(4, 4, device="cuda")

        compiled = torch.compile(func)
        out = compiled(inputs, **self.get_world_trs())
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        FileCheck() \
            .check("buf0 = empty(") \
            .check("buf0.copy_(arg0_1)") \
            .check("buf1 = buf0") \
            .check("buf1_work = dist.all_reduce(buf1") \
            .check("fun_col_impl._register_tensor_work(buf1, buf1_work)") \
            .check("_wait_tensor(buf0)") \
            .check("return (buf3, )") \
            .run(code)
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_inductor_steal_buffer(self):
        """
        it's ok and optimal if inductor allreduce mutates the buffer of an intermediate
        that isn't going to be used again
        """
        torch._inductor.config.debug = True

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
        FileCheck() \
            .check("buf1 = buf0; del buf0  # reuse") \
            .check_not("buf1.copy_(") \
            .check("buf2 = buf1") \
            .check("buf2_work = dist.all_reduce(buf2") \
            .check("fun_col_impl._register_tensor_work(buf2, buf2_work)") \
            .check("_wait_tensor(buf1)") \
            .check("buf3 = buf1") \
            .check("buf4 = empty(") \
            .check("return (buf4, buf5") \
            .run(code)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config.triton, "descriptive_names", False)
    def test_inductor_doesnt_mutate_shared(self):
        """
        make sure that an intermediate that's going to be reuse isn't mutated unless copied
        """
        torch._inductor.config.debug = True

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
        FileCheck() \
            .check("buf0 = empty(") \
            .check("buf5 = empty(") \
            .check("triton_poi__0.run(arg0_1, buf0, buf5") \
            .check_not("copy_(") \
            .check("buf1 = buf0; del buf0  # reuse") \
            .check("buf2 = buf1") \
            .check("buf2_work = dist.all_reduce(buf2") \
            .check("fun_col_impl._register_tensor_work(buf2, buf2_work)") \
            .check("_wait_tensor(buf1)") \
            .check("buf3 = buf1") \
            .check("return (buf4, buf5, buf6") \
            .run(code)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertTrue(same(out, correct))

    def test_dynamo_trace_allreduce(self):

        def func(inp, *, tag, ranks, group_size):
            ar = _functional_collectives.all_reduce(inp, "sum", ranks, tag)
            return ar

        inputs = torch.ones(4, 4, device="cuda")
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertEqual(counter.frame_count, 1)

        # should test more precisely, but the 2 is supposed to be (all_reduce, wait)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    def test_dynamo_trace_all_gather_tensor(self):

        def func(inp, *, tag, ranks, group_size):
            ar = _functional_collectives.all_gather_tensor(inp, 0, ranks, tag)
            return ar

        inputs = torch.ones(4, 4, device="cuda")
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
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

    def test_dynamo_graphbreaks_unsupported_async_op(self):

        def func(inp, out, *, pg):
            work = torch.distributed.reduce_scatter_tensor(
                out,
                inp,
                group=pg,
                async_op=True
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

        def func(inp, *, tag, ranks, group_size):
            ar = _functional_collectives.reduce_scatter_tensor(inp, "sum", 0, ranks, tag)
            return ar

        inputs = torch.ones(4, 4, device="cuda")
        counter = CompileCounter()
        compiled = torch.compile(func, backend=counter)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        self.assertEqual(counter.frame_count, 1)

        # should test more precisely, but the 2 is supposed to be (reduce_scatter, wait)
        self.assertEqual(counter.op_count, 2)
        self.assertTrue(same(out, correct))

    def test_dynamo_trace_allgather_coalesced(self):
        def func(inp, *, tag, ranks, group_size):
            ar = torch.ops.c10d_functional.all_gather_into_tensor_coalesced(inp, tag, ranks, group_size)
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
        def func(inp, *, tag, ranks, group_size):
            ar = _functional_collectives.all_reduce(inp, "sum", ranks, tag)
            return ar

        input = torch.ones(4, 4, device="cuda", requires_grad=True)
        # TODO implement backwards
        with self.assertRaisesRegex(RuntimeError, "element 0 of tensors does not require grad and does not have a grad_fn"):
            compiled = torch.compile(func, backend="aot_eager")  # inductor bug with single-op allreduce graph
            out = compiled(input, **self.get_world_trs())
            out.sum().backward()

            correct_input = input.clone().detach().requires_grad_()
            correct = func(correct_input, **self.get_world_trs())
            correct.sum().backward()
            self.assertTrue(same(out, correct))
            self.assertTrue(same(input.grad, correct_input.grad))

    def test_meta(self):
        x = torch.rand((2, 3, 4), device="meta")
        out = torch.ops.c10d_functional.all_reduce(x, "sum", **self.get_world_trs())
        self.assertEqual(x.size(), out.size())

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config.triton, "descriptive_names", False)
    def test_inductor_all_gather_coalesced(self):
        """
        make sure that an intermediate that's going to be reuse isn't mutated unless copied
        """
        torch._inductor.config.debug = True

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            tensor_list = torch.ops.c10d_functional.all_gather_into_tensor_coalesced([x, inp], tag, ranks, group_size)
            y = x + 2
            ar0 = torch.ops.c10d_functional.wait_tensor(tensor_list[0])
            ar1 = torch.ops.c10d_functional.wait_tensor(tensor_list[1])
            # ensure other is not incorrectly aliasing ar's buffer
            other = torch.ones_like(inp) + 22
            return ar0, y, other, ar1

        inputs = torch.ones(4, 4, device="cuda")

        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        FileCheck() \
            .check("buf0 = empty(") \
            .check("buf6 = empty(") \
            .check("triton_poi__0.run(arg0_1, buf0, buf6") \
            .check("buf1 = empty(") \
            .check("buf2 = empty(") \
            .check_not("copy_(") \
            .check("buf3_inputs = [buf0,arg0_1]") \
            .check("buf3 = [buf1,buf2]") \
            .check("buf3_work = fun_col_impl._all_gather_into_tensor_coalesced_fallback("
                   "output_tensors=buf3, input_tensors=buf3_inputs") \
            .check("fun_col_impl._register_tensor_work(buf3, buf3_work)") \
            .check("_wait_tensor(buf1)") \
            .check("buf4 = buf1") \
            .check("buf5 = buf0; del buf0  # reuse") \
            .check("_wait_tensor(buf2)") \
            .check("buf8 = buf2") \
            .check("return (buf5, buf6, buf7, buf9") \
            .run(code)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert same(out, correct), f"{out} va {correct}"

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config.triton, "descriptive_names", False)
    def test_inductor_reduce_scatter_coalesced(self):
        """
        make sure that an intermediate that's going to be reuse isn't mutated unless copied
        """
        torch._inductor.config.debug = True

        def func(inp, *, tag, ranks, group_size):
            x = inp + 1
            tensor_list = torch.ops.c10d_functional.reduce_scatter_tensor_coalesced([x, inp], "sum", tag, ranks, group_size)
            y = x + 2
            ar0 = torch.ops.c10d_functional.wait_tensor(tensor_list[0])
            ar1 = torch.ops.c10d_functional.wait_tensor(tensor_list[1])
            # ensure other is not incorrectly aliasing ar's buffer
            other = torch.ones_like(inp) + 22
            return ar0, y, other, ar1

        inputs = torch.ones(4, 4, device="cuda")

        compiled = torch.compile(func)
        code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
        FileCheck() \
            .check("buf0 = empty(") \
            .check("buf6 = empty(") \
            .check("triton_poi__0.run(arg0_1, buf0, buf6") \
            .check("buf1 = empty(") \
            .check("buf2 = empty(") \
            .check_not("copy_(") \
            .check("buf3 = [buf1,buf2]") \
            .check("buf3_work = fun_col_impl._reduce_scatter_tensor_coalesced_fallback("
                   "output_tensors=buf3, input_tensors=buf3_inputs") \
            .check("fun_col_impl._register_tensor_work(buf3, buf3_work)") \
            .check("_wait_tensor(buf1)") \
            .check("buf4 = buf1") \
            .check("buf5 = buf0; del buf0  # reuse") \
            .check("_wait_tensor(buf2)") \
            .check("buf8 = buf2") \
            .check("return (buf5, buf6, buf7, buf9") \
            .run(code)
        out = compiled(inputs, **self.get_world_trs())
        correct = func(inputs, **self.get_world_trs())
        assert same(out, correct), f"{out} va {correct}"


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    run_tests()
