import os
from typing import Any, Optional
import torch
from torch._C import DispatchKey
import torch.distributed as dist
from functools import partial
from torch._dynamo.utils import same
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.distributed.distributed_c10d import _get_default_group
from torch._C._distributed_c10d import _register_process_group
from torch._dispatch.python import enable_python_dispatcher

# @impl(aten_cuda_lib, 'all_reduce')
@torch._ops.ops.aten.all_reduce.default.py_impl(DispatchKey.CUDA)
def all_reduce(self, group_id, reduce_op):

    from torch.utils._pytree import tree_map
    from torch._C import _disabled_torch_function_impl

    class AsyncCollectiveTensor(torch.Tensor):
        r"""
        A Tensor subclass that is only used in eager mode, to hold a 'work' object
        and then wait on it before invoking a real op.

        Usage, from inside functional collective:
        def functional_collective(input):
            input = input.clone()
            mutated_input, work = c10d.{inplace_collective}(input)
            return AsyncCollectiveTensor(mutated_input, work)
        """
        _tensor: torch.Tensor
        _work: Optional[torch.distributed._Work]

        # disable __torch_function__ so that CommTensor can recursively dispatch
        # with ProxyTorchDispatchMode in make_fx
        __torch_function__ = _disabled_torch_function_impl

        @staticmethod
        def __new__(cls, tensor: torch.Tensor, work):
            t = tensor
            r = torch.Tensor._make_subclass(cls, t, require_grad=t.requires_grad)
            r._tensor = tensor  # type: ignore[attr-defined]
            r._work = work  # type: ignore[attr-defined]
            return r

        def __repr__(self):
            return f"AsyncCollectiveTensor({self._tensor}, work={self._work})"

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            def unwrap(e: Any):
                if isinstance(e, AsyncCollectiveTensor):
                    # print(f"unwrapping {e}")
                    e._work.wait()
                    return e._tensor
                return e

            # TODO what happens when func is an inplace? (add_) -
            # currently, seems like the mutation works on the underlying tensor,
            # but it remains an AsyncCollectiveTensor so subsequent ops still 'unwrap' again.
            # would be nice to fix this.

            unwrapped_args = tree_map(unwrap, args)
            unwrapped_kwargs = tree_map(unwrap, kwargs)

            out = func(*unwrapped_args, **unwrapped_kwargs)
            # print(f"hit torchdispatch for func {func}, returning {out}")
            return out


    group = torch.ops.c10d.lookup_pg(torch.empty(()), group_id)
    assert reduce_op == "sum", "Unable to convert str to ReduceOp, so only default sum works"

    # without using `lookup_pg` helper, I get this error trying to invoke c10d.allreduce_
    # RuntimeError: c10d::allreduce_() Expected a value of type '__torch__.torch.classes.c10d.ProcessGroup
    # (of Python compilation unit at: 0)' for argument 'process_group' but instead found type 'ProcessGroup'.
    timeout = 100
    inplace_tensor = self.clone()
    _, work = torch.ops.c10d.allreduce_([inplace_tensor], group, torch.classes.c10d.ReduceOp(), timeout)
    c_self = AsyncCollectiveTensor(inplace_tensor, work)
    return c_self

def matmul_cat_col(a, b, c, d, e, f, *, pg_id):
    x = torch.matmul(a, b)
    y = torch.matmul(c, d)
    z = torch.cat((x, y))
    ar = torch.ops.aten.all_reduce(z, group_id=pg_id, reduce_op="sum")
    g = torch.matmul(e, f)
    out = torch.add(ar, g.repeat(2, 1))
    return (out, )

def compile(func, example_inputs):
    graph = make_fx(func)(*example_inputs)
    return inductor_compile_fx(graph, example_inputs)

if __name__ == '__main__':
    os.environ["RANK"] = os.getenv("RANK", "0")
    os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12345")
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl')

    # this is a useless thing to do for the simple case of using default pg.
    # however, i did it to demonstrate the API proposed, whereby pg as int is passed
    # to collective APIs and pg object is recovered in execution layer
    pg = _get_default_group()
    pg_id = _register_process_group(pg)
    matmul_cat_col = partial(matmul_cat_col, pg_id=pg_id)

    inputs = (torch.ones(4, 4, device="cuda") + rank,) * 6

    compiled_matmul_cat_col = compile(matmul_cat_col, inputs)
    inductor_out = compiled_matmul_cat_col(*inputs)

    # non-ideally, i seem to need to enable this at user level in order to construct a torchdispatch subclass
    # inside py registered collective ops
    with enable_python_dispatcher():
        correct_out = matmul_cat_col(*inputs)
        print(f"rank {rank}: {correct_out}, {inductor_out}")
        assert same(correct_out, inductor_out)
