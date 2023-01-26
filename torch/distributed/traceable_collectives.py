import torch
from torch._C import DispatchKey
from torch._C import _disabled_torch_function_impl
from torch.utils._pytree import tree_map
from typing import Any, Optional

"""
New traceable, functional collectives.
  compiler: trace these ops with plain-old-data schemas, then choose how to lower them.
  eager: execute these 'functional' ops which in eager return AsyncCollectiveTensor subclasses,
         automatically calling .wait() on underlying/hidden async 'work' obj only when fed to
         a downstream op.

Issues:
* Where should these ops live? Couldn't `import torch` if putting these ops in existing torch.distributed files
* How can we make these ops work in eager without manually enabling python_dispatacher mode?

"""


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
                e._work.wait()  # type: ignore[union-attr]
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


# can't use py op registration due to some error:
# !check_has_torch_dispatch(obj)
# ...
# This violates the invariant that operations in HermeticPyObject have equivalent C++ implementations
# ...
# @impl(aten_cuda_lib, 'all_reduce')

@torch._ops.ops.aten.all_reduce.default.py_impl(DispatchKey.CPU)
@torch._ops.ops.aten.all_reduce.default.py_impl(DispatchKey.CUDA)
def all_reduce(self, group_id, reduce_op):
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
