import weakref

import torch
from torch._C import DispatchKey
from torch._C import _disabled_torch_function_impl
from torch.utils._pytree import tree_map
from typing import Any, Optional

from .distributed_c10d import _find_pg_by_ranks, get_rank
from .constants import default_pg_timeout
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

"""
FIXME for this to work correctly we need to change Work to internally hold no reference to the tensor.
FIXME wait_tensor should be an op so its traceable
"""
tensor_to_work = weakref.WeakKeyDictionary
def wait_tensor(tensor: torch.Tensor):
    w = tensor_to_work.get(tensor)
    if w:
        w.wait()



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

    # disable __torch_function__ so that CommTensor can recursively dispatch
    # with ProxyTorchDispatchMode in make_fx
    __torch_function__ = _disabled_torch_function_impl

    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        t = tensor
        r = torch.Tensor._make_subclass(cls, t, require_grad=t.requires_grad)
        r._tensor = tensor  # type: ignore[attr-defined]
        return r

    def __repr__(self):
        return f"AsyncCollectiveTensor({self._tensor})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e: Any):
            if isinstance(e, AsyncCollectiveTensor):
                # print(f"unwrapping {e}")
                wait_tensor(e._tensor)
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


@torch._ops.ops.aten.all_reduce.default.py_impl(DispatchKey.CPU)
@torch._ops.ops.aten.all_reduce.default.py_impl(DispatchKey.CUDA)
def all_reduce(self, reduce_op, ranks, tag):
    print("all reducing!")
    assert reduce_op == "sum", "Unable to convert str to ReduceOp, so only default sum works"
    assert tag == "", "No support for non-empty comms tag"

    my_rank = get_rank()
    my_ranks = None
    for rs in ranks:
        if my_rank() in rs:
            my_ranks = rs

    assert my_ranks is not None, "Called all_reduce with a set of ranks that doesn't include the current node"
    group = _find_pg_by_ranks(my_ranks)

    # TODO we take this from tag
    timeout = default_pg_timeout
    inplace_tensor = self.clone()
    _, work = torch.ops.c10d.allreduce_([inplace_tensor], group, torch.classes.c10d.ReduceOp(), timeout)


    tensor_to_work[inplace_tensor] = work
    c_self = AsyncCollectiveTensor(inplace_tensor)
    return c_self
