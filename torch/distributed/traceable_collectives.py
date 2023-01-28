import weakref

import torch
import torch.distributed as dist

from torch._C import DispatchKey

from torch._C import _disabled_torch_function_impl
from torch.utils._pytree import tree_map
from typing import Any, Optional

from .distributed_c10d import _find_pg_by_ranks_and_tag, get_rank
from .constants import default_pg_timeout
from torch._meta_registrations import register_meta


from typing import List
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

#FIXME we do this way cuz we can't use tensor __eq__, we want id()
#TODO use a weakref callback
class StupidWeakDict:
    def __init__(self):
        self.kvs = []

    def add(self, key, val):
        self.kvs.append((weakref.ref(key), val))

    def get_and_remove(self, key):
        new_arr = []
        val = None
        for k, v in self.kvs:
            this_k = k()
            if this_k == None:
                continue
            if id(this_k) == id(key):
                val = v
            else:
                new_arr.append((k, v))
        self.kvs = new_arr
        return val

tensor_to_work = StupidWeakDict()

lib = torch.library.Library("tr_c10d", "DEF")
lib.define("wait(Tensor self) -> Tensor")

impl_lib = torch.library.Library("tr_c10d", "IMPL",  "CPU")

def _wait_tensor(tensor: torch.Tensor):
    print("__wait")
    w = tensor_to_work.get_and_remove(tensor)
    if w:
        w.wait()
    return tensor * 99

impl_lib.impl("wait", _wait_tensor)

def wait_tensor(tensor):
    print("wait")
    return torch._ops.ops.tr_c10d.wait(tensor)

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
                return wait_tensor(e._tensor)
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


# TODO assert if ranks has duplicated entries
# TODO change signature from int[] to int[][]
def _all_reduce_cpu(self, reduceOp, tag, ranks):
    print("all reducing!")
    #TODO accept SUM - lowercase it
    assert reduceOp == "sum", "Unable to convert str to ReduceOp, so only default sum works"
    assert tag == "", "No support for non-empty comms tag"

    my_rank = get_rank()
    my_ranks = ranks


    assert my_ranks is not None, "Called all_reduce with a set of ranks that doesn't include the current node"
    assert my_rank in my_ranks, "Called all_reduce with a set of ranks that doesn't include the current node"
    
    group = _find_pg_by_ranks_and_tag(tag, my_ranks)
    assert group is not None

    inplace_tensor = self.clone()
    work = dist.all_reduce(inplace_tensor,op=dist.ReduceOp.SUM, group=group, async_op=True)

    global tensor_to_work
    tensor_to_work.add(inplace_tensor, work)
    return inplace_tensor

c10_lib = torch.library.Library("aten", "IMPL",  "CPU")
c10_lib.impl("all_reduce", _all_reduce_cpu)

# FIXME not the actual Python API, just here to help try it
def all_reduce(self, reduceOp, tag, ranks):
    tensor = torch.ops.aten.all_reduce(self, reduceOp, tag, ranks)
    return AsyncCollectiveTensor(tensor)