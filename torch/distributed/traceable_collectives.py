import weakref

import torch
import torch.distributed as dist

from torch._C import DispatchKey

from torch._C import _disabled_torch_function_impl
from torch.utils._pytree import tree_map
from typing import Any, Union

import torch.distributed.distributed_c10d as c10d
import torch.distributed._tensor as dt

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
"""

#FIXME we do this way cuz we can't use tensor __eq__, we want id()
#TODO use a weakref callback
class IdKeyWeakDict:
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

tensor_to_work = IdKeyWeakDict()

lib = torch.library.Library("tr_c10d", "DEF")
lib.define("wait(Tensor self) -> Tensor")

impl_lib = torch.library.Library("tr_c10d", "IMPL",  "CPU")

def _wait_tensor(tensor: torch.Tensor):
    w = tensor_to_work.get_and_remove(tensor)
    if w:
        w.wait()
    return tensor

impl_lib.impl("wait", _wait_tensor)

def wait_tensor(tensor):
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
                return wait_tensor(e._tensor)
            return e

        # TODO what happens when func is an inplace? (add_) -
        # currently, seems like the mutation works on the underlying tensor,
        # but it remains an AsyncCollectiveTensor so subsequent ops still 'unwrap' again.
        # would be nice to fix this.

        unwrapped_args = tree_map(unwrap, args)
        unwrapped_kwargs = tree_map(unwrap, kwargs)

        out = func(*unwrapped_args, **unwrapped_kwargs)
        return out


# TODO assert if ranks has duplicated entries
def _all_reduce_cpu(self, reduceOp, tag, ranks, stride):
    reduceOp = reduceOp.lower()
    assert reduceOp == "sum", "Unable to convert str to ReduceOp, so only default sum works"
    assert tag == "", "No support for non-empty comms tag"
    assert len(ranks) % stride == 0, f"Ranks length ({len(ranks)}) must be divisible by stride ({stride})"

    my_rank = dist.get_rank()
    rank_set = None

    for i in range(0, len(ranks), stride):
        rank_set = ranks[i : i + stride]
        if my_rank in rank_set:
            my_ranks = rank_set

    assert my_ranks is not None, "Called all_reduce with a set of ranks that doesn't include the current node"
    assert my_rank in my_ranks, "Called all_reduce with a set of ranks that doesn't include the current node"

    my_ranks.sort()

    group = c10d._find_pg_by_ranks_and_tag(tag, my_ranks)
    assert group is not None

    inplace_tensor = self.clone()
    work = dist.all_reduce(inplace_tensor, op=dist.ReduceOp.SUM, group=group, async_op=True)

    global tensor_to_work
    tensor_to_work.add(inplace_tensor, work)
    return inplace_tensor

c10_lib = torch.library.Library("aten", "IMPL",  "CPU")
c10_lib.impl("all_reduce", _all_reduce_cpu)

RANK_TYPES = Union[List[int], List[List[int]], dist.ProcessGroup, dt.DeviceMesh]

# FIXME not the actual Python API, just here to help try it
def all_reduce(self: torch.Tensor, reduceOp: str, tag: str, group: RANK_TYPES):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    The input tensor is left unmodified.

    rank can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: if you don't like all the new cool stuff or has some legacy weigthing on your shoulder.
        DeviceMesh: Do a SPMD collective over all ranks of a collective
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh
    """
    if isinstance(group, list):
        if isinstance(group[0], list):
            rankset = []
            for rs in group:
                rankset.extend(rs)
                stride = len(rs)
        else:
            rankset = group
            stride = len(rankset)
    elif isinstance(group, dist.ProcessGroup):
        rankset = dist.get_process_group_ranks(group)
        stride = len(rankset)
        tag = tag or c10d._get_group_tag(group)
    elif isinstance(group, dt.DeviceMesh):
        raise ValueError("TODO DeviceMesh")
    else:
        raise ValueError("Invalid type for group")
    tensor = torch.ops.aten.all_reduce(self, reduceOp, tag, rankset, stride)
    return AsyncCollectiveTensor(tensor)