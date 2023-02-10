from typing import Any, Tuple, Union, List, cast

import weakref
import warnings

import torch
import torch.distributed as dist

from torch._C import _disabled_torch_function_impl
from torch.utils._pytree import tree_map

import torch.distributed.distributed_c10d as c10d
"""
New traceable, functional collectives.
  compiler: trace these ops with plain-old-data schemas, then choose how to lower them.
  eager: execute these 'functional' ops which in eager return AsyncCollectiveTensor subclasses,
         automatically calling .wait() on underlying/hidden async 'work' obj only when fed to
         a downstream op.

Issues:
* Where should these ops live? Couldn't `import torch` if putting these ops in existing torch.distributed files
* Proper support for eager requires inplace ops. We should explore having it as an option for the API.
"""

"""
Functional collectives are asynchronous only and we perform implicit stream synchronization
on behalf of the user.

We use AsyncCollectiveTensor to wrap the result tensor of a collective and it lets us witness
first usage of the tensor and insert cross stream sync at the right place.

The above are the easy bit, the hard ones are how we match the Work object returned by
c10d and the tensor AsyncCollectiveTensor wraps. We alloc the tensor inside the collective
op implementation (see ``clone()`` call in ``_all_reduce``) and then it goes into a voyage
of wonders and magic through the dispatcher and eventually something comes out of it.

The tensor the caller of an op receives is not guaranteed to be the same allocated by an
op implenentation as mechanisms like vmap or autograd can produce different instances -
even if they ultimately point to the same underlying data.

Given those constraints we employ the following schema.

We have a dictionary of tensor::data_ptr -> Work that we insert right after we call into c10d.

We use this dictionary during the witness callback to invoke Work::wait()

Finally, we setup a finalizer against the tensor wrapper to observe it getting collected so we
can clean up stale entries in the dictionary.

To eliminate the possiblity of races we have a global version counter that is used by the finalizer.

As a wise man said once: Don't cross the streams (https://www.youtube.com/watch?v=wyKQe_i9yyo)

"""
data_ptr_to_work = dict()
work_version = 0

def _register_tensor_inner(tensor, work):
    global data_ptr_to_work
    global work_version
    data_ptr_to_work[tensor.data_ptr()] = (work_version, work)
    work_version += 1

def _clear_tensor(data_ptr, version):
    global data_ptr_to_work
    version_and_work = data_ptr_to_work.get(data_ptr)

    if version_and_work is not None and version_and_work[0] == version:
        del data_ptr_to_work[data_ptr]

def _register_tensor_wrapper_inner(tensor_wrapper, tensor):
    global data_ptr_to_work
    version, _ = data_ptr_to_work.get(tensor.data_ptr(), (None, None))
    if version is None:
        warnings.warn("Trying to register finalizers to AsyncCollectiveTensor but the inner tensor is already gone")
    else:
        weakref.finalize(tensor_wrapper, _clear_tensor, tensor.data_ptr(), version)

def _wait_tensor_inner(tensor):
    global data_ptr_to_work
    version_and_work = data_ptr_to_work.get(tensor.data_ptr())
    if version_and_work is not None:
        version_and_work[1].wait()
        del data_ptr_to_work[tensor.data_ptr()]

lib = torch.library.Library("tr_c10d", "DEF")
lib.define("wait(Tensor self) -> Tensor")

impl_lib_cpu = torch.library.Library("tr_c10d", "IMPL", "CPU")
impl_lib_cuda = torch.library.Library("tr_c10d", "IMPL", "CUDA")

def _wait_tensor(tensor: torch.Tensor):
    _wait_tensor_inner(tensor)
    return tensor

impl_lib_cpu.impl("wait", _wait_tensor)
impl_lib_cuda.impl("wait", _wait_tensor)

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

        unwrapped_args = tree_map(unwrap, args)
        unwrapped_kwargs = tree_map(unwrap, kwargs)

        out = func(*unwrapped_args, **unwrapped_kwargs)
        return out

def _str_to_reduce_op(reduceOp: str) -> dist.ReduceOp:
    reduceOp = reduceOp.upper()
    op = dist.ReduceOp.RedOpType.__members__.get(reduceOp)
    if op is None:
        raise ValueError(f"Invalid reduce operation {reduceOp}")
    return cast(dist.ReduceOp, op)

# TODO assert if ranks has duplicated entries
def _all_reduce(self, reduceOp, tag, ranks, stride):
    op = _str_to_reduce_op(reduceOp)
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, stride)
    assert group is not None

    inplace_tensor = self.clone()
    work = dist.all_reduce(inplace_tensor, op=op, group=group, async_op=True)
    _register_tensor_inner(inplace_tensor, work)

    return inplace_tensor

c10_lib_cpu = torch.library.Library("aten", "IMPL", "CPU")
c10_lib_cuda = torch.library.Library("aten", "IMPL", "CUDA")

c10_lib_cpu.impl("all_reduce", _all_reduce)
c10_lib_cuda.impl("all_reduce", _all_reduce)

RANK_TYPES = Union[List[int], List[List[int]], dist.ProcessGroup, "dist._tensor.DeviceMesh", Tuple["dist._tensor.DeviceMesh", int]]

def _expand_group(group: RANK_TYPES, tag: str = "") -> Tuple[str, List[int], int]:
    # Cannot import on the top level to avoid circular imports
    import torch.distributed._tensor as dt
    rankset: List[int]
    if isinstance(group, list):
        if isinstance(group[0], list):
            nested_list = cast(List[List[int]], group)
            rankset = []
            stride = -1
            for rs in nested_list:
                rankset.extend(rs)
                if stride != -1 and stride != len(rs):
                    raise ValueError(f"group sizes must be identical found {stride} and {len(rs)}")
                stride = len(rs)
        else:
            rankset = cast(List[int], group)
            stride = len(rankset)
    elif isinstance(group, dist.ProcessGroup):
        rankset = dist.get_process_group_ranks(group)
        stride = len(rankset)
        tag = tag or c10d._get_group_tag(group)
    elif isinstance(group, dt.DeviceMesh):
        rankset = group.mesh.flatten().tolist()
        stride = len(rankset)
        stride = group.mesh.size(0)
        rankset = group.mesh.swapdims(-1, 0).reshape(-1, stride).flatten().tolist()
        tag = tag or c10d._get_group_tag(group.get_dim_groups()[0])
    elif isinstance(group, tuple):
        if len(group) == 2 and isinstance(group[0], dt.DeviceMesh) and isinstance(group[1], int):
            dmesh = group[0]
            dim = group[1]
            stride = dmesh.mesh.size(dim)
            rankset = dmesh.mesh.swapdims(-1, dim).reshape(-1, stride).flatten().tolist()
            tag = tag or c10d._get_group_tag(dmesh.get_dim_groups()[dim])
        else:
            raise ValueError("Invalid tuple for group must be (DeviceMesh, int)")
    else:
        raise ValueError("Invalid type for group, must be one of List, Processgroup, DeviceMesh or (DeviceMesh, int).")

    return (tag, rankset, stride)

def all_reduce(self: torch.Tensor, reduceOp: str, group: RANK_TYPES, tag: str = ""):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    The input tensor is left unmodified.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    tag, rankset, stride = _expand_group(group, tag)
    tensor = torch._C._nn.all_reduce(self, reduceOp, tag, rankset, stride)  # type: ignore[attr-defined]
    res = AsyncCollectiveTensor(tensor)
    _register_tensor_wrapper_inner(res, tensor)
    return res
