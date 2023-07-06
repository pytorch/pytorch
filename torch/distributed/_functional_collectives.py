import warnings
import weakref
import sys
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import Tuple, Union, List, cast, TYPE_CHECKING
from torch.utils._pytree import tree_map_only

from torch.fx.experimental.proxy_tensor import (
    get_innermost_proxy_mode,
)

if torch._running_with_deploy():
    def is_torchdynamo_compiling():
        """Can't import torchdynamo in torchdeploy builds currently."""
        return False
else:
    try:
        from torch._dynamo.external_utils import is_compiling as is_torchdynamo_compiling
    except Exception:
        warnings.warn(
            "Unable to import torchdynamo util `is_torchdynamo_compiling`, so won't support torchdynamo correctly"
        )

        def is_torchdynamo_compiling():
            return False

"""
New traceable, functional collectives.
RFC: https://github.com/pytorch/pytorch/issues/93173

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

The above are the easy bits, the hard one is how we match the Work object returned by
c10d and the tensor AsyncCollectiveTensor wraps. We alloc the tensor inside the collective
op implementation (see ``clone()`` call in ``_all_reduce``) and then it's handled by the
dispatcher which might call other implementations that are allowed to change the returned
tensor - even return a tensor with a different shape (see ``torch.vmap``).

This means the caller of our ops receives a Tensor that is not guaranteed to be the same
allocated by our implementations and that makes pairing The AsyncTensor to the original
tensor a lot harder. This pairing is needed so we can lookup the Work object to use.

Originally, we tried WeakKeyDictionary to map from Tensor to Work, but because Tensor's
identity is not stable across dispatch, the op caller would end up with a different Tensor
instance that would not match any in the dictionary.

With Tensor identity out of the question, we decided use the tensor data pointer, which
should be stable across all the Tensor changes done during dispatch.

We have a dictionary of tensor::data_ptr -> Work that we insert right after we call into c10d.

We use this dictionary when AsyncCollectiveTensor is used to invoke Work::wait()

Finally, we setup a finalizer against the tensor wrapper to observe it getting collected so we
can clean up stale entries in the dictionary.

To eliminate the possibility of races we have a global version counter that is used by the finalizer.

As a wise man said once: Don't cross the streams (https://www.youtube.com/watch?v=wyKQe_i9yyo)

"""

"""
Functional collectives can accept any of these types to describe the ranks participating in collectives.

The different types will be desugared to a canonical format
"""
RANK_TYPES = Union[List[int], List[List[int]], dist.ProcessGroup, "dist._tensor.DeviceMesh", Tuple["dist._tensor.DeviceMesh", int]]


"""
User facing APIs for functional collectives
-------------------------------------------

These apis are called by usercode and expected to work both in eager execution and compilation,
but there are significant differences to how the two modes are implemented underneath.

Eager execution is 'optimized' using a tensor subclass that schedules the synchronization (via wait_tensor() op)
just before the tensor is first used.  Compiled tracing currently relies on the compiler to perform this optimization,
and cannot yet correctly trace the AsyncTensor wrapper class.  In the future, these paths may be unified
if sufficient subclass support is added in dynamo.

Example: all_reduce is an entrypoint API, and other collectives follow a similar pattern.

Here's how it works under torch.compile/dynamo:
all_reduce(...)
  |--> _expand_group(...)               - desugars processgroup into canonical/traceable format
  |--> c10d_functional.all_reduce(...)  - dynamo captures this op call, doesn't trace deeper
  |--> _maybe_wrap_tensor(...)          - wait_tensor() op is immediately called, no AsyncTensor subclass needed

And under eager execution:
all_reduce(...)
  |--> _expand_group(...)               - same as above, but less critical for eager
  |--> c10d_functional.all_reduce(...)  - dispatches to real kernel OR records op in trace
  |--> _maybe_wrap_tensor(...)          - AsyncTensor wrapper applied to returned tensor,
                                          which issues wait_tensor() at the time of first use
"""

def wait_tensor(tensor):
    """
    Wait on a tensor returned by the collectives ops.

    Waiting follows device semantics, which means blocking on CPU and synchronizing streams on CUDA.
    """
    return torch.ops.c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]


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
    tag, rankset, group_size = _expand_group(group, tag)
    tensor = torch.ops.c10d_functional.all_reduce(self, reduceOp, tag, rankset, group_size)  # type: ignore[attr-defined]
    return _maybe_wrap_tensor(tensor)


def all_gather_tensor(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    """
    Gather tensor data across from all machines and concatenate over ``gather_dim``.

    Note that it currently only supports gather_dim = 0.

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
    assert self.is_contiguous()
    tag, rankset, group_size = _expand_group(group, tag)
    tensor = torch.ops.c10d_functional.all_gather_into_tensor(self, tag, rankset, group_size)  # type: ignore[attr-defined]
    res = _maybe_wrap_tensor(tensor)
    # TODO this should be done inside AsyncCollectiveTensor to delay the wait() call
    if gather_dim != 0:
        res = torch.cat(torch.chunk(res, group_size, dim=0), dim=gather_dim)
    return res

def reduce_scatter_tensor(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.


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
    tag, rankset, group_size = _expand_group(group, tag)
    assert (
        self.size(scatter_dim) % group_size == 0
    ), f"input dimension 0 ({self.size(0)} must be a multiple of group_size {group_size}"
    if scatter_dim != 0:
        tensor_list = torch.chunk(self, group_size, dim=scatter_dim)
        self = torch.cat(tensor_list)

    tensor = torch.ops.c10d_functional.reduce_scatter_tensor(self, reduceOp, tag, rankset, group_size)  # type: ignore[attr-defined]
    res = _maybe_wrap_tensor(tensor)
    return res


def all_reduce_coalesced(self: List[torch.Tensor], reduceOp: str, group: RANK_TYPES, tag: str = "") -> List[torch.Tensor]:
    """
    Reduces a list of tensors across all machines in such a way that all get
    the final result.

    The all tensors in the input list are left unmodified.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    tag, rankset, group_size = _expand_group(group, tag)
    tensor_list = torch.ops.c10d_functional.all_reduce_coalesced(self, reduceOp, tag, rankset, group_size)  # type: ignore[attr-defined]
    return list(map(_maybe_wrap_tensor, tensor_list))


def all_gather_into_tensor_coalesced(self: List[torch.Tensor], group: RANK_TYPES, tag: str = "") -> List[torch.Tensor]:
    """
    Gather a list of tensors across from all machines.

    Note that it currently only supports gather_dim = 0.

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
    tag, rankset, group_size = _expand_group(group, tag)
    tensor_list = torch.ops.c10d_functional.all_gather_into_tensor_coalesced(self, tag, rankset, group_size)  # type: ignore[attr-defined]
    return list(map(_maybe_wrap_tensor, tensor_list))


def reduce_scatter_tensor_coalesced(
    inputs: List[torch.Tensor],
    reduceOp: str,
    scatter_dim: List[int],
    group: RANK_TYPES,
    tag: str = "",
) -> List[torch.Tensor]:
    """
    Reduces a list of tensors across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.

    The input tensors are left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    tag, rankset, group_size = _expand_group(group, tag)
    assert len(scatter_dim) == len(inputs)
    for idx, (dim, tensor) in enumerate(zip(scatter_dim, inputs)):
        assert (
            tensor.size(dim) % group_size == 0
        ), f"input dimension {dim} ({tensor.size(dim)} must be a multiple of group_size {group_size} for tensor at index {idx}"
        if dim != 0:
            tensor_list = torch.chunk(tensor, group_size, dim=dim)
            inputs[idx] = torch.cat(tensor_list)

    tensor_list = torch.ops.c10d_functional.reduce_scatter_tensor_coalesced(inputs, reduceOp, tag, rankset, group_size)  # type: ignore[attr-defined]

    return list(map(_maybe_wrap_tensor, tensor_list))


data_ptr_to_work = dict()
work_version = 0

class _WaitRegistration:
    def __init__(self, work):
        global work_version
        self.work = work
        self.version = work_version
        self.ptrs = []
        self.cleanup_count = 0
        work_version += 1

    def _register(self, tensor):
        global data_ptr_to_work
        ptr = tensor.data_ptr()
        data_ptr_to_work[ptr] = self
        self.ptrs.append(ptr)
        self.cleanup_count += 1

    def wait(self):
        if self.work is not None:
            self.work.wait()
            self.work = None
        self.cleanup()

    def decrement_live_tensor(self, ptr):
        self.cleanup_count -= 1
        if self.cleanup_count == 0:
            self.cleanup()
        else:
            if data_ptr_to_work.get(ptr, None) == self:
                del data_ptr_to_work[ptr]

    def cleanup(self):
        for ptr in self.ptrs:
            if data_ptr_to_work.get(ptr, None) == self:
                del data_ptr_to_work[ptr]


def _register_tensor_work(tensor_or_list, work_or_list):
    if not isinstance(tensor_or_list, list):
        tensor_or_list = [tensor_or_list]
    if not isinstance(work_or_list, list):
        reg = _WaitRegistration(work_or_list)
        for tensor in tensor_or_list:
            reg._register(tensor)
    else:
        for tensor, work in zip(tensor_or_list, work_or_list):
            reg = _WaitRegistration(work)
            reg._register(tensor)

def _wait_reg_dec(ptr, wait_reg):
    wait_reg.decrement_live_tensor(ptr)

def _register_wrapper_tensor(tensor_wrapper, tensor):
    global data_ptr_to_work
    # Note: we should NEVER try to trace this, bc it registers runtime stuff during trace.
    # Instead, backends must call this themselves when implementing traced collectives.
    wait_reg = data_ptr_to_work.get(tensor.data_ptr(), None)
    if wait_reg is None:
        warnings.warn(
            "Trying to register finalizers to AsyncCollectiveTensor but the inner tensor is already gone"
        )
    else:
        # We force the collective to be waited in the case this tensor goes away to reduce the change of deadlocks.
        weakref.finalize(tensor_wrapper, _wait_reg_dec, tensor.data_ptr(), wait_reg)

def _wait_tensor(tensor: torch.Tensor) -> torch.Tensor:
    global data_ptr_to_work
    data_ptr = tensor.data_ptr()
    wait_reg = data_ptr_to_work.get(data_ptr)
    if wait_reg is not None:
        wait_reg.wait()
    return tensor

class AsyncCollectiveTensor(torch.Tensor):
    r"""
    A Tensor wrapper subclass that is used to trigger a call to wait
    prior to first use of the underlying tensor.
    Use it inside functional collective pytorch wrappers like the following:
    def functional_collective(self, group, tag):
        tag, rankset, group_size = _expand_group(group, tag)
        tensor = torch.ops.c10d_functional.{collective}(self, tag, rankset, group_size)
        return _maybe_wrap_tensor(tensor)
    """
    elem: torch.Tensor

    __slots__ = ['elem']

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, elem: torch.Tensor):

        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=False
        )
        r.elem = elem
        return r

    def __repr__(self):
        return f"AsyncCollectiveTensor({self.elem})"

    def trigger_wait(self):
        wait_tensor(self.elem)
        return self

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e: AsyncCollectiveTensor):
            # wait_tensor is idepotent and will do stream sync only once
            wait_tensor(e.elem)
            return e.elem

        unwrapped_args = tree_map_only(AsyncCollectiveTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(AsyncCollectiveTensor, unwrap, kwargs)

        # we don't wrap the result as it doesn't need to be waited on.
        out = func(*unwrapped_args, **unwrapped_kwargs)

        return out

def _str_to_reduce_op(reduceOp: str) -> dist.ReduceOp:
    reduceOp = reduceOp.upper()
    op = dist.ReduceOp.RedOpType.__members__.get(reduceOp)
    if op is None:
        raise ValueError(f"Invalid reduce operation {reduceOp}")
    return cast(dist.ReduceOp, op)

"""
Kernel implementations (for eager runtime only) - should never be traced by torch.compile

These functions should all be bound to dispatcher ops.  During tracing, the op itself should be
captured in the graph and the backend should implement the op however it prefers.
"""
# TODO assert if ranks has duplicated entries
def _all_reduce(self, reduceOp, tag, ranks, group_size):
    op = _str_to_reduce_op(reduceOp)
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None

    inplace_tensor = self.clone(memory_format=torch.contiguous_format)
    work = dist.all_reduce(inplace_tensor, op=op, group=group, async_op=True)
    _register_tensor_work(inplace_tensor, work)

    return inplace_tensor

def _all_reduce_coalesced(self, reduceOp, tag, ranks, group_size):
    op = _str_to_reduce_op(reduceOp)
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None

    inplace_tensor_list = [t.clone(memory_format=torch.contiguous_format) for t in self]
    work = dist.all_reduce_coalesced(inplace_tensor_list, op=op, group=group, async_op=True)
    _register_tensor_work(inplace_tensor_list, work)

    return inplace_tensor_list

def _all_gather_into_tensor(shard, tag, ranks, group_size):
    # TODO add dim support?
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None
    out_size = list(shard.size())
    out_size[0] *= group_size
    out_tensor = shard.new_empty(out_size)
    assert out_tensor.is_contiguous()
    # FIXME gloo doesn't support _allgather_base
    if dist.get_backend(group) == dist.Backend.GLOO or shard.is_cpu:
        tensor_list = list(torch.chunk(out_tensor, group_size))
        work = dist.all_gather(tensor_list, shard, group=group, async_op=True)
    else:
        work = dist.all_gather_into_tensor(out_tensor, shard, group=group, async_op=True)
    _register_tensor_work(out_tensor, work)

    return out_tensor


def _all_gather_into_tensor_coalesced(self, tag, rankset, group_size):
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, rankset, group_size)
    assert group is not None

    def mk_out_tensor(shard):
        out_size = list(shard.size())
        out_size[0] *= group_size
        out_tensor = shard.new_empty(out_size)
        assert out_tensor.is_contiguous()
        return out_tensor

    out_tensors = [mk_out_tensor(t) for t in self]

    work_list = _all_gather_into_tensor_coalesced_fallback(
        output_tensors=out_tensors,
        input_tensors=self,
        group=group,
        async_op=False)


    _register_tensor_work(out_tensors, work_list)
    return out_tensors


def _reduce_scatter_tensor(
    input: torch.Tensor,
    reduceOp: str,
    tag: str,
    ranks: List[int],
    group_size: int,
):
    # TODO add dim support?
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None
    op = _str_to_reduce_op(reduceOp)
    out_size = list(input.size())
    out_size[0] //= group_size
    out_tensor = input.new_empty(out_size)
    work = dist.reduce_scatter_tensor(
        out_tensor, input, op=op, group=group, async_op=True
    )
    _register_tensor_work(out_tensor, work)

    return out_tensor

def _reduce_scatter_tensor_coalesced(
    inputs: List[torch.Tensor],
    reduce_op: str,
    tag: str,
    ranks: List[int],
    group_size: int,
):
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None
    op = _str_to_reduce_op(reduce_op)


    def mk_out_tensor(shard):
        out_size = list(shard.size())
        out_size[0] //= group_size
        out_tensor = shard.new_empty(out_size)
        assert out_tensor.is_contiguous()
        return out_tensor

    out_tensors = [mk_out_tensor(t) for t in inputs]

    work_list = _reduce_scatter_tensor_coalesced_fallback(
        output_tensors=out_tensors,
        input_tensors=inputs,
        op=op,
        group=group,
        async_op=False)

    _register_tensor_work(out_tensors, work_list)
    return out_tensors


def _expand_group(group: RANK_TYPES, tag: str = "") -> Tuple[str, List[int], int]:
    # Cannot import on the top level to avoid circular imports
    import torch.distributed._tensor as dt

    # had to define this hack _inside_ expand_group to avoid
    # graph_break [('torch.* op returned non-Tensor int
    # caused by 'cast_*` functions being treated as 'torch.*' ops (iiuc)
    if TYPE_CHECKING:
        def cast_listlistint(x):
            return cast(List[List[int]], x)

        def cast_listint(x):
            return cast(List[int], x)

    else:
        # fake cast op for use at runtime since dynamo doesn't support real cast
        # also, dynamo didn't like encountering 'typing' objects ()
        # NotImplementedError: argument of type: <class 'typing._GenericAlias'>
        def cast_listlistint(x):
            return x

        def cast_listint(x):
            return x

    rankset: List[int]
    if isinstance(group, list):
        if isinstance(group[0], list):
            nested_list = cast_listlistint(group)
            rankset = []
            group_size = -1
            for rs in nested_list:
                rankset.extend(rs)
                if group_size != -1 and group_size != len(rs):
                    raise ValueError(
                        f"group sizes must be identical found {group_size} and {len(rs)}"
                    )
                group_size = len(rs)
        else:
            rankset = cast_listint(group)
            group_size = len(rankset)
    elif isinstance(group, dist.ProcessGroup):
        rankset = dist.get_process_group_ranks(group)
        group_size = len(rankset)
        tag = tag or c10d._get_group_tag(group)
    elif isinstance(group, dt.DeviceMesh):
        assert group.ndim == 1, "Only 1D mesh is supported, pass in (DeviceMesh, int) together if mesh > 1D"
        # TODO: it should run collective in the whole mesh instead of dim 0
        tag, rankset = group._dim_group_infos[0]
        group_size = len(rankset)
    elif isinstance(group, tuple):
        if len(group) == 2 and isinstance(group[0], dt.DeviceMesh) and isinstance(group[1], int):
            dmesh = group[0]
            dim = group[1]
            tag, rankset = dmesh._dim_group_infos[dim]
            group_size = len(rankset)
        else:
            raise ValueError("Invalid tuple for group must be (DeviceMesh, int)")
    else:
        raise ValueError("Invalid type for group, must be one of List, Processgroup, DeviceMesh or (DeviceMesh, int).")

    return (tag, rankset, group_size)

def _are_we_tracing() -> bool:
    if is_torchdynamo_compiling():
        return True
    mode = get_innermost_proxy_mode()
    if mode is None:
        return False
    return mode.tracer is not None

def _maybe_wrap_tensor(self) -> torch.Tensor:
    if _are_we_tracing():
        return wait_tensor(self)
    res = AsyncCollectiveTensor(self)
    _register_wrapper_tensor(res, self)
    return cast(torch.Tensor, res)

def _all_gather_into_tensor_coalesced_meta(self, tag, rankset, group_size):
    def mk_out_tensor(shard):
        out_size = list(shard.size())
        out_size[0] *= group_size
        out_tensor = shard.new_empty(out_size)
        return out_tensor

    return [mk_out_tensor(t) for t in self]

# We now register meta kernels to deal with tracing
def _all_reduce_meta(self, *args):
    return torch.empty_like(self)

def _wait_tensor_meta(self, *args):
    return torch.empty_like(self)

def _all_gather_into_tensor_meta(shard, tag, rankset, group_size):
    out_size = list(shard.size())
    out_size[0] *= group_size
    return shard.new_empty(out_size)

def _reduce_scatter_tensor_meta(input, reduce_op, tag, rankset, group_size):
    out_size = list(input.size())
    out_size[0] //= group_size
    return input.new_empty(out_size)

def _all_reduce_coalesced_meta(self, reduceOp, tag, rankset, group_size):
    return [torch.empty_like(t) for t in self]


def _reduce_scatter_tensor_coalesced_meta(inputs, reduceOp, tag, rankset, group_size):
    def mk_out_tensor(input):
        out_size = list(input.size())
        out_size[0] //= group_size
        out_tensor = input.new_empty(out_size)
        return out_tensor

    return [mk_out_tensor(t) for t in inputs]


def _register_ops():
    ops_defs = [
        "all_reduce(Tensor self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
        "all_reduce_coalesced(Tensor[] self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
        "wait_tensor(Tensor self) -> Tensor",
        "all_gather_into_tensor(Tensor shard, str tag, int[] ranks, int group_size) -> Tensor",
        "all_gather_into_tensor_coalesced(Tensor[] input, str tag, int[] ranks, int group_size) -> Tensor[]",
        "reduce_scatter_tensor(Tensor input, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
        "reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
    ]

    my_module = sys.modules[__name__]
    for op_def in ops_defs:
        op_name = op_def[0:op_def.index('(')]
        backend_impl = getattr(my_module, f"_{op_name}")
        meta_impl = getattr(my_module, f"_{op_name}_meta")
        c10_lib.define(op_def)
        c10_lib_impl.impl(op_name, backend_impl, "CompositeExplicitAutograd")
        c10_lib_impl.impl(op_name, meta_impl, "Meta")


if not torch._running_with_deploy():
    # Library MUST be defined at module scope or it doesn't work
    # Creating a "DEF" Library always crashes torch::deploy so we create our Library instances here
    #   guarded against running inside it
    c10_lib = torch.library.Library("c10d_functional", "DEF")
    c10_lib_impl = torch.library.Library("c10d_functional", "IMPL")
    _register_ops()
else:
    warnings.warn("PyTorch Distributed functional collectives do not work with torch::deploy.")

# We allow torchdynamo to convert calls from legacy inplace APIs into traceable APIs
# via a pseudo-inplace version (like a decomp) that uses the functional collective
# and a copy.
#
# These schemas intentionally match torch.distributed.distributed_c10d.* ops that we are trying to remap from
def all_gather_tensor_inplace(
    output: torch.Tensor,
    input: torch.Tensor,
    group,  # TODO add a type,
    async_op: bool = False,
    tag: str = "",
    gather_dim: int = 0
):
    assert not async_op, "Can't remap async version of inplace op to functional collective"
    return output.copy_(all_gather_tensor(input, gather_dim, group, tag))

def reduce_scatter_tensor_inplace(
    output: torch.Tensor,
    input: torch.Tensor,
    op: str = "sum",  # TODO type is actually c10d ReduceOp. is this ok?
    group=None,  # TODO add a type
    async_op: bool = False,
    scatter_dim: int = 0,
    tag: str = "",
):
    assert not async_op, "Can't remap async version of inplace op to functional collective"
    return output.copy_(reduce_scatter_tensor(input, op, scatter_dim, group, tag))

from torch.distributed.distributed_c10d import (
    all_gather_into_tensor as legacy_allgather,
    reduce_scatter_tensor as legacy_reducescatter,
)

"""
This dict should contain sets of functions that dynamo is allowed to remap.

Functions in this set should accept the same args/kwargs 1:1 as their mapping.
"""

traceable_collective_remaps = {
    legacy_allgather: all_gather_tensor_inplace,
    legacy_reducescatter: reduce_scatter_tensor_inplace,
}

def _all_gather_into_tensor_coalesced_fallback(output_tensors, input_tensors, group, async_op=False):
    # all_gather_coalesced is useless, it doesn't work under NCCL and does lots of copies under Gloo
    # all_gather is useless too because it's single tensor
    # NCCL's PG::all_gather with multiple tensors is broken, it only works for the multi-device setting
    #  and fails if you mix same-size with different-size tensor lists.
    # _coalescing_manager crashed NCCL when used with all_gather_into_tensor.
    work_list = []
    if input_tensors[0].is_cpu:
        out_tensors_sliced = [
            list(torch.chunk(out_tensor, dist.get_world_size(group)))
            for out_tensor in output_tensors
        ]
        for shard, out_tensor in zip(input_tensors, out_tensors_sliced):
            work = c10d.all_gather(out_tensor, shard, group=group, async_op=async_op)
            work_list.append(work)
    else:
        for shard, out_tensor in zip(input_tensors, output_tensors):
            work = c10d.all_gather_into_tensor(out_tensor, shard, group=group, async_op=async_op)
            work_list.append(work)
    return work_list

def _reduce_scatter_tensor_coalesced_fallback(output_tensors, input_tensors, op, group, async_op=False):
    # All the same reasons as the all_gather fallback
    work_list = []
    for shard, out_tensor in zip(input_tensors, output_tensors):
        work = c10d.reduce_scatter_tensor(out_tensor, shard, op=op, group=group, async_op=async_op)
        work_list.append(work)
    return work_list
