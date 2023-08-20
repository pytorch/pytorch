import warnings
import sys
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import Tuple, Union, List, cast, TYPE_CHECKING
from torch.utils._pytree import tree_map_only
from . import _functional_collectives_impl as fun_col_impl
from ._functional_collectives_impl import _register_tensor_wrapper
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
            "Unable to import torchdynamo util `is_torchdynamo_compiling`, so won't support torchdynamo correctly", stacklevel=TO_BE_DETERMINED
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

These apis are called by user code and expected to work both in eager execution and compilation,
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


# This is a bit unsafe: it checks if the first argument in the schema reports as a non-mutable alias.
# Today, this maps 1:1 with "aten ops that are views".
def _is_view_op(tgt):
    assert isinstance(tgt, torch._ops.OpOverload)
    schema = tgt._schema
    if len(schema.arguments) > 0:
        first_arg = schema.arguments[0]
        # check if op is a view
        return first_arg.alias_info is not None and not first_arg.alias_info.is_write

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

    def _get_acs_underlying_tensor(self):
        """This method enables  _functional_collectives_impl to test if a tensor is an ACS"""
        return self.elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        is_view_op = _is_view_op(func)

        def unwrap(e: AsyncCollectiveTensor):
            # wait_tensor is idepotent and will do stream sync only once
            if not is_view_op:
                wait_tensor(e.elem)
            return e.elem

        def wrap(e: torch.Tensor):
            # wait_tensor is idepotent and will do stream sync only once
            assert not isinstance(e, AsyncCollectiveTensor)
            res = AsyncCollectiveTensor(e)
            _register_tensor_wrapper(res)
            return res

        unwrapped_args = tree_map_only(AsyncCollectiveTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(AsyncCollectiveTensor, unwrap, kwargs)

        # we don't wrap the result as it doesn't need to be waited on.
        out = func(*unwrapped_args, **unwrapped_kwargs)

        # View ops dont require a sync, so we should re-wrap the outputs.
        if is_view_op:
            out = tree_map_only(torch.Tensor, wrap, out)

        return out


"""
Utils and infrastructure for tracing support
"""
def _expand_group(group: RANK_TYPES, tag: str = "") -> Tuple[str, List[int], int]:
    """
    _expand_group desugars the different RANK_TYPES types into a canonical format that is traceable.

    By having this be part of the explicit eager codepath, we avoid having to specialize behavior inside
    torchdynamo and can still interoperate with processgroup objects or other untraceable forms.
    """
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
    _register_tensor_wrapper(res)
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
        backend_impl = getattr(fun_col_impl, f"_{op_name}")
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
    warnings.warn("PyTorch Distributed functional collectives do not work with torch::deploy.", stacklevel=TO_BE_DETERMINED)


"""
Dynamo Remappings allow seamless translation from non-functional collectives of supportable form into
functional collective calls followed by inplace copy ops, allowing them to be traced into a functional graph.

We implement this by writing a decomposition and teaching dynamo how to associate it to a corresponding op via
the mapping dict below.

These schemas intentionally match torch.distributed.distributed_c10d.* ops that we are trying to remap from
"""
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

# This dict should contain sets of functions that dynamo is allowed to remap.
# Functions in this set should accept the same args/kwargs 1:1 as their mapping.
traceable_collective_remaps = {
    legacy_allgather: all_gather_tensor_inplace,
    legacy_reducescatter: reduce_scatter_tensor_inplace,
}
