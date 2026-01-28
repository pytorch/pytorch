# mypy: allow-untyped-defs
import contextlib
import sys
import warnings
from typing import Any, cast, TYPE_CHECKING, Union

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch._utils import _maybe_view_chunk_cat
from torch.distributed.device_mesh import DeviceMesh
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from . import _functional_collectives_impl as fun_col_impl


try:
    from torch.utils._cxx_pytree import tree_map_only
except ImportError:
    from torch.utils._pytree import tree_map_only  # type: ignore[no-redef]


try:
    from torch.compiler import is_dynamo_compiling as is_torchdynamo_compiling
except Exception:
    warnings.warn(
        "Unable to import torchdynamo util `is_torchdynamo_compiling`, so won't support torchdynamo correctly",
        stacklevel=2,
    )

    def is_torchdynamo_compiling():  # type: ignore[misc]
        return False
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
RANK_TYPES = Union[
    list[int],
    list[list[int]],
    dist.ProcessGroup,
    DeviceMesh,
    tuple["dist.tensor.DeviceMesh", int],
    c10d.GroupName,
]


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
    return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]


def broadcast(self: torch.Tensor, src: int, group: RANK_TYPES, tag: str = ""):
    """
    Broadcasts the tensor to all processes in the given process group.

    Args:
        src (int): Source rank
        group (ProcessGroup or List[int]): The process group to work on.
        tag (str, optional): A unique identifier for the collective. Default: empty string
    """
    group_name = _resolve_group_name(group, tag)
    tensor = torch.ops._c10d_functional.broadcast(self, src, group_name)
    return _maybe_wrap_tensor(tensor)


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
    group_name = _resolve_group_name(group, tag)
    tensor = torch.ops._c10d_functional.all_reduce(self, reduceOp.lower(), group_name)
    return _maybe_wrap_tensor(tensor)


def all_gather_tensor(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
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
    if not self.is_contiguous():
        raise AssertionError("Tensor must be contiguous for all_gather_tensor")
    group_name = _resolve_group_name(group, tag)
    group_size = c10d._get_group_size_by_name(group_name)
    tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        self, group_size, group_name
    )
    res = _maybe_wrap_tensor(tensor)
    # TODO this should be done inside AsyncCollectiveTensor to delay the wait() call
    if gather_dim != 0:
        # torch.cat access the data so we already need to wait here, first do wait
        # and then chunk + cat avoid us going through ACT dispatching logic again
        if isinstance(res, AsyncCollectiveTensor):
            res = res.wait()  # type: ignore[attr-defined]

        res = _maybe_view_chunk_cat(res, group_size, gather_dim)
    return res


def all_gather_tensor_autograd(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    """
    Gather tensor data across from all machines and concatenate over ``gather_dim``.

    Note that it currently only supports gather_dim = 0.

    This function is the same as all_gather_tensor but will propagate the
    backwards gradient across workers.

    See all_gather_tensor for more details on usage.
    """
    group_name = _resolve_group_name(group, tag)
    group_size = c10d._get_group_size_by_name(group_name)

    tensor = torch.ops._c10d_functional_autograd.all_gather_into_tensor(
        self, group_size, group_name
    )
    res = _FromTorchTensor.apply(tensor)
    # TODO this should be done inside AsyncCollectiveTensor to delay the wait() call
    if gather_dim != 0:
        # torch.cat access the data so we already need to wait here, first do wait
        # and then chunk + cat avoid us going through ACT dispatching logic again
        if isinstance(res, AsyncCollectiveTensor):
            res = res.wait()  # type: ignore[attr-defined]
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
    group_name = _resolve_group_name(group, tag)
    group_size = c10d._get_group_size_by_name(group_name)

    if self.size(scatter_dim) % group_size != 0:
        raise AssertionError(
            f"input dimension 0 ({self.size(0)} must be a multiple of group_size {group_size})"
        )
    if scatter_dim != 0:
        tensor_list = torch.chunk(self, group_size, dim=scatter_dim)
        self = torch.cat(tensor_list)

    tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
        self,
        reduceOp.lower(),
        group_size,
        group_name,  # type: ignore[possibly-undefined]
    )
    res = _maybe_wrap_tensor(tensor)
    return res


def reduce_scatter_tensor_autograd(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.

    This function is the same as reduce_scatter_tensor but will propagate the
    backwards gradient across workers.

    Currently only the "sum" reduceOp is supported.

    See reduce_scatter_tensor for more details on usage.
    """

    group_name = _resolve_group_name(group, tag)
    group_size = c10d._get_group_size_by_name(group_name)

    if self.size(scatter_dim) % group_size != 0:
        raise AssertionError(
            f"input dimension 0 ({self.size(0)} must be a multiple of group_size {group_size}"
        )
    if scatter_dim != 0:
        tensor_list = torch.chunk(self, group_size, dim=scatter_dim)
        self = torch.cat(tensor_list)

    tensor = torch.ops._c10d_functional_autograd.reduce_scatter_tensor(
        self,
        reduceOp.lower(),
        group_size,
        group_name,  # type: ignore[possibly-undefined]
    )
    res = _FromTorchTensor.apply(tensor)
    return res


def all_reduce_coalesced(
    self: list[torch.Tensor], reduceOp: str, group: RANK_TYPES, tag: str = ""
) -> list[torch.Tensor]:
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
    group_name = _resolve_group_name(group, tag)
    tensor_list = torch.ops._c10d_functional.all_reduce_coalesced(  # type: ignore[attr-defined]
        self,
        reduceOp.lower(),
        group_name,
    )
    return list(map(_maybe_wrap_tensor, tensor_list))


def all_gather_into_tensor_coalesced(
    self: list[torch.Tensor], group: RANK_TYPES, tag: str = ""
) -> list[torch.Tensor]:
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
    group_name = _resolve_group_name(group, tag)
    group_size = c10d._get_group_size_by_name(group_name)
    tensor_list = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(  # type: ignore[attr-defined]
        self,
        group_size,
        group_name,
    )
    return list(map(_maybe_wrap_tensor, tensor_list))


def reduce_scatter_tensor_coalesced(
    inputs: list[torch.Tensor],
    reduceOp: str,
    scatter_dim: list[int],
    group: RANK_TYPES,
    tag: str = "",
) -> list[torch.Tensor]:
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
    group_name = _resolve_group_name(group, tag)
    group_size = c10d._get_group_size_by_name(group_name)

    if len(scatter_dim) != len(inputs):
        raise AssertionError(
            f"Length of scatter_dim ({len(scatter_dim)}) must equal length of inputs ({len(inputs)})"
        )
    for idx, (dim, tensor) in enumerate(zip(scatter_dim, inputs)):
        if tensor.size(dim) % group_size != 0:
            raise AssertionError(
                f"input dimension {dim} ({tensor.size(dim)} must be a multiple of group_size {group_size} for tensor at index {idx}"
            )
        if dim != 0:
            tensor_list = torch.chunk(tensor, group_size, dim=dim)
            inputs[idx] = torch.cat(tensor_list)

    tensor_list = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(  # type: ignore[attr-defined]
        inputs,
        reduceOp.lower(),
        group_size,
        group_name,  # type: ignore[possibly-undefined]
    )

    return list(map(_maybe_wrap_tensor, tensor_list))


# This is a bit unsafe: it checks if the first argument in the schema reports as a non-mutable alias.
# Today, this maps 1:1 with "aten ops that are views".
def _is_view_op(tgt):
    if not isinstance(tgt, torch._ops.OpOverload):
        raise AssertionError(f"Expected torch._ops.OpOverload, got {type(tgt)}")
    # Don't apply the view optimization to any `CompositeImplicitAutograd` ops.
    # See issue: https://github.com/pytorch/pytorch/issues/133421
    if torch._C._dispatch_has_kernel_for_dispatch_key(
        tgt.name(), torch.DispatchKey.CompositeImplicitAutograd
    ):
        return False
    schema = tgt._schema
    if len(schema.arguments) > 0:
        first_arg = schema.arguments[0]
        # check if op is a view
        return first_arg.alias_info is not None and not first_arg.alias_info.is_write


def all_to_all_single(
    self: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    """
    Each process splits input tensor and then scatters the split list
    to all processes in a group. Then concatenate the received tensors from all
    the processes in the group and return single output tensor.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    if output_split_sizes is not None:
        if not all(
            isinstance(size, (int, torch.SymInt)) for size in output_split_sizes
        ):
            raise AssertionError(
                f"All output_split_sizes must be int or SymInt, got {output_split_sizes}"
            )
    if input_split_sizes is not None:
        if not all(isinstance(size, (int, torch.SymInt)) for size in input_split_sizes):
            raise AssertionError(
                f"All input_split_sizes must be int or SymInt, got {input_split_sizes}"
            )
    group_name = _resolve_group_name(group, tag)
    group_size = c10d._get_group_size_by_name(group_name)
    if output_split_sizes is None or input_split_sizes is None:
        if not (output_split_sizes is None and input_split_sizes is None):
            raise AssertionError(
                "output_split_sizes and input_split_sizes must either be "
                "specified together or both set to None"
            )
        output_split_sizes = [self.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes
    tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        self,
        output_split_sizes,
        input_split_sizes,
        group_name,
    )
    return _maybe_wrap_tensor(tensor)


def all_to_all_single_autograd(
    self: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    """
    Same as all_to_all_single but supports autograd.
    """
    if output_split_sizes is not None:
        if not all(
            isinstance(size, (int, torch.SymInt)) for size in output_split_sizes
        ):
            raise AssertionError(
                f"All output_split_sizes must be int or SymInt, got {output_split_sizes}"
            )
    if input_split_sizes is not None:
        if not all(isinstance(size, (int, torch.SymInt)) for size in input_split_sizes):
            raise AssertionError(
                f"All input_split_sizes must be int or SymInt, got {input_split_sizes}"
            )

    group_name = _resolve_group_name(group, tag)
    group_size = c10d._get_group_size_by_name(group_name)
    if output_split_sizes is None or input_split_sizes is None:
        if not (output_split_sizes is None and input_split_sizes is None):
            raise AssertionError(
                "output_split_sizes and input_split_sizes must either be "
                "specified together or both set to None"
            )
        output_split_sizes = [self.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes
    tensor = torch.ops._c10d_functional_autograd.all_to_all_single(  # type: ignore[attr-defined]
        self,
        output_split_sizes,
        input_split_sizes,
        group_name,
    )
    return _FromTorchTensor.apply(tensor)


# ============================================================================
# Collecive Autograd Functions / Custom Ops
# ============================================================================


def wait_tensor_backward(ctx, grad_output: torch.Tensor):
    """
    Backward for wait_tensor: identity (no-op).
    Wait is just a synchronization primitive, so gradient flows through unchanged.

    Args:
        ctx: Context object
        grad_output: Gradient from downstream operations

    Returns:
        Gradient unchanged (identity)
    """
    return grad_output


def wait_tensor_setup_context(ctx, inputs, output):
    """
    Setup context for wait_tensor backward.
    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (tensor,)
        output: Output from forward pass
    """
    return


torch.library.register_autograd(
    "_c10d_functional::wait_tensor",
    wait_tensor_backward,
    setup_context=wait_tensor_setup_context,
)


def all_reduce_backward(ctx, grad_output: torch.Tensor):
    """
    Backward for all_reduce: all_reduce with same reduce_op.
    Forward aggregates tensors, backward aggregates gradients.

    Args:
        ctx: Context object
        grad_output: Gradient from downstream operations

    Returns:
        Tuple of (grad_input, grad_group_name, grad_reduce_op)
        grad_group_name and grad_reduce_op are None (not differentiable)
    """
    group_name = ctx.group_name
    reduce_op = ctx.reduce_op

    if reduce_op != "sum":
        raise RuntimeError(
            f"all_reduce backward only supports 'sum' reduction, got '{reduce_op}'"
        )

    # Backward does all_reduce with the same reduce_op
    output = torch.ops._c10d_functional.all_reduce(
        grad_output.contiguous(), reduce_op, group_name
    )
    return wait_tensor(output), None, None


def all_reduce_setup_context(ctx, inputs, output):
    """
    Setup context for all_reduce backward.
    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (input, reduce_op, group_name)
        output: Output from forward pass
    """
    input, reduce_op, group_name = inputs
    ctx.group_name = group_name
    ctx.reduce_op = reduce_op.lower()


torch.library.register_autograd(
    "_c10d_functional::all_reduce",
    all_reduce_backward,
    setup_context=all_reduce_setup_context,
)


def all_gather_into_tensor_backward(ctx, grad_output: torch.Tensor):
    """
    Backward for all_gather_into_tensor: reduce_scatter with sum.

    Forward gathers tensors from all ranks, backward scatters gradients back
    with sum reduction.

    Args:
        ctx: Context object with group_name and group_size
        grad_output: Gradient from downstream operations

    Returns:
        Tuple of (grad_input, grad_group_size, grad_group_name)
        grad_group_size and grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    group_size = ctx.group_size

    # Backward is reduce_scatter with sum
    output = torch.ops._c10d_functional.reduce_scatter_tensor(
        grad_output.contiguous(),
        "sum",
        group_size,
        group_name,
    )
    return wait_tensor(output), None, None


def all_gather_into_tensor_setup_context(ctx, inputs, output):
    """
    Setup context for all_gather_into_tensor backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (input, group_size, group_name)
        output: Output from forward pass
    """
    input, group_size, group_name = inputs
    ctx.group_name = group_name
    ctx.group_size = group_size


torch.library.register_autograd(
    "_c10d_functional::all_gather_into_tensor",
    all_gather_into_tensor_backward,
    setup_context=all_gather_into_tensor_setup_context,
)


def reduce_scatter_tensor_backward(ctx, grad_output: torch.Tensor):
    """
    Backward for reduce_scatter_tensor: all_gather.

    Forward reduces and scatters tensors to ranks, backward gathers gradients
    from all ranks.

    Args:
        ctx: Context object with group_name, group_size, and reduce_op
        grad_output: Gradient from downstream operations

    Returns:
        Tuple of (grad_input, grad_reduce_op, grad_group_size, grad_group_name)
        grad_reduce_op, grad_group_size, grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    group_size = ctx.group_size
    reduce_op = ctx.reduce_op

    # Lazy validation: check reduce_op only when backward is called
    if reduce_op != "sum":
        raise RuntimeError(
            f"reduce_scatter_tensor backward only supports 'sum' reduction, got '{reduce_op}'"
        )

    # Backward is all_gather
    output = torch.ops._c10d_functional.all_gather_into_tensor(
        grad_output.contiguous(),
        group_size,
        group_name,
    )
    return wait_tensor(output), None, None, None


def reduce_scatter_tensor_setup_context(ctx, inputs, output):
    """
    Setup context for reduce_scatter_tensor backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (input, reduce_op, group_size, group_name)
        output: Output from forward pass
    """
    input, reduce_op, group_size, group_name = inputs
    ctx.group_name = group_name
    ctx.group_size = group_size
    ctx.reduce_op = reduce_op.lower()


torch.library.register_autograd(
    "_c10d_functional::reduce_scatter_tensor",
    reduce_scatter_tensor_backward,
    setup_context=reduce_scatter_tensor_setup_context,
)


def all_to_all_single_backward(ctx, grad_output: torch.Tensor):
    """
    Backward for all_to_all_single: all_to_all with reversed split sizes.

    Forward does all-to-all with specified split sizes, backward reverses them.

    Args:
        ctx: Context object with group_name, output_split_sizes, and input_split_sizes
        grad_output: Gradient from downstream operations

    Returns:
        Tuple of (grad_input, grad_output_split_sizes, grad_input_split_sizes, grad_group_name)
        All except grad_input are None (not differentiable)
    """
    group_name = ctx.group_name
    output_split_sizes = ctx.output_split_sizes
    input_split_sizes = ctx.input_split_sizes

    # Backward is all_to_all with reversed split sizes
    output = torch.ops._c10d_functional.all_to_all_single(
        grad_output.contiguous(),
        input_split_sizes,  # Reversed
        output_split_sizes,  # Reversed
        group_name,
    )
    return wait_tensor(output), None, None, None


def all_to_all_single_setup_context(ctx, inputs, output):
    """
    Setup context for all_to_all_single backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (input, output_split_sizes, input_split_sizes, group_name)
        output: Output from forward pass
    """
    input, output_split_sizes, input_split_sizes, group_name = inputs
    ctx.group_name = group_name
    ctx.output_split_sizes = output_split_sizes
    ctx.input_split_sizes = input_split_sizes


torch.library.register_autograd(
    "_c10d_functional::all_to_all_single",
    all_to_all_single_backward,
    setup_context=all_to_all_single_setup_context,
)


def all_reduce_coalesced_backward(ctx, grad_outputs: list[torch.Tensor]):
    """
    Backward for all_reduce_coalesced: all_reduce each gradient.

    Forward aggregates tensors, backward aggregates gradients.

    Args:
        ctx: Context object with group_name and reduce_op
        grad_outputs: Gradients from downstream operations (one per input tensor)

    Returns:
        Tuple of (grad_inputs..., grad_reduce_op, grad_group_name)
        grad_reduce_op and grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    reduce_op = ctx.reduce_op

    if reduce_op != "sum":
        raise RuntimeError(
            f"all_reduce_coalesced backward only supports 'sum' reduction, got '{reduce_op}'"
        )

    # Backward does all_reduce on list of gradients
    grad_inputs = torch.ops._c10d_functional.all_reduce_coalesced(
        [grad_output.contiguous() for grad_output in grad_outputs],
        reduce_op,
        group_name,
    )
    return (list(map(wait_tensor, grad_inputs)), None, None)


def all_reduce_coalesced_setup_context(ctx, inputs, output):
    """
    Setup context for all_reduce_coalesced backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (tensor_list, reduce_op, group_name)
        output: Output from forward pass
    """
    tensor_list, reduce_op, group_name = inputs
    ctx.group_name = group_name
    ctx.reduce_op = reduce_op.lower()


torch.library.register_autograd(
    "_c10d_functional::all_reduce_coalesced",
    all_reduce_coalesced_backward,
    setup_context=all_reduce_coalesced_setup_context,
)


def all_gather_into_tensor_coalesced_backward(ctx, grad_outputs: list[torch.Tensor]):
    """
    Backward for all_gather_into_tensor_coalesced: reduce_scatter each gradient.

    Forward gathers tensors from all ranks, backward scatters gradients back
    with sum reduction.

    Args:
        ctx: Context object with group_name and group_size
        grad_outputs: Gradients from downstream operations (one per input tensor)

    Returns:
        Tuple of (grad_inputs..., grad_group_size, grad_group_name)
        grad_group_size and grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    group_size = ctx.group_size

    # Backward does reduce_scatter on list of gradients
    grad_inputs = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
        [grad_output.contiguous() for grad_output in grad_outputs],
        "sum",
        group_size,
        group_name,
    )
    return (list(map(wait_tensor, grad_inputs)), None, None)


def all_gather_into_tensor_coalesced_setup_context(ctx, inputs, output):
    """
    Setup context for all_gather_into_tensor_coalesced backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (tensor_list, group_size, group_name)
        output: Output from forward pass
    """
    tensor_list, group_size, group_name = inputs
    ctx.group_name = group_name
    ctx.group_size = group_size


torch.library.register_autograd(
    "_c10d_functional::all_gather_into_tensor_coalesced",
    all_gather_into_tensor_coalesced_backward,
    setup_context=all_gather_into_tensor_coalesced_setup_context,
)


def reduce_scatter_tensor_coalesced_backward(ctx, grad_outputs: list[torch.Tensor]):
    """
    Backward for reduce_scatter_tensor_coalesced: all_gather each gradient.

    Forward reduces and scatters tensors to ranks, backward gathers gradients
    from all ranks.

    Args:
        ctx: Context object with group_name, group_size, and reduce_op
        grad_outputs: Gradients from downstream operations (one per input tensor)

    Returns:
        Tuple of (grad_inputs..., grad_reduce_op, grad_group_size, grad_group_name)
        grad_reduce_op, grad_group_size, grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    group_size = ctx.group_size
    reduce_op = ctx.reduce_op

    # Lazy validation: check reduce_op only when backward is called
    if reduce_op != "sum":
        raise RuntimeError(
            f"reduce_scatter_tensor_coalesced backward only supports 'sum' reduction, got '{reduce_op}'"
        )

    # Backward does all_gather on list of gradients
    grad_inputs = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
        [grad_output.contiguous() for grad_output in grad_outputs],
        group_size,
        group_name,
    )
    return (list(map(wait_tensor, grad_inputs)), None, None, None)


def reduce_scatter_tensor_coalesced_setup_context(ctx, inputs, output):
    """
    Setup context for reduce_scatter_tensor_coalesced backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (tensor_list, reduce_op, group_size, group_name)
        output: Output from forward pass
    """
    tensor_list, reduce_op, group_size, group_name = inputs
    ctx.group_name = group_name
    ctx.group_size = group_size
    ctx.reduce_op = reduce_op.lower()


torch.library.register_autograd(
    "_c10d_functional::reduce_scatter_tensor_coalesced",
    reduce_scatter_tensor_coalesced_backward,
    setup_context=reduce_scatter_tensor_coalesced_setup_context,
)


def permute_tensor(
    self: torch.Tensor,
    src_dst: list[int],
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    """
    Permutes the elements of the tensor according to the given source/destination pairs. `src_dst` should
    be defined such that src_dst[m] == n means m sends to n.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one
    """
    t, rankset, group_size = _expand_group(group, tag)
    local_pg = c10d._find_or_create_pg_by_ranks_and_tag(t, rankset, group_size)

    output_split_sizes = [0] * group_size
    input_split_sizes = [0] * group_size
    for src, dst in enumerate(src_dst):
        if src == dist.get_rank(local_pg):
            input_split_sizes[dst] = self.numel()
        if dst == dist.get_rank(local_pg):
            output_split_sizes[src] = self.numel()

    return all_to_all_single(self, output_split_sizes, input_split_sizes, group, tag)


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
    completed: bool

    __slots__ = ["elem", "completed"]

    @staticmethod
    def __new__(cls, elem: torch.Tensor):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )
        r.elem = elem
        r.completed = False
        return r

    def __tensor_flatten__(self):
        return ["elem"], None

    def tolist(self):
        return self.trigger_wait().tolist()

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        if meta is not None:
            raise AssertionError(
                "meta must be None for AsyncCollectiveTensor unflatten"
            )
        elem = inner_tensors["elem"]
        return AsyncCollectiveTensor(elem)

    def __coerce_same_metadata_as_tangent__(
        self, expected_metadata: Any, expected_type: type | None = None
    ):
        if expected_type is not torch.Tensor:
            return None

        return self.trigger_wait()

    def __repr__(self) -> str:  # type: ignore[override]
        return f"AsyncCollectiveTensor({self.trigger_wait()})"

    def trigger_wait(self):
        if not self.completed:
            out = wait_tensor(self.elem)
            self.completed = True
            return out
        else:
            return self.elem

    def wait(self) -> torch.Tensor:
        return wait_tensor(self.elem)

    def _get_acs_underlying_tensor(self):
        """This method enables  _functional_collectives_impl to test if a tensor is an ACS"""
        return self.elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        if func is torch.ops.aten.view.default:
            # Fast handle aten.view as a lot of view related op goes to aten.view
            # eventually, this avoids pytree slowdown

            res = func(args[0].elem, args[1])
            wrapper_res = AsyncCollectiveTensor(res)
            return wrapper_res

        is_view_op = _is_view_op(func)

        def unwrap(e: AsyncCollectiveTensor):
            # wait_tensor is idepotent and will do stream sync only once
            if not is_view_op:
                return e.trigger_wait()
            return e.elem

        def wrap(e: torch.Tensor):
            # wait_tensor is idepotent and will do stream sync only once
            if isinstance(e, AsyncCollectiveTensor):
                raise AssertionError(
                    "Cannot wrap an AsyncCollectiveTensor inside another AsyncCollectiveTensor"
                )
            res = AsyncCollectiveTensor(e)
            return res

        unwrapped_args = tree_map_only(AsyncCollectiveTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(AsyncCollectiveTensor, unwrap, kwargs)

        # we don't wrap the result as it doesn't need to be waited on.
        out = func(*unwrapped_args, **unwrapped_kwargs)

        # View ops dont require a sync, so we should re-wrap the outputs.
        if is_view_op:
            out = tree_map_only(torch.Tensor, wrap, out)

        return out

    def numpy(self):  # type: ignore[override]
        return self.wait().numpy()


"""
Utils and infrastructure for tracing support
"""


def _expand_group(group: RANK_TYPES, tag: str = "") -> tuple[str, list[int], int]:
    """
    _expand_group desugars the different RANK_TYPES types into a canonical format that is traceable.

    By having this be part of the explicit eager codepath, we avoid having to specialize behavior inside
    torchdynamo and can still interoperate with processgroup objects or other untraceable forms.
    """
    # had to define this hack _inside_ expand_group to avoid
    # graph_break [('torch.* op returned non-Tensor int
    # caused by 'cast_*` functions being treated as 'torch.*' ops (iiuc)
    if TYPE_CHECKING:

        def cast_listlistint(x):
            return cast(list[list[int]], x)

        def cast_listint(x):
            return cast(list[int], x)

    else:
        # fake cast op for use at runtime since dynamo doesn't support real cast
        # also, dynamo didn't like encountering 'typing' objects ()
        # NotImplementedError: argument of type: <class 'typing._GenericAlias'>
        def cast_listlistint(x):
            return x

        def cast_listint(x):
            return x

    rankset: list[int]
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
    elif isinstance(group, DeviceMesh):
        if group.ndim != 1:
            raise AssertionError(
                "Only 1D mesh is supported, pass in (DeviceMesh, int) together if mesh > 1D"
            )
        # TODO: it should run collective in the whole mesh instead of dim 0
        pg = group.get_group()
        rankset = dist.get_process_group_ranks(pg)
        group_size = len(rankset)
        tag = tag or c10d._get_group_tag(pg)
    elif isinstance(group, tuple):
        if (
            len(group) == 2
            and isinstance(group[0], DeviceMesh)
            and isinstance(group[1], int)
        ):
            dmesh = group[0]
            dim = group[1]
            pg = dmesh.get_group(dim)
            rankset = dist.get_process_group_ranks(pg)
            group_size = len(rankset)
            tag = tag or c10d._get_group_tag(pg)
        else:
            raise ValueError("Invalid tuple for group must be (DeviceMesh, int)")
    else:
        raise ValueError(
            "Invalid type for group, must be one of List, Processgroup, DeviceMesh or (DeviceMesh, int)."
        )

    return (tag, rankset, group_size)


def _resolve_group_name(group: RANK_TYPES, tag: str = "") -> c10d.GroupName:
    """
    Given group in RANK_TYPES, return the group name.
    """
    # `tag` will be deprecated. See details in:
    # https://github.com/pytorch/pytorch/issues/93173#issuecomment-1907095208
    if isinstance(group, dist.ProcessGroup):
        return group.group_name
    elif isinstance(group, str):
        # In some cases Dynamo doesn't like tracing through NewType constructors
        # - so use a cast instead (the actual newtype representation is
        # literally the underlying type so this is fine). I haven't been able to
        # reproduce it in isolation (see T247631668).
        return cast(c10d.GroupName, group)  # c10d.GroupName(group)
    elif isinstance(group, DeviceMesh):
        if group.ndim != 1:
            raise AssertionError(
                "Only 1D mesh is supported, pass in (DeviceMesh, int) together if mesh > 1D"
            )
        return group._dim_group_names[0]
    elif isinstance(group, tuple):
        if (
            len(group) == 2
            and isinstance(group[0], DeviceMesh)
            and isinstance(group[1], int)
        ):
            dmesh = group[0]
            dim = group[1]
            return dmesh._dim_group_names[dim]
        else:
            raise ValueError("Invalid tuple for group must be (DeviceMesh, int)")
    elif isinstance(group, list):
        if not is_torchdynamo_compiling():
            warnings.warn(
                "The combination of ranks + tag as process group "
                "identifier has been deprecated. Please switch to "
                "using ProcessGroup, DeviceMesh, or group name instead.",
                FutureWarning,
                stacklevel=3,
            )
        # pyrefly: ignore [redundant-cast]
        return c10d._resolve_group_name_by_ranks_and_tag(cast(list[int], group), tag)
    else:
        raise ValueError(f"Unsupported group type: {type(group)}, {group}")


class _FromTorchTensor(torch.autograd.Function):
    """
    _FromTorchTensor allows autograd to propagate from a normal Tensor to an
    AsyncCollectiveTensor.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,  # pyre-ignore[2]: Parameter must be annotated.
        input: torch.Tensor,
    ) -> torch.Tensor:
        return _maybe_wrap_tensor(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return grad_output


@torch.library.custom_op(
    "_c10d_functional::_wrap_tensor_autograd",
    mutates_args=(),
    schema="(Tensor input) -> Tensor",
)
def _wrap_tensor_autograd(input: torch.Tensor) -> torch.Tensor:
    """
    Custom op that allows autograd to propagate
    from a normal Tensor to an AsyncCollectiveTensor.

    This is the low-level implementation. Users should call _maybe_wrap_tensor directly.

    Args:
        input: Input tensor to wrap in AsyncCollectiveTensor

    Returns:
        AsyncCollectiveTensor wrapping the input (or wait_tensor result if tracing)
    """
    return AsyncCollectiveTensor(input)


@_wrap_tensor_autograd.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    """
    Meta kernel for _wrap_tensor_autograd.
    """
    return torch.empty_like(input)


def _wrap_tensor_autograd_backward(ctx, grad_output: torch.Tensor):
    """
    Backward for _wrap_tensor_autograd: identity (no-op).

    The wrapping is just for async optimization, gradients flow through unchanged.

    Args:
        ctx: Context object (unused)
        grad_output: Gradient from downstream operations

    Returns:
        Gradient unchanged (identity)
    """
    return grad_output


def _wrap_tensor_autograd_setup_context(ctx, inputs, output):
    """
    Setup context for _wrap_tensor_autograd backward.

    Args:
        ctx: Context object to save state for backward (nothing to save)
        inputs: Tuple of (input,)
        output: Output from forward pass
    """
    return


_wrap_tensor_autograd.register_autograd(
    _wrap_tensor_autograd_backward,
    setup_context=_wrap_tensor_autograd_setup_context,
)


def _are_we_tracing() -> bool:
    if is_torchdynamo_compiling():
        return True
    # If fake mode is turned on, we are almost definitely compiling/tracing.
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is not None:
        return True
    # See Note [enable_python_dispatcher in dynamo]
    if torch._C._dispatch_tls_is_dispatch_key_included(
        torch._C.DispatchKey.PythonDispatcher
    ):
        return True
    return get_proxy_mode() is not None


def _maybe_wrap_tensor(self) -> torch.Tensor:
    if _are_we_tracing():
        return wait_tensor(self)
    return _wrap_tensor_autograd(self)


@contextlib.contextmanager
def allow_inflight_collective_as_graph_input_ctx(value: bool = True):
    """
    Context manager to temporarily set whether inflight collectives are allowed as torch.compile graph inputs.
    Common use case is when the collective is issued in eager (with `async_op=True`) but waited in compiled region:
    ```
    def all_reduce_eager(x):
        y = x * x
        req = dist.all_reduce(y, op=dist.ReduceOp.SUM, async_op=True)
        return y


    @torch.compile(fullgraph=True)
    def all_reduce_wait_compiled(y):
        torch.ops.c10d_functional.wait_tensor(y)
        return y * y


    x = torch.ones(1280, 1280, device="cuda") + self.rank
    # the context manager ensures that `wait_tensor(y)` will wait on the correct work object
    with allow_inflight_collective_as_graph_input_ctx():
        y = all_reduce_eager(x)
        z = all_reduce_wait_compiled(y)
    ```
    With this context manager, when a collective is called, under the hood the work object of the collective
    will be registered in the work registry, and the wait_tensor() in compiled region called on
    the output tensor of the collective will wait on the correct work object.
    """
    previous = torch._C._distributed_c10d._allow_inflight_collective_as_graph_input()

    try:
        torch._C._distributed_c10d._set_allow_inflight_collective_as_graph_input(value)
        yield
    finally:
        torch._C._distributed_c10d._set_allow_inflight_collective_as_graph_input(
            previous
        )


def _make_all_gather_out_tensor(input, group_size):
    out_size = list(input.size())
    if len(out_size) == 0:
        out_size.append(group_size)
    else:
        out_size[0] *= group_size
    out_tensor = input.new_empty(out_size)
    return out_tensor


def _all_gather_into_tensor_coalesced_meta(self, tag, rankset, group_size):
    return [_make_all_gather_out_tensor(t, group_size) for t in self]


# We now register meta kernels to deal with tracing
def _broadcast_meta(self, *args):
    return torch.empty_like(self)


def _all_reduce_meta(self, *args):
    return torch.empty_like(self)


def _wait_tensor_meta(self, *args):
    return torch.empty_like(self)


def _all_gather_into_tensor_meta(shard, tag, rankset, group_size):
    return _make_all_gather_out_tensor(shard, group_size)


def _reduce_scatter_tensor_meta(input, reduce_op, tag, rankset, group_size):
    out_size = list(input.size())
    out_size[0] //= group_size
    return input.new_empty(out_size)


def _all_reduce_coalesced_meta(self, *args):
    return [torch.empty_like(t) for t in self]


def _all_reduce__meta(inp, *args):
    return inp


def _broadcast__meta(inp, *args):
    return inp


def _all_reduce_coalesced__meta(inputs, *args):
    return inputs


def _reduce_scatter_tensor_coalesced_meta(inputs, reduceOp, tag, rankset, group_size):
    def mk_out_tensor(input):
        out_size = list(input.size())
        out_size[0] //= group_size
        out_tensor = input.new_empty(out_size)
        return out_tensor

    return [mk_out_tensor(t) for t in inputs]


# NB: We often say all_to_all has dynamic output size, but this is not
# technically true: instead, what typically happens is you manually
# communicate the output_split_sizes ahead of time (which is dynamic),
# but then you pass those sizes explicitly, and the all to all itself
# isn't dynamic, it just follows the specified output splits
def _all_to_all_single_meta(
    input, output_split_sizes, input_split_sizes, *args, **kwargs
):
    if output_split_sizes is None:
        return input.new_empty(input.size())
    else:
        for s in output_split_sizes:
            torch._check(s >= 0)
        out_size = list(input.size())
        out_size[0] = sum(output_split_sizes)
        return input.new_empty(out_size)


def _all_gather_into_tensor_out_native_meta(input, group_size, group_name, *, out):
    return _make_all_gather_out_tensor(input, group_size)


def _all_gather_into_tensor_native_meta(input, group_size, group_name):
    return _make_all_gather_out_tensor(input, group_size)


def _all_gather_into_tensor_coalesced_native_meta(inputs, group_size, group_name):
    return [
        _all_gather_into_tensor_native_meta(input, group_size, group_name)
        for input in inputs
    ]


def _reduce_scatter_tensor_native_meta(inp, reduce_op, group_size, group_name):
    shape = list(inp.size())
    shape[0] //= group_size
    return inp.new_empty(shape)


def _reduce_scatter_tensor_out_native_meta(
    inp, reduce_op, group_size, group_name, *, out
):
    shape = list(inp.size())
    shape[0] //= group_size
    return inp.new_empty(shape)


def _reduce_scatter_tensor_coalesced_native_meta(
    inputs, reduce_op, group_size, group_name
):
    return [
        _reduce_scatter_tensor_native_meta(inp, reduce_op, group_size, group_name)
        for inp in inputs
    ]


# Library MUST be defined at module scope or it doesn't work
lib_impl = torch.library.Library("_c10d_functional", "IMPL")
lib_impl.impl("all_reduce", _all_reduce_meta, "Meta")
lib_impl.impl("all_reduce_", _all_reduce__meta, "Meta")
lib_impl.impl("all_reduce_coalesced", _all_reduce_coalesced_meta, "Meta")
lib_impl.impl("all_reduce_coalesced_", _all_reduce_coalesced__meta, "Meta")
lib_impl.impl("wait_tensor", _wait_tensor_meta, "Meta")
lib_impl.impl(
    "all_gather_into_tensor_out", _all_gather_into_tensor_out_native_meta, "Meta"
)
lib_impl.impl("all_gather_into_tensor", _all_gather_into_tensor_native_meta, "Meta")
lib_impl.impl(
    "all_gather_into_tensor_coalesced",
    _all_gather_into_tensor_coalesced_native_meta,
    "Meta",
)
lib_impl.impl("reduce_scatter_tensor", _reduce_scatter_tensor_native_meta, "Meta")
lib_impl.impl(
    "reduce_scatter_tensor_out", _reduce_scatter_tensor_out_native_meta, "Meta"
)
lib_impl.impl(
    "reduce_scatter_tensor_coalesced",
    _reduce_scatter_tensor_coalesced_native_meta,
    "Meta",
)
lib_impl.impl("all_to_all_single", _all_to_all_single_meta, "Meta")
lib_impl.impl("broadcast", _broadcast_meta, "Meta")
lib_impl.impl("broadcast_", _broadcast__meta, "Meta")

# mark these ops has side effect so that they won't be removed by DCE
torch.fx.node.has_side_effect(torch.ops._c10d_functional.wait_tensor.default)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.wait_tensor)  # type: ignore[has-type]

# Register legacy ops for backward compatibility
# TODO(yifu): remove these in functional collective beta release
legacy_lib = torch.library.Library("c10d_functional", "DEF")
legacy_lib_impl = torch.library.Library("c10d_functional", "IMPL")
ops_defs = [
    "broadcast(Tensor self, int src, str tag, int[] ranks, int group_size) -> Tensor",
    "all_reduce(Tensor self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
    "all_reduce_coalesced(Tensor[] self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
    "wait_tensor(Tensor self) -> Tensor",
    "all_gather_into_tensor(Tensor shard, str tag, int[] ranks, int group_size) -> Tensor",
    "all_gather_into_tensor_coalesced(Tensor[] input, str tag, int[] ranks, int group_size) -> Tensor[]",
    "reduce_scatter_tensor(Tensor input, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
    "reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
    "all_to_all_single(Tensor input, SymInt[]? output_split_sizes, SymInt[]? input_split_sizes, str tag, int[] ranks, int group_size) -> Tensor",  # noqa: B950
]

my_module = sys.modules[__name__]
for op_def in ops_defs:
    op_name = op_def[0 : op_def.index("(")]
    backend_impl = getattr(fun_col_impl, f"_{op_name}")
    legacy_lib.define(op_def, tags=torch.Tag.pt2_compliant_tag)
    legacy_lib_impl.impl(op_name, backend_impl, "CompositeImplicitAutograd")


"""
Dynamo Remappings allow seamless translation from non-functional collectives of supportable form into
functional collective calls followed by inplace copy ops, allowing them to be traced into a functional graph.

We implement this by writing a decomposition and teaching dynamo how to associate it to a corresponding op via
the mapping dict below.

These schemas intentionally match torch.distributed.distributed_c10d.* ops that we are trying to remap from
"""


def all_gather_tensor_inplace(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group=None,  # TODO add a type,
    async_op: bool = False,
    tag: str = "",
    gather_dim: int = 0,
):
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    return output_tensor.copy_(all_gather_tensor(input_tensor, gather_dim, group, tag))


def reduce_scatter_tensor_inplace(
    output: torch.Tensor,
    input: torch.Tensor,
    op: str = "sum",  # TODO type is actually c10d ReduceOp. is this ok?
    group=None,  # TODO add a type
    async_op: bool = False,
    scatter_dim: int = 0,
    tag: str = "",
):
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    return output.copy_(reduce_scatter_tensor(input, op, scatter_dim, group, tag))


REDUCE_OP_TO_STR = {
    dist.ReduceOp.SUM: "sum",
    dist.ReduceOp.AVG: "avg",
    dist.ReduceOp.PRODUCT: "product",
    dist.ReduceOp.MIN: "min",
    dist.ReduceOp.MAX: "max",
    dist.ReduceOp.BAND: "band",
    dist.ReduceOp.BOR: "bor",
    dist.ReduceOp.BXOR: "bxor",
}


def all_reduce_inplace(
    tensor: torch.Tensor,
    op: str = "sum",
    group=None,
    async_op: bool = False,
    tag: str = "",
):
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    return tensor.copy_(all_reduce(tensor, op, group, tag))


def all_to_all_inplace(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op=False,
    tag: str = "",
):
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    return output.copy_(
        all_to_all_single(
            input,
            output_split_sizes,
            input_split_sizes,
            group,
            tag,
        )
    )


def all_gather_inplace(
    tensor_list: list[torch.Tensor],
    tensor: torch.Tensor,
    group=None,
    async_op=False,
    tag: str = "",
):
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )
    if tensor.dim() != 0 and not all(t.size(0) == tensor.size(0) for t in tensor_list):
        raise AssertionError("Remapping variable size all_gather is not yet supported")

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    output = all_gather_tensor(tensor, 0, group, tag)

    # Use aten.slice instead of aten.split because the latter causes
    # tensor.shape(0) to be unnecessarily baked in when it's a SymInt.
    output_splits = []
    offset = 0
    for t in tensor_list:
        is_scalar = t.dim() == 0
        t_offset = 1 if is_scalar else t.size(0)

        out = output[offset] if is_scalar else output[offset : offset + t_offset]
        output_splits.append(out)

        offset += t_offset
    for dst, src in zip(tensor_list, output_splits):
        dst.copy_(src)
    return tensor_list


from torch.distributed.distributed_c10d import (
    _all_gather_base as legacy_all_gather_base,
    _reduce_scatter_base as legacy_reduce_scatter_base,
    all_gather as legacy_all_gather,
    all_gather_into_tensor as legacy_allgather,
    all_reduce as legacy_allreduce,
    all_to_all_single as legacy_all_to_all_single,
    reduce_scatter_tensor as legacy_reducescatter,
)


# This dict should contain sets of functions that dynamo is allowed to remap.
# Functions in this set should accept the same args/kwargs 1:1 as their mapping.
traceable_collective_remaps = {
    legacy_allgather: all_gather_tensor_inplace,  # type: ignore[has-type]
    legacy_reducescatter: reduce_scatter_tensor_inplace,  # type: ignore[has-type]
    legacy_allreduce: all_reduce_inplace,  # type: ignore[has-type]
    legacy_all_to_all_single: all_to_all_inplace,  # type: ignore[has-type]
    legacy_all_gather: all_gather_inplace,  # type: ignore[has-type]
    legacy_reduce_scatter_base: reduce_scatter_tensor_inplace,  # type: ignore[has-type]
    legacy_all_gather_base: all_gather_tensor_inplace,  # type: ignore[has-type]
}
