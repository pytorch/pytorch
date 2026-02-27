"""
A LocalTensor is a tensor subclass which simulates a tensor that is
distributed across SPMD ranks.  A LocalTensor might be size N, but in fact
there are world_size shards/replicas of it stored internally.  When you do a
plain PyTorch operation on it, we apply the operation to each shard; when you
do a collective, we do the mathematically equivalent operation on the local
shards.  A LocalTensor is associated with a list of ranks which specify
which ranks it holds local tensors for.

NB, this is NOT a DataParallel like abstraction where you can run operations
on multiple different GPUs. It is intended purely for *debugging* purposes,
the overhead is almost certainly too high to keep eight GPUs (even the C++
autograd needs multithreading to keep up!)  (It might potentially be possible
to trace through this with torch.compile and then compile it with CUDA graphs
but this is currently a non-goal.)

We do not directly handling MPMD. However in practice even in SPMD you may
encounter divergence in behavior per rank (for example, uneven sharding
across ranks). To support scenarios like this, we provide a helper decorator
that allows you to run a function with no side effects for each LocalTensor
shard and combine results back into LocalTensor or LocalIntNode.

NB: This is a torch dispatch Tensor subclass, as we want to assume that autograd
is SPMD, so we run it once, and dispatch the inner autograd calls to the individual
local shards.

NOTE ABOUT MESH:  This subclass requires collectives that are issued to it to
respect a DeviceMesh like abstraction.  The reason for this is that when
DTensor issues us a collective for a particular rank, you will be asked to do
this on a specific process group which involves some ranks.  However, this
will only be for the LOCAL PG that this particular rank is participating in;
there will be a bunch of other PGs for other nodes that you don't get to see.
We need to be able to reverse engineer all of the collectives that don't
involve the current local rank here to actually issue them.  This can be done
two ways: (1) looking at the participating local ranks in the PG and computing
the complement which specifies all the other collectives you have to run, or
(2) retrieving the device mesh axis corresponding to the PG for this rank, and
then running all the fibers for this.
"""

import contextlib
import copy
import functools
import importlib
import operator
import os
import sys
import threading
from ast import Call
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from types import TracebackType
from typing import Any, Optional, ParamSpec, TypeVar, Union


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

import torch
import torch.distributed as dist
from torch import Size, SymBool, SymInt, Tensor
from torch._C import DispatchKey, DispatchKeySet, ScriptObject
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed import DeviceMesh, ProcessGroup
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.distributed_c10d import _get_default_group
from torch.fx.experimental._constant_symnode import ConstantIntNode
from torch.nested._internal.nested_int import NestedIntNode
from torch.utils import _pytree as pytree
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode_stack,
    return_and_correct_aliasing,
    TorchDispatchMode,
)
from torch.utils.checkpoint import get_device_states, set_device_states


_R = TypeVar("_R")
_P = ParamSpec("_P")

not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")


from . import _c10d


def _is_in_fake_tensor_mode() -> bool:
    return any(
        isinstance(mode, FakeTensorMode) for mode in _get_current_dispatch_mode_stack()
    )


def _reduce_multidim_lists(
    lists_to_reduce: list[Any], reduce_func: Callable[[list[Any]], Any]
) -> Any:
    """
    Reduces a list of multi-dimensional lists, assuming they all have
    the exact same shape.

    Args:
        lists_to_reduce (list): A list where each item is a multi-dimensional
                                list (e.g., [md_list_1, md_list_2, ...]).
                                All inner md_lists must have the same shape.
        reduce_func (callable): A function that takes an iterable (list) of
                                values and returns a single reduced value.
                                For example: sum, max, min, or
                                lambda x: sum(x) / len(x) for mean.

    Returns:
        A single multi-dimensional list of the same shape as the inputs,
        where each value is the result of the reduce_func.

    Raises:
        ValueError: If the input list is empty or if shapes are inconsistent
                    (which may also raise IndexError or TypeError).
    """
    if not lists_to_reduce:
        raise ValueError("Input 'lists_to_reduce' cannot be empty.")

    # Get the first list to inspect its structure (shape)
    first_list = lists_to_reduce[0]

    # Check if the first element of this list is *also* a list.
    # This determines if we are at the base case or need to recurse.
    if isinstance(first_list[0], list):
        # --- RECURSIVE STEP ---
        # The elements are lists, so we need to go one level deeper.

        # We find the number of sub-lists from the first list.
        # (e.g., for [[1,2], [3,4]], this is 2)
        num_sublists = len(first_list)

        result = []
        # Iterate by the index of the sub-lists (e.g., i = 0, then i = 1)
        for i in range(num_sublists):
            # Build a new list to pass to the recursive call.
            # This list will contain the i-th sublist from *each* of the
            # input lists.
            # e.g., if lists_to_reduce = [ L1, L2 ] and i = 0,
            # this creates [ L1[0], L2[0] ]
            sublists_to_reduce = [l[i] for l in lists_to_reduce]

            # Recurse and append the result
            result.append(_reduce_multidim_lists(sublists_to_reduce, reduce_func))
        return result
    else:
        # --- BASE CASE ---
        # The elements are values (int, float, etc.), not lists.
        # We are at the innermost dimension.

        # Find the number of values in the innermost list.
        # (e.g., for [1, 2], this is 2)
        num_values = len(first_list)

        result = []
        # Iterate by the index of the values (e.g., i = 0, then i = 1)
        for i in range(num_values):
            # Get the values at this specific position (i) from *all*
            # input lists.
            # e.g., if lists_to_reduce = [ [1,2], [10,20] ] and i = 0,
            # this creates [ 1, 10 ]
            values_at_pos = [l[i] for l in lists_to_reduce]

            # Apply the user-provided reduction function to this list of values
            # and append the single result.
            result.append(reduce_func(values_at_pos))
        return result


def _is_inplace_op(op: OpOverload | Callable[..., Any]) -> bool:
    return (
        isinstance(op, OpOverload)
        # Not precise heuristic to detect inplace operation
        and op._schema.name[-1] == "_"
        # Strengthen the heuristic to check that the first argument and return value are a write
        and len(op._schema.arguments) > 0
        and op._schema.arguments[0].is_write
        and len(op._schema.returns) > 0
        and op._schema.returns[0].is_write
    )


def _int_on_rank(i: "int | LocalIntNode | ConstantIntNode", r: int) -> int:
    if isinstance(i, LocalIntNode):
        return i._local_ints[r]
    elif isinstance(i, ConstantIntNode):
        return i.val
    elif isinstance(i, int):
        return i
    else:
        raise AssertionError(type(i))


def _check_for_subclass(flat_args: Sequence[object]) -> bool:
    return any(_check_for_subclass_arg(x) for x in flat_args)


def _check_for_subclass_arg(x: object) -> bool:
    return (
        not isinstance(x, LocalTensor)
        and isinstance(x, Tensor)
        and type(x)
        not in (
            Tensor,
            FakeTensor,
            torch.nn.Parameter,
            torch.nn.Buffer,
        )
    )


def _map_to_rank_local_val(val: Any, rank: int) -> Any:
    if isinstance(val, LocalTensor):
        return val._local_tensors[rank]
    if isinstance(val, SymInt):
        if isinstance(val.node, LocalIntNode):
            return val.node._local_ints[rank]
        if isinstance(val.node, ConstantIntNode):
            return val.node.val
    return val


def _collect_accelerator_rng_states() -> dict[int, torch.Tensor]:
    """
    Collects RNG state from all available acceleator devices.

    Returns:
        List of RNG state tensors, one for each accelerator device.
        Returns empty list if accelerator is not available.
    """
    if not torch.accelerator.is_available():
        return {}

    if torch.accelerator.is_available():
        device_idx = torch.accelerator.current_device_index()
        with torch.accelerator.device_index(device_idx):
            return {device_idx: torch.get_device_module().get_rng_state()}

    return {}


def _set_accelerator_rng_states(rng_states: dict[int, torch.Tensor]) -> None:
    """
    Sets RNG state for all accelerator devices from a list of states.

    Args:
        rng_states: List of RNG state tensors to restore.
    """
    if not torch.accelerator.is_available():
        return

    if torch.accelerator.is_available():
        for device_idx, device_rng_state in rng_states.items():
            with torch.accelerator.device_index(device_idx):
                torch.get_device_module().set_rng_state(device_rng_state)


def _get_rng_state() -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """
    Gets CPU and accelerator (e.g., CUDA, XPU device) rng states from all devices.
    """
    return (torch.get_rng_state(), _collect_accelerator_rng_states())


def _set_rng_state(
    cpu_state: torch.Tensor, accelerator_states: dict[int, torch.Tensor]
) -> None:
    """
    Sets CPU and accelerator (e.g., CUDA, XPU device) rng states for all devices. If
    the list of accelerator states is shorter than the number of devices only the
    first len(accelerator_states) devices will get their rng state set.
    """
    torch.set_rng_state(cpu_state)
    _set_accelerator_rng_states(accelerator_states)


def _combine_int_rank_results(rank_results: dict[int, int]) -> int | torch.SymInt:
    any_v = next(iter(rank_results.values()))

    if all(v == any_v for v in rank_results.values()):
        return any_v

    return torch.SymInt(LocalIntNode(rank_results))


def _combine_any_rank_results(rank_results: dict[int, Any]) -> Any:
    any_v = next(iter(rank_results.values()))

    if isinstance(any_v, Tensor):
        # pyrefly: ignore [bad-argument-type, bad-argument-count]
        return LocalTensor(rank_results)

    if isinstance(any_v, int):
        return _combine_int_rank_results(rank_results)

    if isinstance(any_v, torch.device):
        if not all(v.type == any_v.type for v in rank_results.values()):
            raise AssertionError("device type should be the same")
        # Just use the first device - the device type is what matters,
        # and LocalTensorMode runs on a single physical device anyway
        return any_v

    if not all(v == any_v for v in rank_results.values()):
        raise AssertionError(
            "Non Tensor or int rank results must be equal for all ranks"
        )

    return any_v


def _combine_rank_results(rank_results: dict[int, Any], default: Any | None) -> Any:
    rank_ids = rank_results.keys()
    rank_value = rank_results[next(iter(rank_ids))]

    if isinstance(rank_value, (list, tuple)):
        max_rank_result_len = max(len(v) for v in rank_results.values())
        ret_list = []
        for i in range(max_rank_result_len):
            rank_col_results = {
                r: v[i] if i < len(v) else default for r, v in rank_results.items()
            }
            ret_list.append(_combine_any_rank_results(rank_col_results))
        return type(rank_value)(ret_list)
    else:
        return _combine_any_rank_results(rank_results)


def _zero_sized_like(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    tensor_size = list(tensor.size())
    tensor_size[dim] = 0
    empty_tensor = torch.empty(*tensor_size, dtype=tensor.dtype, device=tensor.device)
    return empty_tensor


def _for_each_rank_run_func(
    func: OpOverload | Callable[..., Any],
    ranks: frozenset[int],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    alias: bool = True,
) -> Any:
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    flat_args = [
        a.wait() if isinstance(a, AsyncCollectiveTensor) else a for a in flat_args
    ]

    lm = enabled_local_tensor_mode()
    use_per_rank_rng = lm is not None and len(lm._per_rank_rng_states) > 0

    global_rng_state = None if use_per_rank_rng else _get_rng_state()

    flat_rank_rets = {}

    default_value: Tensor | None = None
    for r in sorted(ranks):
        if use_per_rank_rng:
            if lm is None:
                raise AssertionError
            if r in lm._per_rank_rng_states:
                _set_rng_state(*lm._per_rank_rng_states[r])
        else:
            if global_rng_state is None:
                raise AssertionError
            _set_rng_state(*global_rng_state)

        rank_flat_args = [_map_to_rank_local_val(a, r) for a in flat_args]
        rank_args, rank_kwargs = pytree.tree_unflatten(rank_flat_args, args_spec)
        if func is torch.ops.aten.hash_tensor.default and rank_args[0].numel() == 0:
            # Special case for empty tensors, hash_tensor returns an empty tensor
            rank_ret = torch.empty(0, dtype=torch.uint64, device=rank_args[0].device)
        else:
            rank_ret = func(*rank_args, **rank_kwargs)
        flat_rank_rets[r] = rank_ret

        if use_per_rank_rng:
            if lm is None:
                raise AssertionError
            lm._per_rank_rng_states[r] = _get_rng_state()

        if default_value is None and func is torch.ops.aten.split.Tensor:
            # If split happens over the dimension smaller than the number of chunks
            # it is possible that some ranks will produce shorter lists of chunks.
            # In order to make the result across all ranks of the same length we
            # append empty tensors (zero size on the split dimension).
            tensor = rank_flat_args[0]
            split_dim = 0 if len(rank_flat_args) < 3 else rank_flat_args[2]
            default_value = _zero_sized_like(tensor, split_dim)

    if _is_inplace_op(func):
        alias = False
        # For the in-place ops return self
        ret = args[0]
        if isinstance(func, OpOverload) and torch.Tag.inplace_view in func.tags:
            # Ensure that wrapper tensor size is synchronized with its local tensors
            ret._sync_meta()
    else:
        ret = _combine_rank_results(flat_rank_rets, default_value)

    if alias:
        return return_and_correct_aliasing(func, args, kwargs, ret)
    else:
        return ret


def _get_extra_dispatch_keys(t: torch.Tensor) -> DispatchKeySet:
    extra_dispatch_keys = torch._C.DispatchKeySet.from_raw_repr(0)
    if torch._C._dispatch_keys(t).has(torch._C.DispatchKey.Conjugate):
        extra_dispatch_keys = extra_dispatch_keys.add(torch._C.DispatchKey.Conjugate)
    if torch._C._dispatch_keys(t).has(torch._C.DispatchKey.Negative):
        extra_dispatch_keys = extra_dispatch_keys.add(torch._C.DispatchKey.Negative)
    return extra_dispatch_keys


class LocalIntNode:
    """
    Like a LocalTensor, but for an int.  We can't use a 0D tensor to represent this
    because often only a SymInt is accepted where we wish to use this.
    """

    def __new__(cls, local_ints: dict[int, int]) -> "ConstantIntNode | LocalIntNode":  # type: ignore[misc]
        if len(set(local_ints.values())) == 1:
            return ConstantIntNode(next(iter(local_ints.values())))
        return super().__new__(cls)

    def __init__(self, local_ints: dict[int, int]):
        self._local_ints = local_ints

    def maybe_as_int(self) -> int | None:
        return None

    def is_int(self) -> bool:
        return True

    def is_float(self) -> bool:
        return False

    def is_bool(self) -> bool:
        return False

    def is_nested_int(self) -> bool:
        return False

    def clone(self) -> "LocalIntNode":
        return self

    def _str(self) -> str:
        return f"LocalIntNode({self._local_ints})"

    def __str__(self) -> str:
        return self._str()

    def __repr__(self) -> str:
        return self._str()

    def _graph_repr(self) -> str:
        return self._str()

    def is_symbolic(self) -> bool:
        return False

    def is_constant(self) -> bool:
        return False

    def sym_max(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {
                r: max(self._local_ints[r], _int_on_rank(other, r))
                for r in self._local_ints
            }
        )

    def sym_min(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {
                r: min(self._local_ints[r], _int_on_rank(other, r))
                for r in self._local_ints
            }
        )

    def sym_sum(self, other: Sequence[Any]) -> "LocalIntNode | ConstantIntNode":
        t = LocalIntNode(dict.fromkeys(self._local_ints, 0))
        for o in other:
            t = t.add(o)
        return t

    def neg(self) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode({r: -self._local_ints[r] for r in self._local_ints})

    def add(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] + _int_on_rank(other, r) for r in self._local_ints}
        )

    def sub(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] - _int_on_rank(other, r) for r in self._local_ints}
        )

    def mul(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] * _int_on_rank(other, r) for r in self._local_ints}
        )

    def floordiv(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] // _int_on_rank(other, r) for r in self._local_ints}
        )

    def mod(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] % _int_on_rank(other, r) for r in self._local_ints}
        )

    def int_floordiv(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] // _int_on_rank(other, r) for r in self._local_ints}
        )

    def eq(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] == _int_on_rank(other, r) for r in self._local_ints}
        return torch._C._get_constant_bool_symnode(len(r) == 1 and next(iter(r)))

    def ne(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] != _int_on_rank(other, r) for r in self._local_ints}
        return torch._C._get_constant_bool_symnode(len(r) > 1 or next(iter(r)))

    def ge(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] >= _int_on_rank(other, r) for r in self._local_ints}
        if len(r) != 1:
            raise AssertionError((self, other))
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def le(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] <= _int_on_rank(other, r) for r in self._local_ints}
        if len(r) != 1:
            raise AssertionError((self, other))
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def gt(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] > _int_on_rank(other, r) for r in self._local_ints}
        if len(r) != 1:
            raise AssertionError((self, other))
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def lt(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] < _int_on_rank(other, r) for r in self._local_ints}
        if len(r) != 1:
            raise AssertionError((self, other))
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def wrap_int(self, num: int) -> "LocalIntNode | ConstantIntNode":
        return ConstantIntNode(num)


class _LocalDeviceHandle:
    """
    Wrapper around device module (e.g., torch.cuda) with automatic LocalTensor semantics.

    This class wraps device modules and automatically handles per-rank operations in
    LocalTensor mode:
    - get_rng_state() returns a LocalTensor with per-rank states
    - set_rng_state(LocalTensor) sets per-rank states

    When not in LocalTensor mode, it delegates directly to the underlying device handle.
    """

    def __init__(self, device_handle, device_type: str):
        """
        Initialize the local device handle wrapper.

        Args:
            device_handle: The underlying device module (e.g., torch.cuda)
            device_type: Device type string (e.g., "cuda", "cpu")
        """
        self._device_handle = device_handle
        self._device_type = device_type

    def get_rng_state(self):
        """
        Get RNG state, automatically returning LocalTensor in LocalTensor mode.

        Returns:
            LocalTensor in LocalTensor mode, regular Tensor otherwise
        """
        lm = enabled_local_tensor_mode()
        if not lm:
            return self._device_handle.get_rng_state()

        original_state = _get_rng_state()
        per_rank_states = {}

        try:
            for rank in lm.ranks:
                # We need to set-then-get instead of directly copying lm._per_rank_rng_states[rank]
                # because they have different structures:
                # - lm._per_rank_rng_states[rank] is a tuple: (cpu_state, {device_idx: cuda_state})
                # - self._device_handle.get_rng_state() returns just the device-specific tensor
                # So we temporarily restore the full RNG state (CPU + all CUDA devices) for this rank,
                # then extract only the specific device's state tensor that we need.
                if rank in lm._per_rank_rng_states:
                    _set_rng_state(*lm._per_rank_rng_states[rank])

                per_rank_states[rank] = self._device_handle.get_rng_state()
        finally:
            _set_rng_state(*original_state)

        # pyrefly: ignore [bad-argument-type, bad-argument-count]
        return LocalTensor(per_rank_states)

    def set_rng_state(self, state):
        """
        Set RNG state, automatically handling LocalTensor input.

        Args:
            state: Regular Tensor or LocalTensor with per-rank states
        """
        if isinstance(state, LocalTensor):
            lm = enabled_local_tensor_mode()
            if lm is None:
                raise AssertionError

            # Similar to get_rng_state but in reverse: we need to convert from
            # device-specific tensor format to full state tuple format.
            # - state._local_tensors[rank] contains just the device-specific RNG state tensor
            # - lm._per_rank_rng_states[rank] needs a tuple: (cpu_state, {device_idx: cuda_state})
            # So we set the device's state with the rank-specific tensor, then _get_rng_state()
            # captures both CPU and CUDA states into the tuple format that _per_rank_rng_states expects.
            for rank, rank_state in state._local_tensors.items():
                self._device_handle.set_rng_state(rank_state.to("cpu"))
                lm._per_rank_rng_states[rank] = _get_rng_state()
        else:
            self._device_handle.set_rng_state(state.to("cpu"))

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying device module."""
        return getattr(self._device_handle, name)


class _LocalOffsetBasedRNGTracker:
    """
    LocalTensor-specific RNG tracker for DTensor random operations.

    This class manages per-rank RNG states when running in LocalTensor mode,
    using _LocalPhiloxState to track different offsets for each virtual rank.
    It is instantiated and used by OffsetBasedRNGTracker when in LocalTensor mode.

    Much of this is derived from OffsetBasedRNGTracker:
    https://github.com/pytorch/pytorch/blob/402c46503002f98ccfc023a733081fb0719223a1/torch/distributed/tensor/_random.py#L182
    """

    def __init__(self, device_type: str = "cuda"):
        """Initialize the LocalTensor RNG tracker."""
        from torch.distributed.device_mesh import _get_device_handle

        self._device_type = device_type
        self._device_handle = _LocalDeviceHandle(
            _get_device_handle(device_type), device_type
        )
        self.distribute_region_enabled = True
        self._device_mesh = None

    @property
    def _device(self):
        return torch.device(self._device_type, torch.cuda.current_device())

    def _set_pre_op_offset(self, state, spec) -> None:
        """Compute and set per-rank offsets before the random operation."""
        from torch.distributed.tensor._ops.utils import prod
        from torch.distributed.tensor._utils import (
            _compute_local_shape_and_global_offset,
        )
        from torch.distributed.tensor.placement_types import Shard

        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError

        state._per_rank_offsets = {}

        for rank in lm.ranks:
            # compute this rank's coordinate in the mesh
            mesh_coords = []
            for mesh_dim_idx in range(spec.mesh.ndim):
                mesh_dim_size = spec.mesh.size(mesh_dim_idx)
                # calculate rank's coordinate in this mesh dimension
                num_chunks_after = 1
                for j in range(mesh_dim_idx + 1, spec.mesh.ndim):
                    num_chunks_after *= spec.mesh.size(j)
                coord = (rank // num_chunks_after) % mesh_dim_size
                mesh_coords.append(coord)

            # compute shard offset based on placements
            from torch.distributed.tensor._random import (
                _calc_first_shard_size,
                _calc_shard_info,
                _calc_shard_linear_idx,
            )

            # Compute shard index and total number of shards on each tensor dim
            shard_idx_by_dim, total_num_shards_by_dim = _calc_shard_info(
                mesh_coords, spec
            )

            # compute shard linear index
            shard_linear_idx = _calc_shard_linear_idx(
                shard_idx_by_dim, total_num_shards_by_dim
            )

            # get current offset for this rank
            current_offset = int(
                state._per_rank_states[rank][8:].view(dtype=torch.int64).item()
            )

            local_shape = _calc_first_shard_size(spec)
            # compute local size
            local_size = prod(local_shape)

            # compute new offset (must be multiple of 4)
            offset_incr = (shard_linear_idx * local_size + 3) // 4 * 4
            state._per_rank_offsets[rank] = current_offset + offset_incr

    def _set_post_op_offset(self, state, spec, old_offset) -> None:
        """Set per-rank offsets after the random operation."""
        from torch.distributed.tensor._ops.utils import prod

        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError

        dtensor_shape = spec.shape
        numel = prod(dtensor_shape)
        # offset must be multiple of 4
        numel = (numel + 3) // 4 * 4

        if not hasattr(state, "_per_rank_offsets"):
            state._per_rank_offsets = {}

        # handle LocalIntNode old_offset (different values per rank)
        if isinstance(old_offset, SymInt) and isinstance(old_offset.node, LocalIntNode):
            for rank in lm.ranks:
                rank_old_offset = old_offset.node._local_ints[rank]
                state._per_rank_offsets[rank] = rank_old_offset + numel
        else:
            # same old_offset for all ranks
            old_offset_int = (
                int(old_offset) if isinstance(old_offset, SymInt) else old_offset
            )
            for rank in lm.ranks:
                state._per_rank_offsets[rank] = old_offset_int + numel

    @contextlib.contextmanager
    def _distribute_region(self, spec, generator=None):
        """Context manager for LocalTensor mode distribute region."""
        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError

        # get base state
        if generator is not None:
            base_state_tensor = generator.get_state()
            per_rank_states = {rank: base_state_tensor.clone() for rank in lm.ranks}
            # pyrefly: ignore [bad-argument-type, bad-argument-count]
            base_state_tensor = LocalTensor(per_rank_states)
        else:
            base_state_tensor = self._device_handle.get_rng_state()

        state = _LocalPhiloxState(base_state_tensor)

        if self.distribute_region_enabled:
            # sync to rank 0's state if no explicit generator
            if generator is None:
                any_rank_state = lm._any_local_rng_state()
                any_rank_cpu, any_rank_cuda = any_rank_state

                if self._device.type == "cuda":
                    if self._device.index not in any_rank_cuda:
                        raise AssertionError
                    any_rank_device_state = any_rank_cuda[self._device.index]
                else:
                    any_rank_device_state = any_rank_cpu

                from torch.distributed.tensor._random import _PhiloxState

                any_rank_philox = _PhiloxState(any_rank_device_state)
                state.seed = int(any_rank_philox.seed.item())
                state.offset = int(any_rank_philox.offset.item())

            old_offset = state.offset
            self._set_pre_op_offset(state, spec)
            state.apply_to_local_tensor_mode(self._device_handle)

            try:
                yield
            finally:
                self._set_post_op_offset(state, spec, old_offset)
                state.apply_to_local_tensor_mode(self._device_handle)
        else:
            yield

        # maybe reset generator to rank 0's state
        if generator is not None:
            rank_0_state = state._per_rank_states[0]
            generator.set_state(rank_0_state)


_LOCAL_TENSOR_ATTR_PREFIX = "_local_tensor_"


def _is_local_tensor_attr(attr: str) -> bool:
    return attr.startswith(_LOCAL_TENSOR_ATTR_PREFIX)


def _to_local_tensor_attr(rank: int) -> str:
    return f"{_LOCAL_TENSOR_ATTR_PREFIX}{rank}"


def _from_local_tensor_attr(attr: str) -> int:
    if not _is_local_tensor_attr(attr):
        raise AssertionError(f"Invalid local tensor attr {attr}")
    return int(attr[len(_LOCAL_TENSOR_ATTR_PREFIX) :])


def _all_elements_same(values: list[Any]) -> bool:
    if not values:
        return True
    first_value = values[0]
    return all(value == first_value for value in values)


def _compute_local_tensor_meta(
    local_tensors: dict[int, torch.Tensor],
) -> tuple[
    list[torch.SymInt | int],
    list[torch.SymInt | int],
    torch.device,
    torch.dtype,
    torch.layout,
    DispatchKeySet,
]:
    """
    Computes the meta information for a LocalTensor from its local tensors.
    """
    it = iter(local_tensors.values())
    first_local_tensor = next(it)

    first_shape = first_local_tensor.shape
    first_stride = first_local_tensor.stride()
    dtype = first_local_tensor.dtype
    device = first_local_tensor.device
    layout = first_local_tensor.layout

    extra_dispatch_keys = _get_extra_dispatch_keys(first_local_tensor)

    # Assert that all tensors have the same dtype, layout and dispatch keys. Due
    # to uneven sharding, it is possible that tensors will have different shapes.
    for local_tensor in it:
        if dtype != local_tensor.dtype:
            raise AssertionError(
                "Tensors representing LocalTensor shards must have the same dtype"
            )
        if layout != local_tensor.layout:
            raise AssertionError(
                "Tensors representing LocalTensor shards must have the same layout"
            )
        if extra_dispatch_keys != _get_extra_dispatch_keys(local_tensor):
            raise AssertionError(
                "Tensors representing LocalTensor shards must have the "
                "same set of extra dispatch keys"
            )

    # Compute shape/stride.  We allow for non-SPMD'ness here
    local_shapes: dict[int, dict[int, int]] = defaultdict(dict)  # dim => rank => size
    local_strides: dict[int, dict[int, int]] = defaultdict(dict)  # dim => rank => size
    for r, local_tensor in local_tensors.items():
        for d, size in enumerate(local_tensor.shape):
            local_shapes[d][r] = size
            local_strides[d][r] = local_tensor.stride(d)
    shape = [
        (
            first_shape[d]
            if _all_elements_same(list(local_shapes[d].values()))
            else torch.SymInt(LocalIntNode(local_shapes[d]))
        )
        for d in range(len(first_shape))
    ]
    strides = [
        (
            first_stride[d]
            if _all_elements_same(list(local_strides[d].values()))
            else torch.SymInt(LocalIntNode(local_strides[d]))
        )
        for d in range(len(first_shape))
    ]
    return shape, strides, device, dtype, layout, extra_dispatch_keys


class LocalTensor(torch.Tensor):
    """
    LocalTensor is a Tensor subclass that simulates a tensor distributed across multiple SPMD
    (Single Program, Multiple Data) ranks. Each LocalTensor instance internally holds a mapping from
    global rank ids to their corresponding local Tensor shards.Operations performed on a LocalTensor
    are applied independently to each local shard, mimicking distributed computation. Collectives
    and other distributed operations are handled by mapping them to the local shards as appropriate.

    Note:
        This class is primarily intended for debugging and simulating distributed tensor computations
        on a single process.

    """

    # Map from global rank to the local tensor.
    _local_tensors: dict[int, torch.Tensor]
    # Precomputed for speed set of keys from the local tensor map.
    _ranks: frozenset[int]
    _size: list[torch.SymInt | int]
    __slots__ = ["_local_tensors", "_ranks", "_size"]

    @staticmethod
    @torch._disable_dynamo
    def __new__(
        cls,
        local_tensors: dict[int, torch.Tensor],
        requires_grad: bool = False,
    ) -> "LocalTensor":
        if any(t.requires_grad for t in local_tensors.values()):
            raise AssertionError(
                "Internal local_tensors require grad, but we will ignore those autograd graph. "
                "Make a custom autograd function and make sure you detach the inner tensors."
            )

        if len(local_tensors) == 0:
            raise ValueError("LocalTensor cannot be empty!")

        (shape, strides, device, dtype, layout, extra_dispatch_keys) = (
            _compute_local_tensor_meta(local_tensors)
        )

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=strides,
            dtype=dtype,
            device=device,
            layout=layout,
            # In place ops potentially change local tensor sizes (e.g. resize_). While
            # executing an in-place op the return value must be the same as "self" input
            # otherwise we can introduce errors due to tensor identity changes. Hence we
            # need to be able to update wrapper subclass sizes after in-place ops. This
            # dispatch policy allows us to do that.
            dispatch_sizes_strides_policy="sizes",
            requires_grad=requires_grad,
            _extra_dispatch_keys=extra_dispatch_keys,
        )
        # The wrapper has no real storage (data_ptr()=0). Prevent callers
        # (e.g. Triton kernels) from silently reading the null pointer —
        # turn it into a clear RuntimeError instead of a CUDA IMA.
        torch._C._set_throw_on_mutable_data_ptr(r)

        local_tensors = {
            r: v if not isinstance(v, AsyncCollectiveTensor) else v.wait()
            for r, v in local_tensors.items()
        }
        r._local_tensors = local_tensors
        r._ranks = frozenset(local_tensors.keys())
        r._size = shape
        return r

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    def __deepcopy__(self, memo: dict[Any, Any] | None) -> "LocalTensor":
        local_tensors_copy = {
            r: copy.deepcopy(t, memo) for r, t in self._local_tensors.items()
        }
        # pyrefly: ignore [bad-argument-type, bad-argument-count]
        return LocalTensor(local_tensors_copy, self.requires_grad)

    def __repr__(self) -> str:  # type: ignore[override]
        parts = []
        for k, v in self._local_tensors.items():
            parts.append(f"  {k}: {v}")
        tensors_str = ",\n".join(parts)
        return f"LocalTensor(\n{tensors_str}\n)"

    def __getattr__(self, name: str) -> Any:
        if _is_local_tensor_attr(name):
            rank = _from_local_tensor_attr(name)
            if rank not in self._ranks:
                raise AttributeError(f"Local tensor has no knowledge of rank {rank}")
            return self._local_tensors[rank]
        return object.__getattribute__(self, name)

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        local_tensor_attrs = [_to_local_tensor_attr(r) for r in self._ranks]
        return local_tensor_attrs, ()

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Any],
        flatten_spec: tuple[Any, ...],
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
    ) -> "LocalTensor":
        if flatten_spec is None:
            raise AssertionError(
                "Expecting spec to be not None from `__tensor_flatten__` return value!"
            )
        local_tensors = {
            _from_local_tensor_attr(a): t for a, t in inner_tensors.items()
        }
        # pyrefly: ignore [bad-argument-type, bad-argument-count]
        return LocalTensor(local_tensors)

    @classmethod
    @torch._disable_dynamo
    def __torch_dispatch__(  # type: ignore[override]
        cls,
        func: Any,
        types: tuple[Any, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        # This is horribly inefficient
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        local_tensor = None
        for arg in flat_args:
            if isinstance(arg, LocalTensor):
                local_tensor = arg
                break

        if local_tensor is None:
            raise AssertionError("At least one of the arguments must be a LocalTensor")

        # Check for unrecognized tensor subclasses (but allow regular tensors and scalars)
        has_unrecognized_types = _check_for_subclass(flat_args)
        if has_unrecognized_types:
            unrecognized_types = [
                type(x) for x in flat_args if _check_for_subclass_arg(x)
            ]
            not_implemented_log.debug(
                "LocalTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        with LocalTensorMode(local_tensor._ranks):
            return func(*args, **kwargs)

    def numpy(self, *, force: bool = False) -> Any:
        if HAS_NUMPY:
            return self.reconcile().numpy(force=force)
        else:
            raise RuntimeError("Numpy is not available")

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> torch.Tensor:
        return _LocalContiguous.apply(self, memory_format)

    def is_contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> bool:
        return all(
            t.is_contiguous(memory_format=memory_format)
            for t in self._local_tensors.values()
        )

    def tolist(self) -> list[Any]:
        """
        Try to reconcile, if successful convert to list, otherwise if dtype is integer,
        convert to list of local integers.
        """
        equal_obj = self._equal_local_tensors()
        if isinstance(equal_obj, torch.Tensor):
            return equal_obj.tolist()
        if isinstance(equal_obj, torch.Size):
            if not self.dtype.is_floating_point and not self.dtype.is_complex:
                ranks = sorted(self._ranks)
                local_lists = [self._local_tensors[r].tolist() for r in ranks]
                return _reduce_multidim_lists(
                    local_lists,
                    lambda values: torch.SymInt(
                        LocalIntNode(dict(zip(ranks, values, strict=True)))
                    ),
                )

        raise RuntimeError("Cannot convert local tensor to list")

    def reconcile(self) -> torch.Tensor:
        """
        Reconciles the LocalTensor into a single torch.Tensor by ensuring all local
        shards are identical and returning a detached clone of one of them.

        Note:
            This method is useful for extracting a representative tensor from a LocalTensor
            when all shards are expected to be the same, such as after a collective operation
            that synchronizes all ranks.
        """

        # Force all local tensor shards across ranks to be the same
        equal_obj = self._equal_local_tensors()
        if not isinstance(equal_obj, torch.Tensor):
            raise AssertionError("LocalTensor shards must be the same to reconcile")
        cl = equal_obj.clone().detach()
        cl.requires_grad_(self.requires_grad)
        return cl

    def _equal_local_tensors(self) -> torch.Tensor | torch.Size | None:
        it = iter(self._local_tensors.values())
        t1 = next(it)
        if all(t2.equal(t1) for t2 in it):
            return t1
        if all(t2.shape == t1.shape for t2 in it):
            return t1.shape
        return None

    def _sync_meta(self) -> None:
        with no_dispatch():
            (shape, strides, device, dtype, layout, extra_dispatch_keys) = (
                _compute_local_tensor_meta(self._local_tensors)
            )
            self._size = shape


class _LocalContiguous(torch.autograd.Function):
    """Autograd function for LocalTensor.contiguous() that preserves gradient flow."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        input: LocalTensor,
        memory_format: torch.memory_format,
    ) -> LocalTensor:
        # pyrefly: ignore [bad-argument-type]
        return LocalTensor(
            # pyrefly: ignore [bad-argument-count]
            {
                r: t.contiguous(memory_format=memory_format)
                for r, t in input._local_tensors.items()
            },
            input.requires_grad,
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return grad_output, None


# If set to `True` the LocalTensorMode stack will be created for the whole process,
# otherwise it will be created for each thread.
_PROCESS_MODE: bool = True
_PROCESS_LOCAL_TENSOR_MODE: list["LocalTensorMode"] = []
# When running under local runner each thread must create its own local tensor mode
# so that they do not interfere with each other.
_THREAD_LOCAL_TENSOR_MODE: threading.local = threading.local()


def get_local_tensor_mode_list() -> list["LocalTensorMode"]:
    global _PROCESS_MODE
    if _PROCESS_MODE:
        global _PROCESS_LOCAL_TENSOR_MODE
        return _PROCESS_LOCAL_TENSOR_MODE
    global _THREAD_LOCAL_TENSOR_MODE
    if not hasattr(_THREAD_LOCAL_TENSOR_MODE, "value"):
        _THREAD_LOCAL_TENSOR_MODE.value = []
    return _THREAD_LOCAL_TENSOR_MODE.value


# These methods are patched from DeviceMesh to the _LocalDeviceMesh versions.
_PATCHED_DEVICE_MESH_METHODS: Sequence[str] = (
    "get_coordinate",
    "get_local_rank",
    "get_rank",
    "_is_current_rank_part_of_mesh",
    "_sym_get_coordinate",
)

# These random functions are also patched.
_PATCHED_RANDOM_FUNCTIONS: Sequence[tuple[str, str]] = (
    ("torch.random.manual_seed", "torch_manual_seed"),
    ("torch.manual_seed", "torch_manual_seed"),
    ("torch.random.initial_seed", "torch_initial_seed"),
    ("torch.initial_seed", "torch_initial_seed"),
)


class LocalTensorMode(TorchDispatchMode):
    """
    A TorchDispatchMode that simulates SPMD (Single Program, Multiple Data) execution
    for LocalTensor objects across a set of ranks.

    LocalTensorMode enables PyTorch operations to be transparently applied to each
    local shard of a LocalTensor, as if they were distributed across multiple ranks.
    When active, this mode intercepts tensor operations and dispatches them to each
    rank's local tensor, collecting and wrapping the results as LocalTensors. It also
    handles collective operations by mapping them to local implementations.

    This mode is primarily intended for debugging and simulating distributed tensor
    computations on a single process, rather than for high-performance distributed
    training. It maintains a stack of active modes, patches DeviceMesh coordinate
    resolution, and provides utilities for temporarily disabling the mode or mapping
    functions over ranks.
    """

    # What ranks this local tensor mode is operating over
    def __init__(self, ranks: int | frozenset[int]):
        if isinstance(ranks, int):
            # assume is world size
            self.ranks = frozenset(range(ranks))
        else:
            if not isinstance(ranks, frozenset):
                raise AssertionError
            self.ranks = ranks
        self._disable = True
        # Used to store the patched DeviceMesh methods
        self._old_device_mesh_methods: dict[str, Callable[..., object]] | None = None
        # Used to store the patched "random" functions
        self._old_random_functions: dict[str, Callable[..., object]] = {}
        self._per_rank_rng_states: dict[
            int, tuple[torch.Tensor, dict[int, torch.Tensor]]
        ] = {}
        # Cache for get_coordinate results, keyed by mesh id
        # Protected by _coordinate_cache_lock for thread safety in MPMD contexts
        self._coordinate_cache: dict[int, list[SymInt]] = {}
        self._coordinate_cache_lock = threading.Lock()

    def __enter__(self) -> "LocalTensorMode":
        get_local_tensor_mode_list().append(self)
        self.enable_()

        # _distribute_region will compute correct per-shard offsets
        # but we want all ranks to start with the same state
        if not _is_in_fake_tensor_mode():
            cpu_state, cuda_states = _get_rng_state()
            for rank in self.ranks:
                self._per_rank_rng_states[rank] = (
                    cpu_state.clone(),
                    {idx: state.clone() for idx, state in cuda_states.items()},
                )

        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        local_tensor_mode_list = get_local_tensor_mode_list()
        local_tensor_mode_list.pop()
        self.disable_()
        if len(local_tensor_mode_list) > 0:
            if local_tensor_mode_list[-1]._disable:
                local_tensor_mode_list[-1].disable_()
            else:
                local_tensor_mode_list[-1].enable_()
        super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[Any, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        # Find all LocalTensor arguments to determine ranks
        local_tensors = [a for a in flat_args if isinstance(a, LocalTensor)]

        # Check for unrecognized tensor subclasses (but allow regular tensors and scalars)
        has_unrecognized_types = _check_for_subclass(flat_args)
        if has_unrecognized_types:
            unrecognized_types = [
                type(x) for x in flat_args if _check_for_subclass_arg(x)
            ]
            not_implemented_log.debug(
                "LocalTensorMode unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        # Factory functions convert into LocalTensor, so we don't have to
        # transmute a Tensor into a LocalTensor if mutation happens...
        # But if you do an operation on a Tensor, do NOT wrap it into a
        # LocalTensor.  This helps prevent accidents when you're doing Tensor
        # operations on the inner non-wrapped tensors.
        if not local_tensors:
            if self._disable or any(isinstance(a, Tensor) for a in flat_args):
                return func(*args, **kwargs)

        # For LocalTensors, verify they have compatible ranks
        for a in flat_args:
            if isinstance(a, LocalTensor):
                if a._ranks > self.ranks:
                    raise AssertionError(
                        f"Input LocalTensor {a} must be configured for a "
                        f"subset of the LocalTensorMode ranks {self.ranks}"
                    )

        if func.overloadpacket == torch.ops.aten.dim:
            return len(args[0]._size)
        if func.overloadpacket == torch.ops.aten.sym_size:
            return tuple(args[0]._size)

        if func.namespace == "c10d":
            if func is torch.ops.c10d.allreduce_.default:
                return _c10d._local_all_reduce_(*args, **kwargs)
            elif func is torch.ops.c10d.allreduce_coalesced_.default:
                return _c10d._local_allreduce_coalesced_(*args, **kwargs)
            elif func is torch.ops.c10d.reduce_scatter_tensor_coalesced_.default:
                return _c10d._local_reduce_scatter_tensor_coalesced_(*args, **kwargs)
            elif func is torch.ops.c10d.scatter_.default:
                return _c10d._local_scatter_(*args, **kwargs)
            elif func is torch.ops.c10d.broadcast_.default:
                return _c10d._local_broadcast_(*args, **kwargs)
            elif func is torch.ops.c10d.allgather_.default:
                return _c10d._local_all_gather_(*args, **kwargs)
            elif func is torch.ops.c10d.allgather_into_tensor_coalesced_.default:
                return _c10d._local_allgather_into_tensor_coalesced_(*args, **kwargs)
            elif func is torch.ops.c10d._allgather_base_.default:
                return _c10d._local_allgather_base_(*args, **kwargs)
            elif func is torch.ops.c10d._reduce_scatter_base_.default:
                return _c10d._local_reduce_scatter_base_(*args, **kwargs)
            elif func is torch.ops.c10d.gather_.default:
                return _c10d._local_gather_(*args, **kwargs)
            elif func is torch.ops.c10d.alltoall_.default:
                return _c10d._local_alltoall_(*args, **kwargs)
            elif func is torch.ops.c10d.alltoall_base_.default:
                return _c10d._local_alltoall_base_(*args, **kwargs)
            elif func is torch.ops.c10d.barrier.default:
                return _c10d._local_barrier(*args, **kwargs)
            elif func is torch.ops.c10d.monitored_barrier_.default:
                return _c10d._local_monitored_barrier_(*args, **kwargs)
            elif func is torch.ops.c10d.send.default:
                return _c10d._local_send(*args, **kwargs)
            elif func is torch.ops.c10d.recv_.default:
                return _c10d._local_recv_(*args, **kwargs)
            elif func is torch.ops.c10d.recv_any_source_.default:
                return _c10d._local_recv_any_source_(*args, **kwargs)
            raise NotImplementedError(f"{func} not implemented")

        if func.namespace == "_c10d_functional" or func.namespace == "_dtensor":
            if func is torch.ops._dtensor.shard_dim_alltoall.default:
                return _c10d._local_functional_shard_dim_alltoall(*args, **kwargs)
            elif func is torch.ops._c10d_functional.all_gather_into_tensor.default:
                return _c10d._local_functional_all_gather_into_tensor(*args, **kwargs)
            elif func is torch.ops._c10d_functional.reduce_scatter_tensor.default:
                return _c10d._local_functional_reduce_scatter_tensor(*args, **kwargs)
            elif func is torch.ops._c10d_functional.all_to_all_single.default:
                return _c10d._local_functional_all_to_all_single(*args, **kwargs)
            else:
                with LocalTensorMode(self.ranks):
                    return func._op_dk(
                        DispatchKey.CompositeExplicitAutograd, *args, **kwargs
                    )

        if func.namespace == "profiler":
            return func(*args, **kwargs)

        if func.namespace == "_c10d_functional_autograd":
            raise NotImplementedError(f"{func} not implemented")

        if func.namespace == "symm_mem":
            raise NotImplementedError(f"{func} not implemented")

        return _for_each_rank_run_func(func, self.ranks, args, kwargs, alias=True)

    def disable_(self):
        if self._disable:
            return

        self._unpatch_device_mesh()
        self._unpatch_random_functions()
        self._disable = True

    def enable_(self):
        if not self._disable:
            return

        self._patch_device_mesh()
        self._patch_random_functions()
        self._disable = False

    @contextlib.contextmanager
    def disable(self) -> Generator[None, None, None]:
        """
        Disables LocalTensorMode temporarily. Primarily is intended to be used to perform
        rank specific computations and merge results back before enabling LocalTensorMode back.
        """

        # don't unpatch again if already disabled
        if self._disable:
            try:
                yield
            finally:
                # re-disable if the yield messed
                # with the state
                self.disable_()
            return

        self.disable_()
        try:
            yield
        finally:
            self.enable_()

    def rank_map(self, cb: Callable[[int], Tensor]) -> LocalTensor:
        """
        Creates a LocalTensor instance by mapping rank id to ids local shard.
        """

        with self.disable():
            # pyrefly: ignore [bad-argument-type, bad-argument-count]
            return LocalTensor({r: cb(r) for r in self.ranks})

    def tensor_map(
        self, tensor: LocalTensor, cb: Callable[[int, Tensor], Tensor | None]
    ) -> LocalTensor:
        """
        Creates a LocalTensor instance by mapping rank id to ids local shard.
        """

        with self.disable():
            results = {}
            for r in self.ranks:
                if r in tensor._local_tensors:
                    m = cb(r, tensor._local_tensors[r])
                    if m is not None:
                        results[r] = m
            # pyrefly: ignore [bad-argument-type, bad-argument-count]
            return LocalTensor(results)

    def _any_local_rng_state(self) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        return self._per_rank_rng_states[next(iter(self.ranks))]

    def _patch_device_mesh(self) -> None:
        if self._old_device_mesh_methods is not None:
            raise AssertionError
        saved = {}
        for name in _PATCHED_DEVICE_MESH_METHODS:
            saved[name] = getattr(DeviceMesh, name)
            local = getattr(_LocalDeviceMesh, name)
            setattr(DeviceMesh, name, local)
        self._old_device_mesh_methods = saved

    def _unpatch_device_mesh(self) -> None:
        saved, self._old_device_mesh_methods = self._old_device_mesh_methods, None
        if saved is None:
            raise AssertionError
        for name, value in saved.items():
            setattr(DeviceMesh, name, value)

    def _patch_random_functions(self) -> None:
        # TODO: This should either be removed or documented why it's necessary.
        from torch.distributed.tensor import _random as dtensor_random

        for global_name, local_name in _PATCHED_RANDOM_FUNCTIONS:
            if global_name in self._old_random_functions:
                continue
            mod_name, attr_name = global_name.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            old = getattr(mod, attr_name)
            local = getattr(_LocalRandom, local_name)
            setattr(mod, attr_name, local)
            self._old_random_functions[global_name] = old

    def _unpatch_random_functions(self) -> None:
        # TODO: This should either be removed or documented why it's necessary.
        from torch.distributed.tensor import _random as dtensor_random

        for global_name, local_name in _PATCHED_RANDOM_FUNCTIONS:
            value = self._old_random_functions.pop(global_name, None)
            if value is not None:
                mod_name, attr_name = global_name.rsplit(".", 1)
                mod = importlib.import_module(mod_name)
                setattr(mod, attr_name, value)


class _LocalRandom:
    """
    Holds implementations of random functionality that must be patched while running
    under LocalTensorMode.
    """

    @staticmethod
    def torch_manual_seed(seed) -> torch._C.Generator:
        """LocalTensor-aware version of torch.random.manual_seed."""
        if (
            (lm := enabled_local_tensor_mode())
            and isinstance(seed, torch.SymInt)
            and isinstance(seed.node, LocalIntNode)
        ):
            from torch.random import _manual_seed_impl

            for rank in sorted(lm.ranks):
                rank_seed = seed.node._local_ints[rank]
                _manual_seed_impl(rank_seed)
                lm._per_rank_rng_states[rank] = _get_rng_state()
            return torch.random.default_generator
        from torch.random import _manual_seed_impl

        result = _manual_seed_impl(seed)

        if lm is not None and len(lm._per_rank_rng_states) > 0:
            cpu_state, cuda_states = _get_rng_state()
            for rank in lm.ranks:
                lm._per_rank_rng_states[rank] = (
                    cpu_state.clone(),
                    {idx: state.clone() for idx, state in cuda_states.items()},
                )

        return result

    @staticmethod
    def torch_initial_seed():
        """LocalTensor-aware version of torch.random.initial_seed."""
        if lm := enabled_local_tensor_mode():
            if len(lm._per_rank_rng_states) == 0:
                return torch.random.default_generator.initial_seed()
            rank_seeds = {}

            for rank in sorted(lm.ranks):
                _set_rng_state(*lm._per_rank_rng_states[rank])
                rank_seeds[rank] = torch.random.default_generator.initial_seed()

            local_int_node = LocalIntNode(rank_seeds)
            return torch.SymInt(local_int_node)

        return torch.random.default_generator.initial_seed()


# Save the original get_coordinate method before any patching


class _LocalDeviceMesh:
    """
    Holds implementations of DeviceMesh functionality that must be patched while running
    under LocalTensorMode.
    """

    @staticmethod
    def get_coordinate(self: DeviceMesh) -> list[SymInt] | None:
        # NB: In order to support submeshes the code below recreates for each
        # rank submesh with the same mesh dimensions as current mesh. We are
        # doing this because when submesh is created it is created for a particular
        # rank (therefore below we are patching get_rank method). We are trying to
        # limit the invasiveness of local tensor.
        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError("Unexpectedly not in LocalTensorMode")

        # Check cache first (fast path without lock)
        mesh_id = id(self)
        if mesh_id in lm._coordinate_cache:
            return lm._coordinate_cache[mesh_id]

        # Acquire lock for thread safety in MPMD contexts
        with lm._coordinate_cache_lock:
            # Double-check after acquiring lock
            if mesh_id in lm._coordinate_cache:
                return lm._coordinate_cache[mesh_id]

            coords: list[dict[int, int]] = [{} for _ in range(self.ndim)]
            # Clone rank_map to avoid "Cannot set version_counter for inference tensor"
            # error when running under torch.inference_mode()
            rank_map = self._rank_map.clone()
            for r in lm.ranks:
                rank_tensor = self._layout.remap_to_tensor(rank_map)
                rank_coords = (rank_tensor == r).nonzero().tolist()
                if len(rank_coords) != 1:
                    raise AssertionError
                for d, c in enumerate(rank_coords[0][1:]):
                    coords[d][r] = c

            out = [torch.SymInt(LocalIntNode(c)) for c in coords]
            # Cache the result
            lm._coordinate_cache[mesh_id] = out
            # The output contains coordinates for each of the ranks with respect to
            # their meshes formed from root mesh and selecting the same dimensions
            # as the current mesh.
            return out  # type: ignore[return-value]

    @staticmethod
    def _is_current_rank_part_of_mesh(self: DeviceMesh) -> bool:
        my_coordinate = self.get_coordinate()
        return my_coordinate is not None

    @staticmethod
    def _sym_get_coordinate(self: DeviceMesh, index: int) -> int:
        my_coordinate = self.get_coordinate()
        if my_coordinate is None:
            raise AssertionError
        return my_coordinate[index]

    @staticmethod
    def get_rank(self) -> int | SymInt:
        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError("Unexpectedly not in LocalTensorMode")
        return torch.SymInt(LocalIntNode(local_ints={r: r for r in lm.ranks}))

    @staticmethod
    def get_local_rank(self, mesh_dim: int | str | None = None) -> int | SymInt:
        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError("Unexpectedly not in LocalTensorMode")

        if self.ndim > 1 and mesh_dim is None:
            raise RuntimeError(
                f"Found the DeviceMesh have {self.ndim} dimensions",
                "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
            )
        elif mesh_dim is None:
            mesh_dim = 0

        if isinstance(mesh_dim, str):
            mesh_dim = self._mesh_dim_names.index(mesh_dim)

        # Compute local rank for each global rank
        # get_coordinate returns a list of SymInt, one per mesh dimension
        # We need to extract the coordinate for the specified mesh_dim
        coords = _LocalDeviceMesh.get_coordinate(self)
        if coords is None:
            raise AssertionError
        return coords[mesh_dim]


def reconcile_args(args: Any, kwargs: dict[str, Any] | None = None) -> Any:
    """
    Reconciles arguments by converting any LocalTensor instances in the input
    arguments to their underlying torch.Tensor representation.

    This function is typically used to prepare arguments for functions that
    expect standard torch.Tensor objects, by flattening the input arguments,
    replacing LocalTensor instances with their reconciled (standard tensor)
    versions, and then reconstructing the original argument structure.

    Args:
        args: Positional arguments, possibly containing LocalTensor instances.
        kwargs: Keyword arguments, possibly containing LocalTensor instances.

    Returns:
        Any: The arguments with all LocalTensor instances replaced by their reconciled torch.Tensor equivalents,
             preserving the original structure.
    """
    if kwargs is None:
        kwargs = {}
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    reconciled_args = [
        a.reconcile() if isinstance(a, LocalTensor) else a for a in flat_args
    ]
    return pytree.tree_unflatten(reconciled_args, args_spec)


def local_tensor_mode() -> LocalTensorMode | None:
    """
    Returns the current active LocalTensorMode if one exists.

    This function checks the global stack of LocalTensorMode instance. If there
    is at least one LocalTensorMode active, it returns the most recently entered
    (top of the stack) LocalTensorMode. If no LocalTensorMode is active, it returns None.

    Returns:
        Optional[LocalTensorMode]: The current LocalTensorMode if active, else None.
    """
    local_tensor_mode_list = get_local_tensor_mode_list()
    if len(local_tensor_mode_list) > 0:
        return local_tensor_mode_list[-1]
    return None


def enabled_local_tensor_mode() -> LocalTensorMode | None:
    """
    Returns the current active LocalTensorMode only if it's enabled.

    This is a convenience function that combines the common pattern of checking
    if local_tensor_mode() is not None and not disabled.

    Returns:
        Optional[LocalTensorMode]: The current LocalTensorMode if active and enabled, else None.
    """
    lm = local_tensor_mode()
    if lm is not None and not lm._disable:
        return lm
    return None


def maybe_run_for_local_tensor(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Decorator that ensures a function is executed for each local tensor shard
    when running under LocalTensorMode. If not in LocalTensorMode, the function
    is executed normally. When in LocalTensorMode, the function is run for each
    rank, and the results are collected appropriately.

    This decorator is useful for functions that exhibit non-SPMD behavior, such
    as those requiring rank specific actions. For example, a function that computes
    offset into input tensor based on rank.

    Note that the function being decorated must not have any side effects and
    contain operations for a single rank only. For example, wrapping a function
    that performs a collective operation will not work.

    Args:
        func (Callable[..., Any]): The function to be decorated.

    Returns:
        Callable[..., Any]: The wrapped function that handles LocalTensorMode logic.
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not (lm := enabled_local_tensor_mode()):
            return func(*args, **kwargs)
        ret = None
        with lm.disable():
            ret = _for_each_rank_run_func(func, lm.ranks, args, kwargs, alias=False)

        return ret

    return wrapper


def rank_map(cb: Callable[[int], Tensor]) -> Tensor:
    """
    Creates a tensor by mapping a callback over the current rank.

    Under LocalTensorMode, calls cb(rank) for each simulated rank and returns
    a LocalTensor. In real distributed (no LocalTensorMode), calls
    cb(dist.get_rank()) and returns a plain Tensor.
    """
    lm = enabled_local_tensor_mode()
    if lm is not None:
        return lm.rank_map(cb)
    else:
        return cb(dist.get_rank())


def tensor_map(tensor: Tensor, cb: Callable[[int, Tensor], Tensor | None]) -> Tensor:
    """
    Transforms a tensor by mapping a callback over the current rank and its
    local shard.

    Under LocalTensorMode, calls cb(rank, shard) for each simulated rank and
    returns a LocalTensor. In real distributed (no LocalTensorMode), calls
    cb(dist.get_rank(), tensor) and returns the result directly.
    """
    lm = enabled_local_tensor_mode()
    if lm is not None:
        assert isinstance(tensor, LocalTensor)
        return lm.tensor_map(tensor, cb)
    else:
        r = cb(dist.get_rank(), tensor)
        assert r is not None
        return r


def maybe_disable_local_tensor_mode() -> contextlib.AbstractContextManager:
    """
    Context manager that disables LocalTensorMode for the duration of the context.
    """
    lm = local_tensor_mode()
    return lm.disable() if lm is not None else contextlib.nullcontext()


def maybe_enable_local_tracker(
    device_type: str, distribute_region_enabled: bool, spec, generator
):
    """
    Returns a context manager for LocalTensor-mode RNG tracking if local tensor mode is enabled.

    Args:
        device_type: The device type (e.g., "cuda", "cpu")
        distribute_region_enabled: Whether distribute region is enabled
        spec: The DTensorSpec
        generator: Optional torch.Generator

    Returns:
        Context manager from local_tracker._distribute_region if local tensor mode is enabled,
        otherwise None.
    """
    if enabled_local_tensor_mode():
        local_tracker = _LocalOffsetBasedRNGTracker(device_type)
        local_tracker.distribute_region_enabled = distribute_region_enabled
        return local_tracker._distribute_region(spec, generator)

    return None


def get_generator_seed_for_device_type(device_type: str):
    """
    Gets the generator seed for a specific device type, handling LocalTensor mode appropriately.

    Args:
        device_type: The device type (e.g., "cuda", "cpu")

    Returns:
        If in LocalTensor mode with per-rank RNG states:
            - Returns int if all ranks have the same seed
            - Returns SymInt(LocalIntNode) if ranks have different seeds
        Otherwise:
              - Returns int seed from the device's RNG state
    """
    if lm := enabled_local_tensor_mode():
        if len(lm._per_rank_rng_states) == 0:
            device_module = torch.get_device_module(device_type)
            return device_module.get_rng_state()[:8].view(torch.int64).item()
        device_module = torch.get_device_module(device_type)

        original_state = _get_rng_state()

        rank_seeds = {}
        try:
            for rank in sorted(lm.ranks):
                _set_rng_state(*lm._per_rank_rng_states[rank])
                rank_seeds[rank] = int(
                    device_module.get_rng_state()[:8].view(torch.int64).item()
                )
        finally:
            # restore original state
            _set_rng_state(*original_state)

        unique_seeds = set(rank_seeds.values())
        if len(unique_seeds) == 1:
            return next(iter(unique_seeds))
        local_int_node = LocalIntNode(rank_seeds)
        return torch.SymInt(local_int_node)
    else:
        device_module = torch.get_device_module(device_type)
        return device_module.get_rng_state()[:8].view(torch.int64).item()


import threading
from queue import Queue


_LOCAL_RUNNER_MODE: "LocalRunnerMode | None" = None


class _ExceptionRaisingThread(threading.Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None
    ):
        super().__init__(
            target=target, name=name, args=args, kwargs=kwargs, daemon=daemon
        )
        self.exception: BaseException | None = None

    def run(self):
        try:
            super().run()
        except BaseException as e:  # noqa: B036
            self.exception = e

    def join(self, timeout=None):
        super().join(timeout=timeout)
        if self.exception:
            raise self.exception


class LocalRunnerMode:
    """
    A class for running multiple SPMD functions concurrently, however at any point
    in time only one function can be running. The main use case for the local runner
    mode is to enable SPMD functions to be able to use send and recv to communicate
    with each other. Without local runner mode send and recv are not supported.
    """

    runner_context = threading.local()

    def __init__(
        self, ranks: frozenset[int] | int, concurrency: int, fn: Callable[[int], None]
    ):
        if isinstance(ranks, int):
            ranks = frozenset(range(ranks))
        self._ranks = ranks
        self._fn = fn
        self._run_lock = threading.Lock()
        self._run_id = -1
        self._run_cond = threading.Condition(self._run_lock)

        self._recv_objects: dict[int, dict[int, Queue]] = {
            dst: {src: Queue() for src in ranks} for dst in ranks
        }
        self._runners = [
            _ExceptionRaisingThread(target=self._run, args=(i,), name="LocalRunnerMode")
            for i in range(concurrency)
        ]
        self._process_mode = True

    def __enter__(self) -> "LocalRunnerMode":
        global _LOCAL_RUNNER_MODE
        if _LOCAL_RUNNER_MODE is not None:
            raise AssertionError("LocalRunnerMode is already running")
        _LOCAL_RUNNER_MODE = self

        global _PROCESS_MODE
        self._process_mode = _PROCESS_MODE
        _PROCESS_MODE = False
        for r in self._runners:
            r.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        for r in self._runners:
            r.join()
        global _LOCAL_RUNNER_MODE
        _LOCAL_RUNNER_MODE = None

        global _PROCESS_MODE
        _PROCESS_MODE = self._process_mode

    def _run(self, id: int) -> None:
        LocalRunnerMode.runner_context.id = id
        # Only one thread can run at a time, hence must acquire the lock
        try:
            self._acquire_run_lock()
            self._fn(id)
        finally:
            self._release_run_lock()

    def _acquire_run_lock(self) -> None:
        self._run_lock.acquire()
        self._run_id = LocalRunnerMode.runner_context.id

    def _release_run_lock(self) -> None:
        self._run_id = -1
        self._run_lock.release()

    def _assert_holds_run_lock(self) -> None:
        if self._run_id != LocalRunnerMode.runner_context.id:
            raise AssertionError("Calling thread does not hold the run lock")

    def _get_recv_object(self, src: int, dst: int) -> object | None:
        peers = [src] if src != -1 else list(self._ranks)
        recv_objects = self._recv_objects[dst]

        for p in peers:
            if not recv_objects[p].empty():
                return recv_objects[p].get()

        return None

    def _signal_send(self, src: int, dst: int, obj: object) -> None:
        if obj is None:
            raise AssertionError("Cannot signal None")
        # Only a single thread a time executes so it is safe to mutate
        # read objects queue (executing thread is already holding the lock)
        self._recv_objects[dst][src].put(obj)
        # Signal directly condition variable since the calling thread is already
        # holding the lock
        self._run_cond.notify_all()

    def _wait_recv(self, src: int, dst: int, post: Callable[[object], None]) -> None:
        # Wait for the object to be available
        while True:
            obj = self._get_recv_object(src, dst)
            if obj is not None:
                post(obj)
                # Note that we are not releasing the lock here, since the thread
                # will continue to run and therefore must hold the lock
                return
            self._run_cond.wait()

    @staticmethod
    def current() -> "LocalRunnerMode":
        global _LOCAL_RUNNER_MODE
        if _LOCAL_RUNNER_MODE is None:
            raise AssertionError("LocalRunnerMode is not enabled")
        return _LOCAL_RUNNER_MODE


class _LocalPhiloxState:
    """
    LocalTensor-aware version of _PhiloxState that manages per-rank RNG states.
    This class handles the case where the generator state is a LocalTensor, allowing
    different offsets and seeds for different virtual ranks.

    Note: This is designed to be used as a drop-in replacement for _PhiloxState
    when working with LocalTensors in the DTensor random ops implementation.
    """

    def __init__(self, state: torch.Tensor):
        if not isinstance(state, LocalTensor):
            raise AssertionError("_LocalPhiloxState requires a LocalTensor")
        self._local_tensor = state
        self._per_rank_states = {
            rank: local_state.to("cpu")
            for rank, local_state in state._local_tensors.items()
        }

    @property
    def state(self):
        return LocalTensor(self._per_rank_states)  # type: ignore[name-defined]

    @property
    def offset(self) -> int | SymInt:
        from torch.distributed.tensor._random import _PhiloxState

        offsets = {}
        for rank, state in self._per_rank_states.items():
            rank_philox = _PhiloxState(state)
            offsets[rank] = int(rank_philox.offset.item())

        if len(set(offsets.values())) == 1:
            return next(iter(offsets.values()))

        return SymInt(LocalIntNode(offsets))

    @offset.setter
    def offset(self, offset: int | SymInt) -> None:
        from torch.distributed.tensor._random import _PhiloxState

        if isinstance(offset, SymInt) and isinstance(offset.node, LocalIntNode):
            for rank, state in self._per_rank_states.items():
                rank_offset = offset.node._local_ints[rank]
                rank_philox = _PhiloxState(state)
                rank_philox.offset = torch.tensor([rank_offset], dtype=torch.int64)
        else:
            offset_int = int(offset) if isinstance(offset, SymInt) else offset
            offset_tensor = torch.tensor([offset_int], dtype=torch.int64)
            for state in self._per_rank_states.values():
                rank_philox = _PhiloxState(state)
                rank_philox.offset = offset_tensor

    @property
    def seed(self) -> int | SymInt:
        from torch.distributed.tensor._random import _PhiloxState

        seeds = {}
        for rank, state in self._per_rank_states.items():
            rank_philox = _PhiloxState(state)
            seeds[rank] = int(rank_philox.seed.item())

        if len(set(seeds.values())) == 1:
            return next(iter(seeds.values()))
        return SymInt(LocalIntNode(seeds))

    @seed.setter
    def seed(self, seed: int | SymInt) -> None:
        from torch.distributed.tensor._random import _PhiloxState

        if isinstance(seed, SymInt) and isinstance(seed.node, LocalIntNode):
            for rank, state in self._per_rank_states.items():
                rank_seed = seed.node._local_ints[rank]
                rank_philox = _PhiloxState(state)
                rank_philox.seed = torch.tensor([rank_seed], dtype=torch.int64)
        else:
            seed_int = int(seed) if isinstance(seed, SymInt) else seed
            seed_tensor = torch.tensor([seed_int], dtype=torch.int64)
            for state in self._per_rank_states.values():
                rank_philox = _PhiloxState(state)
                rank_philox.seed = seed_tensor

    def apply_to_local_tensor_mode(self, device_handle) -> None:
        """
        Apply per-rank RNG states to the LocalTensorMode's tracked states.
        This updates both the device RNG state and the LocalTensorMode's _per_rank_rng_states.

        Args:
            device_handle: The device handle to use for setting RNG state (_LocalDeviceHandle)
        """
        if not enabled_local_tensor_mode():
            return

        if not hasattr(self, "_per_rank_offsets"):
            raise AssertionError

        for rank in sorted(self._per_rank_states.keys()):
            offset_value = self._per_rank_offsets[rank]
            if isinstance(offset_value, SymInt):
                if isinstance(offset_value.node, LocalIntNode):
                    offset_value = offset_value.node._local_ints[rank]
                else:
                    offset_value = int(offset_value)

            offset_tensor = torch.tensor(
                [offset_value], dtype=torch.uint64, device="cpu"
            ).view(torch.uint8)
            self._per_rank_states[rank][8:] = offset_tensor

        # pyrefly: ignore [bad-argument-type, bad-argument-count]
        device_handle.set_rng_state(LocalTensor(self._per_rank_states))
