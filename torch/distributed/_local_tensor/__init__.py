from ast import Call


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
import functools
import operator
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Callable, Generator, Optional, Union

import numpy as np

import torch
from torch import Size, SymBool, SymInt, Tensor
from torch._C import DispatchKey, DispatchKeySet, ScriptObject
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch.distributed import DeviceMesh, ProcessGroup
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.distributed_c10d import _get_default_group
from torch.fx.experimental._constant_symnode import ConstantIntNode
from torch.nested._internal.nested_int import NestedIntNode
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode
from torch.utils.checkpoint import get_device_states, set_device_states


not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")


from . import _c10d


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
        and type(x) not in (Tensor, torch.nn.Parameter, torch.nn.Buffer)
    )


def _map_to_rank_local_val(val: Any, rank: int) -> Any:
    if isinstance(val, LocalTensor):
        return val._local_tensors[rank]
    if isinstance(val, SymInt) and isinstance(val.node, LocalIntNode):
        return val.node._local_ints[rank]
    return val


def collect_cuda_rng_states() -> list[torch.Tensor]:
    """
    Collects RNG state from all available CUDA devices.

    Returns:
        List of RNG state tensors, one for each CUDA device.
        Returns empty list if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return []

    num_devices = torch.cuda.device_count()
    rng_states = []

    for device_idx in range(num_devices):
        with torch.cuda.device(device_idx):
            rng_state = torch.cuda.get_rng_state()
            rng_states.append(rng_state)

    return rng_states


def set_cuda_rng_states(rng_states: list[torch.Tensor]) -> None:
    """
    Sets RNG state for all CUDA devices from a list of states.

    Args:
        rng_states: List of RNG state tensors to restore.
    """
    if not torch.cuda.is_available():
        return

    num_devices = min(len(rng_states), torch.cuda.device_count())

    for device_idx in range(num_devices):
        with torch.cuda.device(device_idx):
            torch.cuda.set_rng_state(rng_states[device_idx])


def _get_rng_state() -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Gets CPU and CUDA rng states from all devices.
    """
    return (torch.get_rng_state(), collect_cuda_rng_states())


def _set_rng_state(cpu_state: torch.Tensor, cuda_states: list[torch.Tensor]) -> None:
    """
    Sets CPU and CUDA rng states for all devices. If the list of cuda states
    is shorter than the number of devices only the first len(cuda_states) devices
    will get their rng state set.
    """
    torch.set_rng_state(cpu_state)
    set_cuda_rng_states(cuda_states)


def _for_each_rank_run_func(
    func: Callable[..., Any],
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

    # NB: Before invoking an op we are collecting rng states from CPU and
    # CUDA devices such that we can reset to the same before invoking op
    # for each rank. This is not very efficient and will likely be revisited
    # to support per rank rng state.
    rng_state = _get_rng_state()
    flat_rank_rets = {}

    for r in sorted(ranks):
        _set_rng_state(*rng_state)
        rank_flat_args = [_map_to_rank_local_val(a, r) for a in flat_args]
        rank_args, rank_kwargs = pytree.tree_unflatten(rank_flat_args, args_spec)
        rank_ret = func(*rank_args, **rank_kwargs)
        flat_rank_rets[r] = rank_ret

    rr_key = next(iter(flat_rank_rets.keys()))
    rr_val = flat_rank_rets[rr_key]

    if isinstance(rr_val, Tensor):
        ret = LocalTensor({r: flat_rank_rets[r] for r in sorted(ranks)})
    elif isinstance(rr_val, (list, tuple)):
        ret_list = []
        for i in range(len(rr_val)):
            rets = {r: flat_rank_rets[r][i] for r in sorted(ranks)}
            v_it = iter(rets.values())
            v = next(v_it)
            if isinstance(v, Tensor):
                ret_list.append(LocalTensor(rets))
            elif isinstance(v, int) and not all(v == v2 for v2 in v_it):
                ret_list.append(torch.SymInt(LocalIntNode(rets)))
            else:
                assert all(v == v2 for v2 in v_it)
                ret_list.append(v)
        ret = type(rr_val)(ret_list)
    else:
        v_it = iter(flat_rank_rets.values())
        v = next(v_it)
        if all(v == v2 for v2 in v_it):
            return v
        if isinstance(v, int):
            return torch.SymInt(LocalIntNode(flat_rank_rets))
        raise AssertionError(f"Unexpected return type {type(v)}")

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

    def maybe_as_int(self) -> Optional[int]:
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

    def gt(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] > _int_on_rank(other, r) for r in self._local_ints}
        assert len(r) == 1, (self, other)
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def lt(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] < _int_on_rank(other, r) for r in self._local_ints}
        assert len(r) == 1, (self, other)
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def wrap_int(self, num: int) -> "LocalIntNode | ConstantIntNode":
        return ConstantIntNode(num)


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
    __slots__ = ["_local_tensors", "_ranks"]

    @staticmethod
    @torch._disable_dynamo
    def __new__(
        cls,
        local_tensors: dict[int, torch.Tensor],
    ) -> "LocalTensor":
        if any(t.requires_grad for t in local_tensors.values()):
            raise AssertionError(
                "Internal local_tensors require grad, but we will ignore those autograd graph. "
                "Make a custom autograd function and make sure you detach the inner tensors."
            )

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
            assert dtype == local_tensor.dtype, (
                "Tensors representing LocalTensor shards must have the same dtype"
            )
            assert layout == local_tensor.layout, (
                "Tensors representing LocalTensor shards must have the same layout"
            )
            assert extra_dispatch_keys == _get_extra_dispatch_keys(local_tensor), (
                "Tensors representing LocalTensor shards must have the same set of extra dispatch keys"
            )

        # Compute shape/stride.  We allow for non-SPMD'ness here
        local_shapes: dict[int, dict[int, int]] = defaultdict(
            dict
        )  # dim => rank => size
        local_strides: dict[int, dict[int, int]] = defaultdict(
            dict
        )  # dim => rank => size
        for r, local_tensor in local_tensors.items():
            for d, size in enumerate(local_tensor.shape):
                local_shapes[d][r] = size
                local_strides[d][r] = local_tensor.stride(d)
        shape = [
            (
                first_shape[d]
                if len(set(local_shapes[d])) == 1
                else torch.SymInt(LocalIntNode(local_shapes[d]))
            )
            for d in range(len(first_shape))
        ]
        strides = [
            (
                first_stride[d]
                if len(set(local_strides[d])) == 1
                else torch.SymInt(LocalIntNode(local_strides[d]))
            )
            for d in range(len(first_shape))
        ]

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=strides,
            dtype=dtype,
            device=device,
            layout=layout,
            requires_grad=False,
            _extra_dispatch_keys=extra_dispatch_keys,
        )

        local_tensors = {
            r: v if not isinstance(v, AsyncCollectiveTensor) else v.wait()
            for r, v in local_tensors.items()
        }
        r._local_tensors = local_tensors
        r._ranks = frozenset(local_tensors.keys())
        return r

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    def __repr__(self) -> str:  # type: ignore[override]
        parts = []
        for k, v in self._local_tensors.items():
            # pyrefly: ignore  # bad-argument-type
            parts.append(f"  {k}: {v}")
        tensors_str = ",\n".join(parts)
        return f"LocalTensor(\n{tensors_str}\n)"

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        return ["_local_tensors"], ()

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Any],
        flatten_spec: tuple[Any, ...],
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
    ) -> "LocalTensor":
        assert flatten_spec is not None, (
            "Expecting spec to be not None from `__tensor_flatten__` return value!"
        )
        local_tensors = inner_tensors["_local_tensors"]
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

        assert local_tensor is not None, (
            "At least one of the arguments must be a LocalTensor"
        )

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

    def numpy(self, *, force: bool = False) -> np.ndarray:
        return self.reconcile().numpy(force=force)

    def __lt__(
        self, other: torch.Tensor | bool | complex | float | int
    ) -> torch.Tensor:
        self_rec = self.reconcile()
        other_rec = other
        if isinstance(other, LocalTensor):
            other_rec = other.reconcile()
        return self_rec < other_rec

    def __gt__(
        self, other: torch.Tensor | bool | complex | float | int
    ) -> torch.Tensor:
        self_rec = self.reconcile()
        other_rec = other
        if isinstance(other, LocalTensor):
            other_rec = other.reconcile()
        return self_rec > other_rec

    def tolist(self) -> list[Any]:
        """
        Reconcile and convert result to list.
        """

        return self.reconcile().tolist()

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
        it = iter(self._local_tensors.values())
        t1 = next(it)
        for t2 in it:
            assert torch.equal(t1, t2), (
                "LocalTensor shards must be the same to reconcile"
            )
        cl = t1.clone().detach()
        cl.requires_grad_(self.requires_grad)
        return cl


_LOCAL_TENSOR_MODE: list["LocalTensorMode"] = []


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
    def __init__(self, ranks: Union[int, frozenset[int]]):
        if isinstance(ranks, int):
            # assume is world size
            self.ranks = frozenset(range(ranks))
        else:
            assert isinstance(ranks, frozenset)
            self.ranks = ranks
        self._disable = False
        self._old_get_coordinate = None

    def __enter__(self) -> "LocalTensorMode":
        self._disable = False
        self._patch_device_mesh()
        _LOCAL_TENSOR_MODE.append(self)

        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._disable = True
        self._unpatch_device_mesh()
        _LOCAL_TENSOR_MODE.pop()
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
                assert a._ranks <= self.ranks, (
                    f"Input LocalTensor {a} and LocalTensorMode must be configured for the same ranks"
                )

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

    @contextlib.contextmanager
    def disable(self) -> Generator[None, None, None]:
        """
        Disables LocalTensorMode temporarily. Primarily is intended to be used to perform
        rank specific computations and merge results back before enabling LocalTensorMode back.
        """

        old = self._disable
        self._disable = True
        self._unpatch_device_mesh()
        try:
            yield
        finally:
            self._disable = old
            self._patch_device_mesh()

    def rank_map(self, cb: Callable[[int], Tensor]) -> LocalTensor:
        """
        Creates a LocalTensor instance by mapping rank id to ids local shard.
        """

        with self.disable():
            return LocalTensor({r: cb(r) for r in self.ranks})

    def _patch_device_mesh(self) -> None:
        assert self._old_get_coordinate is None
        self._old_get_coordinate = DeviceMesh.get_coordinate  # type: ignore[assignment]
        DeviceMesh.get_coordinate = _LocalDeviceMesh.get_coordinate  # type: ignore[method-assign]

    def _unpatch_device_mesh(self) -> None:
        assert self._old_get_coordinate is not None
        DeviceMesh.get_coordinate = self._old_get_coordinate
        # pyrefly: ignore  # bad-assignment
        self._old_get_coordinate = None


class _LocalDeviceMesh:
    """
    Holds implementations of DeviceMesh functionality that must be patched while running
    under LocalTensorMode.
    """

    @staticmethod
    def get_coordinate(self: DeviceMesh) -> Optional[list[int] | None]:
        # NB: In order to support submeshes the code below recreates for each
        # rank submesh with the same mesh dimensions as current mesh. We are
        # doing this because when submesh is created it is created for a particular
        # rank (therefore below we are patching get_rank method). We are trying to
        # limit the invasiveness of local tensor.
        lm = local_tensor_mode()
        assert lm is not None, "Unexpectedly not in LocalTensorMode"

        coords: list[dict[int, int]] = [{} for _ in range(self.ndim)]
        for r in lm.ranks:
            rank_tensor = self._layout.remap_to_tensor(self._rank_map)
            rank_coords = (rank_tensor == r).nonzero().tolist()
            assert len(rank_coords) == 1
            for d, c in enumerate(rank_coords[0][1:]):
                coords[d][r] = c

        out = [torch.SymInt(LocalIntNode(c)) for c in coords]
        # The output contains coordinates for each of the ranks with respect to
        # their meshes formed from root mesh and selecting the same dimensions
        # as the current mesh.
        return out  # type: ignore[return-value]


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


def local_tensor_mode() -> Optional[LocalTensorMode]:
    """
    Returns the current active LocalTensorMode if one exists.

    This function checks the global stack of LocalTensorMode instance. If there
    is at least one LocalTensorMode active, it returns the most recently entered
    (top of the stack) LocalTensorMode. If no LocalTensorMode is active, it returns None.

    Returns:
        Optional[LocalTensorMode]: The current LocalTensorMode if active, else None.
    """
    if len(_LOCAL_TENSOR_MODE) > 0:
        return _LOCAL_TENSOR_MODE[-1]
    return None


def maybe_run_for_local_tensor(func: Callable[..., Any]) -> Callable[..., Any]:
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
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        lm = local_tensor_mode()
        if lm is None:
            return func(*args, **kwargs)
        ret = None
        with lm.disable():
            ret = _for_each_rank_run_func(func, lm.ranks, args, kwargs, alias=False)

        return ret

    return wrapper
