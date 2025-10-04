from ast import Call


"""
A LocalTensor is a tensor subclass which simulates a tensor that is
distributed across SPMD ranks.  A LocalTensor might be size N, but in fact
there are world_size shards/replicas of it stored internally.  When you do a
plain PyTorch operation on it, we apply the operation to each shard; when you
do a collective, we the mathematically equivalent operation on the local
shards.  A LocalTensor is associated with a list of ranks which specify
which ranks it holds local tensors for.

NB, this is NOT a DataParallel like abstraction where you can run operations
on multiple different GPUs. It is intended purely for *debugging* purposes,
the overhead is almost certainly too high to keep eight GPUs (even the C++
autograd needs multithreading to keep up!)  (It might potentially be possible
to trace through this with torch.compile and then compile it with CUDA graphs
but this is currently a non-goal.)

We do not directly handle MPMD, as in this regime you would have to be able to
have predicated operators which only apply on a subset of ranks.  Instead, you
should generate multiple LocalTensors with disjoint lists of ranks.  However,
in this model, it still seems unavoidable that you would have to modify your
MPMD code to run all of the stages so that you actually have all the compute
for all the stages.

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

Our plan is to assume that the original list of ranks is arange and reverse engineer
the Layout corresponding to the participating ranks, with respect to the global
world size.
"""

import contextlib
import functools
import operator
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Callable, Generator, Optional, Union

import torch
from torch import Size, SymBool, SymInt, Tensor
from torch._C import DispatchKey, DispatchKeySet
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.fx.experimental._constant_symnode import ConstantIntNode
from torch.nested._internal.nested_int import NestedIntNode
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode
from torch.utils.checkpoint import get_device_states, set_device_states


not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")


from . import _c10d


def _int_on_rank(i: "LocalIntNode | ConstantIntNode", r: int) -> int:
    if isinstance(i, LocalIntNode):
        return i._local_ints[r]
    elif isinstance(i, ConstantIntNode):
        return i.val
    else:
        raise AssertionError(type(i))


def _check_for_subclass(flat_args: Sequence[object]) -> bool:
    return any(_check_for_subclass_arg(x) for x in flat_args)


def _check_for_subclass_arg(x: object) -> bool:
    return (
        not isinstance(x, LocalTensor)
        and isinstance(x, Tensor)
        and type(x) is not Tensor
        and type(x) is not torch.nn.Parameter
    )


def _map_to_rank_local_val(val: Any, rank: int) -> Any:
    if isinstance(val, LocalTensor):
        return val._local_tensors[rank]
    if isinstance(val, SymInt) and isinstance(val.node, LocalIntNode):
        return val.node._local_ints[rank]
    return val


def _for_each_rank_run_func(
    func: Callable[..., Any],
    ranks: frozenset[int],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    alias: bool = True,
) -> Any:
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))

    cpu_state = torch.get_rng_state()
    devices, states = get_device_states((args, kwargs))

    flat_rank_rets = {}

    for r in sorted(ranks):
        torch.set_rng_state(cpu_state)
        set_device_states(devices, states)
        rank_flat_args = [_map_to_rank_local_val(a, r) for a in flat_args]
        rank_args, rank_kwargs = pytree.tree_unflatten(rank_flat_args, args_spec)
        rank_ret = func(*rank_args, **rank_kwargs)
        flat_rank_rets[r] = rank_ret

    # Create the LocalTensors
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
        # Single non-tensor return value (scalar, etc.)
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

    # def str(self) -> str:
    #     return self._str()

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
        self, other: "LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {
                r: max(self._local_ints[r], _int_on_rank(other, r))
                for r in self._local_ints
            }
        )

    def add(
        self, other: "LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] + _int_on_rank(other, r) for r in self._local_ints}
        )

    def sub(
        self, other: "LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] - _int_on_rank(other, r) for r in self._local_ints}
        )

    def mul(
        self, other: "LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] * _int_on_rank(other, r) for r in self._local_ints}
        )

    def eq(self, other: "LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] == _int_on_rank(other, r) for r in self._local_ints}
        return torch._C._get_constant_bool_symnode(len(r) == 1 and next(iter(r)))

    def gt(self, other: "LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] > _int_on_rank(other, r) for r in self._local_ints}
        assert len(r) == 1, (self, other)
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def lt(self, other: "LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] < _int_on_rank(other, r) for r in self._local_ints}
        assert len(r) == 1, (self, other)
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def wrap_int(self, num: int) -> "LocalIntNode | ConstantIntNode":
        return ConstantIntNode(num)


class LocalTensor(torch.Tensor):
    # Map from global rank to the local tensor
    # This is a dict because in an MPMD situation, you will only have a subset
    # of ranks that this tensor actually represents (the other ranks may be
    # doing something different)
    _local_tensors: dict[int, torch.Tensor]
    _ranks: frozenset[int]  # precomputed for speed
    __slots__ = ["_local_tensors"]

    @staticmethod
    @torch._disable_dynamo
    def __new__(
        cls,
        local_tensors: dict[int, torch.Tensor],
    ) -> "LocalTensor":
        """
        .. note:: This is not a public API and it's only supposed to be used by the
            operator implementations and internals.

            TODO: guidance on alternate implementations
        """
        if any(t.requires_grad for t in local_tensors.values()):
            raise AssertionError(
                "Internal local_tensors require grad, but we will ignore those autograd graph. "
                "Make a custom autograd function and make sure you detach the inner tensors."
            )

        # Assert that all tensors are consistently the same.  I think maybe we
        # could handle size imbalance but you need to prevent code OUTSIDE of
        # LocalTensor from branching on these differences (and in general you
        # have a problem which is what size do you advertise?  It needs to be
        # some sort of unbacked-like thing).

        it = iter(local_tensors.items())
        first_rank, first_local_tensor = next(it)

        first_shape = first_local_tensor.shape
        first_stride = first_local_tensor.stride()
        ndim = len(first_shape)
        dtype = first_local_tensor.dtype
        device = first_local_tensor.device
        layout = first_local_tensor.layout

        # TODO: make this less horrible
        def get_extra_dispatch_keys(t: torch.Tensor) -> DispatchKeySet:
            extra_dispatch_keys = torch._C.DispatchKeySet.from_raw_repr(0)
            if torch._C._dispatch_keys(t).has(torch._C.DispatchKey.Conjugate):
                extra_dispatch_keys = extra_dispatch_keys.add(
                    torch._C.DispatchKey.Conjugate
                )
            if torch._C._dispatch_keys(t).has(torch._C.DispatchKey.Negative):
                extra_dispatch_keys = extra_dispatch_keys.add(
                    torch._C.DispatchKey.Negative
                )
            return extra_dispatch_keys

        extra_dispatch_keys = get_extra_dispatch_keys(first_local_tensor)

        for _rank, local_tensor in it:
            assert dtype == local_tensor.dtype
            assert layout == local_tensor.layout
            assert extra_dispatch_keys == get_extra_dispatch_keys(local_tensor)

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
            parts.append(f"  {k}: {v}")
        tensors_str = ",\n".join(parts)
        return f"LocalTensor(\n{tensors_str}\n)"

    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        return ["_local_tensors"], ()

    def tolist(self) -> list[Any]:
        return self.reconcile().tolist()

    def reconcile(self) -> torch.Tensor:
        # Force all tensors across ranks to be the same
        it = iter(self._local_tensors.values())
        t1 = next(it)
        for t2 in it:
            assert torch.equal(t1, t2)
        cl = t1.clone().detach()
        cl.requires_grad_(self.requires_grad)
        return cl

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        assert flatten_spec is not None, (
            "Expecting spec to be not None from `__tensor_flatten__` return value!"
        )
        local_tensors = inner_tensors["_local_tensors"]
        return LocalTensor(local_tensors)

    @classmethod
    @torch._disable_dynamo
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # This is horribly inefficient
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        local_tensor = None
        for arg in flat_args:
            if isinstance(arg, LocalTensor):
                local_tensor = arg
                break

        assert local_tensor is not None

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


_LOCAL_TENSOR_MODE: list["LocalTensorMode"] = []


class LocalTensorMode(TorchDispatchMode):
    # Used to temporarily disable local tensor mode
    _disable: bool

    # What ranks this local tensor mode is operating over
    def __init__(self, ranks: Union[int, frozenset[int]]):
        if isinstance(ranks, int):
            # assume is world size
            self.ranks = frozenset(range(ranks))
        else:
            assert isinstance(ranks, frozenset)
            self.ranks = ranks
        self._disable = False

    def __enter__(self) -> "LocalTensorMode":
        self._disable = False
        _LOCAL_TENSOR_MODE.append(self)

        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._disable = True
        _LOCAL_TENSOR_MODE.pop()
        super().__exit__(exc_type, exc_val, exc_tb)

    @contextlib.contextmanager
    def disable(self) -> Generator[None, None, None]:
        old = self._disable
        self._disable = True
        try:
            yield
        finally:
            self._disable = old

    def rank_map(self, cb: Callable[[int], Tensor]) -> LocalTensor:
        with self.disable():
            return LocalTensor({r: cb(r) for r in self.ranks})

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return self._dispatch(func, types, args, kwargs)

    def _dispatch(self, func, types, args=(), kwargs=None):
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
                assert a._ranks == self.ranks

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

        if func.namespace == "_c10d_functional_autograd":
            raise NotImplementedError(f"{func} not implemented")

        if func.namespace == "symm_mem":
            raise NotImplementedError(f"{func} not implemented")

        return _for_each_rank_run_func(func, self.ranks, args, kwargs, alias=True)


def reconcile_args(args: Any, kwargs: dict[str, Any] | None = None):
    if kwargs is None:
        kwargs = {}
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    reconciled_args = [
        a.reconcile() if isinstance(a, LocalTensor) else a for a in flat_args
    ]
    return pytree.tree_unflatten(reconciled_args, args_spec)


def local_tensor_mode() -> Optional[LocalTensorMode]:
    if len(_LOCAL_TENSOR_MODE) > 0:
        return _LOCAL_TENSOR_MODE[-1]
    return None


def maybe_run_for_local_tensor(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        lm = local_tensor_mode()
        if lm is None:
            return func(*args, **kwargs)
        ret = None
        with lm.disable():
            ret = _for_each_rank_run_func(func, lm.ranks, args, kwargs, alias=False)

        lm = local_tensor_mode()
        assert lm is not None
        return ret

    return wrapper
