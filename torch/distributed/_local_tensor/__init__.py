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

from __future__ import annotations

import contextlib
import functools
import operator
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from itertools import product
from typing import Optional, Union

import torch
from torch import Tensor, SymInt
from torch._C import DispatchKey
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch.distributed._distributed_c10d import FakeWork
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.distributed_c10d import ProcessGroup, ReduceOp, Work
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode_stack,
    return_and_correct_aliasing,
    TorchDispatchMode,
)
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.fx.experimental._constant_symnode import ConstantIntNode


not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")


from torch.distributed._pycute.int_tuple import flatten, is_int, is_tuple
from torch.distributed._pycute.layout import complement, Layout

from . import _c10d


_LOCAL_TENSOR_MODE = None


# TODO: this claude code implementation sucks, redo it
def indices_to_layout(indices):
    """
    Convert a sorted list of indices to a pycute Layout using the mathematical
    approach based on the admissible for complement property.

    Args:
        indices: A sorted list of integers starting from 0

    Returns:
        Layout: A pycute Layout that generates the given indices
    """
    if not indices:
        return Layout(0, 0)

    if len(indices) == 1:
        if indices[0] == 0:
            return Layout(1, 1)
        else:
            raise ValueError("Single index must be 0")

    strides = []
    shapes = []
    remaining = set(indices)

    # Always start with stride 1
    current_stride = 1
    max_iterations = len(indices)  # Safety limit
    iteration = 0

    while remaining and iteration < max_iterations:
        iteration += 1

        # Count consecutive multiples of current_stride starting from 0
        size = 0
        while size * current_stride in remaining:
            size += 1

        if size > 0:
            # Found a valid dimension - remove all multiples
            for i in range(size):
                remaining.discard(i * current_stride)
            strides.append(current_stride)
            shapes.append(size)

            # Calculate next stride using admissible property
            current_stride = current_stride * size
        else:
            # No pattern from 0, jump to minimum remaining element
            if remaining:
                current_stride = min(remaining)
            else:
                break

    if iteration >= max_iterations:
        raise RuntimeError(
            f"Algorithm did not converge after {max_iterations} iterations"
        )

    # Convert to proper format for Layout
    if len(shapes) == 1:
        return Layout(shapes[0], strides[0])
    else:
        return Layout(tuple(shapes), tuple(strides))


def _has_local_tensor(*args):
    """Check if any arguments contain LocalTensor."""
    for arg in args:
        if isinstance(arg, LocalTensor):
            return True
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, LocalTensor):
                    return True
    return False


# NB: lifted from https://github.com/pytorch/pytorch/pull/161016
def layout_to_indices(layout):
    return [
        sum(c * s for c, s in zip(coord, flatten(layout.stride)))
        for coord in product(*(range(s) for s in flatten(layout.shape)))
    ]


def int_on_rank(i, r):
    if isinstance(i, LocalIntNode):
        return i._local_ints[r]
    elif isinstance(i, ConstantIntNode):
        return i.val
    else:
        assert False, type(i)


class LocalIntNode:
    """
    Like a LocalTensor, but for an int.  We can't use a 0D tensor to represent this
    because often only a SymInt is accepted where we wish to use this.
    """

    def __new__(cls, local_ints: dict[int, int]):
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

    def clone(self) -> NestedIntNode:
        return self

    def _str(self) -> str:
        return f"LocalIntNode({self._local_ints})"

    def str(self) -> str:
        return self._str()

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

    def add(self, other) -> LocalIntNode:
        return LocalIntNode(
            {r: self._local_ints[r] + int_on_rank(other, r) for r in self._local_ints}
        )

    def sub(self, other) -> LocalIntNode:
        return LocalIntNode(
            {r: self._local_ints[r] - int_on_rank(other, r) for r in self._local_ints}
        )

    def mul(self, other) -> LocalIntNode:
        return LocalIntNode(
            {r: self._local_ints[r] * int_on_rank(other, r) for r in self._local_ints}
        )

    def eq(self, other) -> bool:
        r = {self._local_ints[r] == int_on_rank(other, r) for r in self._local_ints}
        assert len(r) == 1, (self, other)
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def gt(self, other) -> bool:
        r = {self._local_ints[r] > int_on_rank(other, r) for r in self._local_ints}
        assert len(r) == 1, (self, other)
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def wrap_int(self, num: int) -> LocalIntNode:
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
    ) -> LocalTensor:
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
        first_strides = first_local_tensor.stride()
        ndim = len(first_shape)
        dtype = first_local_tensor.dtype
        device = first_local_tensor.device
        layout = first_local_tensor.layout

        # TODO: make this less horrible
        def get_extra_dispatch_keys(t):
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
        local_shapes = defaultdict(dict)  # dim => rank => size
        local_strides = defaultdict(dict)  # dim => rank => size
        for r, local_tensor in local_tensors.items():
            for d, size in enumerate(local_tensor.shape):
                local_shapes[d][r] = size
                local_strides[d][r] = local_tensor.stride(d)
        shape = [
            first_shape[d]
            if len(set(local_shapes[d])) == 1
            else torch.SymInt(LocalIntNode(local_shapes[d]))
            for d in range(len(first_shape))
        ]
        strides = [
            first_stride[d]
            if len(set(local_strides[d])) == 1
            else torch.SymInt(LocalIntNode(local_strides[d]))
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

        assert not any(
            isinstance(v, AsyncCollectiveTensor) for v in local_tensors.values()
        )
        r._local_tensors = local_tensors
        r._ranks = frozenset(local_tensors.keys())
        return r

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __repr__(self):
        parts = []
        for k, v in self._local_tensors.items():
            parts.append(f"  {k}: {v}")
        tensors_str = ",\n".join(parts)
        return f"LocalTensor(\n{tensors_str}\n)"

    def tolist(self):
        # Force result to be SPMD
        it = iter(self._local_tensors.items())
        rank, t = next(it)
        for rank2, t2 in it:
            assert torch.equal(t, t2)
        return t.tolist()

    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        return ["_local_tensors"], ()

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


def _check_for_subclass(flat_args: Sequence[object]) -> bool:
    return any(_check_for_subclass_arg(x) for x in flat_args)


def _check_for_subclass_arg(x: object) -> bool:
    return (
        not isinstance(x, LocalTensor)
        and isinstance(x, Tensor)
        and type(x) is not Tensor
        and type(x) is not torch.nn.Parameter
    )


class LocalTensorMode(TorchDispatchMode):
    _disable: bool

    # What ranks this local tensor mode is operating over
    def __init__(self, ranks: Union[int, frozenset[int]]):
        if isinstance(ranks, int):
            self.ranks = frozenset(range(ranks))  # assume is world size
        else:
            assert isinstance(ranks, frozenset)
            self.ranks = ranks
        self._disable = False

        global _LOCAL_TENSOR_MODE
        _LOCAL_TENSOR_MODE = self

    def __enter__(self):
        super().__enter__()

    @contextlib.contextmanager
    def disable(self):
        old = self._disable
        self._disable = True
        try:
            yield
        finally:
            self._disable = old

    def rank_map(self, cb):
        with self.disable():
            return LocalTensor({r: cb(r) for r in self.ranks})

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # print(func, args, kwargs)
        ret = self._dispatch(func, types, args, kwargs)
        # print("----")
        # print(func, args, kwargs, "-->", ret)
        return ret

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
            elif func is torch.ops.c10d.broadcast_.default:
                return _c10d._local_broadcast_(*args, **kwargs)
            elif func is torch.ops.c10d.allgather_.default:
                return _c10d._local_all_gather_(*args, **kwargs)
            elif func is torch.ops.c10d.scatter_.default:
                return _c10d._local_scatter_(*args, **kwargs)
            elif func is torch.ops.c10d.allgather_into_tensor_coalesced_.default:
                return _c10d._local_allgather_into_tensor_coalesced_(*args, **kwargs)
            elif func is torch.ops.c10d.allreduce_coalesced_.default:
                return _c10d._local_allreduce_coalesced_(*args, **kwargs)
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
            with self:
                # la la la
                return func._op_dk(
                    DispatchKey.CompositeExplicitAutograd, *args, **kwargs
                )

        if func.namespace == "_c10d_functional_autograd":
            raise NotImplementedError(f"{func} not implemented")

        if func.namespace == "symm_mem":
            raise NotImplementedError(f"{func} not implemented")

        flat_rank_rets = {}
        # Handle RNG

        cpu_state = torch.get_rng_state()
        devices, states = get_device_states((args, kwargs))
        for r in sorted(self.ranks):
            torch.set_rng_state(cpu_state)
            set_device_states(devices, states)
            rank_flat_args = [
                a._local_tensors[r] if isinstance(a, LocalTensor) else a
                for a in flat_args
            ]
            rank_args, rank_kwargs = pytree.tree_unflatten(rank_flat_args, args_spec)
            rank_ret = func(*rank_args, **rank_kwargs)
            flat_rank_rets[r] = rank_ret

        # Create the LocalTensors
        rr_key = next(iter(flat_rank_rets.keys()))
        rr_val = flat_rank_rets[rr_key]

        if isinstance(rr_val, Tensor):
            ret = LocalTensor({r: flat_rank_rets[r] for r in sorted(self.ranks)})
        elif isinstance(rr_val, (list, tuple)):
            ret = []
            for i in range(len(rr_val)):
                rets = {r: flat_rank_rets[r][i] for r in sorted(self.ranks)}
                v_it = iter(rets.values())
                v = next(v_it)
                if isinstance(v, Tensor):
                    ret.append(LocalTensor(rets))
                else:
                    assert all(v == v2 for v2 in v_it)
                    ret.append(v)

            ret = tuple(ret) if isinstance(rr_val, tuple) else ret
        else:
            # Single non-tensor return value (scalar, etc.)
            v_it = iter(flat_rank_rets.values())
            v = next(v_it)
            assert all(v == v2 for v2 in v_it)
            return v

        return return_and_correct_aliasing(func, args, kwargs, ret)


def list_tensor_index(xs: list[Tensor], i: torch.SymInt) -> LocalTensor:
    if isinstance(i, int):
        return xs[i]

    assert isinstance(i, torch.SymInt) and isinstance(i.node, LocalIntNode)
    ret = {}
    for r, idx in i.node._local_ints.items():
        cand = xs[idx]
        # TODO: not sure if we need to support non-LocalTensor case
        ret[r] = cand._local_tensors[r] if isinstance(cand, LocalTensor) else cand
    return LocalTensor(ret)


def maybe_unpad_tensor(pad_sizes: list[int], mesh_dim_local_rank: SymInt, output: Tensor, dim: int):
    from torch.distributed.tensor._collective_utils import (
        unpad_tensor,
    )

    if not isinstance(output, LocalTensor):
        assert type(mesh_dim_local_rank) is int
        return unpad_tensor(output, self.dim, pad_sizes[mesh_dim_local_rank]).contiguous()

    assert isinstance(mesh_dim_local_rank, SymInt)

    ret = {}
    for r in output._ranks:
        if pad_sizes[mesh_dim_local_rank.node._local_ints[r]] > 0:
            ret[r] = unpad_tensor(output._local_tensors[r], dim, pad_sizes[mesh_dim_local_rank.node._local_ints[r]]).contiguous()
        else:
            ret[r] = output._local_tensors[r]

    r = LocalTensor(ret)
    print(r)
    print(r.shape)
    return r


def local_tensor_mode() -> Optional[LocalTensorMode]:
    return _LOCAL_TENSOR_MODE
