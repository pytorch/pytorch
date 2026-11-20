"""What a pass over a graph is given, and what it can say with it: the types of the values of the graph, and the
guards a rewrite is valid under."""

import dataclasses
import sys
from typing import Any

import torch


# Under this name in the ``meta`` of a graph: the guards a pass brought.
_GUARDS = "torch.vm.guards"
# Under this name in the ``meta`` of a node: its type, where the graph was
# made from a run on real tensors.
_TYPE = "torch.vm.type"


@dataclasses.dataclass(frozen=True)
class Interval:
    """The integers from ``lo`` to ``hi``, both included. A size that has no
    upper bound has ``sys.maxsize`` as its ``hi``."""

    lo: int
    hi: int

    def __contains__(self, k: object) -> bool:
        return isinstance(k, int) and self.lo <= k <= self.hi

    def samples(self, n: int) -> list[int]:
        """Up to ``n`` numbers of the interval: ``lo``, ``hi``, and evenly
        spaced ones between them."""
        if self.hi >= sys.maxsize:
            raise ValueError(f"{self!r} has no upper end to take samples up to")
        if n <= 1 or self.lo == self.hi:
            return [self.lo]
        step = (self.hi - self.lo) / (n - 1)
        return sorted({self.lo + round(i * step) for i in range(n)})

    def __repr__(self) -> str:
        if self.lo == self.hi:
            return str(self.lo)
        return f"{self.lo}.." + ("" if self.hi >= sys.maxsize else str(self.hi))


class Type:
    """What a value of a graph is like on every call that the graph is run
    for."""


@dataclasses.dataclass(frozen=True, repr=False)
class TensorType(Type):
    """The type of a tensor: its dtype, device and ``requires_grad``, which a
    call is checked for before it gets the graph, and per dimension the sizes
    the graph is for: one number, or an :class:`Interval` where the graph
    leaves the size open."""

    dtype: torch.dtype
    device: torch.device
    requires_grad: bool
    dims: tuple[Interval, ...]

    def dim(self, i: int) -> Interval:
        """The sizes of dimension ``i``."""
        return self.dims[i]

    def nbytes(self, at: str = "hi") -> int:
        """The bytes of a tensor of this type, with every size at its lower
        (``"lo"``) or upper (``"hi"``) end."""
        count = 1
        for dim in self.dims:
            count *= dim.lo if at == "lo" else dim.hi
        return count * self.dtype.itemsize

    def as_checks(self) -> "list[Check]":
        """The type as guards on the tensor it describes: its dtype, its
        device, and the sizes of every dimension."""
        checks: list[Check] = [Dtype(self.dtype), Device(self.device)]
        return checks + [Dim(i, dim) for i, dim in enumerate(self.dims)]

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, torch.Tensor) or value.dim() != len(self.dims):
            return False
        same = value.dtype == self.dtype and value.device == self.device
        return same and all(size in dim for size, dim in zip(value.shape, self.dims))

    def __repr__(self) -> str:
        dims = ", ".join(repr(dim) for dim in self.dims)
        return f"{str(self.dtype).removeprefix('torch.')}[{dims}]"


@dataclasses.dataclass(frozen=True)
class IntType(Type):
    """The type of an integer the graph takes: the values it may have."""

    range: Interval


def type_of_value(value: Any) -> Type | None:
    """The type of a value of a call: of a tensor, with the sizes it has; None
    for what is neither a tensor nor an integer."""
    if isinstance(value, torch.Tensor):
        dims = tuple(Interval(size, size) for size in value.shape)
        return TensorType(value.dtype, value.device, value.requires_grad, dims)
    if isinstance(value, int) and not isinstance(value, bool):
        return IntType(Interval(value, value))
    return None


def type_of(gm: torch.fx.GraphModule, input_types: list[Any], node: Any) -> Type | None:
    """The type of a node of the graph a pass was given: of an input, the one
    in ``input_types``; of an intermediate, what it was in the call the graph
    was made from. None for a node a pass added."""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if node in placeholders and placeholders.index(node) < len(input_types):
        return input_types[placeholders.index(node)]
    return node.meta.get(_TYPE)


def materialize(
    t: TensorType, *, at: str | None = None, dim0: int | None = None
) -> torch.Tensor:
    """Random data of type ``t``, for a pass to try a kernel on: with every
    open size at its lower (``at="lo"``, the default) or upper (``"hi"``) end,
    and the first size ``dim0`` if that is given. It never runs the graph."""
    sizes = [dim.hi if at == "hi" else dim.lo for dim in t.dims]
    if dim0 is not None:
        if dim0 not in t.dim(0):
            raise ValueError(f"{dim0} is no size of dimension 0 of {t!r}")
        sizes[0] = dim0
    if any(size >= sys.maxsize for size in sizes):
        raise ValueError(f"{t!r} has a size without an upper end: say which with dim0=")
    if t.dtype.is_floating_point or t.dtype.is_complex:
        return torch.randn(sizes, dtype=t.dtype, device=t.device)
    if t.dtype == torch.bool:
        return torch.rand(sizes, device=t.device) < 0.5
    return torch.randint(0, 100, sizes, dtype=t.dtype, device=t.device)


class Check:
    """What a guard asks of a tensor. ``holds(tensor)`` looks at the tensor's
    metadata only."""

    def holds(self, tensor: torch.Tensor) -> bool:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class Dtype(Check):
    """The tensor has this dtype."""

    dtype: torch.dtype

    def holds(self, tensor: torch.Tensor) -> bool:
        return tensor.dtype == self.dtype


@dataclasses.dataclass(frozen=True)
class Device(Check):
    """The tensor is on this device."""

    device: torch.device

    def holds(self, tensor: torch.Tensor) -> bool:
        return tensor.device == torch.device(self.device)


@dataclasses.dataclass(frozen=True)
class Dim(Check):
    """Dimension ``i`` has one of the ``allowed`` sizes: a ``range``, a set,
    or an :class:`Interval`."""

    i: int
    allowed: Any

    def holds(self, tensor: torch.Tensor) -> bool:
        return (
            -tensor.dim() <= self.i < tensor.dim()
            and tensor.shape[self.i] in self.allowed
        )


@dataclasses.dataclass(frozen=True)
class Stride(Check):
    """Dimension ``i`` has one of the ``allowed`` strides."""

    i: int
    allowed: Any

    def holds(self, tensor: torch.Tensor) -> bool:
        return (
            -tensor.dim() <= self.i < tensor.dim()
            and tensor.stride(self.i) in self.allowed
        )


@dataclasses.dataclass(frozen=True)
class Contiguous(Check):
    """The tensor is contiguous."""

    def holds(self, tensor: torch.Tensor) -> bool:
        return tensor.is_contiguous()


@dataclasses.dataclass(frozen=True)
class Aligned(Check):
    """The memory of the tensor starts at a multiple of ``nbytes``."""

    nbytes: int

    def holds(self, tensor: torch.Tensor) -> bool:
        return tensor.data_ptr() % self.nbytes == 0


def guard(gm: torch.fx.GraphModule, node: Any, *checks: Check) -> Any:
    """Says that what the pass does to ``gm`` is only right for calls where
    ``node`` passes ``checks``: a call that does not runs the code the pass
    was given, not the graph. ``node`` is an input of the graph (a
    placeholder). The guards are kept with the graph, in ``gm.meta``, and
    checked before every call of what is made of it. Returns ``node``."""
    if node.op != "placeholder":
        raise ValueError(
            f"a guard is on an input of the graph, and {node.name} is computed in it: "
            "guard the inputs it is computed from"
        )
    for check in checks:
        if not isinstance(check, Check):
            raise TypeError(
                f"a guard takes checks (torch.vm.Dim(0, range(16, 33))), not {check!r}"
            )
    gm.meta.setdefault(_GUARDS, []).append((node, checks))
    return node


def guards_of(gm: torch.fx.GraphModule) -> list[tuple[int, Check]]:
    """The guards the passes brought, as (which input, check)."""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    out = []
    for node, checks in gm.meta.get(_GUARDS, ()):
        if node not in placeholders:
            raise ValueError(
                f"the guard on {node.name} is on no input of the graph the pass returned"
            )
        out.extend((placeholders.index(node), check) for check in checks)
    return out
