# mypy: allow-untyped-defs
"""Guards a block pass puts in its graph: ``vm.guard(gm, node, *checks)``.

The VM lifts them out of the graph to the version's entry. A region's own
guard already pins every tensor input's dtype, device, sizes and strides, so
checks on those (Dtype, Dim, Stride, Contiguous, Exact) are decided when the
block is compiled: one that holds for the block's inputs costs nothing, and
one that doesn't means the pass's version can never run, so it's dropped.
Checks on what varies between calls with the same metadata (Aligned, Is,
NotOverlapping) are checked on every call, in front of the version.
"""

import dataclasses
import sys
from typing import Any

import torch
import torch.fx


class Check:
    """One fact about a tensor. ``holds(t, value_of)`` decides it on a value
    (``value_of`` maps a graph node another check names to its value)."""

    # Whether it can differ between calls whose inputs have the same metadata.
    dynamic = False
    # Whether it gives the same answer on every rank by construction (no
    # pointers, no per-rank shapes).
    rank_uniform = True
    # Whether it reads the thread's state rather than the guarded value.
    state = False

    def holds(self, t: torch.Tensor | None, value_of) -> bool:
        """Whether the check holds of ``t`` (None for a check that reads the
        thread's state rather than a value)."""
        raise NotImplementedError

    def place(self, gm: torch.fx.GraphModule, node: torch.fx.Node) -> None:
        """Called when ``vm.guard`` puts the check on ``node`` of ``gm``, for
        a check that starts something then; raises ``vm.CannotGuard`` if the
        check can't be put there."""


def _dims(d: int, t: torch.Tensor) -> int | None:
    return d if -t.dim() <= d < t.dim() else None


@dataclasses.dataclass(frozen=True)
class Dtype(Check):
    dtype: torch.dtype

    def holds(self, t, value_of):
        return t.dtype == self.dtype

    def __repr__(self):
        return f"Dtype({str(self.dtype).removeprefix('torch.')})"


@dataclasses.dataclass(frozen=True)
class Device(Check):
    device: torch.device

    def holds(self, t, value_of):
        return t.device == torch.device(self.device)

    def __repr__(self):
        return f"Device({self.device})"


@dataclasses.dataclass(frozen=True)
class RequiresGrad(Check):
    requires_grad: bool

    def holds(self, t, value_of):
        return t.requires_grad == self.requires_grad


@dataclasses.dataclass(frozen=True)
class Dim(Check):
    """``t.size(d)`` is one of ``sizes`` (a set or a range)."""

    d: int
    sizes: Any

    def holds(self, t, value_of):
        return _dims(self.d, t) is not None and t.size(self.d) in self.sizes

    def __repr__(self):
        s = self.sizes
        shown = (
            f"{s.start}..{s.stop - 1}"
            if isinstance(s, range) and s.step == 1
            else repr(set(s) if not isinstance(s, range) else s)
        )
        return f"Dim({self.d}, {shown})"


@dataclasses.dataclass(frozen=True)
class Stride(Check):
    """``t.stride(d)`` is one of ``strides``."""

    d: int
    strides: Any

    def holds(self, t, value_of):
        return _dims(self.d, t) is not None and t.stride(self.d) in self.strides

    def __repr__(self):
        return f"Stride({self.d}, {set(self.strides)!r})"


@dataclasses.dataclass(frozen=True)
class StrideOrder(Check):
    """``t``'s dimensions, from largest stride to smallest, are ``perm``."""

    perm: tuple[int, ...]

    def holds(self, t, value_of):
        return t.dim() == len(self.perm) and all(
            t.stride(a) >= t.stride(b) for a, b in zip(self.perm, self.perm[1:])
        )


@dataclasses.dataclass(frozen=True)
class Divisible(Check):
    """``t.size(d)`` is a multiple of ``k``."""

    d: int
    k: int

    def holds(self, t, value_of):
        return _dims(self.d, t) is not None and t.size(self.d) % self.k == 0


@dataclasses.dataclass(frozen=True)
class Contiguous(Check):
    memory_format: torch.memory_format = torch.contiguous_format

    def holds(self, t, value_of):
        return t.is_contiguous(memory_format=self.memory_format)

    def __repr__(self):
        return "Contiguous()"


@dataclasses.dataclass(frozen=True)
class Exact(Check):
    """``type(t) is cls``."""

    cls: type

    def holds(self, t, value_of):
        return type(t) is self.cls

    def __repr__(self):
        return f"Exact({self.cls.__name__})"


class PlainTensor(Check):
    """``t`` is exactly a Tensor or Parameter (no subclass)."""

    def holds(self, t, value_of):
        return type(t) in (torch.Tensor, torch.nn.Parameter)

    def __repr__(self):
        return "PlainTensor()"

    def __eq__(self, other):
        return type(other) is PlainTensor

    def __hash__(self):
        return hash(PlainTensor)


@dataclasses.dataclass(frozen=True)
class TensorMeta(Check):
    """``t`` has exactly ``spec``'s dtype, device, sizes and strides
    (``spec``: a tensor, or a type whose sizes didn't vary)."""

    spec: Any

    def holds(self, t, value_of):
        s = self.spec
        sizes = (
            tuple(s.shape)
            if isinstance(s, torch.Tensor)
            else tuple(d.lo for d in s.dims)
        )
        strides = tuple(s.stride()) if isinstance(s, torch.Tensor) else s.strides
        return (
            t.dtype == s.dtype
            and t.device == s.device
            and tuple(t.shape) == sizes
            and tuple(t.stride()) == tuple(strides)
        )

    def __repr__(self):
        return (
            f"TensorMeta({self.spec!r})"
            if not isinstance(self.spec, torch.Tensor)
            else f"TensorMeta({self.spec.dtype}, {list(self.spec.shape)})"
        )


@dataclasses.dataclass(frozen=True)
class Value(Check):
    """The value is ``value`` (None, a bool, int or float): of exactly its
    class and bit for bit the same. Not ``==``: 0.0 and -0.0, True and 1, 1
    and 1.0 are equal, and a graph with one baked in is not a graph for the
    other."""

    value: Any
    dynamic = True

    def holds(self, t, value_of):
        return sys.modules["torch._rust"]._same(t, self.value)


@dataclasses.dataclass(frozen=True)
class Ints(Check):
    """The value is exactly a tuple (``seq=tuple``) or list of these ints."""

    seq: type
    values: tuple[int, ...]
    dynamic = True

    def holds(self, t, value_of):
        return (
            type(t) is self.seq
            and len(t) == len(self.values)
            and all(type(x) is int and x == v for x, v in zip(t, self.values))
        )


class _State(Check):
    """A fact about the thread's state rather than a value: checked on every
    call, on whichever node it guards."""

    dynamic = True
    state = True

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __eq__(self, other):
        return type(other) is type(self)

    def __hash__(self):
        return hash(type(self))


class NoTorchFunctionMode(_State):
    def holds(self, t, value_of):
        return not torch._C._is_torch_function_mode_enabled()


@dataclasses.dataclass(frozen=True, eq=True)
class GradEnabled(_State):
    enabled: bool = True

    def holds(self, t, value_of):
        return torch.is_grad_enabled() == self.enabled

    def __repr__(self):
        return f"GradEnabled({self.enabled})"


class NoForwardAD(_State):
    def holds(self, t, value_of):
        from torch.autograd import forward_ad

        return forward_ad._current_level < 0 or not torch._C._is_fwd_grad_enabled()


class NoSavedTensorHooks(_State):
    def holds(self, t, value_of):
        return torch._C._autograd._top_saved_tensors_default_hooks(False) is None


def address(t: torch.Tensor) -> int:
    """Where ``t``'s first element is, as ``t.data_ptr()`` gives it, read
    without asking PyTorch for a writable pointer: ``data_ptr()`` hands the
    memory out, which ends a ``vm.Unwritten`` on ``t``."""
    return torch._C._data_address(t) + t.storage_offset() * t.element_size()


@dataclasses.dataclass(frozen=True)
class Aligned(Check):
    """``t``'s memory starts at a multiple of ``n`` bytes."""

    n: int
    dynamic = True
    rank_uniform = False

    def holds(self, t, value_of):
        return address(t) % self.n == 0

    def __repr__(self):
        return f"Aligned({self.n})"


@dataclasses.dataclass(frozen=True, eq=False)
class Is(Check):
    """``t is obj``."""

    obj: Any
    dynamic = True

    def holds(self, t, value_of):
        return t is self.obj

    def __repr__(self):
        return f"Is({type(self.obj).__name__}@{id(self.obj):#x})"


@dataclasses.dataclass(frozen=True, eq=False)
class NotOverlapping(Check):
    """``t``'s memory overlaps none of ``others`` (graph nodes)."""

    others: tuple[Any, ...]
    dynamic = True
    rank_uniform = False

    def __init__(self, *others: Any) -> None:
        object.__setattr__(self, "others", others)

    def holds(self, t, value_of):
        lo, hi = _span(t)
        for o in self.others:
            v = value_of(o)
            if isinstance(v, torch.Tensor) and v.device == t.device:
                olo, ohi = _span(v)
                if lo < ohi and olo < hi:
                    return False
        return True

    def __repr__(self):
        return f"NotOverlapping({', '.join(getattr(o, 'name', repr(o)) for o in self.others)})"


def _span(t: torch.Tensor) -> tuple[int, int]:
    if t.numel() == 0:
        return (0, 0)
    start = address(t)
    last = sum((s - 1) * st for s, st in zip(t.shape, t.stride()) if st > 0)
    return (start, start + (last + 1) * t.element_size())


def guard_node(x: Any) -> Any:
    """The target of a guard node, whose checks are in its meta["vm_checks"]:
    passes ``x`` through (the VM removes guard nodes before running a
    graph)."""
    return x


guard_node.__name__ = "vm_guard"


def guard(
    gm: torch.fx.GraphModule, node: torch.fx.Node, *checks: Check
) -> torch.fx.Node:
    """Adds a guard node: this version of the block runs only on calls where
    every check holds for ``node``'s value (else the version below runs)."""
    for c in checks:
        if not isinstance(c, Check):
            raise TypeError(f"vm.guard takes checks (vm.Dtype, vm.Dim, ...); got {c!r}")
        c.place(gm, node)
    with gm.graph.inserting_after(node):
        g = gm.graph.call_function(guard_node, (node,))
    g.meta["vm_checks"] = checks
    return g


@dataclasses.dataclass
class Lifted:
    """A pass's guards, out of its graph: the checks to make on each call (by
    input index), and those decided now."""

    # By input index (-1 for checks on the thread's state).
    dynamic: list[tuple[int, Check]]
    held: list[tuple[str, Check]]
    failed: list[tuple[str, Check]]


def lift(gm: torch.fx.GraphModule, example_inputs: list[Any]) -> Lifted:
    """Removes the guard nodes from ``gm`` and sorts their checks."""
    placeholders = list(gm.graph.find_nodes(op="placeholder"))
    index = {p: i for i, p in enumerate(placeholders)}
    out = Lifted([], [], [])

    def value_of(n):
        if n in index:
            return example_inputs[index[n]]
        return n.meta.get("val")

    for node in list(gm.graph.find_nodes(op="call_function", target=guard_node)):
        (x,) = node.args
        checks = node.meta["vm_checks"]
        val = value_of(x)
        for c in checks:
            if c.state:
                out.dynamic.append((-1, c))
            elif c.dynamic:
                if x not in index:
                    raise NotImplementedError(
                        f"{c!r} on {x.name}: guards that vary between calls go on the block's inputs (placeholders) for now"
                    )
                if isinstance(c, NotOverlapping):
                    if any(o not in index for o in c.others):
                        raise NotImplementedError(
                            f"{c!r}: NotOverlapping compares the block's inputs (placeholders) for now"
                        )
                    # By input index, as holds() reads them.
                    c = NotOverlapping(*(index[o] for o in c.others))
                out.dynamic.append((index[x], c))
            elif isinstance(val, torch.Tensor) and c.holds(val, value_of):
                out.held.append((x.name, c))
            else:
                out.failed.append((x.name, c))
        node.replace_all_uses_with(x)
        gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return out


def failing(dynamic: list[tuple[int, Check]], inputs: Any) -> tuple[int, Check] | None:
    """The first dynamic check that doesn't hold on this call's inputs."""
    for i, c in dynamic:
        if not c.holds(inputs[i] if i >= 0 else None, inputs.__getitem__):
            return (i, c)
    return None
