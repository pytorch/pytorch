# mypy: allow-untyped-defs
"""What a block pass knows about each input: a ``Type``, the join of every
value the VM saw there (from its type lattice, torch/_rust/src/exec/types.rs),
not one example."""

import dataclasses
import math
from typing import Any

import torch
import torch.fx
from torch.vm._examples import raw


@dataclasses.dataclass(frozen=True)
class Interval:
    """The sizes one dimension took (``types::Interval``): ``lo..hi``, equal
    when it never varied. ``seen`` is the size in the run the region was
    traced from, which its graph is specialized to."""

    lo: int
    hi: int
    seen: int

    def __contains__(self, n: object) -> bool:
        return isinstance(n, int) and self.lo <= n <= self.hi

    @property
    def exact(self) -> bool:
        return self.lo == self.hi

    def samples(self, n: int) -> list[int]:
        """Up to ``n`` sizes spread over ``lo..hi``, both ends included."""
        if n <= 1 or self.exact:
            return [self.seen] if n == 1 else [self.lo]
        return sorted(
            {self.lo + round((self.hi - self.lo) * i / (n - 1)) for i in range(n)}
        )

    def as_set(self) -> set[int] | range:
        return {self.lo} if self.exact else range(self.lo, self.hi + 1)

    def __repr__(self) -> str:
        return str(self.lo) if self.exact else f"{self.lo}..{self.hi}"


_SHORT = {
    torch.float32: "f32",
    torch.float64: "f64",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.uint8: "u8",
    torch.bool: "bool",
}


class _Varies:
    """A property that took more than one value (``Konst::Varies``)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Varies"

    def __reduce__(self):
        return (_Varies, ())


Varies: Any = _Varies()


@dataclasses.dataclass(frozen=True)
class DimSet:
    """The sizes one dimension took, when there were few of them: a set,
    rather than the interval spanning them. (The VM's lattice records
    intervals only so far, so ``TensorType.dims`` holds intervals.)"""

    values: frozenset[int]

    def __contains__(self, n: object) -> bool:
        return n in self.values

    @property
    def lo(self) -> int:
        return min(self.values)

    @property
    def hi(self) -> int:
        return max(self.values)

    def samples(self, n: int) -> list[int]:
        ordered = sorted(self.values)
        if n >= len(ordered):
            return ordered
        return sorted(
            {ordered[round((len(ordered) - 1) * i / max(n - 1, 1))] for i in range(n)}
        )

    def as_set(self) -> set[int]:
        return set(self.values)

    def __repr__(self) -> str:
        return "{" + ", ".join(map(str, sorted(self.values))) + "}"


class Type:
    """What the VM knows about the values at one place (``types::Ty``): the
    join of every value it saw there."""

    def join(self, other: "Type") -> "Type":
        return (
            self
            if self == other
            else UnionType((self, other))
            if other is not AnyType
            else other
        )

    def __contains__(self, value: object) -> bool:
        raise NotImplementedError

    def as_checks(self) -> list[Any]:
        return []

    @property
    def const(self) -> Any:
        return None


@dataclasses.dataclass(eq=False)
class TensorType(Type):
    """A tensor's type (``types::TensorTy``). ``dtype``, ``device`` and
    ``requires_grad`` are ``vm.Varies`` when they varied; ``dims`` is None
    when the rank did. A weight (a tensor the region reads off a module rather than
    takes) is a singleton: ``const`` is the tensor itself, which a pass may
    read (to tune on) but must not write."""

    cls: type
    dtype: torch.dtype | _Varies
    device: torch.device | _Varies
    dims: tuple[Interval, ...] | None
    requires_grad: bool | _Varies
    # The traced run's strides (the region checks them exactly; the lattice
    # doesn't record strides yet).
    strides: tuple[int, ...] | None = None
    # (a region's name, which of its outputs) when, at the call that was
    # traced, this input was that very tensor: the fresh output of a region
    # traced before it. What the traced call saw, not a check: a later call
    # may hand the region a tensor from somewhere else.
    made_by: tuple[str, int] | None = None
    _const: torch.Tensor | None = None
    # Told when a pass reads `const` (its output then depends on this tensor).
    _on_const: Any = None

    @property
    def const(self) -> torch.Tensor | None:
        if self._const is not None and self._on_const is not None:
            self._on_const()
        return self._const

    @property
    def is_const(self) -> bool:
        return self._const is not None

    def concrete(self) -> tuple[torch.dtype, torch.device, tuple[Interval, ...]]:
        """Its dtype, device and dims when each was the same on every call
        the VM saw: what making a tensor of this type needs. TypeError when
        the rank, the dtype or the device varied."""
        if (
            isinstance(self.dtype, _Varies)
            or isinstance(self.device, _Varies)
            or self.dims is None
        ):
            raise TypeError(f"{self}: its rank, dtype or device varied")
        return self.dtype, self.device, self.dims

    def dim(self, d: int) -> Interval:
        if self.dims is None:
            raise ValueError(f"{self}: its rank varied")
        return self.dims[d]

    @property
    def ndim(self) -> int:
        if self.dims is None:
            raise ValueError(f"{self}: its rank varied")
        return len(self.dims)

    def nbytes(self, at: str = "hi") -> int:
        """Its bytes at the ``"lo"`` or ``"hi"`` end of every range."""
        if self.dims is None or isinstance(self.dtype, _Varies):
            raise ValueError(f"{self}: its rank or dtype varied")
        return math.prod(getattr(s, at) for s in self.dims) * self.dtype.itemsize

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, torch.Tensor) or type(value) is not self.cls:
            return False
        if self.is_const:
            return value is self._const
        if (self.dtype is not Varies and value.dtype != self.dtype) or (
            self.device is not Varies and value.device != self.device
        ):
            return False
        return self.dims is None or (
            value.dim() == len(self.dims)
            and all(n in s for n, s in zip(value.shape, self.dims))
        )

    def join(self, other: Type) -> Type:
        if not isinstance(other, TensorType) or other.cls is not self.cls:
            return super().join(other)
        dims = None
        if (
            self.dims is not None
            and other.dims is not None
            and len(self.dims) == len(other.dims)
        ):
            dims = tuple(
                Interval(min(a.lo, b.lo), max(a.hi, b.hi), a.seen)
                for a, b in zip(self.dims, other.dims)
            )
        return TensorType(
            self.cls,
            self.dtype if self.dtype == other.dtype else Varies,
            self.device if self.device == other.device else Varies,
            dims,
            self.requires_grad if self.requires_grad == other.requires_grad else Varies,
            self.strides if self.strides == other.strides else None,
            self._const if self._const is other._const else None,
        )

    def as_checks(self) -> list[Any]:
        """The type as guards: what a pass emits to specialize to exactly what
        was seen."""
        from torch.vm._guards import Device, Dim, Dtype

        checks: list[Any] = []
        if not isinstance(self.dtype, _Varies):
            checks.append(Dtype(self.dtype))
        if not isinstance(self.device, _Varies):
            checks.append(Device(self.device))
        for d, s in enumerate(self.dims or ()):
            checks.append(Dim(d, s.as_set()))
        return checks

    def __repr__(self) -> str:
        dtype = (
            "?"
            if isinstance(self.dtype, _Varies)
            else _SHORT.get(self.dtype, str(self.dtype).removeprefix("torch."))
        )
        dims = "[...]" if self.dims is None else f"[{', '.join(map(repr, self.dims))}]"
        device = (
            ""
            if isinstance(self.device, _Varies) or self.device.type == "cpu"
            else f" {self.device}"
        )
        return f"{dtype}{dims}{device}{' const' if self.is_const else ''}"


@dataclasses.dataclass(frozen=True)
class IntType(Type):
    range: Interval

    def __contains__(self, value: object) -> bool:
        return type(value) is int and value in self.range

    def __repr__(self) -> str:
        return f"int[{self.range!r}]"


@dataclasses.dataclass(frozen=True)
class FloatType(Type):
    """A Python float the region takes as an input: its value may differ
    between calls, unless ``value`` is set (the trace specialized on it)."""

    value: float | None = None
    is_const = False

    @property
    def const(self) -> Any:
        return self.value

    def __contains__(self, value: object) -> bool:
        return isinstance(value, float) and (self.value is None or value == self.value)

    def __repr__(self) -> str:
        return "float" if self.value is None else f"float {self.value!r}"


@dataclasses.dataclass(frozen=True, eq=False)
class ConstType(Type):
    """The same object on every call (a module, a function). A weight's type
    is a ``TensorType`` whose ``const`` is set."""

    value: Any

    @property
    def const(self) -> Any:
        return self.value

    def __contains__(self, value: object) -> bool:
        return value is self.value

    def as_checks(self) -> list[Any]:
        from torch.vm._guards import Is

        return [Is(self.value)]

    def __repr__(self) -> str:
        return f"const {type(self.value).__name__}"


@dataclasses.dataclass(frozen=True)
class SeqType(Type):
    """A tuple, list, set, frozenset or dict (``Kind::Tuple`` /
    ``VarTuple`` / ``List`` / ...): its kind, how long it was, and its items'
    types when it's a tuple of fixed length."""

    kind: type
    length: Interval
    items: tuple[Type, ...] | None = None

    def __contains__(self, value: object) -> bool:
        if type(value) is not self.kind or len(value) not in self.length:  # type: ignore[arg-type]
            return False
        return self.items is None or all(v in t for v, t in zip(value, self.items))  # type: ignore[call-overload]

    def __repr__(self) -> str:
        if self.items is not None:
            return f"{self.kind.__name__}[{', '.join(map(repr, self.items))}]"
        return f"{self.kind.__name__}[len {self.length!r}]"


@dataclasses.dataclass(frozen=True)
class ObjectType(Type):
    """Any other object, by its kind or exact class name (``Kind::Object``)."""

    cls: str

    def __contains__(self, value: object) -> bool:
        return type(value).__name__ == self.cls

    def __repr__(self) -> str:
        return self.cls


@dataclasses.dataclass(frozen=True)
class UnionType(Type):
    alternatives: tuple[Type, ...]

    def __contains__(self, value: object) -> bool:
        return any(value in t for t in self.alternatives)

    def __repr__(self) -> str:
        return " | ".join(map(repr, self.alternatives))


class _AnyType(Type):
    """Nothing known (``Ty::Any``)."""

    def __contains__(self, value: object) -> bool:
        return True

    def join(self, other: Type) -> Type:
        return self

    def __repr__(self) -> str:
        return "Any"


AnyType: Type = _AnyType()


def exact(t: torch.Tensor, const: bool = False) -> TensorType:
    """The type of exactly tensor ``t``'s metadata (and ``t`` itself with
    ``const``)."""
    return TensorType(
        cls=type(t),
        dtype=t.dtype,
        device=t.device,
        dims=tuple(Interval(s, s, s) for s in t.shape),
        requires_grad=t.requires_grad,
        strides=tuple(t.stride()),
        _const=t if const else None,
    )


def from_rust(data: Any, example: torch.Tensor) -> TensorType:
    """The type the VM recorded for an input (``Ty::to_py``: None for Any,
    else a list of kinds), given the traced run's tensor there; the
    example's own type when nothing was recorded for it."""
    t = exact(example)
    kinds = [k for k in data or () if k[0] == "tensor"]
    if not kinds:
        return t
    # A union of tensor classes: the one this run had.
    _, _cls, dtype, device, shape, requires_grad = kinds[0]
    dims = None
    if shape is not None and len(shape) == example.dim():
        dims = tuple(Interval(lo, hi, s) for (lo, hi), s in zip(shape, example.shape))
    return dataclasses.replace(
        t,
        dtype=example.dtype if dtype is not None else Varies,
        device=example.device if device is not None else Varies,
        dims=dims,
        requires_grad=Varies if requires_grad is None else requires_grad,
    )


def _scaled_strides(t: TensorType, sizes: list[int]) -> list[int]:
    """Strides for ``sizes`` laid out like the traced tensor: its strides when
    only its outermost dimension changed (a view into a wider buffer keeps
    them), else contiguous."""
    seen = [s.seen for s in t.dims or ()]
    if t.strides is not None and len(seen) == len(sizes):
        changed = [d for d in range(len(sizes)) if sizes[d] != seen[d]]
        outer = max(range(len(sizes)), key=lambda d: t.strides[d], default=None)
        if not changed or changed == [outer]:
            return list(t.strides)
    strides, n = [0] * len(sizes), 1
    for d in reversed(range(len(sizes))):
        strides[d] = n
        n *= max(sizes[d], 1)
    return strides


def materialize(
    t: TensorType,
    *,
    at: str | None = None,
    dim0: int | None = None,
    sizes: list[int] | None = None,
) -> torch.Tensor:
    """Random data of type ``t``, with its strides (a weight's type gives the
    weight itself). Sizes: ``sizes``, else each dimension's traced size, or
    its smallest or largest with ``at="lo"``/``"hi"``; ``dim0`` sets the
    first. Never runs the region."""
    const = t._const
    if const is not None:
        return const
    dtype, device, dims = t.dtype, t.device, t.dims
    if dims is None or isinstance(dtype, _Varies) or isinstance(device, _Varies):
        raise ValueError(f"can't materialize {t}: it varied in rank, dtype or device")
    if sizes is None:
        pick = {"lo": lambda s: s.lo, "hi": lambda s: s.hi, None: lambda s: s.seen}[at]
        sizes = [pick(s) for s in dims]
        if dim0 is not None:
            sizes[0] = dim0
    out = torch.empty_strided(
        sizes, _scaled_strides(t, sizes), dtype=dtype, device=device
    )
    if out.dtype.is_floating_point or out.dtype.is_complex:
        out.copy_(torch.randn(sizes, dtype=dtype, device=device))
    elif out.dtype == torch.bool:
        out.copy_(torch.rand(sizes, device=device) < 0.5)
    else:
        out.copy_(torch.randint(0, 2, sizes, dtype=dtype, device=device))
    return out


def type_of(
    gm: torch.fx.GraphModule, input_types: list[TensorType], node: torch.fx.Node
) -> TensorType:
    """``node``'s type: an input's from ``input_types``; an intermediate's by
    propagating the input types through the graph (fake tensors at each end
    of the inputs' ranges, joined with the traced run's value)."""
    if node.op == "placeholder":
        return input_types[list(gm.graph.find_nodes(op="placeholder")).index(node)]
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        raise TypeError(f"{node} isn't a tensor ({type(val).__name__})")
    t = dataclasses.replace(exact(val), requires_grad=Varies)
    tensors = [it for it in input_types if isinstance(it, TensorType)]
    if all(s.exact for it in tensors if it.dims is not None for s in it.dims):
        return t
    lo, hi = (_propagate(gm, input_types, at).get(node) for at in ("lo", "hi"))
    if (
        not isinstance(lo, torch.Tensor)
        or not isinstance(hi, torch.Tensor)
        or lo.dim() != val.dim()
        or hi.dim() != val.dim()
    ):
        return t
    dims = tuple(
        Interval(min(a, b, s), max(a, b, s), s)
        for a, b, s in zip(lo.shape, hi.shape, val.shape)
    )
    return dataclasses.replace(t, dims=dims)


def _propagate(
    gm: torch.fx.GraphModule, input_types: list[TensorType], at: str
) -> dict[torch.fx.Node, Any]:
    """Every node's fake value with each input at the ``at`` end of its range
    (cached on the graph, per input types); empty if that doesn't run."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    cache = gm.meta.setdefault("vm_propagated", {})
    key = (id(input_types), at, id(gm.graph), len(gm.graph.nodes))
    if key in cache:
        return cache[key]
    env: dict[torch.fx.Node, Any] = {}
    mode = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)

    class Recorder(torch.fx.Interpreter):
        def run_node(self, n):
            env[n] = super().run_node(n)
            return env[n]

    try:
        with mode:
            args: list[Any] = []
            seen = raw(gm) or [None] * len(input_types)
            for t, was in zip(input_types, seen):
                if isinstance(t, FloatType):
                    # A float input: the value the trace specialized on, else
                    # the traced call's (the sizes do not follow a float that
                    # the trace did not specialize on).
                    value = was if t.value is None else t.value
                    if not isinstance(value, float):
                        raise ValueError("no value for a float input")
                    args.append(value)
                    continue
                const = t._const
                if const is not None:
                    args.append(mode.from_tensor(const))
                    continue
                dtype, device, dims = t.concrete()
                sizes = [getattr(s, at) for s in dims]
                args.append(
                    torch.empty_strided(
                        sizes, _scaled_strides(t, sizes), dtype=dtype, device=device
                    )
                )
            Recorder(gm).run(*args)
    except (RuntimeError, TypeError, ValueError, IndexError):
        env = {}
    cache[key] = env
    return env


def nbytes(x: Any) -> int:
    """The most bytes a tensor took: of a TensorType, or of a graph node (by
    its value in the trace)."""
    if isinstance(x, torch.fx.Node):
        x = exact(x.meta["val"])
    return x.nbytes("hi")
