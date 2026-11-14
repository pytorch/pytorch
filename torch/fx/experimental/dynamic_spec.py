"""Spec types for controlling what is dynamic in compiled/exported code.
Currently only supports unbacked dynamic shapes.

"""

from __future__ import annotations

import itertools
import logging
import threading
from typing import Any, cast, TYPE_CHECKING, TypeAlias

from torch import SymBool, SymInt


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import torch.nn
    from torch.fx.experimental.symbolic_shapes import ShapeEnv


__all__ = [
    "ShapeVar",
    "IntVar",
    "STATIC",
    "TensorSpec",
    "ObjectSpec",
    "DictSpec",
    "SeqSpec",
    "ParamsSpec",
    "ShapesSpec",
    "LeafSpec",
    "LeafIntSpec",
    "IntermediateSpec",
    "dynamic_spec",
]


# Indent unit for nested repr output. Two spaces per level.
_INDENT = "  "


# Sentinel meaning "this dim / scalar arg is static" (taken from the
# actual input at runtime, with recompiles on different values).
# Alias for ``None`` so users can write either ``TensorSpec([A, STATIC])``
# (more readable) or ``TensorSpec([A, None])``.
STATIC = None


# ---------------------------------------------------------------------------
# Spec ShapeEnv
#
# IntVar/ShapeVar wrap a SymNode, which requires a ShapeEnv. We use a singleton
# global env for shape specs. This shape env is special and only used for
# expression construction and will fail on any other calls.
# ---------------------------------------------------------------------------


_SPEC_SHAPE_ENV: ShapeEnv | None = None
# Lock guarding the lazy init of _SPEC_SHAPE_ENV so concurrent IntVar()
# calls observe the same singleton. We cannot construct eagerly at module
# import time because ShapeEnv.__init__ pulls in modules that
# transitively import this one.
_SPEC_SHAPE_ENV_LOCK = threading.Lock()


def _get_spec_shape_env() -> ShapeEnv:
    """Lazily build the singleton spec ShapeEnv (thread-safe)."""
    global _SPEC_SHAPE_ENV
    if _SPEC_SHAPE_ENV is not None:
        return _SPEC_SHAPE_ENV
    with _SPEC_SHAPE_ENV_LOCK:
        if _SPEC_SHAPE_ENV is not None:
            return _SPEC_SHAPE_ENV
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        class _SpecShapeEnv(ShapeEnv):
            """Special ShapeEnv used only at spec-definition time.

            Attribute access is blocked by default via ``__getattribute__``;
            only private (``_``-prefix) names and the explicit allowlist in
            ``_ALLOWED_PUBLIC`` pass through. SymInt arithmetic paths read
            a handful of public fields transparently (via the
            ``@record_shapeenv_event`` decorator and FX-cache machinery),
            so those are allowlisted. Everything else — evaluation, guard
            recording, deferred asserts, etc. — is blocked.

            During ``ShapeEnv.__init__`` itself, access is unrestricted
            because the base class touches its own public fields while
            bootstrapping.
            """

            # Public fields read by internal ShapeEnv infra during arithmetic.
            # Grow as new accesses are discovered by tests.
            _ALLOWED_PUBLIC: frozenset[str] = frozenset(
                {
                    "should_record_events",
                    "is_recording",
                    "fx_node_cache",
                    # bound_sympy is read-only (no guards/asserts/mutation).
                    # Needed for `%` (mod) which inspects symbol ranges to
                    # pick between Mod and PythonMod. var_to_range is the
                    # field bound_sympy reads to compute ranges.
                    "bound_sympy",
                    "var_to_range",
                }
            )

            def __init__(self) -> None:
                # Allow everything during super().__init__.
                object.__setattr__(self, "_init_done", False)
                super().__init__()
                object.__setattr__(self, "_init_done", True)

            def __getattribute__(self, name: str) -> Any:
                if name.startswith("_"):
                    return super().__getattribute__(name)
                if not super().__getattribute__("_init_done"):
                    return super().__getattribute__(name)
                if name in type(self)._ALLOWED_PUBLIC:
                    return super().__getattribute__(name)
                raise TypeError(
                    f"_SpecShapeEnv: '{name}' is not allowed at "
                    f"spec-definition time. Use IntVar / ShapeVar only "
                    f"inside TensorSpec / ShapesSpec."
                )

        _SPEC_SHAPE_ENV = _SpecShapeEnv()
        return _SPEC_SHAPE_ENV


class IntVar(SymInt):
    """Indicates that a scalar integer argument is dynamic (no implicit range).

    Unbacked dynamic shapes will represent the underlying value in the
    compiled graph.

    IntVar is a ``SymInt`` subclass backed by ``_SpecShapeEnv``,
    so arithmetic and comparisons compose naturally::

        A = IntVar("a")
        B = IntVar("b")
        A + 1  # SymInt expr=a#0 + 1
        A * B + 1  # SymInt expr=a#0*b#1 + 1
        A > 0  # SymBool

    Such derived ``SymInt`` values may be used directly as leaf specs in
    ``TensorSpec`` / ``ParamsSpec``

    Repr always includes a per-instance uid so two IntVars with the same name
    (or two anonymous ones) are distinguishable in logs::

        IntVar()           -> "IntVar(anon#0)"
        IntVar("offset")   -> "IntVar(offset#1)"

    Example::

        IntVar("num_heads")
        IntVar("offset", min=-100, max=100)
        IntVar("size", min=1, max=2048, optimization_hint=512)
    """

    _uid_counter = itertools.count()

    def __init__(
        self,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        optimization_hint: int | None = None,
    ) -> None:
        import sympy

        from torch.fx.experimental.sym_node import _NO_HINT, SymNode
        from torch.utils._sympy.numbers import int_oo
        from torch.utils._sympy.value_ranges import ValueRanges

        self.name = name if name is not None else "anon"
        self._uid = next(IntVar._uid_counter)
        self.min = min
        self.max = max
        self.optimization_hint = optimization_hint
        self.sympy_sym = sympy.Symbol(f"{self.name}#{self._uid}")
        env = _get_spec_shape_env()
        node = SymNode(
            self.sympy_sym,
            shape_env=env,
            pytype=int,
            hint=_NO_HINT,
        )
        super().__init__(node)
        # Register the user-supplied (or unbounded) range so
        # range-dependent ops like `%` get correct ValueRanges info.
        env.var_to_range[self.sympy_sym] = ValueRanges(
            min if min is not None else -int_oo,
            max if max is not None else int_oo,
        )

    def __repr__(self) -> str:
        # Always include uid so two same-named (or anonymous) instances
        # are distinguishable in logs.
        parts = [f"{self.name}#{self._uid}"]
        if self.min is not None:
            parts.append(f"min={self.min}")
        if self.max is not None:
            parts.append(f"max={self.max}")
        if self.optimization_hint is not None:
            parts.append(f"optimization_hint={self.optimization_hint}")
        return f"{type(self).__name__}({', '.join(parts)})"

    # Hashable by identity — IntVars are unique per construction, and inheriting
    # SymInt.__hash__ would route through node hashing which we don't need here.
    def __hash__(self) -> int:  # type: ignore[override]
        return id(self)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "optimization_hint": self.optimization_hint,
        }


def _validate_spec_sym(v: Any, *, where: str) -> None:
    """If ``v`` is a SymInt or SymBool, validate it originates from the
    spec ShapeEnv.
    """
    if isinstance(v, SymInt):
        kind = "SymInt"
    elif isinstance(v, SymBool):
        kind = "SymBool"
    else:
        return
    if v.node.shape_env is not _get_spec_shape_env():
        raise TypeError(
            f"{where}: {kind} spec values must originate from spec IntVar / "
            f"ShapeVar; got {v!r} backed by a different ShapeEnv."
        )


class ShapeVar(IntVar):
    """Indicates that a dimension size is dynamic and is >= 0.

    Subclass of `IntVar` with ``min=0`` by default.

    Example::

        ShapeVar("batch")
        ShapeVar("batch", max=64)
        ShapeVar("batch", max=64, optimization_hint=32)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        min: int = 0,
        max: int | None = None,
        optimization_hint: int | None = None,
    ) -> None:
        if min < 0:
            raise ValueError(
                f"ShapeVar requires min >= 0 (a shape dim is non-negative); "
                f"got min={min}. Use IntVar(...) for unrestricted scalars."
            )
        super().__init__(name, min=min, max=max, optimization_hint=optimization_hint)


class TensorSpec:
    """Per-dimension shape specification for a tensor.

    A list-like container of ``LeafIntSpec`` with length
    equal to the tensor's dim.

    Example::

        B = ShapeVar("batch")
        TensorSpec([B, STATIC])  # rank 2, dim 0 dynamic
        TensorSpec([B, 10])  # rank 2, dim 0 dynamic, dim 1 static=10
        TensorSpec([B * 2 + 1, STATIC])  # rank 2, dim 0 derived from B
    """

    def __init__(self, dims: Sequence[LeafIntSpec]) -> None:
        for i, d in enumerate(dims):
            if d is not None and not isinstance(d, (int, SymInt)):
                raise TypeError(
                    f"TensorSpec dim {i}: expected LeafIntSpec, got "
                    f"{type(d).__name__}: {d!r}"
                )
            _validate_spec_sym(d, where=f"TensorSpec dim {i}")
        self._specs: list[LeafIntSpec] = list(dims)

    def __getitem__(self, index: int) -> LeafIntSpec:
        return self._specs[index]

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[LeafIntSpec]:
        return iter(self._specs)

    def __repr__(self) -> str:
        lines = ["Tensor:"]
        for i, spec in enumerate(self._specs):
            lines.append(f"{_INDENT}{i}: {spec!r}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "TensorSpec",
            "dims": [
                spec.to_jsonable() if hasattr(spec, "to_jsonable") else spec
                for spec in self._specs
            ],
        }


class ObjectSpec:
    """Spec for any Python object's attributes.

    Constructor::

        ObjectSpec({name: IntermediateSpec, ...})

    Values may be leaves (``TensorSpec`` / ``IntVar`` / ``int`` /
    ``None``) or another ``ObjectSpec`` for recursion.

    Example::

        ObjectSpec({"weight": TensorSpec([ShapeVar("h"), STATIC])})
        ObjectSpec({"inner": ObjectSpec({"weight": TensorSpec([ShapeVar("h")])})})
    """

    def __init__(self, fields: dict[str, IntermediateSpec] | None = None) -> None:
        self._fields: dict[str, IntermediateSpec] = dict(fields) if fields else {}

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def items(self) -> Any:
        return self._fields.items()

    def __repr__(self) -> str:
        lines = ["object_spec:"]
        for name, spec in self._fields.items():
            spec_repr = repr(spec)
            if "\n" in spec_repr:
                lines.append(f"{_INDENT}.{name}:")
                for line in spec_repr.splitlines():
                    lines.append(_INDENT * 2 + line)
            else:
                lines.append(f"{_INDENT}.{name}: {spec_repr}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "ObjectSpec",
            "fields": {
                name: spec.to_jsonable() if hasattr(spec, "to_jsonable") else spec
                for name, spec in self._fields.items()
            },
        }


class DictSpec:
    """Spec for a Python ``dict``-typed value.

    Constructor::

        DictSpec({key: IntermediateSpec, ...})

    Keys may be ``str`` or ``int``.

    Example::

        DictSpec({"x": TensorSpec([ShapeVar("h"), STATIC])})
        DictSpec({"config": DictSpec({"batch": IntVar()})})
        DictSpec({0: TensorSpec([ShapeVar("h"), STATIC])})
    """

    def __init__(
        self, entries: dict[str | int, IntermediateSpec] | None = None
    ) -> None:
        self._entries: dict[str | int, IntermediateSpec] = (
            dict(entries) if entries else {}
        )
        for k in self._entries:
            if not isinstance(k, (str, int)):
                raise TypeError(
                    f"DictSpec entries must have str or int keys, got {type(k).__name__}: {k!r}"
                )

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[str | int]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self) -> Any:
        return self._entries.items()

    def __repr__(self) -> str:
        lines = ["dict_spec:"]
        for key, spec in self._entries.items():
            spec_repr = repr(spec)
            if "\n" in spec_repr:
                lines.append(f"{_INDENT}[{key!r}]:")
                for line in spec_repr.splitlines():
                    lines.append(_INDENT * 2 + line)
            else:
                lines.append(f"{_INDENT}[{key!r}]: {spec_repr}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "DictSpec",
            "entries": {
                str(key): spec.to_jsonable() if hasattr(spec, "to_jsonable") else spec
                for key, spec in self._entries.items()
            },
        }


class SeqSpec:
    """Spec for a Python ``list``- or ``tuple``-typed value.

    Per-position. The spec list may be shorter than the runtime sequence;
    any positions beyond ``len(self)`` are treated as fully static (i.e.
    equivalent to an unspecified slot).

    Constructor::

        SeqSpec([IntermediateSpec, ...])

    Example::

        SeqSpec([TensorSpec([ShapeVar("h"), 10]), 1])
        SeqSpec((TensorSpec([ShapeVar("a")]), TensorSpec([ShapeVar("b")])))
    """

    def __init__(self, entries: Sequence[IntermediateSpec]) -> None:
        self._entries: list[IntermediateSpec] = list(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        lines = ["seq_spec:"]
        for i, spec in enumerate(self._entries):
            spec_repr = repr(spec)
            if "\n" in spec_repr:
                lines.append(f"{_INDENT}[{i}]:")
                for line in spec_repr.splitlines():
                    lines.append(_INDENT * 2 + line)
            else:
                lines.append(f"{_INDENT}[{i}]: {spec_repr}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "SeqSpec",
            "entries": [
                spec.to_jsonable() if hasattr(spec, "to_jsonable") else spec
                for spec in self._entries
            ],
        }


# Per-int-leaf spec type: union of valid values that can appear at a slot
# where the runtime value is an int (tensor dim, scalar arg).
#
# Per-entry semantics:
# - IntVar (or its ShapeVar subclass): unbacked dynamic dimension.
# - SymInt (derived from spec IntVars via arithmetic, e.g. A * 2 + 1):
#   the dim must equal this expression at runtime.
# - int: static dimension pinned to that concrete value.
# - None: unspecified; treated as static. Value is inferred from the
#   example input if the consuming API supports it (e.g. torch.compile),
#   otherwise an error will be thrown.
LeafIntSpec: TypeAlias = IntVar | SymInt | int | None
# Leaf specs (individual argument specifications) — ints plus TensorSpec.
LeafSpec: TypeAlias = LeafIntSpec | TensorSpec
# Includes containers (``ObjectSpec`` / ``DictSpec`` / ``SeqSpec``)
# for nested specs reachable via dynamo source-chain walks.
IntermediateSpec: TypeAlias = LeafSpec | ObjectSpec | DictSpec | SeqSpec

# Value type for the dict passed to ``ParamsSpec``/``ShapesSpec``:
#   - named entries hold a single spec,
#   - ``"*args"`` holds a list of specs,
#   - ``"**kwargs"`` holds a dict of specs.
ParamsSpecValue: TypeAlias = (
    IntermediateSpec | list[IntermediateSpec] | dict[str, IntermediateSpec]
)


class ParamsSpec:
    """Specification for the arguments of a compiled function.

    Describes the dynamic shape behavior for named arguments, ``*args``,
    and ``**kwargs`` of a ``torch.compile``-wrapped function. Takes a
    single dict keyed by parameter name, with two reserved sentinel keys
    for the variadic slots::

        def func(x, n, *args, **kwargs): ...


        ParamsSpec(
            {
                "x": TensorSpec([ShapeVar("batch"), None]),
                "n": IntVar("seq"),
                "*args": [TensorSpec([ShapeVar("a")]), None],
                "**kwargs": {
                    "foo": TensorSpec([ShapeVar("b"), None]),
                    "bar": TensorSpec([ShapeVar("c"), None]),
                },
            }
        )

    Anything not expressed in ``ParamsSpec`` is STATIC.**
    """

    _VARARGS_KEY = "*args"
    _VARKW_KEY = "**kwargs"

    def __init__(
        self,
        params: dict[str, ParamsSpecValue] | None = None,
    ) -> None:
        self._named_args: dict[str, IntermediateSpec] = {}
        self._varargs: list[IntermediateSpec] | None = None
        self._varkw: dict[str, IntermediateSpec] | None = None
        if params is None:
            return
        for key, value in params.items():
            if key == self._VARARGS_KEY:
                if not isinstance(value, list):
                    raise ValueError(
                        f"ParamsSpec {self._VARARGS_KEY!r} value must be a list "
                        f"of leaf specs, got {type(value).__name__}"
                    )
                for i, v in enumerate(value):
                    _validate_spec_sym(v, where=f"ParamsSpec '{key}'[{i}]")
                self._varargs = list(value)
            elif key == self._VARKW_KEY:
                if not isinstance(value, dict):
                    raise ValueError(
                        f"ParamsSpec {self._VARKW_KEY!r} value must be a dict "
                        f"of leaf specs, got {type(value).__name__}"
                    )
                for k, v in value.items():
                    _validate_spec_sym(v, where=f"ParamsSpec '{key}'[{k!r}]")
                self._varkw = dict(value)
            elif key.startswith("*"):
                raise ValueError(
                    f"Unknown sentinel key {key!r} in ParamsSpec; only "
                    f"{self._VARARGS_KEY!r} and {self._VARKW_KEY!r} are reserved"
                )
            else:
                _validate_spec_sym(value, where=f"ParamsSpec[{key!r}]")
                self._named_args[key] = cast(IntermediateSpec, value)

    def __repr__(self) -> str:
        def _indent_lines(text: str, prefix: str = _INDENT) -> str:
            return "\n".join(prefix + line for line in text.splitlines())

        def _entry_repr(key: str, value: Any) -> str:
            v_repr = repr(value)
            if "\n" in v_repr:
                return f"{key}:\n{_indent_lines(v_repr)}"
            return f"{key}: {v_repr}"

        lines = [_entry_repr(k, v) for k, v in self._named_args.items()]
        if self._varargs is not None:
            inner = "\n".join(
                _entry_repr(str(i), v) for i, v in enumerate(self._varargs)
            )
            lines.append(f"{self._VARARGS_KEY}:\n{_indent_lines(inner)}")
        if self._varkw is not None:
            inner = "\n".join(_entry_repr(k, v) for k, v in self._varkw.items())
            lines.append(f"{self._VARKW_KEY}:\n{_indent_lines(inner)}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            name: value.to_jsonable() if hasattr(value, "to_jsonable") else value
            for name, value in self._named_args.items()
        }
        if self._varargs is not None:
            params[self._VARARGS_KEY] = [
                v.to_jsonable() if hasattr(v, "to_jsonable") else v
                for v in self._varargs
            ]
        if self._varkw is not None:
            params[self._VARKW_KEY] = {
                name: value.to_jsonable() if hasattr(value, "to_jsonable") else value
                for name, value in self._varkw.items()
            }
        return {
            "type": "ParamsSpec",
            "params": params,
        }


class ShapesSpec:
    """Top-level shape specification for a ``torch.compile`` call.

    ``params`` describes the arguments of the compiled callable — for a raw
    function this is the function's parameters, for an ``nn.Module`` this
    is the parameters of ``forward`` (excluding ``self``).

    ``assumptions`` is an optional list of ``SymBool`` expressions built
    from spec ``IntVar`` / ``ShapeVar`` values. Each assumption is wired
    into the shape env at compile time and asserted at runtime via the
    deferred-runtime-assert mechanism::

        from torch.fx.experimental.dynamic_spec import (
            ParamsSpec as PARAMS,
            ShapesSpec,
            ShapeVar as VAR,
            TensorSpec as T,
        )

        batch = VAR("batch", min=2, max=128)
        ep = torch.export.export(
            mod,
            (torch.randn(8, 3), torch.randn(16, 3)),
            dynamic_shapes=ShapesSpec(
                params=PARAMS(
                    {
                        "x": T([batch, 3]),
                        "y": T([batch * 2, 3]),  # derived expression
                    }
                ),
                assumptions=[batch % 2 == 0],
            ),
            strict=True,
        )

    ``globals`` is reserved for future use and will raise
    ``NotImplementedError`` if set.
    """

    def __init__(
        self,
        params: ParamsSpec | dict[str, ParamsSpecValue] | None = None,
        *,
        globals: Any = None,
        assumptions: Sequence[SymBool] | None = None,
    ) -> None:
        # Normalize attributes up front so partially-constructed instances
        # (e.g. when a later check raises) still have a stable shape.
        self._params: ParamsSpec | None = None
        self._assumptions: list[SymBool] = []

        if globals is not None:
            raise NotImplementedError("ShapesSpec.globals is not supported yet")
        # Auto-wrap a bare dict so callers can write
        # ``ShapesSpec({"x": ...})`` instead of
        # ``ShapesSpec(params=ParamsSpec({"x": ...}))``.
        if isinstance(params, dict):
            params = ParamsSpec(params)
        elif params is not None and not isinstance(params, ParamsSpec):
            raise TypeError(
                f"shapes_spec must be ParamsSpec, dict, or None, "
                f"got {type(params).__name__}"
            )
        self._params = params

        if assumptions is not None:
            for i, a in enumerate(assumptions):
                if not isinstance(a, SymBool):
                    raise TypeError(
                        f"ShapesSpec.assumptions[{i}]: expected SymBool, "
                        f"got {type(a).__name__}: {a!r}"
                    )
                _validate_spec_sym(a, where=f"ShapesSpec.assumptions[{i}]")
                self._assumptions.append(a)

    @property
    def assumptions(self) -> list[SymBool]:
        return list(self._assumptions)

    def __repr__(self) -> str:
        lines = ["shapes_spec:"]
        if self._params is not None:
            lines.append(f"{_INDENT}params:")
            param_repr = repr(self._params)
            for line in param_repr.splitlines():
                lines.append(_INDENT * 2 + line)
        if self._assumptions:
            lines.append(f"{_INDENT}assumptions:")
            for a in self._assumptions:
                lines.append(_INDENT * 2 + repr(a))
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "ShapesSpec",
            "params": None if self._params is None else self._params.to_jsonable(),
            "assumptions": [repr(a) for a in self._assumptions],
        }


# Shared error message for rejecting `dynamic_shapes=ShapesSpec(...)`
# combined with `prefer_deferred_runtime_asserts_over_guards=True`. Used
# in multiple export entry points; lifted out so the wording can't drift.
_SHAPES_SPEC_VS_DEFERRED_RUNTIME_ASSERTS_MSG = (
    "`prefer_deferred_runtime_asserts_over_guards=True` cannot be "
    "combined with `dynamic_shapes=ShapesSpec(...)`. ShapesSpec "
    "currently uses unbacked symbols only, which already emit "
    "runtime assertions; the flag has no effect."
)


def _coerce_to_shapes_spec(
    x: Any,
) -> ShapesSpec | None:
    """Normalize a user-supplied dynamic-spec value to ``ShapesSpec | None``.

    Accepts ``None``, an existing ``ShapesSpec``, a ``ParamsSpec``, or a plain
    ``dict`` (the same shape ``ParamsSpec`` accepts). The dict and
    ``ParamsSpec`` forms are auto-wrapped by ``ShapesSpec.__init__``.
    """
    if x is None:
        return None
    if isinstance(x, ShapesSpec):
        return x
    if isinstance(x, (dict, ParamsSpec)):
        return ShapesSpec(x)
    raise TypeError(
        f"dynamic spec expects a dict, ShapesSpec, or ParamsSpec, "
        f"got {type(x).__name__}"
    )


# Attribute used to store the resolved ``ShapesSpec`` on a decorated function
# (or ``nn.Module.forward``). Read by ``_resolve_dynamic_shapes`` below.
_DYNAMIC_SPEC_ATTR = "_dynamic_spec"


def dynamic_spec(spec: Any) -> Any:
    """Attach a ``ShapesSpec`` to a function (or ``nn.Module.forward``).

    When :func:`torch.compile`, :func:`torch.export.export`, or
    :func:`torch.fx.experimental.proxy_tensor.make_fx` is invoked on a
    function (or module) that carries this metadata, the attached
    ``ShapesSpec`` is used as the dynamic-shape spec **unless** the caller
    also passes an explicit ``dynamic_shapes=`` kwarg, in which case the
    two-source ambiguity is rejected with ``ValueError``.

    ``spec`` accepts the same types as the call-site
    ``dynamic_shapes=`` a ``dict``, a ``ParamsSpec``, or a ``ShapesSpec``.
    To attach assumptions, build a ``ShapesSpec(params=..., assumptions=[...])``
    and pass it.

    Usage::

        # Simple per-param mapping (no assumptions):
        @dynamic_spec({"x": TensorSpec([ShapeVar("B"), STATIC])})
        def fn(x): ...


        # With assumptions: build a ShapesSpec.
        B = ShapeVar("B")


        @dynamic_spec(
            ShapesSpec(
                ParamsSpec({"x": TensorSpec([B, STATIC])}),
                assumptions=[B % 2 == 0],
            )
        )
        def fn(x): ...

    Equivalent for an ``nn.Module``::

        class M(nn.Module):
            @dynamic_spec({"x": TensorSpec([ShapeVar("B"), STATIC])})
            def forward(self, x): ...
    """
    resolved = _coerce_to_shapes_spec(spec)
    if resolved is None:
        raise TypeError(
            f"dynamic_spec(): `spec` must be a dict, ParamsSpec, or "
            f"ShapesSpec, got {type(spec).__name__}."
        )

    def _decorator(fn: Any) -> Any:
        if hasattr(fn, _DYNAMIC_SPEC_ATTR):
            raise ValueError(
                "@dynamic_spec(...) is already attached to "
                f"{getattr(fn, '__qualname__', repr(fn))}. Stacking multiple "
                "@dynamic_spec decorators on the same function is not "
                "allowed; merge them into a single spec."
            )
        setattr(fn, _DYNAMIC_SPEC_ATTR, resolved)
        return fn

    return _decorator


def _resolve_dynamic_shapes(
    fn_or_module: Callable[..., Any] | torch.nn.Module,
    dynamic_shapes_kwarg: Any,
) -> Any:
    """Resolve the effective dynamic-shapes spec for a call site.

    Reads ``@dynamic_spec(...)`` metadata from ``fn_or_module`` (for an
    ``nn.Module``, the metadata is read from its bound ``forward`` method).
    Raises ``ValueError`` if both the decorator and an explicit
    ``dynamic_shapes_kwarg`` are present, since the precedence is ambiguous.
    Returns ``dynamic_shapes_kwarg`` if the decorator is absent, else the
    attached ``ShapesSpec``.
    """
    fn = fn_or_module
    import torch.nn

    if isinstance(fn_or_module, torch.nn.Module):
        fn = fn_or_module.forward
    attached = getattr(fn, _DYNAMIC_SPEC_ATTR, None)
    if attached is None:
        return dynamic_shapes_kwarg
    if dynamic_shapes_kwarg is not None:
        raise ValueError(
            "Both `@dynamic_spec(...)` is attached to the function/forward "
            "AND a `dynamic_shapes=` argument was passed. Provide only one."
        )
    log.info(
        "Using @dynamic_spec attached to %s: %s",
        getattr(fn, "__qualname__", repr(fn)),
        attached,
    )
    return attached
