"""Spec types for controlling what is dynamic in compiled/exported code.
Currently only supports unbacked dynamic shapes.

Pure data classes — no dependency on dynamo, compile, or export, so this
module can be safely consumed by any layer.
"""

from __future__ import annotations

import itertools
from typing import Any, TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


__all__ = [
    "ShapeVar",
    "IntVar",
    "TensorSpec",
    "ObjectSpec",
    "DictSpec",
    "ParamsSpec",
    "ShapesSpec",
    "LeafSpec",
    "IntermediateSpec",
]


# Indent unit for nested repr output. Two spaces per level.
_INDENT = "  "


class IntVar:
    """Indicates that a scalar integer argument is dynamic (no implicit range).

    Unbacked dynamic shapes will represent the underlying value in the
    compiled graph.

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
        self.name = name if name is not None else "anon"
        self._uid = next(IntVar._uid_counter)
        self.min = min
        self.max = max
        self.optimization_hint = optimization_hint

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

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "optimization_hint": self.optimization_hint,
        }


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
        super().__init__(name, min=min, max=max, optimization_hint=optimization_hint)


class TensorSpec:
    """Per-dimension shape specification for a tensor.

    A list-like container of ``ShapeVar | int | None`` with length equal to the
    tensor's dim.

    Per-entry semantics:
    - ``ShapeVar``: unbacked dynamic dimension.
    - ``int``: static dimension pinned to that concrete value.
    - ``None``: unspecified; treated as static. Value is inferred from the
      example input if the consuming API supports it (e.g. ``torch.compile``),
      otherwise an error will be thrown (e.g. ``torch.export`` requires an
      explicit value).

    Example::

        B = ShapeVar("batch")
        TensorSpec([B, None])  # rank 2, dim 0 dynamic
        TensorSpec([B, 10])  # rank 2, dim 0 dynamic, dim 1 static=10
    """

    _DimSpec = ShapeVar | int | None

    def __init__(self, dims: Sequence[TensorSpec._DimSpec]) -> None:
        self._specs: list[TensorSpec._DimSpec] = list(dims)

    def __getitem__(self, index: int) -> TensorSpec._DimSpec:
        return self._specs[index]

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[TensorSpec._DimSpec]:
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

        ObjectSpec({"weight": TensorSpec([ShapeVar("h"), None])})
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

        DictSpec({"x": TensorSpec([ShapeVar("h"), None])})
        DictSpec({"config": DictSpec({"batch": IntVar()})})
        DictSpec({0: TensorSpec([ShapeVar("h"), None])})
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


# Type alias for leaf specs (individual argument specifications)
LeafSpec: TypeAlias = TensorSpec | IntVar | int | None
# Includes containers (``ObjectSpec`` / ``DictSpec``; future ``ListSpec``)
# for nested specs reachable via dynamo source-chain walks.
IntermediateSpec: TypeAlias = LeafSpec | ObjectSpec | DictSpec


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
        params: dict[str, IntermediateSpec] | None = None,
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
                self._varargs = list(value)
            elif key == self._VARKW_KEY:
                if not isinstance(value, dict):
                    raise ValueError(
                        f"ParamsSpec {self._VARKW_KEY!r} value must be a dict "
                        f"of leaf specs, got {type(value).__name__}"
                    )
                self._varkw = dict(value)
            elif key.startswith("*"):
                raise ValueError(
                    f"Unknown sentinel key {key!r} in ParamsSpec; only "
                    f"{self._VARARGS_KEY!r} and {self._VARKW_KEY!r} are reserved"
                )
            else:
                self._named_args[key] = value

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

    Currently only ``params`` is supported::

        ShapesSpec(params=ParamsSpec({"x": TensorSpec([ShapeVar("batch"), None])}))

    ``globals`` and ``assumptions`` are reserved for future use and will
    raise ``NotImplementedError`` if set.
    """

    def __init__(
        self,
        params: ParamsSpec | dict[str, Any] | None = None,
        *,
        globals: Any = None,
        assumptions: Any = None,
    ) -> None:
        if globals is not None:
            raise NotImplementedError("ShapesSpec.globals is not supported yet")
        if assumptions is not None:
            raise NotImplementedError("ShapesSpec.assumptions is not supported yet")
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

    def __repr__(self) -> str:
        lines = ["shapes_spec:"]
        if self._params is not None:
            lines.append(f"{_INDENT}params:")
            param_repr = repr(self._params)
            for line in param_repr.splitlines():
                lines.append(_INDENT * 2 + line)
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "ShapesSpec",
            "params": None if self._params is None else self._params.to_jsonable(),
        }
