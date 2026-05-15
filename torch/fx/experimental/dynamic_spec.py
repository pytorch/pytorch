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


# Type alias for leaf specs (individual argument specifications)
LeafSpec: TypeAlias = TensorSpec | IntVar | int | None
# This will include ListSpec, DictSpec and ObjectSpec
IntermediateSpec: TypeAlias = LeafSpec


class ParamsSpec:
    """Specification for the arguments of a compiled function.

    Describes the dynamic shape behavior for named arguments, *args, and
    **kwargs of a ``torch.compile``-wrapped function::

        def f(x, y, *args, **kwargs):
        #    ^^^^  named_args
        #           ^^^^^  varargs
        #                   ^^^^^^  varkw

    Example::

        ParamsSpec({"x": TensorSpec([ShapeVar("batch"), None])})
    """

    def __init__(
        self,
        named_args: dict[str, IntermediateSpec] | None = None,
        *,
        varargs: list[IntermediateSpec] | None = None,
        varkw: dict[str, IntermediateSpec] | None = None,
    ) -> None:
        self._named_args: dict[str, LeafSpec] = dict(named_args) if named_args else {}
        if varargs is not None:
            raise NotImplementedError("varargs is not supported yet")
        if varkw is not None:
            raise NotImplementedError("varkw is not supported yet")
        self._varargs: list[IntermediateSpec] | None = None
        self._varkw: dict[str, IntermediateSpec] | None = None

    def __repr__(self) -> str:
        lines = []
        for k, v in self._named_args.items():
            v_repr = repr(v)
            if "\n" in v_repr:
                indented = "\n".join(_INDENT + line for line in v_repr.splitlines())
                lines.append(f"{k}:\n{indented}")
            else:
                lines.append(f"{k}: {v_repr}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "ParamsSpec",
            "named_args": {
                name: value.to_jsonable() if hasattr(value, "to_jsonable") else value
                for name, value in self._named_args.items()
            },
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
        params: ParamsSpec | None = None,
        *,
        globals: Any = None,
        assumptions: Any = None,
    ) -> None:
        if globals is not None:
            raise NotImplementedError("ShapesSpec.globals is not supported yet")
        if assumptions is not None:
            raise NotImplementedError("ShapesSpec.assumptions is not supported yet")
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
