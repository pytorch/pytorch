"""Spec types for controlling what is dynamic in compiled/exported code.
Currently only supports unbacked dynamic shapes.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

from torch._dynamo.source import LocalSource


__all__ = [
    "ShapeVar",
    "IntVar",
    "TensorSpec",
    "ParamsSpec",
    "ShapesSpec",
    "lookup_spec_from_dynamo_source",
]


class IntVar:
    """Indicates that a scalar integer argument is dynamic (no implicit range).

    Unbacked dynamic shapes will represent the underlying value in the
    compiled graph.

    Example::

        IntVar("num_heads")
        IntVar("offset", min=-100, max=100)
        IntVar("size", min=1, max=2048, optimization_hint=512)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        optimization_hint: int | None = None,
    ) -> None:
        self.name = name if name is not None else f"_intvar_{id(self):x}"
        self.min = min
        self.max = max
        self.optimization_hint = optimization_hint

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}"]
        if self.min is not None:
            parts.append(f"min={self.min}")
        if self.max is not None:
            parts.append(f"max={self.max}")
        if self.optimization_hint is not None:
            parts.append(f"optimization_hint={self.optimization_hint}")
        return f"{type(self).__name__}({', '.join(parts)})"


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


# Type alias for leaf specs (individual argument specifications)
LeafSpec: TypeAlias = "TensorSpec | IntVar | None"
# Any spec — what public APIs accept.
IntermediateSpec: TypeAlias = LeafSpec


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
        lines = ["TensorSpec("]
        for i, spec in enumerate(self._specs):
            lines.append(f"  {i}: {spec!r}")
        lines.append(")")
        return "\n".join(lines)


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
        lines = ["ParamsSpec("]
        for k, v in self._named_args.items():
            v_repr = repr(v)
            if "\n" in v_repr:
                indented = "\n".join("    " + line for line in v_repr.splitlines())
                lines.append(f"  {k}:\n{indented}")
            else:
                lines.append(f"  {k}: {v_repr}")
        if self._varargs is not None:
            lines.append(f"  *args: {self._varargs!r}")
        if self._varkw is not None:
            lines.append(f"  **kwargs: {self._varkw!r}")
        lines.append(")")
        return "\n".join(lines)


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
        globals: Any = None,
        assumptions: Any = None,
    ) -> None:
        if globals is not None:
            raise NotImplementedError("ShapesSpec.globals is not supported yet")
        if assumptions is not None:
            raise NotImplementedError("ShapesSpec.assumptions is not supported yet")
        self._params = params

    @property
    def params(self) -> ParamsSpec | None:
        return self._params

    def __repr__(self) -> str:
        lines = ["ShapesSpec("]
        if self._params is not None:
            param_repr = repr(self._params)
            indented = "\n".join("  " + line for line in param_repr.splitlines())
            lines.append(f"  params:\n{indented}")
        lines.append(")")
        return "\n".join(lines)


def lookup_spec_from_dynamo_source(source, shapes_spec: ShapesSpec | None) -> LeafSpec:
    """Look up the spec for a function input arg from the shapes_spec.

    Only supports LocalSource with is_input=True (direct function args).
    Returns TensorSpec, IntVar, or None.
    """
    if shapes_spec is None or shapes_spec.params is None:
        return None
    if not isinstance(source, LocalSource) or not source.is_input:
        return None
    return shapes_spec.params._named_args.get(source.local_name)
