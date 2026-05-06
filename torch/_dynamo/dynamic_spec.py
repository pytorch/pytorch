"""Spec types for controlling what is dynamic in compiled/exported code.
Currently only supports unbacked dynamic shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


__all__ = ["ShapeVar", "IntVar", "TensorSpec"]


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
        entries = ", ".join(repr(spec) for spec in self._specs)
        return f"TensorSpec([{entries}])"
