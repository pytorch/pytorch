from __future__ import annotations

import contextlib
import dataclasses
from typing import Any

import sympy

import torch


_SECTION_ORDER = (
    "body",
    "prologue",
    "indexing",
    "loads",
    "compute",
    "stores",
    "post_loop_combine",
    "post_loop_store",
)
_LOOP_SECTIONS = frozenset(("indexing", "loads", "compute", "stores"))


@dataclasses.dataclass
class StructuredTritonValue:
    name: str
    dtype: str | None = None
    shape: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape is not None else None,
        }


@dataclasses.dataclass
class StructuredTritonOperand:
    kind: str
    value: Any
    dtype: str | None = None
    shape: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape is not None else None,
        }


@dataclasses.dataclass
class StructuredTritonScope:
    id: int
    kind: str
    label: str
    parent: int | None
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
    loop_carried: list[StructuredTritonValue] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "parent": self.parent,
            "attrs": self.attrs,
            "loop_carried": [value.to_dict() for value in self.loop_carried],
        }


@dataclasses.dataclass
class StructuredTritonNode:
    id: int
    kind: str
    op: str
    section: str
    scope: int
    inputs: list[StructuredTritonOperand]
    outputs: list[StructuredTritonValue]
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
    origin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "op": self.op,
            "section": self.section,
            "scope": self.scope,
            "inputs": [operand.to_dict() for operand in self.inputs],
            "outputs": [value.to_dict() for value in self.outputs],
            "attrs": self.attrs,
            "origin": self.origin,
        }


class StructuredTritonKernelIR:
    """
    A sidecar IR that mirrors Triton codegen without changing the existing
    string-based pipeline.  Nodes retain typed values and are attached to an
    explicit scope tree so analyses do not need to reverse-engineer the final
    source text.
    """

    def __init__(
        self,
        *,
        kernel_name: str | None,
        kernel_kind: str,
        numels: dict[str, sympy.Expr],
        range_trees: list[dict[str, Any]],
    ) -> None:
        self.kernel_name = kernel_name
        self.kernel_kind = kernel_kind
        self.numels = {
            prefix: self._normalize_attr_value(numel)
            for prefix, numel in numels.items()
        }
        self.range_trees = [
            {
                str(key): self._normalize_attr_value(value)
                for key, value in range_tree.items()
            }
            for range_tree in range_trees
        ]
        self.section_order = list(_SECTION_ORDER)
        self.nodes: list[StructuredTritonNode] = []
        self.scopes: list[StructuredTritonScope] = []
        self.section_scopes: dict[str, int] = {}
        self._active_scopes: list[int] = []
        self._producer_by_output: dict[str, int] = {}

        root = self._new_scope(
            kind="kernel",
            label=kernel_name or kernel_kind,
            parent=None,
            attrs={"kernel_kind": kernel_kind, "numels": self.numels},
        )
        self.root_scope = root.id
        for section in _SECTION_ORDER:
            self.section_scopes[section] = self.root_scope

        self.reduction_loop_scope: int | None = None
        loop_ranges = [
            range_tree for range_tree in self.range_trees if range_tree["is_loop"]
        ]
        if loop_ranges:
            loop_scope = self._new_scope(
                kind="loop",
                label="reduction_loop",
                parent=self.root_scope,
                attrs={"ranges": loop_ranges},
            )
            self.reduction_loop_scope = loop_scope.id
            for section in _LOOP_SECTIONS:
                self.section_scopes[section] = loop_scope.id

    def set_kernel_name(self, kernel_name: str) -> None:
        self.kernel_name = kernel_name
        self.scopes[self.root_scope].label = kernel_name

    def scope(
        self,
        kind: str,
        label: str,
        *,
        section: str = "compute",
        attrs: dict[str, Any] | None = None,
    ) -> contextlib.AbstractContextManager[None]:
        @contextlib.contextmanager
        def ctx():
            parent = (
                self._active_scopes[-1]
                if self._active_scopes
                else self.section_scopes.get(section, self.root_scope)
            )
            scope = self._new_scope(kind, label, parent, attrs)
            self._active_scopes.append(scope.id)
            try:
                yield
            finally:
                popped_scope = self._active_scopes.pop()
                assert popped_scope == scope.id

        return ctx()

    def register_loop_carried(self, *values: Any) -> None:
        if self.reduction_loop_scope is None:
            return
        loop_scope = self.scopes[self.reduction_loop_scope]
        for value in self._flatten(values):
            normalized = self._normalize_output(value)
            if all(
                existing.name != normalized.name for existing in loop_scope.loop_carried
            ):
                loop_scope.loop_carried.append(normalized)

    def record_node(
        self,
        *,
        kind: str,
        op: str,
        section: str,
        inputs: Any = (),
        outputs: Any = (),
        attrs: dict[str, Any] | None = None,
        origin: str | None = None,
        dedupe_outputs: bool = False,
    ) -> StructuredTritonNode | None:
        normalized_outputs = [
            self._normalize_output(value) for value in self._flatten(outputs)
        ]
        output_names = [value.name for value in normalized_outputs]
        if (
            dedupe_outputs
            and output_names
            and all(name in self._producer_by_output for name in output_names)
        ):
            return None

        node = StructuredTritonNode(
            id=len(self.nodes),
            kind=kind,
            op=op,
            section=section,
            scope=self._current_scope(section),
            inputs=[self._normalize_operand(value) for value in self._flatten(inputs)],
            outputs=normalized_outputs,
            attrs=self._normalize_attr_value(attrs or {}),
            origin=origin,
        )
        self.nodes.append(node)
        for value in normalized_outputs:
            self._producer_by_output.setdefault(value.name, node.id)
        return node

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "kernel_kind": self.kernel_kind,
            "numels": self.numels,
            "range_trees": self.range_trees,
            "section_order": list(self.section_order),
            "section_scopes": dict(self.section_scopes),
            "scopes": [scope.to_dict() for scope in self.scopes],
            "nodes": [node.to_dict() for node in self.nodes],
        }

    def _new_scope(
        self,
        kind: str,
        label: str,
        parent: int | None,
        attrs: dict[str, Any] | None = None,
    ) -> StructuredTritonScope:
        scope = StructuredTritonScope(
            id=len(self.scopes),
            kind=kind,
            label=label,
            parent=parent,
            attrs=self._normalize_attr_value(attrs or {}),
        )
        self.scopes.append(scope)
        return scope

    def _current_scope(self, section: str) -> int:
        if self._active_scopes:
            return self._active_scopes[-1]
        return self.section_scopes.get(section, self.root_scope)

    def _normalize_output(self, value: Any) -> StructuredTritonValue:
        if isinstance(value, StructuredTritonValue):
            return value
        name = getattr(value, "name", None)
        if name is None:
            name = self._normalize_attr_value(value)
        return StructuredTritonValue(
            name=str(name),
            dtype=self._normalize_dtype(getattr(value, "dtype", None)),
            shape=self._normalize_shape(getattr(value, "shape", None)),
        )

    def _normalize_operand(self, value: Any) -> StructuredTritonOperand:
        if isinstance(value, StructuredTritonOperand):
            return value
        if getattr(value, "name", None) is not None:
            kind = "value"
            normalized = str(value.name)
        elif isinstance(value, sympy.Expr):
            kind = "sympy"
            normalized = str(value)
        elif isinstance(value, torch.dtype):
            kind = "dtype"
            normalized = str(value)
        elif callable(value):
            kind = "callable"
            normalized = getattr(
                value, "__qualname__", getattr(value, "__name__", repr(value))
            )
        elif isinstance(value, str):
            kind = "expr"
            normalized = value
        else:
            kind = type(value).__name__
            normalized = self._normalize_attr_value(value)
        return StructuredTritonOperand(
            kind=kind,
            value=normalized,
            dtype=self._normalize_dtype(getattr(value, "dtype", None)),
            shape=self._normalize_shape(getattr(value, "shape", None)),
        )

    def _normalize_shape(self, shape: Any) -> tuple[str, ...] | None:
        if shape is None:
            return None
        if isinstance(shape, tuple):
            return tuple(str(self._normalize_attr_value(dim)) for dim in shape)
        if isinstance(shape, list):
            return tuple(str(self._normalize_attr_value(dim)) for dim in shape)
        return (str(self._normalize_attr_value(shape)),)

    @staticmethod
    def _normalize_dtype(dtype: torch.dtype | None) -> str | None:
        return None if dtype is None else str(dtype)

    def _normalize_attr_value(self, value: Any) -> Any:
        if isinstance(value, (bool, float, int, str)) or value is None:
            return value
        if isinstance(value, sympy.Expr):
            return str(value)
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, StructuredTritonValue):
            return value.to_dict()
        if isinstance(value, StructuredTritonOperand):
            return value.to_dict()
        if isinstance(value, dict):
            return {
                str(key): self._normalize_attr_value(inner)
                for key, inner in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._normalize_attr_value(inner) for inner in value]
        if getattr(value, "name", None) is not None:
            return self._normalize_output(value).to_dict()
        if callable(value):
            return getattr(
                value, "__qualname__", getattr(value, "__name__", repr(value))
            )
        return repr(value)

    def _flatten(self, values: Any) -> list[Any]:
        if values is None:
            return []
        if isinstance(values, (list, tuple)):
            result: list[Any] = []
            for value in values:
                result.extend(self._flatten(value))
            return result
        return [values]
