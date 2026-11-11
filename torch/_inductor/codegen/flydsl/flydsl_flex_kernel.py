# mypy: allow-untyped-defs
"""Render lowered flex score_mod subgraphs into FlyDSL template source."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import hashlib
import os
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet

from ...ir import ComputedBuffer, InputBuffer
from ...virtualized import V
from ..common import CSE, IndentedBuffer, PythonPrinter, ValueRanges
from .flydsl_kernel import FlyDSLTemplateKernel
from .flydsl_op_overrides import (
    FlyDSLCSEVariable,
    FlyDSLOpOverrides,
    RUNTIME_ALIAS,
    RUNTIME_MODULE,
)
from .flydsl_template import FlyDSLTemplate


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ...ir import Buffer


_COORDINATES = ("b", "h", "m", "n")


class FlyDSLPrinter(PythonPrinter):
    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("ToFloat expects exactly one argument")
        return f"{RUNTIME_ALIAS}.to_f32({self.doprint(expr.args[0])})"

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        return f"({self.doprint(x)} // {self.doprint(div)})"


flydsl_pexpr = FlyDSLPrinter().doprint


@functools.cache
def _runtime_source_hash() -> str:
    # FlyDSL's JIT cache does not see the shim the generated mods call into.
    path = os.path.join(os.path.dirname(__file__), "flydsl_mod_runtime.py")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _hierarchical_indexer(size: Sequence[Any]) -> Callable[[Sequence[Any]], Any]:
    # Captured views reach the kernel as reinterpreted tensors, so ignore the offset.
    from ...kernel.flex.flex_flash_attention import HierarchicalIndex

    def indexer(indices: Sequence[Any]) -> Any:
        if len(indices) != len(size):
            raise AssertionError(
                f"Rank mismatch: got {len(indices)} indices for rank {len(size)}"
            )
        if not indices:
            return sympy.Integer(0)
        if len(indices) == 1:
            return indices[0]
        return HierarchicalIndex(*indices)

    return indexer


@contextlib.contextmanager
def _patch_capture_indexer():
    from ...kernel.flex.flex_flash_attention import patch_fixed_layout_indexer

    with patch_fixed_layout_indexer(
        lambda size, stride, offset: _hierarchical_indexer(size)
    ):
        yield


@dataclasses.dataclass
class FlyDSLSubgraphInfo:
    body: IndentedBuffer
    cse: CSE[Any, str] | None = None

    def to_dict(self):
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


class FlyDSLFlexTemplateKernel(FlyDSLTemplateKernel):
    """FlyDSL template kernel that inlines lowered score_mod subgraphs."""

    overrides = FlyDSLOpOverrides  # type: ignore[assignment]

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
        subgraphs: Sequence[Any] | None = None,
        aux_buffers: Sequence[Buffer] | None = None,
    ) -> None:
        super().__init__(
            kernel_name=kernel_name, input_nodes=input_nodes, output_node=output_node
        )
        self.subgraphs = list(subgraphs) if subgraphs is not None else None
        self.aux_buffers = list(aux_buffers or ())
        self.subgraph_bodies: dict[str, FlyDSLSubgraphInfo] = {}
        self.body: IndentedBuffer = IndentedBuffer()
        self.cse = CSE(name_prefix="tmp")
        self._aux_slots: dict[tuple[int, tuple[int, ...]], int] = {}
        self._aux_slot_specs: list[tuple[int, tuple[int, ...], int]] = []
        self._mod_scalars: dict[str, sympy.Expr] = {}

    def create_cse_var(self, *args, **kwargs):
        return FlyDSLCSEVariable(*args, **kwargs)

    def kexpr(self, expr: sympy.Expr) -> str:
        return flydsl_pexpr(expr)

    def gen_imports(self) -> str:
        imports = super().gen_imports()
        if not self.subgraph_bodies:
            return imports
        return imports + f"import {RUNTIME_MODULE} as {RUNTIME_ALIAS}\n"

    def template_env(self, **kwargs) -> dict[str, Any]:
        return {
            **super().template_env(**kwargs),
            "modification": self.modification,
            "mod_key": self.mod_key,
            "aux_slots": self.aux_slots,
            "scalar_defines": self.scalar_defines,
        }

    def aux_buffer_index(self, name: str) -> int:
        matches = [
            index
            for index, buffer in enumerate(self.aux_buffers)
            if buffer.get_name() == name
        ]
        if len(matches) != 1:
            raise NotImplementedError(
                f"score_mod reads {name}, which is not exactly one captured tensor"
            )
        return matches[0]

    def aux_slot(
        self, name: str, dim_indices: Sequence[sympy.Expr], coord_names: Sequence[str]
    ) -> tuple[int, dict[int, str]]:
        """Return the aux slot for one read and any computed reader arguments.

        A slot reads ``sum(stride_j * coord_j)`` with one stride per coordinate (0
        broadcasts), so a tensor read with two index patterns takes two slots. An axis
        indexed by a computed value takes a reader position no coordinate uses.
        """
        buffer_index = self.aux_buffer_index(name)
        buffer = self.aux_buffers[buffer_index]
        sizes = [V.graph.sizevars.guard_int(s) for s in buffer.get_size()]
        strides = [V.graph.sizevars.guard_int(s) for s in buffer.get_stride()]
        if len(dim_indices) != len(sizes):
            raise AssertionError(
                f"captured tensor {name} has rank {len(sizes)} but was indexed with "
                f"{len(dim_indices)} coordinates"
            )
        coord_positions = {
            coord: position for position, coord in enumerate(coord_names)
        }
        spec = [0, 0, 0, 0]
        claimed: OrderedSet[int] = OrderedSet()
        computed: list[tuple[int, sympy.Expr]] = []
        for axis, index_expr in enumerate(dim_indices):
            expr = self.rename_indexing(index_expr)
            if expr == sympy.Integer(0) or sizes[axis] == 1:
                continue
            position = (
                coord_positions.get(str(expr))
                if isinstance(expr, sympy.Symbol)
                else None
            )
            if position is None:
                computed.append((axis, expr))
                continue
            if position in claimed:
                raise NotImplementedError(
                    f"captured tensor {name} is indexed on two axes by {expr}; the "
                    "FlyDSL aux reader has one stride per coordinate"
                )
            claimed.add(position)
            spec[position] = strides[axis]

        overrides: dict[int, str] = {}
        for axis, expr in computed:
            position = next((p for p in range(4) if p not in claimed), None)
            if position is None:
                raise NotImplementedError(
                    f"captured tensor {name} is indexed by {expr} on axis {axis}, but "
                    "every aux reader position is already taken by a coordinate"
                )
            claimed.add(position)
            spec[position] = strides[axis]
            overrides[position] = self.kexpr(expr)

        numel = 1 + sum((size - 1) * stride for size, stride in zip(sizes, strides))
        key = (buffer_index, tuple(spec))
        slot = self._aux_slots.get(key)
        if slot is None:
            slot = len(self._aux_slot_specs)
            self._aux_slots[key] = slot
            self._aux_slot_specs.append((buffer_index, tuple(spec), numel))
        return slot, overrides

    def aux_slots(self) -> tuple[tuple[int, tuple[int, ...], int], ...]:
        """``(buffer index, (stride_b, stride_h, stride_q, stride_kv), numel)`` per slot."""
        return tuple(self._aux_slot_specs)

    def scalar_defines(self) -> str:
        """Bind size symbols referenced by the mods as (guarded) module constants."""
        return "\n".join(
            f"{arg_name} = {V.graph.sizevars.guard_int(expr)}"
            for arg_name, expr in self._mod_scalars.items()
        )

    def mod_key(self, *extra: Any) -> str:
        """Hash of the rendered mod bodies, the runtime shim and ``extra``."""
        if not self.subgraph_bodies:
            raise AssertionError("mod_key() called before any modification() rendered")
        payload = "\x00".join(
            [
                *(
                    f"{name}\x00{info.body.getvalue()}"
                    for name, info in sorted(self.subgraph_bodies.items())
                ),
                _runtime_source_hash(),
                *(str(e) for e in extra),
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):
        old_state = {
            key.name: getattr(self, key.name)
            for key in dataclasses.fields(FlyDSLSubgraphInfo)
        }
        subgraph = self.subgraph_bodies[body_name]
        for key, value in subgraph.to_dict().items():
            if value is None and key == "cse":
                continue
            setattr(self, key, value)
        try:
            yield
        finally:
            self.subgraph_bodies[body_name] = FlyDSLSubgraphInfo(
                **{
                    key.name: getattr(self, key.name)
                    for key in dataclasses.fields(FlyDSLSubgraphInfo)
                }
            )
            for key, value in old_state.items():
                setattr(self, key, value)

    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str):
        if body_name in self.subgraph_bodies:
            raise AssertionError(f"Subgraph body '{body_name}' already exists")
        self.subgraph_bodies[body_name] = FlyDSLSubgraphInfo(
            body=IndentedBuffer(), cse=self.cse.clone()
        )
        with self.set_subgraph_body(body_name):
            yield

    def modification(
        self, subgraph_number: int, output_name: str, **fixed_inputs: str
    ) -> str:
        """Render a subgraph as FlyDSL statements binding ``output_name``.

        ``fixed_inputs`` maps placeholder names to template variables (``m`` -> ``q_idx``).
        """
        if self.subgraphs is None or not 0 <= subgraph_number < len(self.subgraphs):
            raise AssertionError(f"Invalid subgraph number {subgraph_number}")
        subgraph = self.subgraphs[subgraph_number]
        num = 0
        while f"mod_{subgraph_number}_{num}" in self.subgraph_bodies:
            num += 1
        with self.create_subgraph_body(f"mod_{subgraph_number}_{num}"):
            handler = ModificationWrapperFlyDSL(self, subgraph_number, fixed_inputs)
            sizevars_before = OrderedSet(self.args.sizevars)
            with (
                V.set_kernel_handler(self),
                V.set_ops_handler(handler),
                _patch_capture_indexer(),
            ):
                if isinstance(subgraph, list):
                    raise NotImplementedError(
                        "FlyDSL flex mods do not support scatter graphs"
                    )
                if not isinstance(subgraph, ComputedBuffer):
                    raise AssertionError(
                        f"Expected ComputedBuffer, got {type(subgraph)}"
                    )
                if isinstance(subgraph.data, InputBuffer):
                    out = subgraph.data.make_loader()(())
                else:
                    out = subgraph.data.inner_fn(())
            for expr, arg_name in self.args.sizevars.items():
                if expr not in sizevars_before:
                    self._mod_scalars[arg_name] = expr
            if out is None:
                raise AssertionError(f"Expected a result for {output_name}")
            self.body.writeline(f"{output_name} = {out}")
            return self.body.getvalue()


class ModificationWrapperFlyDSL(V.WrapperHandler):  # type: ignore[name-defined]
    """Resolve subgraph placeholders and captured-tensor reads."""

    def __init__(
        self,
        kernel: FlyDSLFlexTemplateKernel,
        subgraph_number: int,
        fixed_inputs: dict[str, str],
    ):
        super().__init__(FlyDSLOpOverrides())
        self.name = f"FlyDSLPlaceholderSubstitution_{subgraph_number}"
        self.kernel = kernel
        self.fixed_inputs = dict(fixed_inputs)

    def load(self, name: str, index: sympy.Expr):
        if name in self.fixed_inputs:
            value = self.fixed_inputs[name]
            dtype = torch.int32 if name in _COORDINATES else torch.float32
            result = self.kernel.cse.generate(
                self.kernel.body, value, bounds=ValueRanges.unknown(), dtype=dtype
            )
            if isinstance(result, FlyDSLCSEVariable) and dtype == torch.int32:
                from ...utils import sympy_index_symbol

                result.index_expr = sympy_index_symbol(value)
            return result
        return self._load_captured_tensor(name, index)

    def _load_captured_tensor(self, name: str, index: sympy.Expr):
        from ...kernel.flex.flex_flash_attention import HierarchicalIndex

        missing = [axis for axis in _COORDINATES if axis not in self.fixed_inputs]
        if missing:
            raise AssertionError(f"modification() needs coordinates {missing}")
        buffer = self.kernel.aux_buffers[self.kernel.aux_buffer_index(name)]
        dim_indices = index.args if isinstance(index, HierarchicalIndex) else (index,)
        if not buffer.get_size():
            dim_indices = ()
        coord_names = [self.fixed_inputs[axis] for axis in _COORDINATES]
        slot, overrides = self.kernel.aux_slot(name, dim_indices, coord_names)
        reader_args = list(coord_names)
        for position, expr in overrides.items():
            reader_args[position] = f"{RUNTIME_ALIAS}.to_i32({expr})"
        return self.kernel.cse.generate(
            self.kernel.body,
            f"aux[{slot}]({', '.join(reader_args)})",
            dtype=torch.float32,
            bounds=ValueRanges.unknown(),
        )

    def store(self, name, index, value, mode=None):
        raise NotImplementedError("Stores are not supported in FlyDSL flex mods")

    def indirect_indexing(self, index_var, size, check=True, wrap_neg=True):
        from ...utils import sympy_index_symbol

        if isinstance(index_var, (int, sympy.Integer)):
            if wrap_neg and index_var < 0:
                return V.graph.sizevars.simplify(sympy.Integer(index_var) + size)
            return sympy.Integer(index_var)

        index_expr = getattr(index_var, "index_expr", None)
        if wrap_neg and (
            index_expr is None
            or not V.graph.sizevars.statically_known_geq(index_expr, 0)
        ):
            wrapped = self.kernel.cse.newvar(dtype=torch.int32)
            size_expr = f"{RUNTIME_ALIAS}.const_i32({self.kernel.kexpr(size)})"
            self.kernel.body.writeline(
                f"{wrapped} = {RUNTIME_ALIAS}.where("
                f"{RUNTIME_ALIAS}.lt({index_var}, {RUNTIME_ALIAS}.const_i32(0)), "
                f"{RUNTIME_ALIAS}.add({index_var}, {size_expr}), {index_var})"
            )
            return sympy_index_symbol(str(wrapped))

        for expr, var in self.kernel.cse._cache.items():
            if var is index_var and isinstance(expr, str) and expr.isidentifier():
                return sympy_index_symbol(expr)
        return sympy_index_symbol(str(index_var))


class FlyDSLFlexTemplate(FlyDSLTemplate):
    """FlyDSL template whose kernel can inline score_mod subgraphs."""

    kernel_type: type[Any] = FlyDSLFlexTemplateKernel

    def generate(self, **kwargs: Any) -> Any:
        # Bind on a copy: the base builds kernels here and in make_kernel_render.
        subgraphs = kwargs.pop("subgraphs", None)
        aux_buffers = kwargs.pop("aux_buffers", None)
        bound = copy.copy(self)
        bound.kernel_type = functools.partial(  # type: ignore[assignment]
            self.kernel_type, subgraphs=subgraphs, aux_buffers=aux_buffers
        )
        return FlyDSLTemplate.generate(bound, **kwargs)

    def render_kernel(
        self,
        kernel_name: str,
        input_nodes: list[Any],
        output_node: Any,
        subgraphs: Sequence[Any] | None = None,
        aux_buffers: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> tuple[FlyDSLFlexTemplateKernel, str]:
        kernel = self.kernel_type(
            kernel_name=kernel_name,
            input_nodes=input_nodes,
            output_node=output_node,
            subgraphs=subgraphs,
            aux_buffers=aux_buffers,
        )
        partial = kernel.render(self.template, **kwargs)
        return kernel, partial.finalize_all()
