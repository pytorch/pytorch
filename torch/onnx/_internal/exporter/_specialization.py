"""Detect dynamic input dimensions specialized before ONNX translation."""

from __future__ import annotations

import dataclasses
import re
import warnings
from typing import TYPE_CHECKING

import sympy

import torch
from torch.export.exported_program import _get_shape_env
from torch.export.graph_signature import InputKind
from torch.onnx._internal.exporter import _decomp, _errors
from torch.utils._sympy.value_ranges import bound_sympy


if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch.onnx._internal.exporter import _registration
    from torch.utils._sympy.value_ranges import ValueRanges


@dataclasses.dataclass(frozen=True)
class _Snapshot:
    n_guards: int
    deferred_asserts: set[sympy.Expr]
    replacements: set[sympy.Symbol]
    var_to_range: tuple[tuple[sympy.Symbol, ValueRanges], ...]
    input_dims: dict[sympy.Symbol, str]
    candidate_ops: tuple[str, ...]


def _input_dim_names(
    exported_program: torch.export.ExportedProgram, shape_env: ShapeEnv
) -> dict[sympy.Symbol, str]:
    user_inputs = {
        spec.arg.name
        for spec in exported_program.graph_signature.input_specs
        if spec.kind == InputKind.USER_INPUT
    }
    names: dict[sympy.Symbol, str] = {}
    for node in exported_program.graph.nodes:
        if node.op != "placeholder" or node.name not in user_inputs:
            continue
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            continue
        for i, dim in enumerate(value.shape):
            if not isinstance(dim, torch.SymInt):
                continue
            for symbol in sorted(dim.node.expr.free_symbols, key=str):
                sources = shape_env.var_to_sources.get(symbol, ())
                if sources:
                    names[symbol] = re.sub(r"L\['([^']+)'\]", r"\1", sources[0].name)
                else:
                    names.setdefault(symbol, f"{node.name}.shape[{i}]")
    return names


def snapshot(
    exported_program: torch.export.ExportedProgram,
    registry: _registration.ONNXRegistry,
) -> _Snapshot | None:
    """Save constraints and input dimensions before decomposition mutates ShapeEnv."""
    shape_env = _get_shape_env(exported_program.graph_module)
    if shape_env is None:
        return None
    implemented = set(_decomp.get_onnx_implemented_overloads(registry))
    candidate_ops = {
        str(node.target)
        for node in exported_program.graph.nodes
        if node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and node.target not in implemented
        and not node.target._schema.name.startswith(
            ("aten::_assert_scalar", "aten::sym_constrain_range")
        )
    }
    return _Snapshot(
        n_guards=len(shape_env.guards),
        deferred_asserts={
            ra.expr
            for values in shape_env.deferred_runtime_asserts.values()
            for ra in values
        },
        replacements=set(shape_env.replacements),
        var_to_range=tuple(shape_env.var_to_range.items()),
        input_dims=_input_dim_names(exported_program, shape_env),
        candidate_ops=tuple(sorted(candidate_ops)),
    )


def warn_if_specialized(
    exported_program: torch.export.ExportedProgram, before: _Snapshot | None
) -> None:
    """Warn once about specialization constraints that the ONNX graph cannot enforce."""
    if before is None:
        return
    shape_env = _get_shape_env(exported_program.graph_module)
    if shape_env is None:
        return
    new_constraints = {guard.expr for guard in shape_env.guards[before.n_guards :]}
    new_constraints.update(
        ra.expr
        for values in shape_env.deferred_runtime_asserts.values()
        for ra in values
        if ra.expr not in before.deferred_asserts
    )
    new_constraints.update(
        sympy.Eq(symbol, value)
        for symbol, value in shape_env.replacements.items()
        if symbol not in before.replacements
    )
    input_symbols = set(before.input_dims)
    ranges = dict(before.var_to_range)
    constraints = set()
    for expr in new_constraints:
        if not expr.free_symbols <= input_symbols:
            continue
        if expr.free_symbols <= ranges.keys():
            # ShapeEnv substitutions may already include the new specialization.
            vr = bound_sympy(expr, ranges)
            if vr.is_singleton() and vr.lower is sympy.true:
                continue
        constraints.add(expr)
    if not constraints:
        return

    affected: set[sympy.Symbol] = set()
    for expr in constraints:
        affected.update(expr.free_symbols)
    dims = ", ".join(
        f"{before.input_dims[symbol]} ({symbol})"
        for symbol in sorted(affected, key=str)
    )
    lines = [
        "The exported ONNX model declares these input dimensions as dynamic, but the "
        "graph was specialized to the example input's sizes and will give wrong results "
        "(or fail) for other sizes:",
        f"  {dims}",
        "Constraints added while decomposing ops that have no ONNX implementation "
        "(removed from the ONNX graph):",
    ]
    rendered = [str(expr) for expr in sorted(constraints, key=str)]
    if len(rendered) > 10:
        rendered = [*rendered[:10], f"... and {len(rendered) - 10} more"]
    lines.append("  " + "; ".join(rendered))
    if before.candidate_ops:
        lines.append(
            "Ops decomposed for ONNX that may be responsible: "
            + ", ".join(before.candidate_ops)
        )
    lines.append(
        "To fix: provide an ONNX implementation for the op (custom_translation_table), "
        "or export these dimensions as static."
    )
    warnings.warn(
        "\n".join(lines),
        category=_errors.DynamicShapeSpecializationWarning,
        stacklevel=3,
    )
