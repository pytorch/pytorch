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
from torch.utils._sympy.functions import Mod, PythonMod
from torch.utils._sympy.value_ranges import bound_sympy


if TYPE_CHECKING:
    from collections.abc import Callable

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
    candidate_targets: frozenset[torch._ops.OpOverload]
    shape_env: ShapeEnv
    by_op: dict[str, list[sympy.Expr]] = dataclasses.field(default_factory=dict)


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
        node.target
        for node in exported_program.graph.nodes
        if node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and not node.target._schema.is_mutable
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
        candidate_ops=tuple(sorted(map(str, candidate_ops))),
        candidate_targets=frozenset(candidate_ops),
        shape_env=shape_env,
    )


def attribution_hook(before: _Snapshot | None) -> Callable[[dict], dict] | None:
    """Record constraints added by each outermost candidate decomposition."""
    if before is None:
        return None
    shape_env = before.shape_env
    candidate_targets = before.candidate_targets
    by_op = before.by_op
    depth = 0

    def wrap(op, decompose):
        def attributed(*args, **kwargs):
            nonlocal depth
            if depth:
                return decompose(*args, **kwargs)
            n_guards = len(shape_env.guards)
            assert_ids = {
                id(ra)
                for values in shape_env.deferred_runtime_asserts.values()
                for ra in values
            }
            replacements = set(shape_env.replacements)
            depth += 1
            try:
                return decompose(*args, **kwargs)
            finally:
                depth -= 1
                exprs = [guard.expr for guard in shape_env.guards[n_guards:]]
                exprs.extend(
                    ra.expr
                    for values in shape_env.deferred_runtime_asserts.values()
                    for ra in values
                    if id(ra) not in assert_ids
                )
                exprs.extend(
                    sympy.Eq(symbol, value)
                    for symbol, value in shape_env.replacements.items()
                    if symbol not in replacements
                )
                if exprs:
                    by_op.setdefault(str(op), []).extend(exprs)

        return attributed

    def hook(table: dict) -> dict:
        return {
            op: wrap(op, decompose) if op in candidate_targets else decompose
            for op, decompose in table.items()
        }

    return hook


_CONSTRAINT_PRIORITY = {"bound": 0, "relation": 1, "divisibility": 2, "pin": 3}


def _constraint_kind(expr: sympy.Expr) -> str:
    if isinstance(expr, sympy.And):
        return max(
            map(_constraint_kind, expr.args), key=_CONSTRAINT_PRIORITY.__getitem__
        )
    if isinstance(expr, sympy.Eq):
        lhs, rhs = expr.args
        mod_types = (sympy.Mod, Mod, PythonMod)
        if isinstance(lhs, mod_types) or isinstance(rhs, mod_types):
            return "divisibility"
        if (lhs.is_number and rhs.free_symbols) or (rhs.is_number and lhs.free_symbols):
            return "pin"
        if lhs.free_symbols and rhs.free_symbols:
            return "relation"
    return "bound"


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
            # Evaluate against the pre-decomposition ranges only: the ShapeEnv now
            # treats the new asserts and replacements as facts, so its own evaluation
            # would report every new constraint as implied.
            vr = bound_sympy(expr, ranges)
            if vr.is_singleton() and vr.lower is sympy.true:
                continue
        if _constraint_kind(expr) != "bound":
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
    consequence = (
        "will give wrong results (or fail) for other sizes"
        if any(_constraint_kind(expr) == "pin" for expr in constraints)
        else "may give wrong results for sizes that do not satisfy these constraints"
    )
    lines = [
        "The exported ONNX model declares these input dimensions as dynamic, but the "
        f"graph was specialized to the example input's sizes and {consequence}:",
        f"  {dims}",
        "Constraints added while decomposing ops that have no ONNX implementation "
        "(removed from the ONNX graph):",
    ]
    ordered = sorted(
        constraints,
        key=lambda expr: (-_CONSTRAINT_PRIORITY[_constraint_kind(expr)], str(expr)),
    )
    by_op: dict[str, list[str]] = {}
    unattributed = []
    for expr in ordered[:10]:
        ops = sorted(op for op, exprs in before.by_op.items() if expr in exprs)
        if ops:
            for op in ops:
                by_op.setdefault(op, []).append(str(expr))
        else:
            unattributed.append(str(expr))
    lines.extend(f"  {op}: {'; '.join(exprs)}" for op, exprs in sorted(by_op.items()))
    if unattributed:
        lines.append("  " + "; ".join(unattributed))
    if len(ordered) > 10:
        lines.append(f"  ... and {len(ordered) - 10} more")
    if unattributed and before.candidate_ops:
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
