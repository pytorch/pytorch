# mypy: allow-untyped-defs
"""Symbolic shape helpers shared by GEMM epilogue frontends."""

from collections.abc import Sequence
from typing import Any

import sympy

import torch
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import (
    guard_int,
    has_guarding_hint,
    statically_known_true as fx_statically_known_true,
)


def normalize_shape(shape: Any) -> Any:
    """Canonicalize sequence-like shapes to tuples."""
    return tuple(shape) if isinstance(shape, (list, tuple, torch.Size)) else shape


def guarded_int(value: Any) -> int | None:
    """Return an integer after guarding backed symbolic values."""
    if isinstance(value, torch.fx.Node):
        value = value.meta.get("val")
    if isinstance(value, torch.SymInt):
        if not has_guarding_hint(value):
            return None
        # TODO: Defer speculative guards if they become a meaningful source of
        # recompilation.
        return guard_int(value)
    return value if isinstance(value, int) else None


def statically_known(expr: Any) -> bool:
    """Return whether a symbolic predicate is known true without adding guards."""
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, sympy.Basic):
        return V.graph.sizevars.statically_known_true(expr)
    return fx_statically_known_true(expr)


def statically_known_equal(lhs: Any, rhs: Any) -> bool:
    """Return whether symbolic shape values are known equal without adding guards."""
    return statically_known(lhs == rhs)


def statically_known_shape_equal(
    actual_shape: Sequence[Any], expected_shape: Sequence[Any]
) -> bool:
    """Compare possibly symbolic shape tuples without adding guards."""
    return len(actual_shape) == len(expected_shape) and all(
        statically_known_equal(actual, expected)
        for actual, expected in zip(actual_shape, expected_shape)
    )
