# mypy: allow-untyped-defs
"""Symbolic shape helpers shared by GEMM epilogue frontends."""

from collections.abc import Sequence

import sympy

import torch
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import (
    statically_known_true as fx_statically_known_true,
)


def normalize_shape(shape: object) -> object:
    """Canonicalize sequence-like shapes to tuples."""
    return tuple(shape) if isinstance(shape, (list, tuple, torch.Size)) else shape


def statically_known(expr: object) -> bool:
    """Return whether a symbolic predicate is known true without adding guards."""
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, sympy.Basic):
        return V.graph.sizevars.statically_known_true(expr)
    if not isinstance(expr, torch.SymBool):
        raise AssertionError(f"expected a boolean predicate, got {type(expr)}")
    return fx_statically_known_true(expr)


def statically_known_equal(lhs: object, rhs: object) -> bool:
    """Return whether symbolic shape values are known equal without adding guards."""
    return statically_known(lhs == rhs)


def statically_known_shape_equal(
    actual_shape: Sequence[object], expected_shape: Sequence[object]
) -> bool:
    """Compare possibly symbolic shape tuples without adding guards."""
    return len(actual_shape) == len(expected_shape) and all(
        statically_known_equal(actual, expected)
        for actual, expected in zip(actual_shape, expected_shape)
    )
