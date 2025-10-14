"""
Constraint types for ModelSpec.

This module defines the basic constraint types that conditions compile down to.
Each constraint knows how to generate guards and torch._check calls.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from torch._subclasses.fake_tensor import _MetadataIntLike


# ============================================================================
# Enums
# ============================================================================


class Context(Enum):
    """Execution context for compilation rules."""

    GRAD = "grad"
    NO_GRAD = "no_grad"
    INFERENCE = "inference"
    AUTOCAST = "autocast"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"Context.{self.name}"


class CompileAction(Enum):
    """Action to take when condition matches in a custom dispatcher."""

    COMPILE = "compile"
    RAISE_ERROR = "raise_error"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"CompileAction.{self.name}"


# Export for user convenience
COMPILE = CompileAction.COMPILE
RAISE_ERROR = CompileAction.RAISE_ERROR

# ============================================================================
# Constraint Classes
# ============================================================================


class Constraint(ABC):
    """
    Base class for all constraints.

    Each constraint type knows how to convert itself to:
    1. Guards (via to_guard_expression)
    2. torch._check calls (via to_check)
    """

    @abstractmethod
    def to_guard_expression(self, arg_name: str) -> str:
        """
        Generate a guard expression for this constraint.

        Args:
            arg_name: The name of the argument to guard

        Returns:
            str: Python expression that will be used in guard generation
        """

    @abstractmethod
    def to_check(self, arg_name: str) -> str:
        """
        Generate a torch._check call for this constraint.

        Args:
            arg_name: The name of the argument to check

        Returns:
            str: Python code for the torch._check call
        """

    @abstractmethod
    def __repr__(self) -> str:
        """Return a readable representation of this constraint."""


@dataclass
class ShapeConstraint(Constraint):
    """
    Constraint on tensor shape.

    shape can contain:
    - int: Static dimension size
    - _MetadataIntLike: Supports both concrete ints and symbolic ints (SymInt)
    """

    shape: tuple[_MetadataIntLike, ...]
    dynamic_dims: Optional[list[int]] = None  # Which dimensions are dynamic

    def to_guard_expression(self, arg_name: str) -> str:
        parts = []
        for i, dim in enumerate(self.shape):
            # Skip dimensions marked as dynamic
            if self.dynamic_dims and i in self.dynamic_dims:
                parts.append(f"True  # {arg_name}.shape[{i}] is dynamic: s{i}")
            else:
                # Static dimensions get equality guards
                parts.append(f"{arg_name}.shape[{i}] == {dim}")
        return " and ".join(parts) if parts else "True"

    def to_check(self, arg_name: str) -> str:
        checks = []
        for i, dim in enumerate(self.shape):
            # Only add checks for static dimensions
            if not (self.dynamic_dims and i in self.dynamic_dims):
                checks.append(
                    f"torch._check({arg_name}.shape[{i}] == {dim}, "
                    f'"Expected dimension {i} to be {dim}")'
                )
        return "\n".join(checks) if checks else ""

    def __repr__(self) -> str:
        return f"ShapeConstraint(shape={self.shape}, dynamic_dims={self.dynamic_dims})"


@dataclass
class DtypeConstraint(Constraint):
    """Constraint on tensor dtype."""

    dtype: torch.dtype

    def to_guard_expression(self, arg_name: str) -> str:
        return f"{arg_name}.dtype == {self.dtype}"

    def to_check(self, arg_name: str) -> str:
        return (
            f"torch._check({arg_name}.dtype == {self.dtype}, "
            f'"Expected dtype {self.dtype}")'
        )

    def __repr__(self) -> str:
        return f"DtypeConstraint(dtype={self.dtype})"


@dataclass
class RankConstraint(Constraint):
    """Constraint on tensor rank (number of dimensions)."""

    rank: int

    def to_guard_expression(self, arg_name: str) -> str:
        return f"{arg_name}.dim() == {self.rank}"

    def to_check(self, arg_name: str) -> str:
        return (
            f"torch._check({arg_name}.dim() == {self.rank}, "
            f'"Expected rank {self.rank}")'
        )

    def __repr__(self) -> str:
        return f"RankConstraint(rank={self.rank})"


@dataclass
class NoneConstraint(Constraint):
    """Constraint on whether argument is None."""

    is_none: bool  # True for isNone, False for notNone

    def to_guard_expression(self, arg_name: str) -> str:
        if self.is_none:
            return f"{arg_name} is None"
        else:
            return f"{arg_name} is not None"

    def to_check(self, arg_name: str) -> str:
        if self.is_none:
            return f'torch._check({arg_name} is None, "Expected None")'
        else:
            return f'torch._check({arg_name} is not None, "Expected non-None")'

    def __repr__(self) -> str:
        return f"NoneConstraint(is_none={self.is_none})"


@dataclass
class TypeConstraint(Constraint):
    """Constraint on argument type (for non-tensor types)."""

    expected_type: type

    def to_guard_expression(self, arg_name: str) -> str:
        type_name = self.expected_type.__name__
        return f"isinstance({arg_name}, {type_name})"

    def to_check(self, arg_name: str) -> str:
        type_name = self.expected_type.__name__
        return (
            f"torch._check(isinstance({arg_name}, {type_name}), "
            f'"Expected type {type_name}")'
        )

    def __repr__(self) -> str:
        return f"TypeConstraint(expected_type={self.expected_type})"


@dataclass
class TensorSubclassConstraint(Constraint):
    """Constraint on tensor subclass type."""

    subclass_type: type

    def to_guard_expression(self, arg_name: str) -> str:
        type_name = self.subclass_type.__name__
        return f"isinstance({arg_name}, {type_name})"

    def to_check(self, arg_name: str) -> str:
        type_name = self.subclass_type.__name__
        return (
            f"torch._check(isinstance({arg_name}, {type_name}), "
            f'"Expected tensor subclass {type_name}")'
        )

    def __repr__(self) -> str:
        return f"TensorSubclassConstraint(subclass_type={self.subclass_type})"


@dataclass
class LayoutConstraint(Constraint):
    """Constraint on tensor layout."""

    layout: torch.layout

    def to_guard_expression(self, arg_name: str) -> str:
        return f"{arg_name}.layout == {self.layout}"

    def to_check(self, arg_name: str) -> str:
        return (
            f"torch._check({arg_name}.layout == {self.layout}, "
            f'"Expected layout {self.layout}")'
        )

    def __repr__(self) -> str:
        return f"LayoutConstraint(layout={self.layout})"


@dataclass
class DeviceConstraint(Constraint):
    """Constraint on tensor device."""

    device: torch.device

    def to_guard_expression(self, arg_name: str) -> str:
        return f"{arg_name}.device == {self.device}"

    def to_check(self, arg_name: str) -> str:
        return (
            f"torch._check({arg_name}.device == {self.device}, "
            f'"Expected device {self.device}")'
        )

    def __repr__(self) -> str:
        return f"DeviceConstraint(device={self.device})"


# ============================================================================
# Custom Dispatcher Support
# ============================================================================


@dataclass
class SymbolicCondition:
    """
    Condition on symbolic properties like shapes and strides.

    These conditions are typically derived from custom dispatcher functions
    and represent runtime checks on tensor properties that can be symbolically
    evaluated.
    """

    expression: str  # Python expression (e.g., "x.shape[0] % 8 == 0")
    arg_deps: list[str]  # Arguments this condition depends on

    def to_guard_expression(self) -> str:
        """Convert to guard expression (typically SHAPE_ENV guard)."""
        return self.expression

    def to_check(self) -> str:
        """Convert to torch._check call."""
        return f'torch._check({self.expression}, "Condition failed: {self.expression}")'

    def __repr__(self) -> str:
        return f"SymbolicCondition({self.expression!r}, deps={self.arg_deps})"


@dataclass
class DispatchBranch:
    """
    Single branch in a custom dispatcher.

    Each branch represents a possible execution path with:
    - A set of conditions that must all be true (AND-ed together)
    - An action to take (COMPILE or RAISE_ERROR)
    - A unique identifier for this branch
    """

    conditions: list[SymbolicCondition]  # AND-ed together
    action: CompileAction
    branch_id: str

    def __repr__(self) -> str:
        conditions_repr = " AND ".join(c.expression for c in self.conditions)
        return f"DispatchBranch({conditions_repr!r} -> {self.action.name}, id={self.branch_id})"
