from __future__ import annotations

import dataclasses
import weakref

from typing import Optional

import sympy

import torch
import torch.utils.checkpoint

from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
from torch.utils._sympy.value_ranges import ValueRanges


@dataclasses.dataclass
class ConstraintTarget:
    """
    This represents input tensor dimensions.  Don't create this
    class directly; instead, use :func:`torch._export.dynamic_dim`.
    """

    w_tensor: weakref.ReferenceType[torch.Tensor]
    # TODO: We don't need t_id; we can get it off of w_tensor
    t_id: int
    dim: int


@dataclasses.dataclass
class Constraint(ConstraintTarget):
    """
    This represents constraints on input tensor dimensions, e.g., requiring
    them to be fully polymorphic or within some range.  Don't create this
    class directly; instead, use :func:`torch._export.dynamic_dim`.
    """

    # NOTE(avik): In the future, this could be Union[StrictMinMaxConstraint, <other kinds>]
    constraint_range: StrictMinMaxConstraint
    # Represent that `constraint_range` is shared with another ConstraintTarget, which
    # typically arises because of a specified equality with another dynamic dimension.
    shared: Optional[ConstraintTarget] = None

    def _clone_with_range(self, lower=2, upper=sympy.oo):
        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & ValueRanges(lower=lower, upper=upper),
            warn_only=False,
        )
        return Constraint(
            self.w_tensor, self.t_id, self.dim, constraint_range, self.shared
        )

    def __ge__(self, lower):
        return self._clone_with_range(lower=lower)

    def __gt__(self, lower):
        return self._clone_with_range(lower=lower + 1)

    def __le__(self, upper):
        return self._clone_with_range(upper=upper)

    def __lt__(self, upper):
        return self._clone_with_range(upper=upper - 1)

    def __bool__(self):
        # NOTE(avik): We do not support compound expressions like a <= x <= b.
        # This is because Python implicitly desugars them into bool(a <= x) and bool(x <= b),
        # and moreover, enforces that any overload of __bool__ must return True or False.
        # FWIW, sympy also raises TypeError in this case.
        raise TypeError(
            "Cannot determine truth value of Constraint. "
            "If you are trying to combine Constraints with logical connectives, "
            "you can specify them separately instead."
        )

    @property
    def serializable_spec(self):
        # We need a serialization compatible format of the constraint so that it
        # can be savedin the graph module w/o breaking the module serialization.
        # The saved constraints will be used directly for the post-exporting pass
        # that converts constraints to runtime assertion. The saved constraints
        # will not be saved in the serialized module.
        # TODO: A better way is needed. Currently we use 't_id' to map the constraint,
        # which is not reliable
        return {
            "t_id": self.t_id,
            "dim": self.dim,
            "min": self.constraint_range.vr.lower,
            "max": self.constraint_range.vr.upper,
            "shared": (
                None
                if self.shared is None
                else {
                    "t_id": self.shared.t_id,
                    "dim": self.shared.dim,
                }
            ),
        }

    def __eq__(self, other):
        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & other.constraint_range.vr,
            warn_only=False,
        )
        return Constraint(
            self.w_tensor,
            self.t_id,
            self.dim,
            constraint_range,
            shared=ConstraintTarget(other.w_tensor, other.t_id, other.dim),
        )
