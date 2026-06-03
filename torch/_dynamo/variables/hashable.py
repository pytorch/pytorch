"""
Hashability utilities for PyTorch Dynamo variable tracking.

This module provides the HashableTracker wrapper class and associated utilities
for making VariableTracker instances usable as dictionary keys and set elements
during symbolic execution. Used by both ConstDictVariable and SetVariable.
"""

import collections
from typing import Any, TYPE_CHECKING

import torch

from .. import variables
from ..exc import raise_observed_exception
from ..utils import specialize_symnode
from .base import VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


def raise_unhashable(
    arg: VariableTracker, tx: "InstructionTranslatorBase | None" = None
) -> None:
    if tx is None:
        from torch._dynamo.symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()

    try:
        arg_type = arg.python_type()
    except Exception:
        arg_type = None

    # Safety check: if we know the real Python type and it IS hashable,
    # our is_hashable() disagrees with CPython. Graph-break rather than
    # raising a wrong TypeError.
    if arg_type is not None and arg_type.__hash__ is not None:
        from .. import graph_break_hints
        from ..exc import unimplemented

        unimplemented(
            gb_type="Hashability mismatch",
            context=f"raise_unhashable {arg}",
            explanation=f"Dynamo thinks {arg_type.__name__} is unhashable, "
            f"but its __hash__ is not None. This likely indicates a missing "
            f"or incorrect is_hashable() override.",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    type_name = arg_type.__name__ if arg_type is not None else type(arg).__name__
    raise_observed_exception(
        TypeError,
        tx,
        args=[
            f"unhashable type: '{type_name}'",
        ],
    )


def is_hashable(x: VariableTracker) -> bool:
    # LazyVT optimization: check hashability without realizing the VT to avoid
    # accidentally inserting guards.
    if (
        isinstance(x, variables.LazyVariableTracker)
        and not x.is_realized()
        and x.is_hashable_lazy()
    ):
        return True

    return x.is_hashable()


def _contains_unrealized_source_backed_lazy_constant(value: Any) -> bool:
    # Unlike PyCodegen's container-only scan, hash deferral starts from a single
    # candidate key. Walk the full VT object here so supported composite keys can
    # defer their hash guard when any item reloads from source.
    cache: set[int] = set()

    def visit(obj: Any) -> bool:
        idx = id(obj)
        if idx in cache:
            return False
        cache.add(idx)

        if isinstance(obj, variables.LazyConstantVariable):
            return not obj.is_realized() and obj.source is not None

        if isinstance(obj, HashableTracker):
            return visit(obj.vt)

        if isinstance(obj, VariableTracker):
            obj = obj.unwrap()
            if visit(obj):
                return True
            return any(
                visit(subvalue)
                for key, subvalue in obj.__dict__.items()
                if key not in obj._nonvar_fields
            )

        if isinstance(obj, (list, tuple, set, frozenset)):
            return any(visit(subvalue) for subvalue in obj)

        if isinstance(obj, (dict, collections.OrderedDict)):
            return any(visit(key) or visit(subvalue) for key, subvalue in obj.items())

        return False

    return visit(value)


def _can_defer_hash_guard(vt: VariableTracker) -> bool:
    # Keep these local: list/set variable modules import HashableTracker.
    from .lists import SliceVariable, TupleVariable
    from .sets import FrozensetVariable

    def can_hash_item(item: VariableTracker) -> bool:
        if _contains_unrealized_source_backed_lazy_constant(item):
            return _can_defer_hash_guard(item)
        return is_hashable(item)

    if (
        isinstance(vt, variables.LazyConstantVariable)
        and not vt.is_realized()
        and vt.source is not None
    ):
        return True

    if isinstance(vt, TupleVariable):
        return all(can_hash_item(item) for item in vt.items)

    if isinstance(vt, SliceVariable):
        return vt.is_hashable() and all(can_hash_item(item) for item in vt.items)

    if isinstance(vt, FrozensetVariable):
        return all(can_hash_item(item.vt) for item in vt.set_items)

    return False


class RawHash:
    """Wraps a pre-computed hash value to bypass int.__hash__'s modular reduction.

    When building a tuple/frozenset of per-item hashes, using bare ints would
    apply long_hash (mod sys.hash_info.modulus), corrupting the values.
    Wrapping in RawHash makes tuplehash/frozenset_hash see the original hash.
    """

    __slots__ = ("h",)

    def __init__(self, h: int) -> None:
        self.h = h

    def __hash__(self) -> int:
        return self.h

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RawHash) and self.h == other.h


class HashableTracker:
    """
    Class that wraps a VariableTracker and makes it hashable.
    Note that it's fine to put VTs into dictionaries and sets, but doing so
    does not take into account aliasing.
    """

    _MISSING = object()

    def __init__(self, vt: VariableTracker, *, defer_guard: bool = False) -> None:
        # We specialize SymNodes
        vt = specialize_symnode(vt)
        self.vt: VariableTracker
        self._defer_guard = False

        if defer_guard and _can_defer_hash_guard(vt):
            # Dict side-effect replay can reload this key from its
            # source later. Use identity hashing until a dict operation actually
            # observes key equality and needs the real constant value.
            self._hash = id(self)
            self.vt = vt
            self._defer_guard = True
            return

        # Fast path for unrealized LazyVariableTrackers: check and hash without
        # realizing, to avoid inserting guards.  If the fast-path check fails,
        # fall through to realize the VT and try the full is_hashable check.
        if (
            isinstance(vt, variables.LazyVariableTracker)
            and not vt.is_realized()
            and vt.is_hashable_lazy()
        ):
            self._hash = hash(vt.original_value())
            self.vt = vt
            return

        # Compute hash via the tp_hash slot (generic_hash_impl).
        # For unhashable types, hash_impl raises ObservedTypeError.
        from torch._dynamo.symbolic_convert import InstructionTranslator

        from .object_protocol import generic_hash_impl

        tx = InstructionTranslator.current_tx()
        self._hash, _ = generic_hash_impl(tx, vt)
        self.vt = vt

    @classmethod
    def _maybe_constant_torch_size(cls, vt: VariableTracker) -> object:
        from .lists import SizeVariable
        from .tensor import TensorVariable

        if (
            isinstance(vt, variables.LazyVariableTracker)
            and not vt.is_realized()
            and isinstance(vt.original_value(), torch.Size)
        ):
            return vt.original_value()

        if not isinstance(vt, SizeVariable):
            return cls._MISSING

        items = []
        for item in vt.items:
            if item.is_python_constant():
                items.append(item.as_python_constant())
                continue

            if isinstance(item, TensorVariable):
                proxy = getattr(item, "proxy", None)
                node = getattr(proxy, "node", None)
                meta = getattr(node, "meta", None) if node is not None else None
                example_value = (
                    meta.get("example_value") if isinstance(meta, dict) else None
                )
                constant = getattr(example_value, "constant", None)

                if isinstance(constant, torch.Tensor) and constant.numel() == 1:
                    items.append(constant.item())
                    continue

            return cls._MISSING

        return torch.Size(items)

    def __hash__(self) -> int:
        return self._hash

    def materialize(self) -> "HashableTracker":
        if not self._defer_guard:
            return self
        return HashableTracker(variables.LazyVariableTracker.realize_all(self.vt))

    def __eq__(self, other: object) -> bool:
        """
        Checks equality between two HashableTracker instances.

        Delegates to the VariableTracker's is_python_equal method to compare
        the underlying variable trackers for Python-level equality.

        Args:
            other: Another HashableTracker instance to compare with

        Returns:
            True if the underlying variable trackers are Python-equal, False otherwise
        """
        if not isinstance(other, HashableTracker):
            return False
        if self._defer_guard or other._defer_guard:
            return self is other
        if self.vt is other.vt:
            return True

        self_constant = self._maybe_constant_torch_size(self.vt)
        other_constant = self._maybe_constant_torch_size(other.vt)
        if self_constant is not self._MISSING and other_constant is not self._MISSING:
            return self_constant == other_constant

        return self.vt.is_python_equal(other.vt)
