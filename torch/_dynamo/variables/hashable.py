"""
Hashability utilities for PyTorch Dynamo variable tracking.

This module provides the HashableTracker wrapper class and associated utilities
for making VariableTracker instances usable as dictionary keys and set elements
during symbolic execution. Used by both ConstDictVariable and SetVariable.
"""

from typing import TYPE_CHECKING

import torch

from .. import variables
from ..exc import raise_observed_exception
from ..utils import specialize_symnode
from .base import VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


def raise_unhashable(
    arg: VariableTracker, tx: "InstructionTranslator | None" = None
) -> None:
    if tx is None:
        from torch._dynamo.symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()
    try:
        arg_type = arg.python_type()
    except Exception:
        arg_type = type(arg)

    raise_observed_exception(
        TypeError,
        tx,
        args=[
            f"unhashable type: {arg_type!r} and variable tracker = {type(arg.realize())}",
        ],
    )


def is_hashable(x: VariableTracker) -> bool:
    # NB - performing isinstance check on a LazVT realizes the VT, accidentally
    # inserting the guard. To avoid this, lazyVT `is_hashable` methods looks at
    # the underlying value without realizing the VT. Consider updating the
    # lazyVT `is_hashable` method if you see unnecessary guarding for a key VT.
    if (
        isinstance(x, variables.LazyVariableTracker)
        and not x.is_realized()
        and x.is_hashable()
    ):
        return True
    return x.is_python_hashable()


class HashableTracker:
    """
    Class that wraps a VariableTracker and makes it hashable.
    Note that it's fine to put VTs into dictionaries and sets, but doing so
    does not take into account aliasing.
    """

    _MISSING = object()

    def __init__(self, vt: VariableTracker) -> None:
        # We specialize SymNodes
        vt = specialize_symnode(vt)

        # If Dynamo does not know the hashability of the vt, it will raise unsupported here
        # TODO(follow-up): check tp_hash via C-level slot detection — unhashable keys
        # (e.g. list) should raise TypeError, not graph break via is_python_hashable/unimplemented.
        if not is_hashable(vt):
            raise_unhashable(vt)
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
        """
        Computes the hash value for the wrapped VariableTracker.

        For unrealized LazyVariableTrackers, uses the hash of the original value
        to avoid realizing the tracker and inserting unnecessary guards.
        For all other cases, delegates to the VariableTracker's get_python_hash method.

        Returns:
            The hash value of the underlying variable tracker
        """
        if (
            isinstance(self.vt, variables.LazyVariableTracker)
            and not self.vt.is_realized()
            and self.vt.is_hashable()
        ):
            return hash(self.vt.original_value())

        maybe_constant = self._maybe_constant_torch_size(self.vt)
        if maybe_constant is not self._MISSING:
            return hash(maybe_constant)

        return self.vt.get_python_hash()

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
        if self.vt is other.vt:
            return True

        self_constant = self._maybe_constant_torch_size(self.vt)
        other_constant = self._maybe_constant_torch_size(other.vt)
        if self_constant is not self._MISSING and other_constant is not self._MISSING:
            return self_constant == other_constant

        return self.vt.is_python_equal(other.vt)
