import functools
from typing import Optional

from .base import MutableLocalBase, MutableLocalSource, VariableTracker


class LazyMutableLocal(MutableLocalBase):
    """Container to cache the real VariableTracker"""

    def __init__(self):
        super().__init__(MutableLocalSource.Local)
        self.vt: Optional[VariableTracker] = None


class LazyVariableTracker(VariableTracker):
    """
    A structure that defers the creation of the actual VariableTracker
    for a given underlying value until it is accessed.

    The `realize` function invokes VariableBuilder to produce the real object.
    Once a LazyVariableTracker has been realized, internal bookkeeping will
    prevent double realization.

    This object should be utilized for processing containers, or objects that
    reference other objects where we may not want to take on creating all the
    VariableTrackers right away.
    """

    _nonvar_fields = {"_value", *VariableTracker._nonvar_fields}

    def __init__(self, _value, source, **kwargs):
        super().__init__(source=source, **kwargs)
        self._value = _value
        if self.mutable_local is None:
            self.mutable_local = LazyMutableLocal()
        assert (
            self.source
        ), "Illegal construction. LazyVariableTracker deferred creation utilizes VariableBuilder."

    def realize(self) -> VariableTracker:
        """Force construction of the real VariableTracker"""
        if self.mutable_local.vt is None:
            from ..symbolic_convert import InstructionTranslator
            from .builder import VariableBuilder

            tx = InstructionTranslator.current_tx()
            self.mutable_local.vt = VariableBuilder(tx, self.source)(self._value)
            self.mutable_local.vt.parents_tracker.add(self.parents_tracker)
            self._value = None
        return self.mutable_local.vt.add_options(self)

    def unwrap(self):
        """Return the real VariableTracker if it already exists"""
        if self.is_realized():
            return self.mutable_local.vt
        return self

    def is_realized(self):
        return self.mutable_local.vt is not None

    def clone(self, **kwargs):
        if (
            kwargs.get("source", self.source) is not self.source
            or kwargs.get("_value", self._value) is not self._value
        ):
            self.realize()
        return VariableTracker.clone(self.unwrap(), **kwargs)

    def __str__(self):
        return VariableTracker.__str__(self.unwrap())

    def __getattr__(self, item):
        return getattr(self.realize(), item)

    # most methods are auto-generated below, these are the ones we want to exclude
    add_options = VariableTracker.add_options
    apply = VariableTracker.apply
    copy = VariableTracker.copy
    __post_init__ = VariableTracker.__post_init__
    propagate = VariableTracker.propagate
    __repr__ = VariableTracker.__repr__


def _create_realize_and_forward(name):
    @functools.wraps(getattr(VariableTracker, name))
    def realize_and_forward(self, *args, **kwargs):
        return getattr(self.realize(), name)(*args, **kwargs)

    return realize_and_forward


def _populate():
    for name, value in VariableTracker.__dict__.items():
        if name not in LazyVariableTracker.__dict__:
            if callable(value):
                setattr(LazyVariableTracker, name, _create_realize_and_forward(name))


_populate()
