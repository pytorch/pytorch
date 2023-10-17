import functools
from typing import Optional

from .base import MutableLocalBase, MutableLocalSource, VariableTracker


class LazyMutableLocal(MutableLocalBase):
    """Container to cache the real VariableTracker"""

    def __init__(self):
        super().__init__(MutableLocalSource.Local)
        self.vt: Optional[VariableTracker] = None


class LazyVariableTracker(VariableTracker):
    _nonvar_fields = ["_value"]

    def __init__(self, _value, source, **kwargs):
        super().__init__(source=source, **kwargs)
        self._value = _value
        if self.mutable_local is None:
            self.mutable_local = LazyMutableLocal()
        assert self.source

    def realize(self) -> VariableTracker:
        """Force construction of the real VariableTracker"""
        if self.mutable_local.vt is None:
            from ..symbolic_convert import InstructionTranslator
            from .builder import VariableBuilder

            tx = InstructionTranslator.current_tx()
            self.mutable_local.vt = VariableBuilder(tx, self.source)(self._value)
            self._value = None
            tx.output.guards.update(self.mutable_local.vt.guards)
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
        vt = self.realize()
        if item not in vt.__dict__:
            raise AttributeError(item)
        return vt.__dict__[item]

    # most methods are auto-generated below, these are the ones we want to exclude
    add_guards = VariableTracker.add_guards
    add_guard = VariableTracker.add_guard
    add_options = VariableTracker.add_options
    _aggregate_mutables = VariableTracker._aggregate_mutables
    apply = VariableTracker.apply
    copy = VariableTracker.copy
    __post_init__ = VariableTracker.__post_init__
    propagate = VariableTracker.propagate
    __repr__ = VariableTracker.__repr__
    _update_contains = VariableTracker._update_contains


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
