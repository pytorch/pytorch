# mypy: ignore-errors
import collections
import functools
from typing import Optional

from .base import VariableTracker


class LazyCache:
    """Container to cache the real VariableTracker"""

    def __init__(self, value, source):
        assert source
        self.value = value
        self.source = source
        self.vt: Optional[VariableTracker] = None

    def realize(self):
        assert self.vt is None
        from ..symbolic_convert import InstructionTranslator
        from .builder import VariableBuilder

        tx = InstructionTranslator.current_tx()
        self.vt = VariableBuilder(tx, self.source)(self.value)

        del self.value
        del self.source


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

    _nonvar_fields = {"_cache", *VariableTracker._nonvar_fields}

    @staticmethod
    def create(value, source, **options):
        return LazyVariableTracker(LazyCache(value, source), source=source, **options)

    def __init__(self, _cache, **kwargs):
        assert isinstance(_cache, LazyCache)
        super().__init__(**kwargs)
        self._cache = _cache

    def realize(self) -> VariableTracker:
        """Force construction of the real VariableTracker"""
        if self._cache.vt is None:
            self._cache.realize()
        return self._cache.vt

    def unwrap(self):
        """Return the real VariableTracker if it already exists"""
        if self.is_realized():
            return self._cache.vt
        return self

    def is_realized(self):
        return self._cache.vt is not None

    def clone(self, **kwargs):
        assert kwargs.get("_cache", self._cache) is self._cache
        if kwargs.get("source", self.source) is not self.source:
            self.realize()
        return VariableTracker.clone(self.unwrap(), **kwargs)

    def __str__(self):
        if self.is_realized():
            return self.unwrap().__str__()
        return VariableTracker.__str__(self.unwrap())

    def __getattr__(self, item):
        return getattr(self.realize(), item)

    # most methods are auto-generated below, these are the ones we want to exclude
    visit = VariableTracker.visit
    __repr__ = VariableTracker.__repr__

    @classmethod
    def realize_all(
        cls,
        value,
        cache=None,
    ):
        """
        Walk an object and realize all LazyVariableTrackers inside it.
        """
        if cache is None:
            cache = dict()

        idx = id(value)
        if idx in cache:
            return cache[idx][0]

        value_cls = type(value)
        if issubclass(value_cls, LazyVariableTracker):
            result = cls.realize_all(value.realize(), cache)
        elif issubclass(value_cls, VariableTracker):
            # update value in-place
            result = value
            value_dict = value.__dict__
            nonvars = value._nonvar_fields
            for key in value_dict:
                if key not in nonvars:
                    value_dict[key] = cls.realize_all(value_dict[key], cache)
        elif value_cls is list:
            result = [cls.realize_all(v, cache) for v in value]
        elif value_cls is tuple:
            result = tuple(cls.realize_all(v, cache) for v in value)
        elif value_cls in (dict, collections.OrderedDict):
            result = {k: cls.realize_all(v, cache) for k, v in list(value.items())}
        else:
            result = value

        # save `value` to keep it alive and ensure id() isn't reused
        cache[idx] = (result, value)
        return result


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
