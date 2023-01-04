import weakref
from weakref import ref
from _weakrefset import _IterationGuard  # type: ignore[attr-defined]
from collections.abc import MutableMapping, Mapping
from typing import Dict
import collections.abc as _collections_abc


__all__ = ['WeakIdRef', 'WeakIdKeyDictionary', 'WeakTensorKeyDictionary']


# This file defines a variant of WeakKeyDictionary that overrides the hashing
# behavior of the key to use object identity, rather than the builtin
# __eq__/__hash__ functions.  This is useful for Tensor weak keys, as their
# __eq__ implementation return a Tensor (elementwise equality), which means
# you can't use them directly with the WeakKeyDictionary in standard library.
#
# Our implementation strategy is to create a wrapper weak key object, which we
# use as a key in a stock Python dictionary.  This is similar to how weakref
# implements WeakKeyDictionary, but instead of using weakref.ref as the
# wrapper, we use a custom wrapper that has different __eq__ and __hash__
# behavior.  Note that we subsequently store this weak key directly in an
# ORDINARY dictionary, since the newly constructed WeakIdKey's only use would
# be a dictionary so it would have no strong references.  Ensuring that
# only live WeakIdKeys are in the map is handled by putting finalizers on the
# original key object.


# It is simpler to implement this with composition, but if we want to
# directly reuse the callback mechanism on weakref, we need the weakref
# and the key to be exactly the same object.  Reusing the callback mechanism
# minimizes the divergence between our implementation and Lib/weakref.py
#
# NB: Prefer using this when working with weakrefs of Tensors; e.g., do
# WeakIdRef(tensor) rather than weakref.ref(tensor); it handles a number of
# easy to get wrong cases transparently for you.
class WeakIdRef(weakref.ref):
    __slots__ = ['_id']

    def __init__(self, key, callback=None):
        # Unlike stock weakref, which preserves hash semantics of the
        # original object but lazily defers hash calls until the first
        # time the user attempts to hash the weakref, we can eagerly
        # cache the id of the key as we know this is definitely the hash
        # method
        self._id = id(key)
        super().__init__(key, callback)

    def __call__(self):
        r = super().__call__()
        # Special logic for Tensor PyObject resurrection
        if hasattr(r, '_fix_weakref'):
            r._fix_weakref()  # type: ignore[union-attr]
        return r

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        # An attractive but wrong alternate implementation is to only test if
        # the stored _ids match.  This can lead to an ABA problem if you have:
        #
        #   a1 = A()
        #   w1 = WeakIdRef(a)
        #   del a1
        #   a2 = A()  # suppose it gets the same ID as a1
        #   w2 = WeakIdRef(a2)
        #   print(w1 == w2)
        #
        # This should be False, as a1 and a2 are unrelated (and a1 is
        # dead anyway)
        a = self()
        b = other()
        if a is not None and b is not None:
            return a is b
        return self is other

# This is directly adapted from cpython/Lib/weakref.py
class WeakIdKeyDictionary(MutableMapping):
    data: Dict[WeakIdRef, object]

    def __init__(self, dict=None):
        self.data = {}

        def remove(k, selfref=ref(self)):
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(k)
                else:
                    try:
                        del self.data[k]
                    except KeyError:
                        pass
        self._remove = remove
        # A list of dead weakrefs (keys to be removed)
        self._pending_removals = []
        self._iterating = set()
        self._dirty_len = False
        if dict is not None:
            self.update(dict)

    def _commit_removals(self):
        # NOTE: We don't need to call this method before mutating the dict,
        # because a dead weakref never compares equal to a live weakref,
        # even if they happened to refer to equal objects.
        # However, it means keys may already have been removed.
        pop = self._pending_removals.pop
        d = self.data
        while True:
            try:
                key = pop()
            except IndexError:
                return

            try:
                del d[key]
            except KeyError:
                pass

    def _scrub_removals(self):
        d = self.data
        self._pending_removals = [k for k in self._pending_removals if k in d]
        self._dirty_len = False

    def __delitem__(self, key):
        self._dirty_len = True
        del self.data[WeakIdRef(key)]  # CHANGED

    def __getitem__(self, key):
        return self.data[WeakIdRef(key)]  # CHANGED

    def __len__(self):
        if self._dirty_len and self._pending_removals:
            # self._pending_removals may still contain keys which were
            # explicitly removed, we have to scrub them (see issue #21173).
            self._scrub_removals()
        return len(self.data) - len(self._pending_removals)

    def __repr__(self):
        return "<%s at %#x>" % (self.__class__.__name__, id(self))

    def __setitem__(self, key, value):
        self.data[WeakIdRef(key, self._remove)] = value  # CHANGED

    def copy(self):
        new = WeakIdKeyDictionary()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = value
        return new

    __copy__ = copy

    def __deepcopy__(self, memo):
        from copy import deepcopy
        new = self.__class__()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = deepcopy(value, memo)
        return new

    def get(self, key, default=None):
        return self.data.get(WeakIdRef(key), default)  # CHANGED

    def __contains__(self, key):
        try:
            wr = WeakIdRef(key)
        except TypeError:
            return False
        return wr in self.data

    def items(self):
        with _IterationGuard(self):
            for wr, value in self.data.items():
                key = wr()
                if key is not None:
                    yield key, value

    def keys(self):
        with _IterationGuard(self):
            for wr in self.data:
                obj = wr()
                if obj is not None:
                    yield obj

    __iter__ = keys

    def values(self):
        with _IterationGuard(self):
            for wr, value in self.data.items():
                if wr() is not None:
                    yield value

    def keyrefs(self):
        """Return a list of weak references to the keys.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the keys around longer than needed.

        """
        return list(self.data)

    def popitem(self):
        self._dirty_len = True
        while True:
            key, value = self.data.popitem()
            o = key()
            if o is not None:
                return o, value

    def pop(self, key, *args):
        self._dirty_len = True
        return self.data.pop(WeakIdRef(key), *args)  # CHANGED

    def setdefault(self, key, default=None):
        return self.data.setdefault(WeakIdRef(key, self._remove), default)  # CHANGED

    def update(self, dict=None, **kwargs):
        d = self.data
        if dict is not None:
            if not hasattr(dict, "items"):
                dict = type({})(dict)
            for key, value in dict.items():
                d[WeakIdRef(key, self._remove)] = value  # CHANGED
        if len(kwargs):
            self.update(kwargs)

    def __ior__(self, other):
        self.update(other)
        return self

    def __or__(self, other):
        if isinstance(other, _collections_abc.Mapping):
            c = self.copy()
            c.update(other)
            return c
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, _collections_abc.Mapping):
            c = self.__class__()
            c.update(other)
            c.update(self)
            return c
        return NotImplemented

    # Default Mapping equality will tests keys for equality, but
    # we want to test ids for equality
    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return {id(k): v for k, v in self.items()} == {id(k): v for k, v in other.items()}

# Convenience alias
WeakTensorKeyDictionary = WeakIdKeyDictionary
