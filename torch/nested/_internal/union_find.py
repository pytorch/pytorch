import weakref

import torch
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403


# Union find over versions of tensors
# -----------------------------------
# TensorUnionFind implements union-find over "versions of tensors". What this
# means is that if TensorUnionFind holds a set of two tensors {`a`, `b`} with
# `a` being canonical, we should really think of it as holding {(`a`, 0),
# (`b`, 0)} with (`a`, 0) being canonical, where the 0 is the version of the
# tensor.
#
# If the Tensor `a` is mutated, the version of `a` is increments to 1, the
# the set continues continues to be {(`a`, 0), (`b`, 0)}, and the canonical
# entry of that set continues to be (`a`, 0).
#
# If someone does uf.find(a) now, they are really querying for uf.find((`a`, 1))
# and that would return (`a`, 1) because (`a`, 1) is in a new set of its own
# and is the canonical entry of that set.
#
# A more problematic case if someone does uf.find(b), i.e., uf.find((`b`, 0)).
# Since the canonical entry of the set continues to be point to (`a`, 0), which
# has now become outdated, we raise an error.

# Storing metadata on canonical entries
# -------------------------------------
# Each set has a metadata dict associated with it. This dict is stored on the
# canonical entry of the set.
#
# When two sets are merged, one of the two canonical entries is chosen to be
# the canonical entry of the union. The other entry is no longer the canonical
# entry of any set. When a merge happens, we also merge their metadata dicts
# and store the result on the new canonical entry. The metadata of the
# no-longer-canonical entry is invalidated.
#
# In the case where the two dicts have different entries for
# the same key, we have a choice of which metadata to favor. In this class, we
# somewhat arbitrarily chose to favor the metadata of the NON-canonical tensor!
# This might be the opposite of what one would expect, but that shouldn't matter
# because which tensor between a and b union-find chooses to be canonical is an
# implementation detail anyway. The the user should careful to make sure that no
# such conflict can exist.


class DefaultWeakTensorKeyDictionary(WeakTensorKeyDictionary):
    def __init__(self, default_cls):
        super().__init__()
        self._default_cls = default_cls

    def __getitem__(self, key):
        if not super().__contains__(key):
            super().__setitem__(key, self._default_cls())
        return super().__getitem__(key)


class TensorIntMap:
    # Assigns Tensor objects to unique ints in an incrementing fashion.
    # The int given corresponds to a particular version of a Tensor.
    # If a Tensor has been mutated, its original int is invalidated, and
    # it will be assigned a new int upon the next get_int.
    # We try to be careful to NOT hold any owning references.
    _incrementing_id = 0
    _tensor_to_int_and_version = WeakTensorKeyDictionary()
    _int_to_tensor: Dict[int, weakref.ReferenceType] = dict()

    def get_int(self, t):
        mb_data = self._tensor_to_int_and_version.get(t)
        if mb_data is None or mb_data[1] != t._version:
            self._tensor_to_int_and_version[t] = (self._incrementing_id, t._version)
            self._int_to_tensor[self._incrementing_id] = weakref.ref(t)
            self._incrementing_id += 1
        return self._tensor_to_int_and_version[t][0]

    def get_opt_tensor(self, i):
        # This function may not always succeed. If that Tensor is no longer
        # alive or is no longer the same version i.e. it was mutated, None is
        # returned.
        mb_weak_t = self._int_to_tensor.get(i)
        if mb_weak_t is None:
            return None
        mb_t = mb_weak_t()
        if mb_t is None or (
            self._tensor_to_int_and_version[mb_t][1] != mb_t._version
            or self._tensor_to_int_and_version[mb_t][0] != i
        ):
            del self._int_to_tensor[i]
            return None
        return mb_t


class TensorUnionFind:
    # Union-find over tensors with some extra functionality:
    #
    # - Maintains a metadata dict object for each set. See
    #   "Storing metadata on canonical entries".
    # - Keeps track of all alive tensors in each set (see _equiv_sets).
    # - Maintains a tree of owning references such the canonical tensor is
    #   always alive as long as any tensor in its set is alive (see _refs).
    #
    def __init__(self, union_find_int=None, tensor_int_map=None):
        # Allow user to pass in an existing union-find datastructure to use as
        # a shared source of truth. This is useful for the NestedInt case where
        # the source of truth must exist in cpp.
        self._union_find_int = (
            union_find_int if union_find_int is not None else torch._C._UnionFind()
        )
        self._tensor_int_map = (
            tensor_int_map if tensor_int_map is not None else TensorIntMap()
        )
        # See note "Storing metadata on canonical entries"
        self._metadata = DefaultWeakTensorKeyDictionary(dict)
        self._equiv_sets = DefaultWeakTensorKeyDictionary(set)
        # Used to manage lifetime of tensors in sets
        self._refs = WeakTensorKeyDictionary()
        # Sentinel value to indicate that an entry has been invalidated
        self._INVALID_ENTRY = object()

    def merge(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root is y_root:
            return
        self._union_find_int.merge(
            self._tensor_int_map.get_int(x), self._tensor_int_map.get_int(y)
        )
        # src and tgt depend on which direction we merged in the actual impl
        tgt, src = (x_root, y_root) if self.find(x_root) is x_root else (y_root, x_root)

        # See note "Storing metadata on canonical entries"
        self._metadata[tgt].update(self._metadata[src])
        self._metadata[src] = self._INVALID_ENTRY
        self._get_equiv_set(tgt).update(self._get_equiv_set(src))
        self._equiv_sets[src] = self._INVALID_ENTRY

        # Maintains that the the canonical tensor and by extension the metadata
        # and equiv sets are kept alive by any tensors alive in the set.
        self._refs[src] = tgt

    def find(self, tensor):
        canonical_id = self._union_find_int.find(self._tensor_int_map.get_int(tensor))
        mb_tensor = self._tensor_int_map.get_opt_tensor(canonical_id)
        if mb_tensor is None:
            # See note "Union find over versions of tensors".
            raise RuntimeError("The canonical tensor of this set has been mutated.")
        return mb_tensor

    def _get_equiv_set(self, canonical_tensor):
        self._equiv_sets[canonical_tensor].add(weakref.ref(canonical_tensor))
        return self._equiv_sets[canonical_tensor]

    def get_metadata(self, tensor):
        ret = self._metadata[self.find(tensor)]
        assert ret is not self._INVALID_ENTRY
        return ret

    def get_equiv_tensors(self, tensor):
        equiv_set = self._get_equiv_set(self.find(tensor))
        assert equiv_set is not self._INVALID_ENTRY
        to_remove = set()
        for weak_tensor in equiv_set:
            mb_tensor = weak_tensor()
            if mb_tensor is not None:
                yield mb_tensor
            else:
                to_remove.add(weak_tensor)
        equiv_set -= to_remove

    def validate_invariants(self):
        # for testing only
        for t, v in self._metadata.items():
            assert (self.find(t) is t) == (v is not self._INVALID_ENTRY)
        for t, v in self._equiv_sets.items():
            assert (self.find(t) is t) == (v is not self._INVALID_ENTRY)
