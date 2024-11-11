import weakref
from typing import *

import torch
from torch.nested._internal.tensor_registry import TensorRegistry
from torch.nested._internal.utils import assert_not_fake, try_get_fake_mode


# Thin wrapper around a dict to help us enable weak references, then handle
# extra things?
# Do not construct/update this directly! Search for the "Composite Cache APIs".
class MetadataCache:
    # I feel like this makes more sense anyway.
    def __init__(self, *, data, cache_id: int, fake_mode=None):
        self.data = data
        self.id: int = cache_id

        # You must use the special factory function in the nested namespace which
        # allows auxiliary arguments
        # torch.nested.zeros((B, njt.shape[1], D), cuda_offsets=cuda_offsets))

    def state(self):
        # TODO(soulitzer): revisit guards
        return tuple(k is not None for k in self.data.values())


# Maintains a MetadataCache to int map
class CacheRegistry:
    def __init__(self):
        self._cache_id_to_cache_ref: Dict[int, weakref.ReferenceType] = {}
        self._cache_id_counter = 0

    def try_get_cache(self, mb_cache_id: int):
        cache_ref = self._cache_id_to_cache_ref.get(mb_cache_id)
        if cache_ref is not None:
            cache = cache_ref()
            if cache is not None:
                return cache
            else:
                del self._cache_id_to_cache_ref
        return None

    def create_cache(self, data):
        cache = MetadataCache(data=data, cache_id=self._cache_id_counter)
        self._cache_id_counter += 1
        self._cache_id_to_cache_ref[cache.id] = weakref.ref(cache)
        return cache

    def copy(self):
        ret = CacheRegistry()
        ret._cache_id_to_cache_ref = self._cache_id_to_cache_ref.copy()
        ret._cache_id_counter = self._cache_id_counter
        return ret


# TODO(soulitzer): Update the compile story for global state.
class CacheState:
    # Encapsulates all global state for the cache. Only used in eager.
    # Compile uses something simpler without weak ref handling
    #
    # CacheState is responsible for the mapping from tensor id to cache id
    # in eager. Maintenance of the id mappings state is left to the tensor/cache
    # registeries.
    def __init__(self, keys):
        # tensor <-> cache_id
        self._tensor_registry = TensorRegistry()
        # tensor_id -> cache_id (can be swapped out with something like union-find)
        self._tensor_id_to_cache_id: Dict[int, int] = {}
        # cache_id <-> cache
        self._cache_registry = CacheRegistry()
        self.keys = keys

    def maybe_update_tensor_id_to_cache_id(self, cache, k, v):
        assert_not_fake(v)
        if k not in self.keys:
            return
        tensor_id = self._tensor_registry.get_int(v)
        # Require that the tensor not be already registered.
        assert tensor_id not in self._tensor_id_to_cache_id
        self._tensor_id_to_cache_id[tensor_id] = cache.id

    def try_get_cache(self, tensor) -> Optional[MetadataCache]:
        # What is the None story here...
        assert_not_fake(tensor)
        tensor_id = self._tensor_registry.get_int(tensor)
        mb_cache_id = self._tensor_id_to_cache_id.get(tensor_id)
        if mb_cache_id is None:
            return None
        return self._cache_registry.try_get_cache(mb_cache_id)

    def add_entry(self, cache_id, key, value):
        # Take cache_id instead of MetadataCache because this must be called
        # through our custom op.
        cache = self._cache_registry.try_get_cache(cache_id)
        assert cache is not None, "add_entry: Expected cache to exist"
        cache.data[key] = value
        self.maybe_update_tensor_id_to_cache_id(cache, key, value)

    def create_cache(self, data: Dict):
        # Don't create a fresh dict because we created this one.
        cache = self._cache_registry.create_cache(data)
        for k, v in data.items():
            self.maybe_update_tensor_id_to_cache_id(cache, k, v)
        return cache

    def copy(self):
        ret = CacheState(self.keys)
        ret._cache_registry = self._cache_registry.copy()
        ret._tensor_id_to_cache_id = self._tensor_id_to_cache_id.copy()
        ret._tensor_registry = self._tensor_registry.copy()
        return ret


_global_cache_state = None


def get_global_cache_state():
    # By default use the nested tensor ragged source keys
    from torch.nested._internal.nested_tensor import RAGGED_SOURCE_KEYS

    global _global_cache_state

    if _global_cache_state is None:
        _global_cache_state = CacheState(RAGGED_SOURCE_KEYS)

    return _global_cache_state


# Custom op registration for operations that perform side effect during compile
# e.g. adding a new cache entry to an existing cache.

lib = torch.library.Library("nested", "FRAGMENT")

lib.define("_add_cache_entry(Tensor val, str key, int cache_id) -> ()")


def _add_cache_entry_impl(val, key, cache_id):
    # Not sure if it actually matters, but the signatures are different because I want
    # the first argument of the custom op to be a tensor.
    get_global_cache_state().add_entry(cache_id, key, val)


def _add_cache_entry_meta(val, key, cache_id):
    # Update the cache in the FakeMode
    mb_fake_mode = try_get_fake_mode(val)
    assert mb_fake_mode is not None
    mb_fake_mode.add_entry(cache_id, key, val)


lib.impl("_add_cache_entry", _add_cache_entry_impl, "CPU")
lib.impl("_add_cache_entry", _add_cache_entry_impl, "CUDA")
lib.impl("_add_cache_entry", _add_cache_entry_meta, "Meta")

from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

# Make sure this does not get DCE'd
_register_effectful_op(torch.ops.nested._add_cache_entry.default, _EffectType.ORDERED)


#
# "Composite Cache APIs"
#
# If you are running code that is "composite", e.g. shared between
# the compile/eager paths you should call into these wrappers, which
# handle the fake/non-fake routing for you.
#
# NB: Today the NestedTensor constructor is the only user of these APIs.
#
# NB: We don't have all of these as custom-ops so that aot dispatch can
# trace through.
#
def try_get_cache(data: Dict[str, Optional[torch.Tensor]]):
    mb_fake_mode = try_get_fake_mode(data)
    if mb_fake_mode is not None:
        return mb_fake_mode.get_nested_cache_if_exists(data)
    else:
        return get_global_cache_state().try_get_cache(data)


def create_cache(data: Dict[str, Optional[torch.Tensor]]):
    mb_fake_mode = try_get_fake_mode(data)
    if mb_fake_mode is not None:
        return mb_fake_mode.create_cache(data)
    else:
        return get_global_cache_state().create_cache(data)


def add_entry(cache, key, value):
    torch.ops.nested._add_cache_entry(value, key, cache.id)


def try_get_entry(cache, key):
    return cache.data.get(key)
