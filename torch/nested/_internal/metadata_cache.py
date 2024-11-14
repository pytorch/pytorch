import abc
import weakref
from math import e
from typing import *

import torch
from torch.nested._internal.utils import (
    assert_not_fake,
    MaybeUnwrapKeysWeakKeyDict,
    try_get_fake_mode,
)

# Note [ CacheState tmp refs ]
#
# The problem this tries to solve:
#
# 1. The _add_cache_entry is a custom op (so that side effects can persist)
# 2. The adding a cache entry creates a new cache if none currently exists.
# 3. Op signatures prevent us from directly return the cache object, so unless
#    we keep the cache alive somewhere, it would die. That is because
#    the cache_registry ordinarily only stores a weak reference to cache
#    because we don't want to have things offsets/lengths keeping alive
#    the cache, which itself may have offsets/lengths.
#
# Solution:
#
# 1. If the _add_cache_entry custom op created a new cache, it keeps
#    a temporary reference around to keep the cache alive
# 2. After we leave the custom op, we get a strong reference to the cache
#    from the registry to hold onto ourselves.
# 3. Clear the temporary reference to cache (make sure to do this in a second
#    custom op so that it doesn't get traced away!)


# Thin wrapper around a dict to help us enable weak references, then handle
# extra things?
# Do not construct/update this directly! Search for the "Composite Cache APIs".
class MetadataCache:
    def __init__(self, *, data, cache_id: int):
        self.data = data
        self.id: int = cache_id
        self.eq_id: int = cache_id

        # You must use the special factory function in the nested namespace which
        # allows auxiliary arguments
        # torch.nested.zeros((B, njt.shape[1], D), cuda_offsets=cuda_offsets))

    def state(self):
        # TODO(soulitzer): revisit guards
        return tuple(k is not None for k in self.data.values())


class CacheRegistry(abc.ABC):
    def __init__(self, is_weak=False):
        self._incrementing_id = 0
        self.is_weak = is_weak
        self._cache_id_to_cache: Dict[int, Any] = dict()

    def create_cache(self, data) -> MetadataCache:
        cache = MetadataCache(data=data, cache_id=self._incrementing_id)
        self._incrementing_id += 1
        self._cache_id_to_cache[cache.id] = (
            weakref.ref(cache) if self.is_weak else cache
        )
        return cache

    def try_get_cache(self, cache_id: int) -> Optional[MetadataCache]:
        cache_ref = self._cache_id_to_cache.get(cache_id)
        if cache_ref is not None:
            if not self.is_weak:
                return cache_ref
            cache = cache_ref()
            if cache is not None:
                return cache
            else:
                del self._cache_id_to_cache[cache_id]
        return None

    def copy(self) -> "CacheRegistry":
        ret = self.__class__()
        ret._cache_id_to_cache = self._cache_id_to_cache.copy()
        ret._incrementing_id = self._incrementing_id
        return ret


# Encapsulates all global state for the cache.
# Eager and compile subclass this to add their own logic.
class GenericCacheState(abc.ABC):
    _cache_registry: CacheRegistry
    _keys: List[str]

    def __init__(self, keys):
        self._keys = keys
        self._tensor_to_cache_id = MaybeUnwrapKeysWeakKeyDict()

    def _maybe_update_tensor_to_cache_id(self, cache, k, v):
        if k not in self._keys:
            return

        existing_cache_id = self._tensor_to_cache_id.get(v)

        if existing_cache_id is not None:
            existing_cache = self._cache_registry.try_get_cache(existing_cache_id)
            # TODO(soulitzer): I'm not sure we should always swap? Why cannot I assert
            # that the existing one does not exist.
            del self._tensor_to_cache_id[v]

        # Require that the tensor not be already registered.
        if v not in self._tensor_to_cache_id:
            # TODO(soulitzer): figure out why this is happening
            # We can probably do this unconditionally, now that we do the del above.
            self._tensor_to_cache_id[v] = cache.id

    def try_get_cache(self, tensor) -> Optional[MetadataCache]:
        mb_cache_id = self._tensor_to_cache_id.get(tensor)
        if mb_cache_id is None:
            return None
        return self._cache_registry.try_get_cache(mb_cache_id)

    def add_entry(
        self, key_tensor: Optional[torch.Tensor], key, value
    ) -> MetadataCache:
        cache = None
        if key_tensor is not None:
            cache = self.try_get_cache(key_tensor)
        if cache is None:
            # See Note [ CacheState tmp refs ]
            # The first thing add to a cache is always a "key tensor"
            # Q: should we be making this distiction anyway?
            cache = self._cache_registry.create_cache({key: value})
        else:
            cache.data[key] = value
        self._maybe_update_tensor_to_cache_id(cache, key, value)
        return cache

    def copy(self) -> "GenericCacheState":
        ret = self.__class__(self._keys)
        ret._cache_registry = self._cache_registry.copy()
        ret._tensor_to_cache_id = self._tensor_to_cache_id.copy()
        return ret


class EagerCacheState(GenericCacheState):
    def __init__(self, keys) -> None:
        # cache_id <-> cache
        self._cache_registry = CacheRegistry(is_weak=True)
        # See Note [ CacheState tmp refs ]
        self._tmp_refs: Dict[int, MetadataCache] = {}
        super().__init__(keys)

    # Eager needs extra logic to keep the cache alive via a temporary reference.
    def add_entry(
        self, key_tensor: Optional[torch.Tensor], key, value
    ) -> MetadataCache:
        cache = super().add_entry(key_tensor, key, value)
        self._tmp_refs[cache.id] = cache
        return cache

    def clear_tmp_ref(self, key_tensor: torch.Tensor) -> None:
        assert_not_fake(key_tensor)
        cache = self.try_get_cache(key_tensor)
        assert cache is not None
        del self._tmp_refs[cache.id]


class TracingCacheState(GenericCacheState):
    def __init__(self, keys) -> None:
        # cache_id <-> cache
        self._cache_registry = CacheRegistry(is_weak=False)
        super().__init__(keys)

    # Any extra methods that we only need during tracing belong here.
    @staticmethod
    def init_from_eager(eager_cache_state: EagerCacheState) -> "TracingCacheState":
        ret = TracingCacheState(eager_cache_state._keys)
        ret._cache_registry._incrementing_id = (
            eager_cache_state._cache_registry._incrementing_id
        )
        return ret

    def register_cache(self, data, cache_id):
        cache = MetadataCache(data=data, cache_id=cache_id)
        self._cache_registry._cache_id_to_cache[cache_id] = cache
        for k, v in data.items():
            if k in self._keys:
                self._tensor_to_cache_id[v] = cache_id
        return cache

    # This is used for the .detach() case.
    def maybe_alias_tensor(self, new_tensor, old_tensor):
        # TODO(soulitzer): locally have a detach cache for aot autograd
        if old_tensor not in self._tensor_to_cache_id:
            return
        old_cache = self._cache_registry.try_get_cache(
            self._tensor_to_cache_id[old_tensor]
        )
        assert old_cache is not None
        # Create a new cache where every entry is a view of the corresponding
        # entry in the old cache to preserve invariant: v in get_cache(v)
        new_cache_data = dict()
        for k, v in old_cache.data.items():
            if v is old_tensor:
                new_cache_data[k] = new_tensor
            else:
                new_cache_data[k] = v.detach()
        new_cache = self._cache_registry.create_cache(new_cache_data)
        for k, v in new_cache.data.items():
            self._maybe_update_tensor_to_cache_id(new_cache, k, v)
        # Make sure the symints compare equal
        new_cache.eq_id = old_cache.eq_id


_global_cache_state = None


def get_global_cache_state():
    # By default use the nested tensor ragged source keys
    from torch.nested._internal.nested_tensor import RAGGED_SOURCE_KEYS

    global _global_cache_state

    if _global_cache_state is None:
        _global_cache_state = EagerCacheState(RAGGED_SOURCE_KEYS)

    return _global_cache_state


# Custom op registration for operations that perform side effect during compile
# e.g. adding a new cache entry to an existing cache.

lib = torch.library.Library("nested", "FRAGMENT")

lib.define("_add_cache_entry(Tensor? key_tensor, Tensor val, str key) -> Tensor")


def _add_cache_entry_impl(
    key_tensor: Optional[torch.Tensor], val: torch.Tensor, key: str
):
    # Q: when I write a custom op, does the first argument need to be a tensor?
    # Can I return a
    cache = get_global_cache_state().add_entry(key_tensor, key, val)
    if key_tensor is None:
        key_tensor = _maybe_unpack_first(cache.data)
    return key_tensor


def _add_cache_entry_meta(
    key_tensor: Optional[torch.Tensor], val: torch.Tensor, key: str
):
    mb_fake_mode = try_get_fake_mode(val)
    if mb_fake_mode is not None:
        # TODO(soulitzer): is this called for the eager path too?
        cache = mb_fake_mode.nested_cache_state.add_entry(key_tensor, key, val)
    else:
        # What is the dummy tensor story?
        assert val.device.type == "meta"
        cache = get_global_cache_state().add_entry(key_tensor, key, val)
    if key_tensor is None:
        key_tensor = _maybe_unpack_first(cache.data)
    return key_tensor


lib.impl("_add_cache_entry", _add_cache_entry_impl, "CPU")
lib.impl("_add_cache_entry", _add_cache_entry_impl, "CUDA")
lib.impl("_add_cache_entry", _add_cache_entry_meta, "Meta")


lib.define("_clear_tmp_ref(Tensor cached_tensor) -> ()")


def _clear_tmp_ref_impl(cached_tensor):
    get_global_cache_state().clear_tmp_ref(cached_tensor)


def _clear_tmp_ref_meta(cached_tensor):
    # Nothing needs to be done here.
    pass


lib.impl("_clear_tmp_ref", _clear_tmp_ref_impl, "CPU")
lib.impl("_clear_tmp_ref", _clear_tmp_ref_impl, "CUDA")
lib.impl("_clear_tmp_ref", _clear_tmp_ref_meta, "Meta")


from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

# Make sure this does not get DCE'd
# They should share the same token so that clearing the tmp ref happens after adding cache entry
_register_effectful_op(torch.ops.nested._add_cache_entry.default, _EffectType.ORDERED)
_register_effectful_op(torch.ops.nested._clear_tmp_ref.default, _EffectType.ORDERED)


def _maybe_unpack_first(
    data_or_tensor: Union[Dict[str, Optional[torch.Tensor]], torch.Tensor]
) -> torch.Tensor:
    # Gets the first tensor in the cache data dict according to
    # priority in RAGGED_SOURCE_KEYS. If a tensor is directly
    # passed in, then return as-is.
    from torch.nested._internal.nested_tensor import RAGGED_SOURCE_KEYS

    if isinstance(data_or_tensor, torch.Tensor):
        return data_or_tensor
    cached_tensor = None
    for k in RAGGED_SOURCE_KEYS:
        if k in data_or_tensor:
            cached_tensor = data_or_tensor[k]
            break
    assert cached_tensor is not None
    return cached_tensor


#
# "Composite Cache APIs"
#
# If you are running code that is "composite", e.g. shared between
# the compile/eager paths you should call into these wrappers, which
# handle the fake/non-fake routing for you.
#
# These are NOT public APIs, you should not call them directly.
# Dynamo should graph break TODO(soulitzer).
#
def try_get_cache(
    data_or_tensor: Union[Dict[str, Optional[torch.Tensor]], torch.Tensor]
):
    mb_fake_mode = try_get_fake_mode(data_or_tensor)
    cached_tensor = _maybe_unpack_first(data_or_tensor)

    if mb_fake_mode is not None:
        return mb_fake_mode.nested_cache_state.try_get_cache(cached_tensor)
    else:
        return get_global_cache_state().try_get_cache(cached_tensor)


# Are users allowed to call this directly where dynamo can see it?
def add_entry(cache, key, value):
    key_tensor = None
    if cache is not None:
        key_tensor = _maybe_unpack_first(cache.data)

    new_key_tensor = torch.ops.nested._add_cache_entry(key_tensor, value, key)
    assert new_key_tensor is not None

    if key_tensor is None:
        # See Note [ CacheRegistry tmp refs ]
        # But wait, if key is not actually the key? this wouldn't work
        # Can I return a proper key tensor?
        cache = try_get_cache(new_key_tensor)
        assert cache is not None
        torch.ops.nested._clear_tmp_ref(new_key_tensor)

    return cache
