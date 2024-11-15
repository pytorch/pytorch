import abc
import weakref
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
    def __init__(self, *, data, cache_id: int, is_mutable: bool = True):
        self.data = data
        self.id: int = cache_id
        self.eq_id: int = cache_id
        self.is_mutable: bool = is_mutable

    def state(self):
        # TODO(soulitzer): revisit guards
        return tuple(k is not None for k in self.data.values())


class CacheRegistry(abc.ABC):
    def __init__(self, is_weak=False):
        self._incrementing_id = 0
        self.is_weak = is_weak
        self._cache_id_to_cache: Dict[int, Any] = dict()

    def create_cache(self, data, is_mutable=True) -> MetadataCache:
        cache = MetadataCache(
            data=data, cache_id=self._incrementing_id, is_mutable=is_mutable
        )
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

    def __init__(self):
        self._tensor_to_cache_id = MaybeUnwrapKeysWeakKeyDict()

    # TODO(soulitzer): maybe update this name
    def _maybe_update_tensor_to_cache_id(self, cache, v):
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
        self, ref_tensor: Optional[torch.Tensor], key, value
    ) -> MetadataCache:
        cache = self.try_get_cache(ref_tensor) if ref_tensor is not None else None
        assert cache.is_mutable, "Cannot add to an immutable cache"
        if cache is None:
            cache = self._cache_registry.create_cache({key: value})
        else:
            cache.data[key] = value
        self._maybe_update_tensor_to_cache_id(cache, value)
        return cache

    def copy(self) -> "GenericCacheState":
        ret = self.__class__()
        ret._cache_registry = self._cache_registry.copy()
        ret._tensor_to_cache_id = self._tensor_to_cache_id.copy()
        return ret


class EagerCacheState(GenericCacheState):
    def __init__(self) -> None:
        super().__init__()
        # cache_id <-> cache
        self._cache_registry = CacheRegistry(is_weak=True)
        # See Note [ CacheState tmp refs ]
        self._tmp_refs: Dict[int, MetadataCache] = {}

    # Eager needs extra logic to keep the cache alive via a temporary reference.
    def add_entry(
        self, ref_tensor: Optional[torch.Tensor], key, value
    ) -> MetadataCache:
        cache = super().add_entry(ref_tensor, key, value)
        self._tmp_refs[cache.id] = cache
        return cache

    def clear_tmp_ref(self, ref_tensor: torch.Tensor) -> None:
        assert_not_fake(ref_tensor)
        cache = self.try_get_cache(ref_tensor)
        assert cache is not None
        del self._tmp_refs[cache.id]


class TracingCacheState(GenericCacheState):
    def __init__(self) -> None:
        super().__init__()
        # cache_id <-> cache
        self._cache_registry = CacheRegistry(is_weak=False)

    # Any extra methods that we only need during tracing belong here.
    @staticmethod
    def init_from_eager(eager_cache_state: EagerCacheState) -> "TracingCacheState":
        ret = TracingCacheState()
        ret.set_counter(
            0
            if eager_cache_state is None
            else eager_cache_state._cache_registry._incrementing_id
        )
        return ret

    def set_counter(self, counter: int):
        self._cache_registry._incrementing_id = counter

    def register_cache(self, data, cache_id):
        cache = MetadataCache(data=data, cache_id=cache_id)
        self._cache_registry._cache_id_to_cache[cache_id] = cache
        for v in data.values():
            self._tensor_to_cache_id[v] = cache_id
        return cache

    # This is used for the .detach() case.
    def maybe_alias_tensor(self, new_tensor, old_tensor):
        # For AOT Autograd, we need to .detach() all the inner tensors in order
        # to avoid the inputs/outputs being collapsed in the fx graph.
        # See Note [AOT Autograd: Views to avoid tangents aliasing inputs]
        #
        # This is something we ordinarily do not do; ordinarily, doing a view
        # returns new NestedTensor that shares the same offsets instance.
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

        new_cache = self._cache_registry.create_cache(new_cache_data, is_mutable=False)
        for v in new_cache.data.values():
            self._maybe_update_tensor_to_cache_id(new_cache, v)

        new_cache.eq_id = old_cache.eq_id


_global_cache_state = None


def get_global_cache_state():
    global _global_cache_state

    if _global_cache_state is None:
        _global_cache_state = EagerCacheState()

    return _global_cache_state


# Custom op registration for operations that perform side effect during compile
# e.g. adding a new cache entry to an existing cache.

lib = torch.library.Library("nested", "FRAGMENT")

lib.define("_add_cache_entry(Tensor? opt_ref_tensor, Tensor val, str key) -> Tensor")


def _add_cache_entry_impl(
    opt_ref_tensor: Optional[torch.Tensor], val: torch.Tensor, key: str
):
    cache = get_global_cache_state().add_entry(opt_ref_tensor, key, val)
    ref_tensor = (
        _get_ref_tensor(cache.data) if opt_ref_tensor is None else opt_ref_tensor
    )
    return ref_tensor


def _add_cache_entry_meta(
    opt_ref_tensor: Optional[torch.Tensor], val: torch.Tensor, key: str
):
    mb_fake_mode = try_get_fake_mode(val)
    if mb_fake_mode is not None:
        cache = mb_fake_mode.nested_cache_state.add_entry(opt_ref_tensor, key, val)
    else:
        # Called with meta tensors in eager (nested dummy tensors)
        assert val.device.type == "meta"
        cache = get_global_cache_state().add_entry(opt_ref_tensor, key, val)
    ref_tensor = (
        _get_ref_tensor(cache.data) if opt_ref_tensor is None else opt_ref_tensor
    )
    return ref_tensor


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


def _get_ref_tensor(
    data_or_tensor: Union[Dict[str, torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    if isinstance(data_or_tensor, torch.Tensor):
        return data_or_tensor
    return next(iter(data_or_tensor.values()))


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
def try_get_cache(data_or_tensor: Union[Dict[str, torch.Tensor], torch.Tensor]):
    mb_fake_mode = try_get_fake_mode(data_or_tensor)
    cached_tensor = _get_ref_tensor(data_or_tensor)

    if mb_fake_mode is not None:
        return mb_fake_mode.nested_cache_state.try_get_cache(cached_tensor)
    else:
        return get_global_cache_state().try_get_cache(cached_tensor)


# Are users allowed to call this directly where dynamo can see it?
def add_entry(cache, key, value):
    ref_tensor = None
    if cache is not None:
        ref_tensor = _get_ref_tensor(cache.data)

    new_ref_tensor = torch.ops.nested._add_cache_entry(ref_tensor, value, key)
    assert new_ref_tensor is not None

    if ref_tensor is None:
        # See Note [ CacheRegistry tmp refs ]
        cache = try_get_cache(new_ref_tensor)
        assert cache is not None
        torch.ops.nested._clear_tmp_ref(new_ref_tensor)

    return cache
