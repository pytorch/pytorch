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
# 2. Adding a cache entry creates a new cache if none currently exists.
# 3. Op signatures prevent us from directly returning the cache object, so unless
#    we keep the cache alive somewhere, it would die. That is because
#    the cache_registry ordinarily only stores a weak reference to cache
#    because we don't want to have things offsets/lengths keeping alive
#    the cache, which itself may have offsets/lengths.
#
# Solution:
#
# 1. If the _add_cache_entry custom op created a new cache, it keeps
#    a temporary reference around to keep the cache alive.
# 2. After we leave the custom op, we get a strong reference to the cache
#    from the registry to hold onto ourselves.
# 3. Clear the temporary reference to cache (make sure to do this in a second
#    custom op so that it doesn't get traced away!)

# The user could've passed in an older offsets, we know that
# one along with the cache, which doens't even necessarly contain that tensor
# would be fakified.
# We'd register a single fake cache, the canonical one
# we'd specialize on the aliasing relation between the tensors
# is that not desired? I think that's fine.


# TreeCache class combines UnionFind and MetadataCache functionalities
class TreeCache:
    def __init__(self, cache_id: int, data: Dict[str, torch.Tensor]):
        self.id: int = cache_id
        self.parent: "TreeCache" = self  # Parent in the Union-Find tree
        self.size: int = 1  # For union by size
        self.data: Dict[str, torch.Tensor] = data

    def find(self) -> "TreeCache":
        if self.parent != self:
            self.parent = self.parent.find()  # Path compression
        return self.parent

    def merge(self, other: "TreeCache"):
        root1 = self.find()
        root2 = other.find()
        if root1 == root2:
            return root1
        # Union by size to keep tree shallow
        if root1.size < root2.size:
            root1, root2 = root2, root1
        root2.parent = root1
        root1.size += root2.size
        # Merge cache data from root2 into root1
        root1.data.update(root2.data)
        return root1

    def try_get_entry(self, key) -> Optional[torch.Tensor]:
        return self.find().data.get(key)

    def set_entry(self, key: str, value: torch.Tensor):
        root = self.find()
        root.data[key] = value

    def state(self):
        # TODO: Revisit guards if necessary
        return tuple(k is not None for k in self.data.values())


# Holds TreeCache and allows one to query for them via id. Holds
# TreeCache weakly in eager, and strongly during compile.
class CacheRegistry:
    def __init__(self, is_weak=False):
        self._next_id = 0
        self.is_weak = is_weak
        self._cache_id_to_cache: Dict[int, Union[weakref.ref, TreeCache]] = dict()

    def create_cache(self, data) -> TreeCache:
        cache = TreeCache(cache_id=self._next_id, data=data)
        # print("incrementing next_id", self._next_id, "to", self._next_id + 1)
        # breakpoint()
        self._next_id += 1
        self._cache_id_to_cache[cache.id] = (
            weakref.ref(cache) if self.is_weak else cache
        )
        return cache

    def try_get_cache(self, cache_id: int) -> Optional[TreeCache]:
        # Returns the canonical cache for a given id, or None if it doesn't exist.
        cache_ref = self._cache_id_to_cache.get(cache_id)
        if cache_ref is None:
            return None
        if self.is_weak:
            assert isinstance(cache_ref, weakref.ref)
            cache: TreeCache = cache_ref()
            if cache is None:
                del self._cache_id_to_cache[cache_id]
                return None
        else:
            assert isinstance(cache_ref, TreeCache)
            cache = cache_ref
        return cache.find()

    def merge(self, cache_id_a: int, cache_id_b: int):
        cache_a = self.try_get_cache(cache_id_a)
        cache_b = self.try_get_cache(cache_id_b)
        if cache_a is not None and cache_b is not None:
            cache_a.merge(cache_b)

    def set_next_id(self, next_id: int):
        self._next_id = next_id

    def get_next_id(self):
        return self._next_id

    def copy(self) -> "CacheRegistry":
        ret = CacheRegistry(is_weak=self.is_weak)
        ret._cache_id_to_cache = self._cache_id_to_cache.copy()
        ret._next_id = self._next_id
        return ret


# Encapsulates all global state for the cache.
# Eager and compile subclass this to add their own logic.
class BaseCacheState(abc.ABC):
    _cache_registry: CacheRegistry

    def __init__(self):
        self._tensor_to_cache_id = MaybeUnwrapKeysWeakKeyDict()

    @abc.abstractmethod
    def register_tensor(self, tensor, cache_id): ...

    def try_get_cache(self, tensor) -> Optional[TreeCache]:
        assert isinstance(tensor, torch.Tensor)
        mb_cache_id = self._tensor_to_cache_id.get(tensor)
        if mb_cache_id is None:
            return None
        return self._cache_registry.try_get_cache(mb_cache_id)

    def add_entry(self, ref_tensor: Optional[torch.Tensor], key, value) -> TreeCache:
        # ref_tensor is None if we are creating a new cache
        cache = self.try_get_cache(ref_tensor) if ref_tensor is not None else None
        if cache is None:
            cache = self._cache_registry.create_cache({key: value})
        else:
            cache.set_entry(key, value)
        self.register_tensor(value, cache.id)
        return cache

    @abc.abstractmethod
    def try_get_entry(self, tensor_ref, key) -> Optional[torch.Tensor]: ...

    def merge(self, tensor_ref_a: torch.Tensor, tensor_ref_b: torch.Tensor):
        cache_a = self.try_get_cache(tensor_ref_a)
        cache_b = self.try_get_cache(tensor_ref_b)
        if cache_a is None or cache_b is None:
            # TODO(soulitzer): When does this happen
            return
        cache_a.merge(cache_b)

    def get_next_id(self):
        return self._cache_registry.get_next_id()

    def copy(self) -> "BaseCacheState":
        ret = self.__class__()
        ret._cache_registry = self._cache_registry.copy()
        ret._tensor_to_cache_id = self._tensor_to_cache_id.copy()
        return ret


class EagerCacheState(BaseCacheState):
    def __init__(self) -> None:
        super().__init__()
        self._cache_registry = CacheRegistry(is_weak=True)
        # See Note [ CacheState tmp refs ]
        self._tmp_refs: Dict[int, TreeCache] = {}

    def register_tensor(self, tensor, cache_id):
        if tensor in self._tensor_to_cache_id and self.try_get_cache(tensor):
            # Don't allow changing if exists
            # if not self._tensor_to_cache_id[tensor] == cache_id:
            #     breakpoint()
            assert self._tensor_to_cache_id[tensor] == cache_id
            return
        self._tensor_to_cache_id[tensor] = cache_id

    # Hack: Eager needs extra logic to keep the cache alive via a temporary reference.
    #       We can replace this with torchbind eventually.
    def add_entry(self, ref_tensor: Optional[torch.Tensor], key, value) -> TreeCache:
        cache = super().add_entry(ref_tensor, key, value)
        self._tmp_refs[cache.id] = cache
        return cache

    def clear_tmp_ref(self, ref_tensor: torch.Tensor) -> None:
        assert_not_fake(ref_tensor)
        cache = self.try_get_cache(ref_tensor)
        assert cache is not None
        del self._tmp_refs[cache.id]

    def try_get_entry(self, tensor_ref, key) -> Optional[torch.Tensor]:
        cache = self.try_get_cache(tensor_ref)
        assert cache is not None
        return cache.try_get_entry(key)


class TracingCacheState(BaseCacheState):
    def __init__(self) -> None:
        super().__init__()
        self._cache_registry = CacheRegistry(is_weak=False)
        self._need_detached = MaybeUnwrapKeysWeakKeyDict()

    @staticmethod
    def init_from_eager(eager_cache_state: EagerCacheState) -> "TracingCacheState":
        ret = TracingCacheState()
        ret.set_next_id(
            0 if eager_cache_state is None else eager_cache_state.get_next_id()
        )
        return ret

    def register_tensor(self, tensor, cache_id):
        # Don't allow changing in compile
        if tensor in self._tensor_to_cache_id:
            assert self._tensor_to_cache_id[tensor] == cache_id
        self._tensor_to_cache_id[tensor] = cache_id

    def set_next_id(self, next_id: int):
        self._cache_registry.set_next_id(next_id)

    def try_get_entry(self, tensor_ref, key) -> Optional[torch.Tensor]:
        cache = self.try_get_cache(tensor_ref)
        assert cache is not None

        if self._need_detached.get(tensor_ref, False):
            detached_key = key + "_detached"
            ret = cache.try_get_entry(detached_key)
            if ret is None:
                entry = cache.try_get_entry(key)
                if entry is not None:
                    ret = torch.detach(entry)
                    cache.set_entry(detached_key, ret)
            return ret
        return cache.try_get_entry(key)

    def is_registered(self, cache_id):
        return cache_id in self._cache_registry._cache_id_to_cache

    def register_cache(self, data, cache_id):
        assert cache_id not in self._cache_registry._cache_id_to_cache
        cache = TreeCache(cache_id=cache_id, data=data)
        self._cache_registry._cache_id_to_cache[cache_id] = cache
        for tensor in data.values():
            self.register_tensor(tensor, cache_id)
        return cache

    def register_detached_tensor(self, new_tensor, old_tensor):
        if old_tensor not in self._tensor_to_cache_id:
            return
        cache = self.try_get_cache(old_tensor)
        assert cache is not None
        self.register_tensor(new_tensor, cache.id)
        self._need_detached[new_tensor] = True


_global_cache_state = None


def get_global_cache_state():
    global _global_cache_state

    if _global_cache_state is None:
        _global_cache_state = EagerCacheState()

    return _global_cache_state


# Custom op registration for operations that perform side effects during compile
# e.g., adding a new cache entry to an existing cache.

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

lib.define("_clear_tmp_ref(Tensor ref_tensor) -> ()")


def _clear_tmp_ref_impl(ref_tensor):
    get_global_cache_state().clear_tmp_ref(ref_tensor)


def _clear_tmp_ref_meta(ref_tensor):
    # Nothing needs to be done here.
    pass


lib.impl("_clear_tmp_ref", _clear_tmp_ref_impl, "CPU")
lib.impl("_clear_tmp_ref", _clear_tmp_ref_impl, "CUDA")
lib.impl("_clear_tmp_ref", _clear_tmp_ref_meta, "Meta")


lib.define("_merge_caches(Tensor ref_tensor_a, Tensor ref_tensor_b) -> ()")


def _merge_caches_impl(ref_tensor_a, ref_tensor_b):
    get_global_cache_state().merge(ref_tensor_a, ref_tensor_b)


def _merge_caches_meta(ref_tensor_a, ref_tensor_b):
    mb_fake_mode = try_get_fake_mode(ref_tensor_a)
    if mb_fake_mode is not None:
        mb_fake_mode.nested_cache_state.merge(ref_tensor_a, ref_tensor_b)
    else:
        # Called with meta tensors in eager (nested dummy tensors)
        assert ref_tensor_a.device.type == "meta"
        get_global_cache_state().merge(ref_tensor_a, ref_tensor_b)


lib.impl("_merge_caches", _merge_caches_impl, "CPU")
lib.impl("_merge_caches", _merge_caches_impl, "CUDA")
lib.impl("_merge_caches", _merge_caches_meta, "Meta")


# This does kind of look like merge
lib.define("_register_tensor(Tensor ref_tensor, Tensor tensor) -> ()")


def _register_tensor_impl_inner(cache_state, ref_tensor, tensor):
    cache = cache_state.try_get_cache(ref_tensor)
    assert cache is not None
    cache_state.register_tensor(tensor, cache.id)


def _register_tensor_impl(ref_tensor, tensor):
    _register_tensor_impl_inner(get_global_cache_state(), ref_tensor, tensor)


def _register_tensor_meta(ref_tensor, tensor):
    mb_fake_mode = try_get_fake_mode(ref_tensor)
    if mb_fake_mode is not None:
        _register_tensor_impl_inner(mb_fake_mode.nested_cache_state, ref_tensor, tensor)
    else:
        # assert ref_tensor.device.type == "meta"
        print(ref_tensor.device.type)
        _register_tensor_impl_inner(get_global_cache_state(), ref_tensor, tensor)


lib.impl("_register_tensor", _register_tensor_impl, "CPU")
lib.impl("_register_tensor", _register_tensor_impl, "CUDA")
lib.impl("_register_tensor", _register_tensor_meta, "Meta")


from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

# Ensure these ops are not DCE'd
# Using the same token is okay?
_register_effectful_op(torch.ops.nested._add_cache_entry.default, _EffectType.ORDERED)
_register_effectful_op(torch.ops.nested._clear_tmp_ref.default, _EffectType.ORDERED)
_register_effectful_op(torch.ops.nested._merge_caches.default, _EffectType.ORDERED)
# _register_effectful_op(torch.ops.nested._register_tensor.default, _EffectType.ORDERED)


def _get_ref_tensor(
    data_or_tensor: Union[Dict[str, torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    if isinstance(data_or_tensor, torch.Tensor):
        return data_or_tensor
    return next(iter(data_or_tensor.values()))


#
# "Composite Cache APIs"
#
# If you are running code that is "composite", e.g., shared between
# the compile/eager paths, you should call into these wrappers, which
# handle the fake/non-fake routing for you.
#
# These are NOT public APIs; you should not call them directly.
# Dynamo should graph break TODO.
#
def try_get_cache(data_or_tensor: Union[Dict[str, torch.Tensor], torch.Tensor]):
    mb_fake_mode = try_get_fake_mode(data_or_tensor)
    cached_tensor = _get_ref_tensor(data_or_tensor)

    if not isinstance(cached_tensor, torch.Tensor):
        breakpoint()
    if mb_fake_mode is not None:
        return mb_fake_mode.nested_cache_state.try_get_cache(cached_tensor)
    else:
        return get_global_cache_state().try_get_cache(cached_tensor)


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


def merge_caches(cache_a, cache_b):
    if cache_a is None:
        return cache_b
    if cache_b is None:
        return cache_a
    # Can we ban accessing .data directly?
    ref_tensor_a = _get_ref_tensor(cache_a.data)
    ref_tensor_b = _get_ref_tensor(cache_b.data)
    # breakpoint()
    if ref_tensor_a is ref_tensor_b:
        return cache_a.find()
    torch.ops.nested._merge_caches(ref_tensor_a, ref_tensor_b)
    return cache_a.find()


# I have an existing cache, and want to make it so that when I construct a tensor
# with. (you could almost think of this as merging...)
def register_tensor(cache, tensor):
    ref_tensor = _get_ref_tensor(cache.data)
    torch.ops.nested._register_tensor(ref_tensor, tensor)
