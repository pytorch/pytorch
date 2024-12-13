import weakref
from typing import *

import torch
from torch.nested._internal.nested_int import NestedIntNode
from torch.nested._internal.tensor_registry import TensorRegistry


RAGGED_SOURCE_KEYS = [
    "cpu_lengths",
    "cpu_offsets",
    "device_lengths",
    "device_offsets",
]

# If we accept the input in the form of offsets/lengths/cpu_offsets/cpu_lengtsh
# we need to do some normalization at some point. Where should it be done?
def process_cpu_device_offsets_lengths(
    *, offsets=None, lengths=None, cpu_offsets=None, cpu_lengths=None
):
    # The device of offsets/lengths determines the device of the NestedTensor.
    # CPU NestedTensor cannot have device offsets/lengths cached.
    device_offsets, device_lengths = None, None
    if offsets is not None:
        if offsets.is_cpu:
            cpu_offsets = offsets
        else:
            device_offsets = offsets
    if lengths is not None:
        if lengths.is_cpu:
            cpu_lengths = lengths
        else:
            device_lengths = lengths
    ret = {
        # Duplicate :(
        "cpu_offsets": cpu_offsets,
        "cpu_lengths": cpu_lengths,
        "device_offsets": device_offsets,
        "device_lengths": device_lengths,
    }
    for k, v in ret.items():
        if v is None:
            del ret[k]
    return ret


# Thin wrapper around a dict to help us enable weak references, then handle
# extra things?
# Do not construct/update this directly! Search for the "Composite Cache APIs".
class NestedCache:
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


# Maintains a NestedCache to int map
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
        cache = NestedCache(data=data, cache_id=self._cache_id_counter)
        self._cache_id_counter += 1
        self._cache_id_to_cache_ref[cache.id] = weakref.ref(cache)
        return cache

    def copy(self):
        ret = CacheRegistry()
        ret._cache_id_to_cache_ref = self._cache_id_to_cache_ref.copy()
        ret._cache_id_counter = self._cache_id_counter
        return ret


def _assert_not_fake(t):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    t = mb_unwrap_functional_tensor(t)
    assert not isinstance(t, FakeTensor)


# TODO(soulitzer): Update the compile story for global state.
class NestedTensorState:
    # Encapsulates all global state for NestedTensors. Only used in eager.
    # Compile uses something simpler without weak ref handling
    #
    # NestedTensorState is responsible for the mapping from tensor id to cache id
    # in eager. Maintenance of the id mappings state is left to the tensor/cache
    # registeries.
    def __init__(self):
        # tensor <-> cache_id
        self._tensor_registry = TensorRegistry()
        # tensor_id -> cache_id (can be swapped out with something like union-find)
        self._tensor_id_to_cache_id: Dict[int, int] = {}
        # cache_id <-> cache
        self._cache_registry = CacheRegistry()

    def maybe_update_tensor_id_to_cache_id(self, cache, k, v):
        _assert_not_fake(v)
        if k not in RAGGED_SOURCE_KEYS:
            return
        tensor_id = self._tensor_registry.get_int(v)
        # Require that the tensor not be already registered.
        assert tensor_id not in self._tensor_id_to_cache_id
        self._tensor_id_to_cache_id[tensor_id] = cache.id

    def get_cache_if_exists(self, tensor) -> Optional[NestedCache]:
        # What is the None story here...
        _assert_not_fake(tensor)
        tensor_id = self._tensor_registry.get_int(tensor)
        mb_cache_id = self._tensor_id_to_cache_id.get(tensor_id)
        if mb_cache_id is None:
            return None
        return self._cache_registry.try_get_cache(mb_cache_id)

    def add_entry_to_cache(self, cache_id, key, value):
        # Take cache_id instead of NestedCache because this must be called
        # through our custom op.
        cache = self._cache_registry.try_get_cache(cache_id)
        assert cache is not None, "add_entry_to_cache: Expected cache to exist"
        cache.data[key] = value
        self.maybe_update_tensor_id_to_cache_id(cache, key, value)

    def create_cache_with_data(self, data: Dict):
        # Don't create a fresh dict because we created this one.
        cache = self._cache_registry.create_cache(data)
        for k, v in data.items():
            self.maybe_update_tensor_id_to_cache_id(cache, k, v)
        return cache

    def copy(self):
        ret = NestedTensorState()
        ret._cache_registry = self._cache_registry.copy()
        ret._tensor_id_to_cache_id = self._tensor_id_to_cache_id.copy()
        ret._tensor_registry = self._tensor_registry.copy()
        return ret


_global_nested_state = NestedTensorState()


def _try_get_fake_mode(obj):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    if isinstance(obj, dict):
        for v in obj.values():
            if v is not None:
                fake_mode = _try_get_fake_mode(v)
                if fake_mode is not None:
                    return fake_mode
    elif isinstance(obj, torch.Tensor):
        t = mb_unwrap_functional_tensor(obj)
        if isinstance(t, FakeTensor):
            return t.fake_mode
        else:
            return None
    elif isinstance(obj, NestedCache):
        # TODO(soulitzer): revisit assumptions
        # Assume that I have a cache that is registered somewhere.
        # Assume that anything in the cache has been registered.
        # Every NestedCache must contain at least one tensor
        return _try_get_fake_mode(obj.data)
    else:
        assert False, f"get_fake_mode: got unexpected type {type(obj)}"


# Custom op registration for operations that perform side effect during compile
# e.g. adding a new cache entry to an existing cache.

lib = torch.library.Library("nested", "FRAGMENT")

lib.define("_add_nested_cache_entry(Tensor val, str key, int cache_id) -> ()")


def _add_nested_cache_entry_impl(val, key, cache_id):
    # Not sure if it actually matters, but the signatures are different because I want
    # the first argument of the custom op to be a tensor.
    _global_nested_state.add_entry_to_cache(cache_id, key, val)


def _add_nested_cache_entry_meta(val, key, cache_id):
    # Update the cache in the FakeMode
    mb_fake_mode = _try_get_fake_mode(val)
    assert mb_fake_mode is not None
    mb_fake_mode.add_entry_to_cache(cache_id, key, val)


lib.impl("_add_nested_cache_entry", _add_nested_cache_entry_impl, "CPU")
lib.impl("_add_nested_cache_entry", _add_nested_cache_entry_impl, "CUDA")
lib.impl("_add_nested_cache_entry", _add_nested_cache_entry_meta, "Meta")

from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

# Make sure this does not get DCE'd
_register_effectful_op(
    torch.ops.nested._add_nested_cache_entry.default, _EffectType.ORDERED
)


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
def get_cache_if_exists(data: Dict[str, Optional[torch.Tensor]]):
    mb_fake_mode = _try_get_fake_mode(data)
    if mb_fake_mode is not None:
        return mb_fake_mode.get_nested_cache_if_exists(data)
    else:
        return _global_nested_state.get_cache_if_exists(data)


def create_cache_with_data(data: Dict[str, Optional[torch.Tensor]]):
    mb_fake_mode = _try_get_fake_mode(data)
    if mb_fake_mode is not None:
        return mb_fake_mode.create_cache_with_data(data)
    else:
        return _global_nested_state.create_cache_with_data(data)


def add_entry_to_cache(cache, key, value):
    torch.ops.nested._add_nested_cache_entry(value, key, cache.id)


def try_get_cache_entry(cache, key):
    return cache.data.get(key)


def get_nested_symint(cache: NestedCache, *, coeff=1):
    mb_fake_mode = _try_get_fake_mode(cache)
    if mb_fake_mode is not None:
        # In compile, keep the same instance of nested int around
        return mb_fake_mode.get_nested_symint(cache) * coeff
    else:
        # In eager, always create a fresh nested int.
        return torch.SymInt(NestedIntNode(cache, coeff=coeff))
