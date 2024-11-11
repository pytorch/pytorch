from typing import *
import weakref

import torch
from torch.utils.weak import WeakTensorKeyDictionary


# TODO(soulitzer): Inference tensors should be banned from registering.
# If user has offsets/lengths, they should copy-on-write create a new non-inference tensor.
class TensorRegistry:
    # Assigns Tensor objects to unique ints in an incrementing fashion.
    # The int given corresponds to a particular version of a Tensor.
    # If a Tensor has been mutated, its original int is invalidated, and
    # it will be assigned a new int upon the next get_int.
    # We try to be careful to NOT hold any owning references.
    _incrementing_id = 0
    _tensor_to_int_and_version = WeakTensorKeyDictionary()
    _int_to_tensor: Dict[int, weakref.ReferenceType] = dict()

    # TODO(soulitzer): Why are these allow-in-graph?
    @torch._dynamo.allow_in_graph
    def get_int(self, t) -> int:
        mb_data = self._tensor_to_int_and_version.get(t)
        if mb_data is None or mb_data[1] != t._version:
            self._tensor_to_int_and_version[t] = (self._incrementing_id, t._version)
            self._int_to_tensor[self._incrementing_id] = weakref.ref(t)
            self._incrementing_id += 1
        return self._tensor_to_int_and_version[t][0]

    @torch._dynamo.allow_in_graph
    def try_get_tensor(self, i):
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

    # TODO(soulitzer): do we still need this?
    def is_registered(self, t):
        return t in self._tensor_to_int_and_version

    def copy(self):
        new_map = TensorRegistry()
        new_map._incrementing_id = self._incrementing_id
        # TODO(soulitzer): double check that copying such dicts are okay.
        new_map._tensor_to_int_and_version = self._tensor_to_int_and_version.copy()
        new_map._int_to_tensor = self._int_to_tensor.copy()
        return new_map
