from functools import reduce
from typing import List
from torch._mock_dispatcher.dispatch_key_set import (
    DispatchKeySet, backend_bitset_mask)
from torch._mock_dispatcher.dispatch_key import (
    DispatchKey, NUM_BACKENDS, toBackendKey, toFunctionalityKey,
    isPerBackendFunctionalityKey)

class DispatchKeyExtractor:
    # Why do we have a separate field for fallthrough keys, and per-backend fallthrough keys?
    # This is an optimization.
    # In almost all cases, an operator will have the same fallthrough bitset for every backend.
    # We can take advantage of this to directly read from nonFallthroughKeys_ at runtime,
    # instead of having to do a load from nonFallthroughKeysPerBackend_.
    # Note: If we make Autocast a per-backend functionality key, then this optimization will be slightly less useful,
    # since there are a few ops that get fallthrough kernels registered separately for cpu vs cuda.
    nonFallthroughKeys_: DispatchKeySet
    # Invariant: if requiresBitsetPerBackend_ is True, then we can use nonFallthroughKeys_
    # to perform the runtime fallthrough key masking.
    # If it's false, then requiresBitsetPerBackend_ is in an invalid state.
    requiresBitsetPerBackend_: bool
    nonFallthroughKeysPerBackend_: List[DispatchKeySet]
    # missing dispatch_arg_indices_reverse

    def __init__(self) -> None:
        # Every backend gets its own fallthrough key bitset.
        # This is because different backends can independently choose to fallthrough for a given piece of functionality,
        # E.g. AutogradCPU vs AutogradCUDA
        self.nonFallthroughKeysPerBackend_ = [DispatchKeySet(full=True) for _ in range(NUM_BACKENDS)]
        self.nonFallthroughKeys_ = DispatchKeySet(full=True)
        self.requiresBitsetPerBackend_ = False

    def setOperatorHasFallthroughForKey(self, k: DispatchKey, has_fallthrough: bool) -> None:
        # (1) Update nonFallthroughKeys_
        if has_fallthrough:
            self.nonFallthroughKeys_ = self.nonFallthroughKeys_.removeFunctionalityBit(k)
        else:
            self.nonFallthroughKeys_ = self.nonFallthroughKeys_.add(k)

        # (2) Update the per-backend nonfallthrough keys.
        if k != DispatchKey.Undefined and isPerBackendFunctionalityKey(toFunctionalityKey(k)):
            # This is a per-backend functionality key.
            # We need to figure out what the currenty backend is,
            # and only update the bitset for that backend.
            idx = toBackendKey(k).value
            if has_fallthrough:
                self.nonFallthroughKeysPerBackend_[idx] = self.nonFallthroughKeysPerBackend_[idx].removeFunctionalityBit(k)
            else:
                self.nonFallthroughKeysPerBackend_[idx] = self.nonFallthroughKeysPerBackend_[idx].add(k)
        else:
            # Otherwise, if a fallthrough is set for a functionality that isn't per backend,
            # Then we update the fallthrough bitset for EVERY backend.
            for i in range(NUM_BACKENDS):
                if has_fallthrough:
                    self.nonFallthroughKeysPerBackend_[i] = self.nonFallthroughKeysPerBackend_[i].removeFunctionalityBit(k)
                else:
                    self.nonFallthroughKeysPerBackend_[i] = self.nonFallthroughKeysPerBackend_[i].add(k)

        # (3) set requiresBitsetPerBackend accordingly
        for i in range(NUM_BACKENDS - 1):
            if self.nonFallthroughKeysPerBackend_[i] != self.nonFallthroughKeysPerBackend_[i + 1]:
                self.requiresBitsetPerBackend_ = True
                return
        self.requiresBitsetPerBackend_ = False

    def getDispatchKeySetUnboxed(self, args: List[DispatchKeySet]) -> DispatchKeySet:
        # Note: skipping TLS logic from C++
        reduced_keys = reduce(lambda k1, k2: k1 | k2, args, DispatchKeySet(full=False))
        if not self.requiresBitsetPerBackend_:
            return reduced_keys & self.nonFallthroughKeys_
        else:
            # Note: We're technically redundantly doing the work to calculate the backend from the bitset here.
            # We probably don't need to worry about de-duplicating for now, since this path should almost never be taken.
            backend_idx = ((reduced_keys & backend_bitset_mask) >> 1).highestPriorityTypeId().value
            return reduced_keys & self.nonFallthroughKeysPerBackend_[backend_idx]
