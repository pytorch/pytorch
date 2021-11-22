from functools import reduce
from typing import List
from torch._mock_dispatcher.dispatch_key_set import DispatchKeySet
from torch._mock_dispatcher.dispatch_key import DispatchKey, NUM_BACKENDS, toBackendKey, toFunctionalityKey, isPerBackendFunctionalityKey

class DispatchKeyExtractor:
    nonFallthroughKeys_: List[DispatchKeySet]
    # missing dispatch_arg_indices_reverse

    def __init__(self) -> None:
        # Every backend gets its own fallthrough key bitset.
        # This is because different backends can independently choose to fallthrough for a given piece of functionality,
        # E.g. AutogradCPU vs AutogradCUDA
        self.nonFallthroughKeys_ = [DispatchKeySet(full=True) for _ in range(NUM_BACKENDS)]

    def setOperatorHasFallthroughForKey(self, k: DispatchKey, has_fallthrough: bool) -> None:
        if k != DispatchKey.Undefined and isPerBackendFunctionalityKey(toFunctionalityKey(k)):
            # This is a per-backend functionality key.
            # We need to figure out what the currenty backend is,
            # and only update the bitset for that backend.
            idx = toBackendKey(k).value
            if has_fallthrough:
                self.nonFallthroughKeys_[idx] = self.nonFallthroughKeys_[idx].removeFunctionalityBit(k)
            else:
                self.nonFallthroughKeys_[idx] = self.nonFallthroughKeys_[idx].add(k)
            return
        # Otherwise, if a fallthrough is set for a functionality that isn't per backend,
        # Then we update the fallthrough bitset for EVERY backend.
        for i in range(NUM_BACKENDS):
            if has_fallthrough:
                self.nonFallthroughKeys_[i] = self.nonFallthroughKeys_[i].removeFunctionalityBit(k)
            else:
                self.nonFallthroughKeys_[i] = self.nonFallthroughKeys_[i].add(k)
        return

    def getDispatchKeySetUnboxed(self, args: List[DispatchKeySet]) -> DispatchKeySet:
        # Note: skipping TLS logic from C++
        return reduce(lambda k1, k2: k1 | k2, args, DispatchKeySet(full=False))
