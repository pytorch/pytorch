from functools import reduce
from typing import List
from torch._mock_dispatcher.dispatch_key_set import DispatchKeySet
from torch._mock_dispatcher.dispatch_key import DispatchKey

class DispatchKeyExtractor:
    nonFallthroughKeys_: DispatchKeySet
    # missing dispatch_arg_indices_reverse

    def __init__(self) -> None:
        self.nonFallthroughKeys_ = DispatchKeySet(full=True)

    def setOperatorHasFallthroughForKey(self, k: DispatchKey, has_fallthrough: bool) -> None:
        if has_fallthrough:
            self.nonFallthroughKeys_ = self.nonFallthroughKeys_.remove(k)
        else:
            self.nonFallthroughKeys_ = self.nonFallthroughKeys_.add(k)

    def getDispatchKeySetUnboxed(self, args: List[DispatchKeySet]) -> DispatchKeySet:
        # Note: skipping TLS logic from C++
        return reduce(lambda k1, k2: k1 | k2, args)
