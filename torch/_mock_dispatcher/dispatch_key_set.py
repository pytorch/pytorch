from typing import List, Optional
from functools import reduce
from torch._mock_dispatcher.dispatch_key import DispatchKey, getDispatchTableIndexForDispatchKey, isAliasDispatchKey

class DispatchKeySet:
    # This class represents a 64-bitset
    repr_: int

    # In the C++ code, this is originally just the number of runtime keys
    # in the DispatchKey enum.
    # That's just a coincidence though in the long-run: this will always be 64 bits.
    SIZE: int = 64

    def __init__(self, k: Optional[DispatchKey] = None, *, full: Optional[bool] = None) -> None:
        if k is not None:
            self.repr_ = 0 if k == DispatchKey.Undefined else 1 << (k.value - 1)
        else:
            assert full is not None
            if full:
                # this converts to a 64-length str of 1's in binary
                self.repr_ = 18446744073709551615
            else:
                self.repr_ = 0

    def __iter__(self):
        for k in DispatchKey:
            if k != DispatchKey.Undefined and not isAliasDispatchKey(k):
                if self.has(k):
                    yield k

    @staticmethod
    def from_keys(keys: List[DispatchKey]) -> 'DispatchKeySet':
        as_ks_list = [DispatchKeySet(k) for k in keys]
        return reduce(lambda k1, k2: k1 | k2, as_ks_list, DispatchKeySet(full=False))

    def to_padded_bin_str(self) -> str:
        bin_str = "{0:b}".format(self.repr_)
        leading_zeros_needed = 64 - len(bin_str)
        return ("0" * leading_zeros_needed) + bin_str

    def highestPriorityTypeId(self) -> 'DispatchKey':
        padded_str = self.to_padded_bin_str()

        leading_zeros = 0
        for bit in padded_str:
            if bit == '0':
                leading_zeros += 1
            else:
                break
        return DispatchKey(64 - leading_zeros)

    def copy(self) -> 'DispatchKeySet':
        out = DispatchKeySet(full=False)
        out.repr_ = self.repr_
        return out

    def add(self, k: DispatchKey) -> 'DispatchKeySet':
        return self | DispatchKeySet(k)

    def remove(self, k: DispatchKey) -> 'DispatchKeySet':
        out = self.copy()
        # implementing differently from C++ because ~ in python is weird
        if out.has(k):
            out.repr_ -= k.value
        return out

    def has(self, k: DispatchKey) -> bool:
        assert k is not DispatchKey.Undefined
        return self.repr_ & DispatchKeySet(k).repr_ > 0

    def __or__(self, other: 'DispatchKeySet') -> 'DispatchKeySet':
        out = self.copy()
        out.repr_ = self.repr_ | other.repr_
        return out

    def __and__(self, other: 'DispatchKeySet') -> 'DispatchKeySet':
        out = self.copy()
        out.repr_ = self.repr_ & other.repr_
        return out

autograd_dispatch_keyset: DispatchKeySet = DispatchKeySet.from_keys([
    DispatchKey.AutogradCPU,
    DispatchKey.AutogradCUDA,
    DispatchKey.AutogradXLA,
    DispatchKey.AutogradLazy,
    DispatchKey.AutogradNestedTensor,
    DispatchKey.AutogradMLC,
    DispatchKey.AutogradHPU,
    DispatchKey.AutogradXPU,
    DispatchKey.AutogradPrivateUse1,
    DispatchKey.AutogradPrivateUse2,
    DispatchKey.AutogradPrivateUse3,
    DispatchKey.AutogradOther,
])

backend_dispatch_keyset: DispatchKeySet = DispatchKeySet.from_keys([
    DispatchKey.CPU,
    DispatchKey.CUDA,
    DispatchKey.XLA,
    DispatchKey.Lazy,
    DispatchKey.XPU,
    DispatchKey.PrivateUse1,
    DispatchKey.PrivateUse2,
    DispatchKey.PrivateUse3,
    DispatchKey.MLC,
    DispatchKey.HPU,
    DispatchKey.ORT,
    DispatchKey.Meta,
])

math_dispatch_keyset: DispatchKeySet = backend_dispatch_keyset | autograd_dispatch_keyset

autogradother_backends = DispatchKeySet.from_keys([
    DispatchKey.CPU,
    DispatchKey.CUDA,
    DispatchKey.XLA,
    DispatchKey.Lazy,
    DispatchKey.XPU,
    DispatchKey.PrivateUse1,
    DispatchKey.PrivateUse2,
    DispatchKey.PrivateUse3,
    DispatchKey.MLC,
    DispatchKey.HPU,
    DispatchKey.ORT,
    DispatchKey.Meta,
])

def num_entries() -> int:
    return len(DispatchKey)

def isBackendDispatchKey(k: DispatchKey) -> bool:
    return k is not DispatchKey.Undefined and not isAliasDispatchKey(k) and backend_dispatch_keyset.has(k)

def getRuntimeDispatchKeySet(k: DispatchKey) -> DispatchKeySet:
    if k is DispatchKey.Undefined:
        raise AssertionError()
    if k is DispatchKey.Autograd:
        return autograd_dispatch_keyset
    elif k is DispatchKey.CompositeImplicitAutograd:
        return math_dispatch_keyset
    elif k is DispatchKey.CompositeExplicitAutograd:
        return backend_dispatch_keyset
    else:
        return DispatchKeySet(k)

def runtimeDispatchKeySetHas(alias: DispatchKey, k: DispatchKey) -> bool:
    assert alias is not DispatchKey.Undefined
    if alias is DispatchKey.Autograd:
        return autograd_dispatch_keyset.has(k)
    elif alias is DispatchKey.CompositeImplicitAutograd:
        return math_dispatch_keyset.has(k)
    elif alias is DispatchKey.CompositeExplicitAutograd:
        return backend_dispatch_keyset.has(k)
    return alias == k

def isIncludedInAlias(k: DispatchKey, alias: DispatchKey) -> bool:
    return k is not DispatchKey.Undefined and runtimeDispatchKeySetHas(alias, k)

def getBackendKeySetFromAutograd(k: DispatchKey) -> DispatchKeySet:
    if k is DispatchKey.AutogradCPU:
        return DispatchKeySet(DispatchKey.CPU)
    if k is DispatchKey.AutogradCUDA:
        return DispatchKeySet(DispatchKey.CUDA)
    if k is DispatchKey.AutogradXLA:
        return DispatchKeySet(DispatchKey.XLA)
    if k is DispatchKey.AutogradLazy:
        return DispatchKeySet(DispatchKey.Lazy)
    if k is DispatchKey.AutogradMLC:
        return DispatchKeySet(DispatchKey.MLC)
    if k is DispatchKey.AutogradHPU:
        return DispatchKeySet(DispatchKey.HPU)
    if k is DispatchKey.AutogradNestedTensor:
        return DispatchKeySet(DispatchKey.NestedTensor)
    if k is DispatchKey.AutogradXPU:
        return DispatchKeySet(DispatchKey.XPU)
    if k is DispatchKey.AutogradPrivateUse1:
        return DispatchKeySet(DispatchKey.PrivateUse1)
    if k is DispatchKey.AutogradPrivateUse2:
        return DispatchKeySet(DispatchKey.PrivateUse2)
    if k is DispatchKey.AutogradPrivateUse3:
        return DispatchKeySet(DispatchKey.PrivateUse3)
    if k is DispatchKey.AutogradOther:
        return autogradother_backends
    return DispatchKeySet(full=False)
