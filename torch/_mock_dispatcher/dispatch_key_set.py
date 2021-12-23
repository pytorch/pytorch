from typing import List, Optional, Dict, Tuple, Iterator
from functools import reduce
from torch._mock_dispatcher.dispatch_key import (
    DispatchKey, isAliasDispatchKey, toBackendKey, toFunctionalityKey, isPerBackendFunctionalityKey,
    isRuntimeDispatchKey, NUM_BACKENDS)

class DispatchKeySet:
    # This class represents a 64-bitset
    repr_: int

    # In the C++ code, this is originally just the number of runtime keys
    # in the DispatchKey enum.
    # That's just a coincidence though in the long-run: this will always be 64 bits.
    SIZE: int = 64
    SIXTY_FOUR_ONES: int = 18446744073709551615

    # This function is large because it subsumes 4 different constructors in C++:
    # - DispatchKeySet(FULL)
    # - DispatchKeySet(EMPTY)
    # - DispatchKeySet(RAW)
    # - DispatchKeySet(DispatchKey k)
    def __init__(self, k: Optional[DispatchKey] = None, *, raw_repr: Optional[int] = None, full: Optional[bool] = None) -> None:
        # raw_repr constructor
        if raw_repr:
            assert full is None
            assert k is None
            self.repr_ = raw_repr
            return

        # full/empty constructor
        if full is not None:
            assert k is None
            if full:
                # this converts to a 64-length str of 1's in binary
                self.repr_ = self.SIXTY_FOUR_ONES
            else:
                self.repr_ = 0
            return

        # DispatchKey constructor
        assert k is not None
        assert full is None and raw_repr is None
        assert not isAliasDispatchKey(k)
        if k == DispatchKey.Undefined:
            # special case 1: handle Undefined
            backend_val = 0
            functionality_val = 0
        elif k.value <= DispatchKey.EndOfBackendKeys.value:
            # special case 2: if you directly pass e.g. DispatchKey.CPUBit,
            # then we'll set the cpu "backend" bit without setting any functionality bits
            backend_val = 1 << k.value
            functionality_val = 0
        elif isPerBackendFunctionalityKey(k):
            # special case 3: if you directly pass e.g. DispatchKey.Dense,
            # then we'll set the dense "functionality" bit without setting any backend bits
            backend_val = 0
            # The - 1 is because Undefined is technically a "functionality" that doesn't show up in the bitset.
            # So Dense is technically the second functionality, but the lowest functionality bit.
            functionality_val = 1 << (k.value - 1)
        else:
            # first compute which bit to flip for the backend
            backend_k = toBackendKey(k)
            if backend_k.value <= DispatchKey.EndOfBackendKeys.value:
                # This is a per-backend key, so set the backend bit
                backend_val = 1 << backend_k.value
            else:
                # Not a per-backend key, so not backend bit to set.
                # Note that for "fake" backends (e.g. FPGA, Meta), they don't get a backend bit.
                # That means that it's possible for a tensor to have 0 backend bits set.
                backend_val = 0

            # then compute which bit to flip for the functionality.
            functionality_k = toFunctionalityKey(k)
            assert functionality_k.value > DispatchKey.EndOfBackendKeys.value
            assert functionality_k.value <= DispatchKey.EndOfFunctionalityKeys.value
            # The - 1 is because Undefined is technically a "functionality" that doesn't show up in the bitset.
            # So Dense is technically the second functionality, but the lowest functionality bit.
            functionality_val = 1 << (functionality_k.value - 1)
        # Finally, combine the backend and functionality bits
        self.repr_ = backend_val + functionality_val

    def __iter__(self) -> Iterator['DispatchKey']:
        # Iterating through a DispatchKeySet should only look at values
        # That are allowed to correspond to indices of runtime kernels.
        for k in DispatchKey:
            if k.value <= DispatchKey.EndOfBackendKeys.value:
                continue
            if k == DispatchKey.Undefined:
                continue
            if isPerBackendFunctionalityKey(k):
                continue
            if isAliasDispatchKey(k):
                continue
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

    def removeFunctionalityBit(self, k: DispatchKey) -> 'DispatchKeySet':
        # NOTE: removing from a dispatch key set has weird semantics.
        # This functionality is needed to deal with fallthrough keys, but otherwise should be used with EXTREME caution.
        # For now, we're only allowing removal of "functionality bits" from the keyset,
        # which is specifically needed by the fallthrough key calculation logic.
        # Why is removing backend bits problematic? Consider this example:
        #
        # DispatchKeySet([DispatchKey.CPU, DispatchKey.AutogradCUDA, DispatchKey.CUDA]).remove(DispatchKey.AutogradCUDA)
        # DispatchKeySet([DispatchKey.CPU, DispatchKey.AutogradCUDA]).remove(DispatchKey.AutogradCUDA)
        #
        # What do we want to happen?
        # Technically, we'd like it to be true that after removal,
        # the first keyset still has the CUDA dispatch key while the second doesn't.
        # Unfortunately there's no way to represent that, because the two keysets are represented the same way internally:
        # functionality bits: Autograd, Dense
        # backned bits: CPU, CUDA
        assert k.value > DispatchKey.EndOfBackendKeys.value
        functionality_key = toFunctionalityKey(k)
        functionality_keyset = DispatchKeySet(functionality_key)

        out = self.copy()
        if out.has(functionality_key):
            out.repr_ -= functionality_keyset.repr_
        return out

    def __eq__(self, ks: object) -> bool:
        return isinstance(ks, DispatchKeySet) and self.repr_ == ks.repr_

    def has(self, k: DispatchKey) -> bool:
        assert k is not DispatchKey.Undefined
        ks = DispatchKeySet(k)
        return self & ks == ks

    def __rshift__(self, shift: int) -> 'DispatchKeySet':
        out = self.copy()
        out.repr_ = self.repr_ >> shift
        return out

    def __lshift__(self, shift: int) -> 'DispatchKeySet':
        out = self.copy()
        out.repr_ = self.repr_ << shift
        return out

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

backend_bitset_mask: DispatchKeySet = DispatchKeySet(raw_repr=(1 << DispatchKey.EndOfBackendKeys.value + 1) - 1)


# maps:
# - 0 (Undefined) -> 0, FALSE
# - 1 (Dense) -> 1 (...12), TRUE
# - 2...12 (FPGA...Meta) -> 13...23, FALSE
# - 13 (Quantized) -> 24 (...35), TRUE
# - 14 (Sparse) -> 36 (...47), TRUE
# - 15 (SparseCsrCPU) -> 48, FALSE
# - 16 (SparseCsrCUDA) -> 49, FALSE
# - 17 (BackendSelect) -> 50, FALSE
# ...
# In C++ this should be an array of packed pairs of (int64_t, int64_t)
functionality_key_offset_and_mask: Dict[int, Tuple[int, DispatchKeySet]] = {}
# Directly add DispatchKey.Undefined first
functionality_key_offset_and_mask[0] = (0, DispatchKeySet(full=False),)
# Loop through every functionality key, starting AFTER undefined (that's why we have the + 2)
for k_val in range(DispatchKey.EndOfBackendKeys.value + 2, DispatchKey.EndOfFunctionalityKeys.value + 1):
    key = DispatchKey(k_val)
    idx = k_val - NUM_BACKENDS  # this should map e.g. Undefined -> 0, Dense -> 1, FPGA -> 1, ...

    prev_offset, prev_mask = functionality_key_offset_and_mask[idx - 1]

    # If the previous functionality is defined per-backend, then it will take up NUM_BACKEND
    # slots in the runtime operator table.
    next_offset = prev_offset + (1 if prev_mask == DispatchKeySet(full=False) else NUM_BACKENDS)
    next_mask = backend_bitset_mask if isPerBackendFunctionalityKey(key) else DispatchKeySet(full=False)
    functionality_key_offset_and_mask[idx] = (next_offset, next_mask,)

def num_entries() -> int:
    num_functionalities = len(functionality_key_offset_and_mask)
    last_offset, last_mask = functionality_key_offset_and_mask[num_functionalities - 1]
    return last_offset + (1 if last_mask == DispatchKeySet(full=False) else NUM_BACKENDS)

def isBackendDispatchKey(k: DispatchKey) -> bool:
    return k is not DispatchKey.Undefined and not isAliasDispatchKey(k) and backend_dispatch_keyset.has(k)

def getRuntimeDispatchKeySet(k: DispatchKey) -> DispatchKeySet:
    if k is DispatchKey.Undefined:
        raise AssertionError()
    if k is DispatchKey.AutogradAlias:
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

# This is used at runtime to compute the offset into the operator table
# total perf cost:
# 1x load
# 2x highestPriorityTypeId()
# 2x >>
# 2x &
# 1x +
def getDispatchTableIndexForDispatchKeySet(ks: DispatchKeySet) -> int:
    functionality_idx = (ks >> NUM_BACKENDS).highestPriorityTypeId().value
    functionality_offset, is_per_backend_functionality_mask = functionality_key_offset_and_mask[functionality_idx]

    # Mask the functionality bits out first, then right-shift by 1.
    # right-shifting by 1 because everything is zero-indexed.
    # E.g. 000001 (CPU) should give us an offset of 0, 000010 (CUDA) should give us an offset of 1, etc.
    backend_idx = ((ks & is_per_backend_functionality_mask) >> 1).highestPriorityTypeId().value
    return functionality_offset + backend_idx

# This is used at registration time to map registrations to a given DispatchKey to a spot in the runtime operator table
def getDispatchTableIndexForDispatchKey(k: DispatchKey) -> int:
    assert isRuntimeDispatchKey(k)
    # Undefined is a runtime key since it gets a slot in the runtime operator table, but it does not belong in the DispatchKeySet.
    if k == DispatchKey.Undefined:
        return 0
    return getDispatchTableIndexForDispatchKeySet(DispatchKeySet(k))
