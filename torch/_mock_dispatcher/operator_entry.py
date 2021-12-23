from typing import Optional, List, Dict
from torch._mock_dispatcher.dispatch_key import (
    DispatchKey, getAutogradKeyFromBackend,
    isAliasDispatchKey, isRuntimeDispatchKey)
from torch._mock_dispatcher.dispatch_key_set import (
    DispatchKeySet, num_entries, isIncludedInAlias, getBackendKeySetFromAutograd,
    autogradother_backends, getRuntimeDispatchKeySet,
    isBackendDispatchKey, getDispatchTableIndexForDispatchKeySet,
    getDispatchTableIndexForDispatchKey)
from torch._mock_dispatcher.dispatch_key_extractor import DispatchKeyExtractor
from torch._mock_dispatcher.kernel_function import KernelFunction

def _ambiguous_autograd_fn() -> None:
    raise AssertionError("ambiguous autograd")
ambiguousAutogradOtherKernel = KernelFunction(_ambiguous_autograd_fn)

def _missing_fn() -> None:
    raise AssertionError("missing kernel")
missingKernel = KernelFunction(_missing_fn)

class OperatorEntry:

    name_: str  # technically should be an OperatorName class
    dispatchTable_: List[KernelFunction]
    dispatchKeyExtractor_: DispatchKeyExtractor
    # kernels normally stores a list per entry to deal with Jupyter notebooks. Ignoring here.
    kernels_: Dict[DispatchKey, KernelFunction]
    # missing schema_, cpp_signature_, is_observed_

    def __init__(self, op: str, fallbacks: Dict[int, KernelFunction]) -> None:
        self.name_ = op
        self.dispatchTable_ = [KernelFunction() for _ in range(num_entries())]
        self.dispatchKeyExtractor_ = DispatchKeyExtractor()
        self.kernels_ = {}
        self.updateDispatchTableFull_(fallbacks)

    def dumpRuntimeState(self, k: Optional[DispatchKey] = None) -> str:
        if k is not None:
            return self.dispatchTable_[getDispatchTableIndexForDispatchKey(k)].name()

        def key_to_name(k: DispatchKey) -> str:
            if k in self.kernels_:
                return self.kernels_[k].name()
            return '[None]'

        # We no longer have a convenient way to go from "table offset index" to "DispatchKey".
        # For simplicity I'm computing it the slow brute force way, which is fine because this function is only used for debugging.
        def brute_force_idx_to_key(idx: int) -> str:
            for k in DispatchKey:
                if isRuntimeDispatchKey(k) and getDispatchTableIndexForDispatchKey(k) == idx:
                    return str(k)
            return 'Key does not exist'

        kernels_str = "\n".join(f"{brute_force_idx_to_key(i)}: {self.dispatchTable_[i].name()}" for i in range(num_entries()))
        return f"""name: {self.name_}

{kernels_str}"""

    def dumpRegistrationState(self, k: Optional[DispatchKey] = None) -> str:
        if k is not None:
            return self.kernels_[k].name() if k in self.kernels_ else "[None]"

        def key_to_name(k: DispatchKey) -> str:
            if k in self.kernels_:
                return self.kernels_[k].name()
            return '[None]'
        kernels_str = "\n".join(f"{str(k)}: {key_to_name(k)}" for k in DispatchKey
                                if isRuntimeDispatchKey(k) or isAliasDispatchKey(k))
        return f"""name: {self.name_}

{kernels_str}"""

    def lookup(self, ks: DispatchKeySet) -> KernelFunction:
        idx = getDispatchTableIndexForDispatchKeySet(ks)
        return self.dispatchTable_[idx]

    def hasKernelForAnyDispatchKey(self, ks: DispatchKeySet) -> bool:
        assert not any([k is DispatchKey.Undefined for k in self.kernels_])
        for key in self.kernels_:
            if not isAliasDispatchKey(key) and ks.has(key):
                return True
        return False

    def hasKernelForDispatchKey(self, k: DispatchKey) -> bool:
        assert not any([k is DispatchKey.Undefined for k in self.kernels_])
        for key in self.kernels_:
            if key == k:
                return True
        return False

    def getKernelForDispatchKey(self, dispatch_key: DispatchKey) -> Optional[KernelFunction]:
        if dispatch_key in self.kernels_:
            return self.kernels_[dispatch_key]
        return None

    def computeDispatchTableEntry(self, fallbacks: Dict[int, KernelFunction], dispatch_key: DispatchKey) -> KernelFunction:
        # 1. Operator registration
        direct_registration = self.getKernelForDispatchKey(dispatch_key)
        if direct_registration is not None:
            return direct_registration

        # 2.1 Use CompositeExplicitAutograd kernel if available.
        #     See Note [Undefined in dispatchTable_] for the special handling for Undefined.
        if dispatch_key is DispatchKey.Undefined or isIncludedInAlias(dispatch_key, DispatchKey.CompositeExplicitAutograd):
            default_backend_registration = self.getKernelForDispatchKey(DispatchKey.CompositeExplicitAutograd)
            if default_backend_registration is not None:
                return default_backend_registration

        # Note when there's direct registration to CompositeExplicitAutograd, this code path will only be hit by
        # non backend keys (e.g AutogradXXX, Batched etc) due to (2.1).
        has_backend_kernel = self.hasKernelForAnyDispatchKey(getBackendKeySetFromAutograd(dispatch_key)) \
            or self.hasKernelForDispatchKey(DispatchKey.CompositeExplicitAutograd)

        # 2.2. Use CompositeImplicitAutograd kernel if available.
        # For autograd keys, we only use kernel from CompositeImplicitAutograd
        #      when there's no direct registration to its corresponding backend key or CompositeExplicitAutograd.
        #      For AutogradOther, we return ambiguousAutogradOtherKernel() if there's registration
        #      to any of its backends.
        #      See Note [Undefined in dispatchTable_] for the special handling for Undefined.
        if dispatch_key is DispatchKey.Undefined or isIncludedInAlias(dispatch_key, DispatchKey.CompositeImplicitAutograd):
            math_registration = self.getKernelForDispatchKey(DispatchKey.CompositeImplicitAutograd)
            if math_registration is not None:
                if dispatch_key is DispatchKey.AutogradOther and self.hasKernelForAnyDispatchKey(autogradother_backends):
                    return ambiguousAutogradOtherKernel
                elif not has_backend_kernel:
                    return math_registration

        # 2.3. For autograd backend keys, use kernel from DispatchKey::Autograd if available
        if isIncludedInAlias(dispatch_key, DispatchKey.Autograd):
            autograd_registration = self.getKernelForDispatchKey(DispatchKey.Autograd)
            if autograd_registration is not None:
                return autograd_registration

        # 3. Backend fallback
        dispatch_idx = getDispatchTableIndexForDispatchKey(dispatch_key)
        if fallbacks[dispatch_idx].isValid():
            return fallbacks[dispatch_idx]

        # 4. Default to error
        return missingKernel

    def updateDispatchTableEntry_(self, fallbacks: Dict[int, KernelFunction], dispatch_key: DispatchKey) -> None:
        idx = getDispatchTableIndexForDispatchKey(dispatch_key)
        if idx == -1:
            return
        self.dispatchTable_[idx] = self.computeDispatchTableEntry(fallbacks, dispatch_key)
        self.dispatchKeyExtractor_.setOperatorHasFallthroughForKey(dispatch_key, self.dispatchTable_[idx].isFallthrough())
        return

    def updateDispatchTable_(self, fallbacks: Dict[int, KernelFunction], dispatch_key: DispatchKey) -> None:
        # Handle Undefined separately since it isn't a runtime key but we have an entry in dispatchTable_.
        # See Note [Undefined in dispatchTable_]
        if dispatch_key is DispatchKey.Undefined:
            self.updateDispatchTableEntry_(fallbacks, dispatch_key)
            return

        for k in getRuntimeDispatchKeySet(dispatch_key):
            self.updateDispatchTableEntry_(fallbacks, k)

        # Registration to CompositeExplicitAutograd and CompositeImplicitAutograd should be populated to Undefined.
        # We cannot do this above since Undefined cannot be represented in DispatchKeySet.
        if dispatch_key is DispatchKey.CompositeImplicitAutograd or dispatch_key is DispatchKey.CompositeExplicitAutograd:
            self.updateDispatchTableEntry_(fallbacks, DispatchKey.Undefined)

        # Note [Refresh Runtime Autograd entries in dispatchTable_]
        # Registering to backend key might affect computed entry at its Autograd backend key due to (2.1) & (2.3).
        if isBackendDispatchKey(dispatch_key):
            autograd_key = getAutogradKeyFromBackend(dispatch_key)
            self.updateDispatchTableEntry_(fallbacks, autograd_key)

    def updateDispatchTableFull_(self, fallbacks: Dict[int, KernelFunction]) -> None:
        for k in DispatchKey:
            if isRuntimeDispatchKey(k):
                self.updateDispatchTable_(fallbacks, k)

    def updateFallback(self, fallbacks: Dict[int, KernelFunction], k: DispatchKey) -> None:
        self.updateDispatchTable_(fallbacks, k)

    def registerKernel(self, fallbacks: Dict[int, KernelFunction], k: Optional[DispatchKey], f: KernelFunction) -> None:
        # Add the kernel to the kernels list,
        # possibly creating the list if this is the first kernel.
        # Redirect catchAll registrations to CompositeImplicitAutograd.
        key = k if k is not None else DispatchKey.CompositeImplicitAutograd
        if key in self.kernels_ and self.kernels_[key].isValid():
            # This is a warning in the actual dispatcher, but I'm not testing it in python for now.
            raise AssertionError("Overriding a previously registered kernel for the same operator and dispatch key")
        self.kernels_[key] = f
        if k is not None:
            self.updateDispatchTable_(fallbacks, k)
        else:
            self.updateDispatchTableFull_(fallbacks)
