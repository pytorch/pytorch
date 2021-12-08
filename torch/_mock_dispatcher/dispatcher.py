from typing import Dict, List, Optional
from torch._mock_dispatcher.operator_entry import OperatorEntry
from torch._mock_dispatcher.dispatch_key import DispatchKey
from torch._mock_dispatcher.dispatch_key_set import DispatchKeySet, num_entries, getDispatchTableIndexForDispatchKey
from torch._mock_dispatcher.kernel_function import KernelFunction

class Dispatcher:
    # Skipping OperatorHandle, OperatorDef
    operatorLookupTable_: Dict[str, OperatorEntry]
    # skipping "AnnotatedKernel"
    backendFallbackKernels_: Dict[int, KernelFunction]
    # missing: operators_, libraries_, listeners_, mutex_

    def __init__(self) -> None:
        self.operatorLookupTable_ = {}
        self.backendFallbackKernels_ = {i: KernelFunction() for i in range(num_entries())}

    # Using this instead of RAII for deregistering
    def reset(self) -> None:
        self.operatorLookupTable_ = {}
        self.backendFallbackKernels_ = {i: KernelFunction() for i in range(num_entries())}

    def dumpRuntimeState(self, op: str, k: Optional[DispatchKey] = None) -> str:
        assert op in self.operatorLookupTable_
        return self.operatorLookupTable_[op].dumpRuntimeState(k)

    def dumpRegistration(self, op: str, k: Optional[DispatchKey] = None) -> str:
        assert op in self.operatorLookupTable_
        return self.operatorLookupTable_[op].dumpRegistrationState(k)

    def callBoxed(self, op: str, args: List[DispatchKeySet]) -> None:
        assert op in self.operatorLookupTable_, f"nothing registered for {op}"
        op_entry = self.operatorLookupTable_[op]
        dispatchKeySet = op_entry.dispatchKeyExtractor_.getDispatchKeySetUnboxed(args)
        kernel = op_entry.lookup(dispatchKeySet)
        kernel.callBoxed()

    def call(self, op: str, args: List[DispatchKeySet]) -> None:
        assert op in self.operatorLookupTable_, f"nothing registered for {op}"
        op_entry = self.operatorLookupTable_[op]
        dispatchKeySet = op_entry.dispatchKeyExtractor_.getDispatchKeySetUnboxed(args)
        kernel = op_entry.lookup(dispatchKeySet)
        kernel.call()

    def registerImpl(self, op: str, k: Optional[DispatchKey], f: KernelFunction) -> None:
        entry = self.findOrRegisterName_(op)
        # registerKernel() in C++ takes in the entire dispatcher
        # but in python that creates circular imports.
        # Instead, I'm just passing in what's needed (The fallback kernels)
        entry.registerKernel(self.backendFallbackKernels_, k, f)

    def registerFallback(self, k: DispatchKey, f: KernelFunction) -> None:
        idx = getDispatchTableIndexForDispatchKey(k)
        assert not self.backendFallbackKernels_[idx].isValid()
        self.backendFallbackKernels_[idx] = f
        for op in self.operatorLookupTable_:
            # updateFallback() in C++ takes in the entire dispatcher
            # but in python that creates circular imports.
            # Instead, I'm just passing in what's needed (The fallback kernels)
            self.operatorLookupTable_[op].updateFallback(self.backendFallbackKernels_, k)

    def findOrRegisterName_(self, op: str) -> OperatorEntry:
        if op not in self.operatorLookupTable_:
            self.operatorLookupTable_[op] = OperatorEntry(op, self.backendFallbackKernels_)
        return self.operatorLookupTable_[op]

    @staticmethod
    def singleton() -> 'Dispatcher':
        return _dispatcher

# singleton dispatcher instance
_dispatcher = Dispatcher()
