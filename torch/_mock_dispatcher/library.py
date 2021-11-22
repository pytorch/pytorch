from typing import Callable, Any, Optional
from torch._mock_dispatcher.dispatch_key import DispatchKey
from torch._mock_dispatcher.dispatch_key_set import getRuntimeDispatchKeySet
from torch._mock_dispatcher.kernel_function import KernelFunction
from torch._mock_dispatcher.dispatcher import Dispatcher

class CppFunction:
    func_: KernelFunction
    # Missing:  dispatch_key_, cpp_signature_, schema_, debug_

    def __init__(self, f: Callable[..., Any], *, boxed: bool) -> None:
        if boxed:
            self.func_ = KernelFunction.makeFromBoxed(f)
        else:
            self.func_ = KernelFunction.makeFromUnboxed(f)

class Library:
    ns_: str
    dispatch_key: Optional[DispatchKey]
    # Missing: kind_, file_, line_, registrars_

    def __init__(self, ns: str, k: Optional[DispatchKey] = None) -> None:
        self.ns_ = ns
        self.dispatch_key = k

    def impl(self, op: str, f: Callable[..., Any]) -> None:
        func: CppFunction = CppFunction(f, boxed=False)
        Dispatcher.singleton().registerImpl(op, self.dispatch_key, func.func_)

    def fallback(self, f: Callable[..., Any]) -> None:
        # Note if dispatch_key is DispatchKey::Undefined, it'll be ignored here since Undefined
        # isn't a runtime key, you shouldn't register anything to it at all.
        func: CppFunction = CppFunction(f, boxed=True)
        if self.dispatch_key is None:
            raise AssertionError()
        for k in getRuntimeDispatchKeySet(self.dispatch_key):
            Dispatcher.singleton().registerFallback(k, func.func_)
