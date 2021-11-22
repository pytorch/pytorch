from typing import Callable, Any, Optional
from torch._mock_dispatcher.dispatch_key import DispatchKey
from torch._mock_dispatcher.dispatch_key_set import getRuntimeDispatchKeySet
from torch._mock_dispatcher.kernel_function import KernelFunction, fallthrough_kernel
from torch._mock_dispatcher.dispatcher import Dispatcher

class CppFunction:
    func_: KernelFunction
    # Missing:  dispatch_key_, cpp_signature_, schema_, debug_

    def __init__(self, f: Optional[Callable[..., Any]] = None, *, boxed: bool, fallthrough: bool = False) -> None:
        if fallthrough:
            assert f is None
            self.func_ = fallthrough_kernel
            return

        assert f is not None
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

    def impl_fallthrough(self, op: str) -> None:
        func: CppFunction = CppFunction(boxed=False, fallthrough=True)
        Dispatcher.singleton().registerImpl(op, self.dispatch_key, func.func_)

    def fallback(self, f: Callable[..., Any]) -> None:
        # Note if dispatch_key is DispatchKey::Undefined, it'll be ignored here since Undefined
        # isn't a runtime key, you shouldn't register anything to it at all.
        func: CppFunction = CppFunction(f, boxed=True)
        if self.dispatch_key is None:
            raise AssertionError()
        for k in getRuntimeDispatchKeySet(self.dispatch_key):
            Dispatcher.singleton().registerFallback(k, func.func_)
