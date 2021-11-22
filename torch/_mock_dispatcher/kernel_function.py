from typing import Callable, Any, Optional

class KernelFunction:
    _boxed_kernel_func_: Optional[Callable[..., Any]]
    _unboxed_kernel_func_: Optional[Callable[..., Any]]
    # missing: functor_

    def __init__(self, f: Optional[Callable[..., Any]] = None) -> None:
        self._boxed_kernel_func_ = f
        self._unboxed_kernel_func_ = f

    def isValid(self) -> bool:
        return self._boxed_kernel_func_ is not None

    def call(self) -> None:
        assert self._unboxed_kernel_func_ is not None
        self._unboxed_kernel_func_()

    def callBoxed(self) -> None:
        assert self._boxed_kernel_func_ is not None
        self._boxed_kernel_func_()

    def isFallthrough(self) -> bool:
        return self == fallthrough_kernel

    def name(self) -> str:
        if self._unboxed_kernel_func_ is not None:
            return self._unboxed_kernel_func_.__name__
        return "[None]"

    @staticmethod
    def makeFromUnboxed(f: Callable[..., Any]) -> 'KernelFunction':
        return KernelFunction(f)

    @staticmethod
    def makeFromBoxed(f: Callable[..., Any]) -> 'KernelFunction':
        return KernelFunction(f)

def _fallthrough_fn() -> None:
    raise AssertionError("fallthrough")
fallthrough_kernel = KernelFunction(_fallthrough_fn)
