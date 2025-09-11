from typing import Any, Callable
from typing_extensions import Protocol, runtime_checkable


class TimerClass(Protocol):
    """This is the portion of the `timeit.Timer` API used by benchmark utils."""
    def __init__(
        self,
        stmt: str,
        setup: str,
        timer: Callable[[], float],
        globals: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        ...

    def timeit(self, number: int) -> float:
        ...


@runtime_checkable
class TimeitModuleType(Protocol):
    """Modules generated from `timeit_template.cpp`."""
    def timeit(self, number: int) -> float:
        ...


class CallgrindModuleType(Protocol):
    """Replicates the valgrind endpoints in `torch._C`.

    These bindings are used to collect Callgrind profiles on earlier versions
    of PyTorch and will eventually be removed.
    """
    __file__: str
    __name__: str

    def _valgrind_supported_platform(self) -> bool:
        ...

    def _valgrind_toggle(self) -> None:
        ...
