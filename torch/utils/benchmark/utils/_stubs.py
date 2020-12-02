import sys
from typing import TYPE_CHECKING


if TYPE_CHECKING or sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


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
