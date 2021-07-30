import sys
import typing

if typing.TYPE_CHECKING or sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


class CompiledTemplate(Protocol):

    @staticmethod
    def call(n_iter: int) -> None: ...

    @staticmethod
    @typing.overload
    def measure_wall_time(
        n_iter: int,
        n_warmup_iter: int,
        cuda_sync: bool,
        timer: typing.Callable[[], float],
    ) -> float:
        # Python overload.
        ...

    @staticmethod
    @typing.overload
    def measure_wall_time(
        n_iter: int,
        n_warmup_iter: int,
        cuda_sync: bool,
    ) -> float:
        # C++ overload
        ...

    @staticmethod
    def collect_callgrind(n_iter: int, n_warmup_iter: int) -> None:
        ...
