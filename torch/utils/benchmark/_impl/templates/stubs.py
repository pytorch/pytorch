import sys
import typing

if typing.TYPE_CHECKING or sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class CompiledTemplate(Protocol):

    @staticmethod
    def call(n_iter: int) -> None:
        ...

    # There are two overloads for `measure_wall_time`: One takes a `timer`
    # arg (Python), while the other doesn't (C++). Unfortunately
    # `@typing.overload` doesn't work with `@staticmethod`
    # (https://github.com/python/mypy/issues/7781), so we define a single
    # method in the stub. (`@runtime_checkable` does not check the signature.)

    @staticmethod
    def measure_wall_time(
        n_iter: int,
        n_warmup_iter: int,
        cuda_sync: bool,
        timer: typing.Callable[[], float],
    ) -> float:
        ...

    @staticmethod
    def collect_callgrind(n_iter: int, n_warmup_iter: int) -> None:
        ...
