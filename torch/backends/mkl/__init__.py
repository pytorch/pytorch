from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import torch


__all__ = ["is_available", "VERBOSE_OFF", "VERBOSE_ON", "verbose"]


if TYPE_CHECKING:
    from types import TracebackType


def is_available() -> bool:
    r"""Return whether PyTorch is built with MKL support."""
    return torch._C.has_mkl


VERBOSE_OFF = 0
VERBOSE_ON = 1


class verbose:
    """
    On-demand oneMKL verbosing functionality.

    To make it easier to debug performance issues, oneMKL can dump verbose
    messages containing execution information like duration while executing
    the kernel. The verbosing functionality can be invoked via an environment
    variable named `MKL_VERBOSE`. However, this methodology dumps messages in
    all steps. Those are a large amount of verbose messages. Moreover, for
    investigating the performance issues, generally taking verbose messages
    for one single iteration is enough. This on-demand verbosing functionality
    makes it possible to control scope for verbose message dumping. In the
    following example, verbose messages will be dumped out for the second
    inference only.

    .. highlight:: python
    .. code-block:: python

        import torch
        model(data)
        with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):
            model(data)

    Args:
        level: Verbose level
            - ``VERBOSE_OFF``: Disable verbosing
            - ``VERBOSE_ON``:  Enable verbosing
    """

    def __init__(self, enable: int) -> None:
        self.enable = enable

    def __enter__(self) -> verbose | None:
        if self.enable == VERBOSE_OFF:
            return None
        st = torch._C._verbose.mkl_set_verbose(self.enable)
        assert (
            st
        ), "Failed to set MKL into verbose mode. Please consider to disable this verbose scope."
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        torch._C._verbose.mkl_set_verbose(VERBOSE_OFF)
        return False
