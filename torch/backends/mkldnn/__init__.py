# mypy: allow-untyped-defs
import sys
from contextlib import contextmanager

from typing import TYPE_CHECKING

import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


def is_available():
    r"""Return whether PyTorch is built with MKL-DNN support."""
    return torch._C._has_mkldnn


VERBOSE_OFF = 0
VERBOSE_ON = 1
VERBOSE_ON_CREATION = 2


class verbose:
    """
    On-demand oneDNN (former MKL-DNN) verbosing functionality.

    To make it easier to debug performance issues, oneDNN can dump verbose
    messages containing information like kernel size, input data size and
    execution duration while executing the kernel. The verbosing functionality
    can be invoked via an environment variable named `DNNL_VERBOSE`. However,
    this methodology dumps messages in all steps. Those are a large amount of
    verbose messages. Moreover, for investigating the performance issues,
    generally taking verbose messages for one single iteration is enough.
    This on-demand verbosing functionality makes it possible to control scope
    for verbose message dumping. In the following example, verbose messages
    will be dumped out for the second inference only.

    .. highlight:: python
    .. code-block:: python

        import torch
        model(data)
        with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
            model(data)

    Args:
        level: Verbose level
            - ``VERBOSE_OFF``: Disable verbosing
            - ``VERBOSE_ON``:  Enable verbosing
            - ``VERBOSE_ON_CREATION``: Enable verbosing, including oneDNN kernel creation
    """

    def __init__(self, level):
        self.level = level

    def __enter__(self):
        if self.level == VERBOSE_OFF:
            return
        st = torch._C._verbose.mkldnn_set_verbose(self.level)
        assert (
            st
        ), "Failed to set MKLDNN into verbose mode. Please consider to disable this verbose scope."
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch._C._verbose.mkldnn_set_verbose(VERBOSE_OFF)
        return False


def set_flags(_enabled, _deterministic=None):
    orig_flags = (torch._C._get_mkldnn_enabled(), torch._C._get_mkldnn_deterministic())
    torch._C._set_mkldnn_enabled(_enabled)
    if _deterministic is not None:
        torch._C._set_mkldnn_deterministic(_deterministic)
    return orig_flags


@contextmanager
def flags(enabled=False, deterministic=False):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, deterministic)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


class MkldnnModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)

    enabled = ContextProp(torch._C._get_mkldnn_enabled, torch._C._set_mkldnn_enabled)
    deterministic = ContextProp(
        torch._C._get_mkldnn_deterministic, torch._C._set_mkldnn_deterministic
    )


if TYPE_CHECKING:
    enabled: ContextProp
    deterministic: ContextProp

sys.modules[__name__] = MkldnnModule(sys.modules[__name__], __name__)
