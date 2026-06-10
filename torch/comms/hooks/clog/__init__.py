# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
# patternlint-disable fbcode-nonempty-init-py

"""
clog module for TorchComm hooks.

Logs collective operation signatures and lifecycle events to a
pipe-delimited log file.

Example:
    >>> # xdoctest: +SKIP("requires a configured communicator")
    >>> from torch.comms.hooks import clog
    >>> logger = clog(output="/tmp/clog.log", events=["ALL"])
    >>> logger.register_with_comm(comm)
    >>> # ... run collectives ...
    >>> comm.finalize()
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch.comms.hooks.clog._clog import clog  # pyrefly: ignore[missing-import]
else:
    import torch._C._comms as _comms_mod

    clog = _comms_mod.hooks.clog.clog
    # Point __module__ at this module so it is recognized as the public,
    # canonical location by test_public_bindings (the underlying pybind class
    # otherwise reports its C-extension module).
    clog.__module__ = "torch.comms.hooks.clog"

__all__ = [
    "clog",
]
