# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
# patternlint-disable fbcode-nonempty-init-py

"""
Flight Recorder module for TorchComm hooks.

This module provides the FlightRecorderHook class for tracking all collective
operations in flight for TorchComm communicators. The output format matches
the OSS FlightRecorder format from PyTorch's distributed module, so traces
can be analyzed using the same fr_trace analysis tools.

Example:
    >>> # xdoctest: +SKIP("requires a CUDA device and a configured communicator")
    >>> from torch.comms.hooks import fr
    >>> import torch.comms
    >>> comm = torch.comms.new_comm("nccl", device, "world")
    >>> recorder = fr.FlightRecorderHook(max_entries=1024)
    >>> recorder.register_with_comm(comm)
    >>> # ... run some collectives ...
    >>> json_trace = recorder.dump_json()
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    # pyrefly: ignore[missing-import]
    from torch.comms.hooks.fr._fr import FlightRecorderHook
else:
    import torch._C._comms as _comms_mod

    FlightRecorderHook = _comms_mod.hooks.fr.FlightRecorderHook
    # Point __module__ at this module so it is recognized as the public,
    # canonical location by test_public_bindings (the underlying pybind class
    # otherwise reports its C-extension module).
    FlightRecorderHook.__module__ = "torch.comms.hooks.fr"

__all__ = [
    "FlightRecorderHook",
]
