# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
# patternlint-disable fbcode-nonempty-init-py

"""
NanCheck module for TorchComm hooks.

This module provides the NanCheckHook class for detecting NaN values
in tensors before collective operations.

Example:
    >>> # xdoctest: +SKIP("requires a CUDA device and a configured communicator")
    >>> from torch.comms.hooks import NanCheckHook
    >>> import torch.comms
    >>> comm = torch.comms.new_comm("nccl", device, "world")
    >>> nan_check = NanCheckHook()
    >>> nan_check.register_with_comm(comm)
    >>> # NaN in tensors will now raise RuntimeError before collective runs
"""

from torch.comms.hooks.nan_check.nan_check import NanCheckHook


__all__ = [
    "NanCheckHook",
]
