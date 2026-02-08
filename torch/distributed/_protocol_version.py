# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed Protocol Version Guardrail (M11)

This module defines the distributed protocol version used during
init_process_group to detect cross-version incompatibilities early.

The protocol version is checked via the rendezvous Store before backend
initialization proceeds. If ranks have mismatched protocol versions,
initialization fails immediately with a clear error message.

This is an internal module; the protocol version and override mechanism
are not part of PyTorch's public API.
"""

import os
from typing import Final

__all__ = ["PROTOCOL_VERSION", "get_protocol_version"]

# The distributed protocol version.
# Increment this when making breaking changes to the distributed wire protocol.
# M11 establishes version 1 as the baseline.
PROTOCOL_VERSION: Final[int] = 1

# Environment variable for test-only override of protocol version.
# This allows mismatch simulation in tests without requiring multiple PyTorch versions.
_PROTOCOL_VERSION_OVERRIDE_ENV: Final[str] = "TORCH_DISTRIBUTED_PROTOCOL_VERSION_OVERRIDE"


def get_protocol_version() -> int:
    """
    Return the current distributed protocol version.

    If the environment variable TORCH_DISTRIBUTED_PROTOCOL_VERSION_OVERRIDE is set,
    its value is used instead of PROTOCOL_VERSION. This is intended for testing only.

    Returns:
        The protocol version as an integer.

    Raises:
        RuntimeError: If the override environment variable is set but cannot be
            parsed as an integer.
    """
    override = os.environ.get(_PROTOCOL_VERSION_OVERRIDE_ENV)
    if override is not None:
        try:
            return int(override)
        except ValueError as e:
            raise RuntimeError(
                f"Invalid value for {_PROTOCOL_VERSION_OVERRIDE_ENV}: '{override}'. "
                f"Expected an integer."
            ) from e
    return PROTOCOL_VERSION

