# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Entry-point shim for the gloo backend.

The gloo bindings are compiled into libtorch_python and exposed at
``torch._C._comms._comms_gloo``; the backend itself is registered with the
factory in ``torch.comms`` initialization. This module exists so the
``torch.comms.backends`` entry point has an importable target; importing
``torch.comms`` (its parent) is what performs registration.
"""

from torch._C._comms._comms_gloo import *  # noqa: F403  # pyrefly: ignore[missing-import]
