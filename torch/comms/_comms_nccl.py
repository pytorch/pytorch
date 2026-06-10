# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Entry-point shim for the nccl backend.

The nccl bindings are compiled into libtorch_python (CUDA builds) and exposed at
``torch._C._comms._comms_nccl``; the backend itself is registered with the
factory in ``torch.comms`` initialization. This module exists so the
``torch.comms.backends`` entry point has an importable target; importing
``torch.comms`` (its parent) is what performs registration.
"""

try:
    from torch._C._comms._comms_nccl import *  # noqa: F403  # pyrefly: ignore[missing-import]
except ImportError:
    # The nccl backend is compiled only into CUDA builds, so this submodule is
    # absent on CPU/ROCm builds. The entry point still resolves (so backend
    # discovery via built_backends() works) and importing this shim never fails.
    pass
