# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
import sys
from datetime import timedelta
from importlib.metadata import entry_points

import torch
from torch.comms.functional import (
    is_torch_compile_supported as _is_torch_compile_supported,
    is_torch_compile_supported_and_enabled as _is_torch_compile_supported_and_enabled,
)


if _is_torch_compile_supported():
    from torch._opaque_base import OpaqueBaseMeta

    # make the metaclass available to the pybind module
    sys.modules["torch.comms._opaque_meta"] = type(
        "module", (), {"OpaqueBaseMeta": OpaqueBaseMeta}
    )()

    # to support opaque registration for time delta.
    class Timeout(timedelta, metaclass=OpaqueBaseMeta):
        pass

else:
    # When compile support is disabled, define Timeout without the metaclass
    class Timeout(timedelta):
        pass


# The comms C++ bindings are compiled into libtorch_python (torch._C). Calling
# _comms_init() creates the torch._C._comms submodule and registers the in-tree
# gloo/nccl backends with the factory (mirrors torch.distributed's _c10d_init).
# It must run after torch.comms._opaque_meta is installed above, since the
# bindings look it up while registering classes.
if not hasattr(torch._C, "_comms_init"):
    raise ImportError(
        "torch.comms requires PyTorch to be built with USE_TORCH_COMMS=ON"
    )
torch._C._comms_init()

import torch.comms.hooks as hooks
import torch.comms.objcol as objcol
from torch._C._comms import *  # noqa: F403


if _is_torch_compile_supported_and_enabled():
    # Import collectives first to ensure all operations are registered
    # This must happen before patch_torchcomm() so that window operations
    # and other collectives are registered and can be patched
    from torch.comms.functional import collectives


# The documentation uses __all__ to determine what is documented and in what
# order.
__all__ = [
    "new_comm",
    "TorchComm",
    "ReduceOp",
    "TorchWork",
    "Timeout",
    "BatchP2POptions",
    "BatchSendRecv",
    "P2POp",
    "CommOptions",
    "TorchCommWindow",
    "register_backend",
    "TorchCommBackend",
    "is_backend_built",
    "built_backends",
]


def _load_backend(backend: str) -> None:
    """Used to load backends lazily from C++

    C++ calls this only when the backend is not already registered via
    register_backend. The in-tree gloo/nccl backends are registered eagerly in
    _comms_init() above, so this is only exercised for out-of-tree backends
    registered through the ``torch.comms.backends`` entry point group.
    """
    found = entry_points(group="torch.comms.backends", name=backend)
    if not found:
        raise ModuleNotFoundError(
            f"failed to find backend {backend}, is it registered via entry_points?"
        )
    wheel = next(iter(found))
    wheel.load()


def is_backend_built(backend: str) -> bool:
    """True if torch was built with this backend's extension."""
    return bool(entry_points(group="torch.comms.backends", name=backend))


def built_backends() -> list[str]:
    """Names of all backends torch was built with."""
    return [ep.name for ep in entry_points(group="torch.comms.backends")]


# Re-point __module__ of the re-exported pybind classes/functions at torch.comms
# so reprs, docs and test_public_bindings treat torch.comms as the canonical
# public location (the functions defined here are already torch.comms).
for _name in __all__:
    globals()[_name].__module__ = "torch.comms"
del _name
