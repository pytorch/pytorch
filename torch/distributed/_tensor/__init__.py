"""
NOTICE: DTensor has moved to torch.distributed.tensor

This file is a shim to redirect to the new location, and
we keep the old import path starts with `_tensor` for
backward compatibility.
"""
import importlib
import sys

import torch.distributed.tensor


def _populate():  # type: ignore[no-untyped-def]
    for name in (
        # TODO: _utils here mainly for checkpoint imports BC, remove it
        "_utils",
        "api",
        "debug",
        "device_mesh",
        "experimental",
        "placement_types",
        "random",
    ):
        try:
            globals()[name] = sys.modules[
                f"torch.distributed._tensor.{name}"
            ] = importlib.import_module(f"torch.distributed.tensor.{name}")
        except ImportError as e:
            import traceback

            traceback.print_exc()
            raise ImportError(
                f"Failed to import torch.distributed.tensor.{name} due to {e}"
            ) from e

    for name, val in torch.distributed.tensor.__dict__.items():
        # Skip private names and tensor parallel package
        if not name.startswith("_") and name != "parallel":
            globals()[name] = val


_populate()
