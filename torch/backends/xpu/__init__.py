# mypy: allow-untyped-defs
import contextlib
from typing import Union
from typing_extensions import deprecated

import torch


__all__ = [
    "matmul",
]


class XPUModule:
    def __getattr__(self, name):
        if name == "allow_tf32":
            return torch.get_float32_matmul_precision() == "high"
        raise AttributeError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            assert isinstance(
                value, bool
            ), f"allow_tf32 should be a boolean, but got {type(value)}"
            if value:
                torch.set_float32_matmul_precision("high")
            else:
                torch.set_float32_matmul_precision("highest")
            return
        raise AttributeError("Unknown attribute " + name)


matmul = XPUModule()
