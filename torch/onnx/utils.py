"""Backward compatibility module for torch.onnx.utils."""

from __future__ import annotations


__all__: list[str] = []


from torch.onnx._internal.torchscript_exporter.utils import *  # noqa: F401,F403
