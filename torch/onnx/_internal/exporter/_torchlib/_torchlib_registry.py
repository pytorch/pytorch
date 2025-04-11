# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Registry for aten functions."""

from __future__ import annotations


__all__ = ["onnx_impl", "get_torchlib_ops"]

import logging
from collections.abc import Sequence
from typing import Any, Callable, TypeVar

import onnxscript

import torch
from torch.onnx._internal.exporter import _constants, _registration


_T = TypeVar("_T", bound=Callable)

logger = logging.getLogger("__name__")


_registry: list[_registration.OnnxDecompMeta] = []


def onnx_impl(
    target: _registration.TorchOp | tuple[_registration.TorchOp, ...],
    *,
    trace_only: bool = False,
    complex: bool = False,
    opset_introduced: int = 18,
    no_compile: bool = False,
    private: bool = False,
) -> Callable[[_T], _T]:
    """Register an ONNX implementation of a torch op."""

    if isinstance(target, torch._ops.OpOverloadPacket):
        raise TypeError(
            f"Target '{target}' should be provided as an OpOverload instead of an "
            "OpOverloadPacket. You can get the default overload with "
            "<op>.default"
        )

    def wrapper(
        func: _T,
    ) -> _T:
        processed_func: Any
        if no_compile:
            processed_func = func
        else:
            torchlib_opset = onnxscript.values.Opset(
                domain=_constants.TORCHLIB_DOMAIN, version=1
            )

            if not trace_only:
                # Compile the function
                processed_func = onnxscript.script(opset=torchlib_opset)(func)
            else:
                processed_func = onnxscript.TracedOnnxFunction(torchlib_opset, func)

        if not private:
            # TODO(justinchuby): Simplify the logic and remove the private attribute
            # Skip registration if private
            if not isinstance(target, Sequence):
                targets = (target,)
            else:
                targets = target  # type: ignore[assignment]

            for t in targets:
                _registry.append(
                    _registration.OnnxDecompMeta(
                        onnx_function=processed_func,
                        fx_target=t,
                        signature=None,
                        is_complex=complex,
                        opset_introduced=opset_introduced,
                        skip_signature_inference=no_compile,
                    )
                )
        return processed_func  # type: ignore[return-value]

    return wrapper


def get_torchlib_ops() -> tuple[_registration.OnnxDecompMeta, ...]:
    # Trigger op registration
    from torch.onnx._internal.exporter._torchlib import ops

    del ops
    assert len(_registry) != 0
    return tuple(_registry)
