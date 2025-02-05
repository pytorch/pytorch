# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Registry for aten functions."""

from __future__ import annotations


__all__ = ["onnx_impl", "get_torchlib_ops"]

import logging
from typing import Any, Callable, Sequence, TypeVar

import onnxscript

import torch
from torch.onnx._internal.exporter import _constants, _registration, _schemas


_T = TypeVar("_T", bound=Callable)

logger = logging.getLogger("__name__")


_registry: list[_registration.OnnxDecompMeta] = []


def onnx_impl(
    target: _registration.TorchOp | tuple[_registration.TorchOp, ...],
    *,
    trace_only: bool = False,
    complex: bool = False,
    no_compile: bool = False,
    private: bool = False,
) -> Callable[[_T], _T]:
    """Register an ONNX implementation of a torch op."""

    if isinstance(target, torch._ops.OpOverloadPacket):
        raise TypeError(
            "Please provide an overload instead of an OpOverloadPacket. "
            "You can get the default overload with torch.ops.aten.<op>.default."
        )

    def wrapper(
        func: _T,
    ) -> _T:
        signature: _schemas.OpSignature | None = None
        processed_func: Any
        if no_compile:
            processed_func = func
        else:
            custom_opset = onnxscript.values.Opset(
                domain=_constants.TORCHLIB_DOMAIN, version=1
            )

            if not trace_only:
                # Compile the function
                processed_func = onnxscript.script(opset=custom_opset)(func)
            else:
                processed_func = func

            # TODO(justinchuby): Simplify the logic and remove the private attribute
            if isinstance(processed_func, onnxscript.OnnxFunction):
                opset_version = processed_func.opset.version
            else:
                opset_version = 1

            signature = _schemas.OpSignature.from_function(
                processed_func,
                _constants.TORCHLIB_DOMAIN,
                getattr(processed_func, "name", func.__name__),
                opset_version=opset_version,
            )
            processed_func._pt_onnx_signature = signature  # type: ignore[assignment]

        if not private:
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
                        signature=signature,
                        is_complex=complex,
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
