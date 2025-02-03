# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Registry for aten functions."""

from __future__ import annotations


__all__ = ["onnx_impl", "get_torchlib_ops"]

import logging
from typing import Callable, Sequence, TypeVar

import onnxscript

import torch
from torch.onnx._internal.exporter import _registration, _schemas
from torch.onnx._internal.exporter._torchlib import _constants


_T = TypeVar("_T", bound=Callable)

logger = logging.getLogger("__name__")


_registry: list[_registration.OnnxDecompMeta] = []


def onnx_impl(
    target: _registration.TorchOp
    | torch._ops.OpOverloadPacket
    | tuple[_registration.TorchOp | torch._ops.OpOverloadPacket, ...],
    *,
    trace_only: bool = False,
    complex: bool = False,
    no_compile: bool = False,  # TODO: Complete this
    private: bool = False,
) -> Callable[[_T], _T]:
    """Register an ONNX implementation of a torch op."""

    if isinstance(target, torch._ops.OpOverloadPacket):
        target = target.default

    def wrapper(
        func: _T,
    ) -> _T:
        try:
            # NOTE: This is heavily guarded with try-except because we don't want
            # to fail the entire registry population if one function fails.
            custom_opset = onnxscript.values.Opset(domain=_constants.DOMAIN, version=1)

            processed_func: (
                onnxscript.OnnxFunction | onnxscript.values.TracedOnnxFunction
            )
            if trace_only:
                # TODO(justinchuby): Simplify this implementation
                processed_func = onnxscript.values.TracedOnnxFunction(
                    custom_opset, func
                )
            else:
                # Compile the function
                processed_func = onnxscript.script(opset=custom_opset)(func)

            if not private:
                # Skip registration if private
                # TODO(justinchuby): Simplify the logic and remove the private attribute
                if isinstance(processed_func, onnxscript.OnnxFunction):
                    opset_version = processed_func.opset.version
                else:
                    opset_version = 1

                signature = _schemas.OpSignature.from_function(
                    processed_func,
                    _constants.DOMAIN,
                    processed_func.name,
                    opset_version=opset_version,
                )

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
        except Exception:
            logger.exception(
                "Failed to register function '%s' to target '%s'. Skipped", func, target
            )
            return func

    return wrapper


def get_torchlib_ops() -> tuple[_registration.OnnxDecompMeta, ...]:
    # Trigger op registration
    from torch.onnx._internal.exporter._torchlib import ops

    del ops
    assert len(_registry) != 0
    return tuple(_registry)
