"""Registry for aten functions."""

from __future__ import annotations


__all__ = ["onnx_impl", "get_torchlib_ops"]

import logging
from collections.abc import Callable, Sequence
from typing import TypeVar
from typing_extensions import ParamSpec

import torch
from torch.onnx._internal.exporter import _custom_translation, _registration


# Use ParamSpec for better type preservation instead of bound Callable TypeVar
_P = ParamSpec("_P")
_R = TypeVar("_R")

logger = logging.getLogger("__name__")


_registry: list[_registration.OnnxDecompMeta] = []


def onnx_impl(
    target: _registration.TorchOp | str | tuple[_registration.TorchOp | str, ...],
    *,
    complex: bool = False,
    opset_introduced: int = 18,
    private: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Register an ONNX implementation of a torch op."""

    if isinstance(target, torch._ops.OpOverloadPacket):
        raise TypeError(
            f"Target '{target}' should be provided as an OpOverload instead of an "
            "OpOverloadPacket. You can get the default overload with "
            "<op>.default"
        )

    def wrapper(
        func: Callable[_P, _R],
    ) -> Callable[_P, _R]:
        if not private:
            # TODO(justinchuby): Simplify the logic and remove the private attribute
            # Skip registration if private
            if isinstance(target, str) or not isinstance(target, Sequence):
                targets = (target,)
            else:
                targets = target  # type: ignore[assignment]

            for t in targets:
                _registry.append(
                    _registration.OnnxDecompMeta(
                        onnx_function=_custom_translation.NativeTranslation(
                            t, func, complex, opset_introduced, None
                        ),
                        fx_target=t,
                        signature=None,
                        is_custom=False,
                        is_complex=complex,
                        opset_introduced=opset_introduced,
                        skip_signature_inference=True,
                    )
                )
        return func

    return wrapper


def get_torchlib_ops() -> tuple[_registration.OnnxDecompMeta, ...]:
    # Trigger op registration
    from torch.onnx._internal.exporter._torchlib import ops

    del ops
    if len(_registry) == 0:
        raise AssertionError("_registry must not be empty")
    return tuple(_registry)
