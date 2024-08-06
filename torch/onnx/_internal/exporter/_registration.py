"""Module for handling ATen to ONNX functions registration.

https://github.com/pytorch/pytorch/blob/6aa5bb1a76dee8112f1a9e7c194c790b5cdc6462/torch/onnx/_internal/fx/registration.py
"""

# NOTE: Why do we need a different registry than the one in torchlib?
# The registry in torchlib is used to register functions that are already implemented in
# torchlib, and is designed to be a static singleton. It does not take into account custom ops or different
# opsets etc. The registry implemented for the exporter is designed to be modifiable at
# export time by users, and is designed with dispatching in mind.

# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import logging
import math
import operator
import types
import typing
from typing import Callable, Literal, Mapping, Union
from typing_extensions import TypeAlias

import torch
import torch._ops
from torch.onnx._internal.exporter import _schemas


if typing.TYPE_CHECKING:
    import onnxscript
    from onnxscript.function_libs.torch_lib import registration as torchlib_registration

_DEFAULT_OPSET_VERSION = 18


TorchOp: TypeAlias = Union[torch._ops.OpOverload, types.BuiltinFunctionType, Callable]

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class OnnxDecompMeta:
    """A wrapper of onnx-script function with additional metadata.

    onnx_function: The onnx-script function from torchlib.
    fx_target: The PyTorch node callable target.
    is_custom: Whether the function is a custom function.
    is_complex: Whether the function is a function that handles complex valued inputs.
    device: The device the function is registered to. If None, it is registered to all devices.
    """

    onnx_function: onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction
    fx_target: TorchOp
    is_custom: bool = False
    is_complex: bool = False
    device: Literal["cuda", "cpu"] | str | None = None  # noqa: PYI051


def _get_overload(qualified_name: str) -> torch._ops.OpOverload | None:
    """Obtain the torch op from <namespace>::<op_name>[.<overload>]"""
    # TODO(justinchuby): Handle arbitrary custom ops
    namespace, opname_overload = qualified_name.split("::")
    op_name, *overload = opname_overload.split(".", 1)
    if namespace == "_operator":
        # Builtin functions
        return getattr(operator, op_name)
    if namespace == "math":
        return getattr(math, op_name)
    if namespace == "torchvision":
        try:
            import torchvision.ops
        except ImportError:
            logger.warning("torchvision is not installed. Skipping %s", qualified_name)
            return None
        try:
            return getattr(torchvision.ops, op_name)
        except AttributeError:
            logger.warning("Failed to find torchvision op '%s'", qualified_name)
            return None
        except Exception:
            logger.exception("Failed to find torchvision op '%s'", qualified_name)
    try:
        op_packet = getattr(getattr(torch.ops, namespace), op_name)
        if overload:
            overload = overload[0]
        elif "default" in op_packet._overload_names or "" in op_packet._overload_names:
            # Has a default overload
            overload = "default"
        else:
            logger.warning(
                "'%s' does not have a 'default' overload. This could be an error in specifying the op name. Ignoring.",
                qualified_name,
                stacklevel=1,
            )
            return None

        return getattr(op_packet, overload)
    except AttributeError:
        if qualified_name.endswith("getitem"):
            # This is a special case where we registered the function incorrectly,
            # but for BC reasons (pt<=2.4) we need to keep it.
            return None
        logger.info("'%s' is not found in this version of PyTorch.", qualified_name)
        return None
    except Exception:
        logger.exception("Failed to find torch op '%s'", qualified_name)
        return None


class ONNXRegistry:
    """Registry for ONNX functions.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It supports registering custom onnx-script functions and for
    dispatcher to dispatch calls to the appropriate function.

    """

    def __init__(self) -> None:
        """Initializes the registry"""

        # TODO: Design multi-opset version support
        self._opset_version = _DEFAULT_OPSET_VERSION

        self.functions: dict[TorchOp | str, list[OnnxDecompMeta]] = {}

    @property
    def opset_version(self) -> int:
        """The ONNX opset version the exporter should target.

        Defaults to the latest supported ONNX opset version: 18.
        The default version will increment over time as ONNX continues to evolve.
        """

        return self._opset_version

    @classmethod
    def from_torchlib(
        cls,
        torchlib_registry: Mapping[str, torchlib_registration.OverloadedFunction]
        | None = None,
    ) -> ONNXRegistry:
        """Populates the registry with ATen functions from torchlib.

        Args:
            torchlib_registry: The torchlib registry to use for populating the registry.
        """
        registry = cls()
        if torchlib_registry is None:
            from onnxscript.function_libs.torch_lib import (
                registration as torchlib_registration,
            )

            torchlib_registry = torchlib_registration.default_registry
        for qualified_name, aten_overloads_func in torchlib_registry.items():
            try:
                # NOTE: This is heavily guarded with try-except because we don't want
                # to fail the entire registry population if one function fails.
                if qualified_name.startswith("internal::"):
                    # Skip the custom defined internal functions
                    continue
                target = _get_overload(qualified_name)
                if target is None:
                    continue
                for overload_func in aten_overloads_func.overloads:
                    overload_func.signature = _schemas.OpSignature.from_function(
                        overload_func,
                        overload_func.function_ir.domain,
                        overload_func.name,
                    )
                    onnx_decomposition = OnnxDecompMeta(
                        onnx_function=overload_func,
                        fx_target=target,
                        is_custom=False,
                        is_complex=False,
                    )
                    registry._register(target, onnx_decomposition)

                for complex_func in aten_overloads_func.complex:
                    overload_func.signature = _schemas.OpSignature.from_function(
                        overload_func,
                        overload_func.function_ir.domain,
                        overload_func.name,
                    )
                    onnx_decomposition = OnnxDecompMeta(
                        onnx_function=complex_func,
                        fx_target=target,
                        is_custom=False,
                        is_complex=True,
                    )
                    registry._register(target, onnx_decomposition)
            except Exception:
                logger.exception("Failed to register '%s'. Skipped", qualified_name)
                continue
        return registry

    def _register(
        self,
        target: TorchOp,
        onnx_decomposition: OnnxDecompMeta,
    ) -> None:
        """Registers a OnnxDecompMeta to an operator.

        Args:
            target: The PyTorch node callable target.
            onnx_decomposition: The OnnxDecompMeta to register.
        """
        if isinstance(target, torch._ops.OpOverload):
            # Get the qualified name of the aten op because torch._ops.OpOverload lookup in
            # a dictionary is unreliable for some reason.
            target_or_name: str | TorchOp = target.name()
        else:
            target_or_name: str | TorchOp = target
        if onnx_decomposition.is_custom:
            self.functions.setdefault(target_or_name, []).insert(0, onnx_decomposition)
        else:
            self.functions.setdefault(target_or_name, []).append(onnx_decomposition)

    def register_op(
        self,
        target: TorchOp,
        function: onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction,
        is_complex: bool = False,
    ) -> None:
        """Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            target: The PyTorch node callable target.
            function: The onnx-script function to register.
            is_complex: Whether the function is a function that handles complex valued inputs.
        """
        onnx_decomposition = OnnxDecompMeta(
            onnx_function=function,
            fx_target=target,
            is_custom=True,
            is_complex=is_complex,
        )
        self._register(target, onnx_decomposition)

    def get_decomps(self, target: TorchOp) -> list[OnnxDecompMeta]:
        """Returns a list of OnnxDecompMeta for the given op: torch.ops.<namespace>.<op_name>.<overload>.

        The list is ordered by the time of registration. The custom operators should come
        first in the list.

        Args:
            target: The PyTorch node callable target.
        Returns:
            A list of OnnxDecompMeta corresponding to the given name, or None if
            the name is not in the registry.
        """
        if isinstance(target, torch._ops.OpOverload):
            # Get the qualified name of the aten op because torch._ops.OpOverload lookup in
            # a dictionary is unreliable for some reason.
            target_or_name: str | TorchOp = target.name()
        else:
            target_or_name: str | TorchOp = target
        decomps = self.functions.get(target_or_name, [])
        return sorted(decomps, key=lambda x: x.is_custom, reverse=True)

    def is_registered(self, target: TorchOp) -> bool:
        """Returns whether the given op is registered: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            target: The PyTorch node callable target.

        Returns:
            True if the given op is registered, otherwise False.
        """
        return bool(self.get_decomps(target))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(functions={self.functions})"
