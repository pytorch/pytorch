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
import importlib.util
import logging
import math
import operator
import types
from typing import Callable, Literal, Union
from typing_extensions import TypeAlias

import torch
import torch._ops
from torch.onnx._internal._lazy_import import onnxscript, onnxscript_apis
from torch.onnx._internal.exporter import _constants, _schemas
from torch.onnx._internal.exporter._torchlib import _torchlib_registry


TorchOp: TypeAlias = Union[torch._ops.OpOverload, types.BuiltinFunctionType, Callable]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OnnxDecompMeta:
    """A wrapper of onnx-script function with additional metadata.

    onnx_function: The onnx-script function from torchlib.
    fx_target: The PyTorch node callable target.
    signature: The ONNX signature of the function. When None, the signature is inferred.
    is_custom: Whether the function is a custom function.
    is_complex: Whether the function is a function that handles complex valued inputs.
    opset_introduced:
        The ONNX opset version in which the function was introduced.
        Its specifies the minimum ONNX opset version required to use the function.
    device: The device the function is registered to. If None, it is registered to all devices.
    skip_signature_inference: Whether to skip signature inference for the function.
    """

    onnx_function: Callable
    fx_target: TorchOp
    signature: _schemas.OpSignature | None
    is_custom: bool = False
    is_complex: bool = False
    opset_introduced: int = 18
    device: Literal["cuda", "cpu"] | str | None = None  # noqa: PYI051
    skip_signature_inference: bool = False

    def __post_init__(self) -> None:
        if self.signature is None and not self.skip_signature_inference:
            try:
                if isinstance(self.onnx_function, onnxscript.OnnxFunction):
                    signature = _schemas.OpSignature.from_function(  # type: ignore[attr-defined]
                        self.onnx_function,
                        self.onnx_function.function_ir.domain,
                        self.onnx_function.name,
                        opset_version=self.onnx_function.opset.version,
                    )
                else:
                    signature = _schemas.OpSignature.from_function(
                        self.onnx_function, "__traced", self.onnx_function.__name__
                    )
            except Exception as e:
                # Log an warning if the op is custom. Raise exception for builtin ops.
                if not self.is_custom:
                    raise
                else:
                    # When the function is targeting an HOP, for example, it will accept
                    # functions as arguments and fail to generate an ONNX signature.
                    # In this case we set signature to None and dispatch to this function always.
                    logger.warning(
                        "Failed to infer the signature for function '%s' because '%s'"
                        "All nodes targeting `%s` will be dispatched to this function",
                        self.onnx_function,
                        e,
                        self.fx_target,
                    )
            else:
                self.signature = signature
                self.onnx_function._pt_onnx_signature = signature  # type: ignore[attr-defined]


def _get_overload(qualified_name: str) -> torch._ops.OpOverload | None:
    """Obtain the torch op from <namespace>::<op_name>[.<overload>]"""
    # TODO(justinchuby): Handle arbitrary custom ops
    namespace, opname_overload = qualified_name.split("::")
    op_name, *maybe_overload = opname_overload.split(".", 1)
    if namespace == "_operator":
        # Builtin functions
        return getattr(operator, op_name)
    if namespace == "math":
        return getattr(math, op_name)
    if namespace == "torchvision":
        if importlib.util.find_spec("torchvision") is None:
            logger.warning("torchvision is not installed. Skipping %s", qualified_name)
            return None
    try:
        op_packet = getattr(getattr(torch.ops, namespace), op_name)
        if maybe_overload:
            overload = maybe_overload[0]
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

        return getattr(op_packet, overload)  # type: ignore[call-overload]
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
        self._opset_version = _constants.TORCHLIB_OPSET
        self.functions: dict[TorchOp | str, list[OnnxDecompMeta]] = {}

    @property
    def opset_version(self) -> int:
        """The ONNX opset version the exporter should target."""
        return self._opset_version

    @classmethod
    def from_torchlib(cls, opset_version=_constants.TORCHLIB_OPSET) -> ONNXRegistry:
        """Populates the registry with ATen functions from torchlib.

        Args:
            torchlib_registry: The torchlib registry to use for populating the registry.
        """
        registry = cls()
        registry._opset_version = opset_version
        for meta in _torchlib_registry.get_torchlib_ops():
            registry._register(meta.fx_target, meta)

        # TODO(justinchuby): Remove this once torchlib is migrated to PyTorch
        torchlib_ops = onnxscript_apis.get_torchlib_ops()

        for torchlib_meta in torchlib_ops:
            qualified_name = torchlib_meta.qualified_name
            overload_func = torchlib_meta.function
            try:
                # NOTE: This is heavily guarded with try-except because we don't want
                # to fail the entire registry population if one function fails.
                target = _get_overload(qualified_name)
                if target is None:
                    continue

                meta = OnnxDecompMeta(
                    onnx_function=overload_func,
                    fx_target=target,
                    signature=None,
                    is_custom=False,
                    is_complex=torchlib_meta.is_complex,
                )
                registry._register(target, meta)
            except Exception:
                logger.exception("Failed to register '%s'. Skipped", qualified_name)
                continue

        registry._cleanup_registry_based_on_opset_version()
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
        target_or_name: str | TorchOp
        if isinstance(target, torch._ops.OpOverload):
            # Get the qualified name of the aten op because torch._ops.OpOverload lookup in
            # a dictionary is unreliable for some reason.
            target_or_name = target.name()
        else:
            target_or_name = target
        if onnx_decomposition.is_custom:
            self.functions.setdefault(target_or_name, []).insert(0, onnx_decomposition)
        else:
            self.functions.setdefault(target_or_name, []).append(onnx_decomposition)

    def register_op(
        self,
        target: TorchOp,
        function: Callable,
        is_complex: bool = False,
    ) -> None:
        """Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            target: The PyTorch node callable target.
            function: The onnx-script function to register.
            is_complex: Whether the function is a function that handles complex valued inputs.
        """
        if isinstance(target, torch._ops.OpOverloadPacket):
            raise TypeError(
                f"Target '{target}' should be provided as an OpOverload instead of an "
                "OpOverloadPacket. You can get the default overload with "
                "<op>.default"
            )

        self._register(
            target,
            OnnxDecompMeta(
                onnx_function=function,
                fx_target=target,
                signature=None,
                is_custom=True,
                is_complex=is_complex,
            ),
        )

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
        target_or_name: str | TorchOp
        if isinstance(target, torch._ops.OpOverload):
            # Get the qualified name of the aten op because torch._ops.OpOverload lookup in
            # a dictionary is unreliable for some reason.
            target_or_name = target.name()
        else:
            target_or_name = target
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

    def _cleanup_registry_based_on_opset_version(self) -> None:
        """Pick the implementation with the highest opset version valid until the current opset version."""
        cleaned_functions = {}
        for target_or_name, decomps in self.functions.items():
            # Filter decompositions to only include those with opset_introduced <= opset_version
            decomps = [d for d in decomps if d.opset_introduced <= self.opset_version]

            # Keep only the decomposition with the highest opset_introduced
            if decomps:
                # Find the maximum opset_introduced
                max_opset = max(d.opset_introduced for d in decomps)

                # Keep all decompositions with the maximum opset_introduced
                cleaned_functions[target_or_name] = [
                    d for d in decomps if d.opset_introduced == max_opset
                ]

        self.functions = cleaned_functions

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(functions={self.functions})"
