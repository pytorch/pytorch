# mypy: allow-untyped-defs
"""A context manager that disables the decomposition of certain ops during dynamo tracing.

The approach is to temporarily hijack the operator callable with PT2 custom operator.
The custom operator will not be decomposed and will show up as a single node to be exported to ONNX.

For the time being the decomposition of these ops is otherwise unavoidable.

https://github.com/pytorch/pytorch/issues/116684
https://github.com/pytorch/pytorch/issues/115883

This solution will no longer be required once the issue is resolved.
"""
from __future__ import annotations

import abc
import contextlib
from typing import Callable, Sequence

from onnxscript.function_libs.torch_lib.ops import (  # type: ignore[import-not-found]
    core as torchlib_core,
    nn as torchlib_nn,
)

import torch
from torch._decomp import decompositions


_NEW_OP_NAMESPACE: str = "onnx_export"
"""The namespace for the custom operator."""


class DecompSkip(abc.ABC):
    op_callable: Callable
    """The original operator callable to skip decomposition."""
    onnxscript_function: Callable
    """The ONNXScript function to be registered for exporting the custom operator."""

    new_op_name: str
    """The name for the custom operator."""
    new_op_schema: str
    """The schema for the custom operator. This should match with the signature of the original operator."""

    @classmethod
    @abc.abstractmethod
    def register(cls, export_options: torch.onnx.ExportOptions):
        """Registers the custom operator and overrides the original operator.

        It should do the following steps in order:

        1. Register the custom operator.
        2. Override the original operator with the replacement callable.
        3. Register the ONNXScript function for exporting the custom operator.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def unregister(cls):
        """Restores the original operator callable."""
        ...

    @classmethod
    @abc.abstractmethod
    def abstract(cls, *args, **kwargs):
        """An abstract impl (meta kernel) for the operator."""
        ...

    @classmethod
    def register_custom_op(cls):
        """Registers the custom operator."""
        new_op_qualname = f"{_NEW_OP_NAMESPACE}::{cls.new_op_name}"
        torch.library.define(new_op_qualname, cls.new_op_schema)
        torch.library.impl(new_op_qualname, "default", cls.replacement)
        torch.library.register_fake(new_op_qualname, cls.abstract)

    @classmethod
    def replacement(cls, *args, **kwargs):
        """A replacement callable for the operator to be hijacked.

        This has the same signature and eager behavior as the original operator.
        """
        return cls.op_callable(*args, **kwargs)


class UpsampleBilinear2DDecompSkip(DecompSkip):
    op_callable = torch._C._nn.upsample_bilinear2d  # type: ignore[attr-defined]
    onnxscript_function = torchlib_nn.aten_upsample_bilinear2d_vec  # type: ignore[attr-defined]
    new_op_name = "upsample_bilinear2d"
    new_op_schema = "(Tensor self, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> (Tensor)"

    @classmethod
    def register(cls, export_options: torch.onnx.ExportOptions):
        if not hasattr(torch.ops, _NEW_OP_NAMESPACE) or not hasattr(
            torch.ops.onnx_export, cls.new_op_name
        ):
            cls.register_custom_op()
        torch._C._nn.upsample_bilinear2d = torch.ops.onnx_export.upsample_bilinear2d  # type: ignore[attr-defined]
        if export_options.onnx_registry is None:
            export_options.onnx_registry = torch.onnx.OnnxRegistry()
        registry = export_options.onnx_registry
        registry.register_op(
            function=cls.onnxscript_function,
            namespace=_NEW_OP_NAMESPACE,
            op_name=cls.new_op_name,
        )

    @classmethod
    def unregister(cls):
        torch._C._nn.upsample_bilinear2d = cls.op_callable  # type: ignore[attr-defined]

    @classmethod
    def abstract(cls, input, output_size, align_corners, scale_factors):
        osize = decompositions.upsample_compute_output_size(
            input.size(), output_size, scale_factors
        )
        return torch.empty(
            (input.size(0), input.size(1), *osize),
            dtype=input.dtype,
            device=input.device,
        )


class UpsampleTrilinear3DDecompSkip(DecompSkip):
    op_callable = torch._C._nn.upsample_trilinear3d  # type: ignore[attr-defined]
    onnxscript_function = torchlib_nn.aten_upsample_trilinear3d_vec  # type: ignore[attr-defined]
    new_op_name = "upsample_trilinear3d"
    new_op_schema = "(Tensor self, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> (Tensor)"

    @classmethod
    def register(cls, export_options: torch.onnx.ExportOptions):
        if not hasattr(torch.ops, _NEW_OP_NAMESPACE) or not hasattr(
            torch.ops.onnx_export, cls.new_op_name
        ):
            cls.register_custom_op()
        torch._C._nn.upsample_trilinear3d = torch.ops.onnx_export.upsample_trilinear3d  # type: ignore[attr-defined]
        if export_options.onnx_registry is None:
            export_options.onnx_registry = torch.onnx.OnnxRegistry()
        registry = export_options.onnx_registry
        registry.register_op(
            function=cls.onnxscript_function,
            namespace=_NEW_OP_NAMESPACE,
            op_name=cls.new_op_name,
        )

    @classmethod
    def unregister(cls):
        torch._C._nn.upsample_trilinear3d = cls.op_callable  # type: ignore[attr-defined]

    @classmethod
    def abstract(cls, input, output_size, align_corners, scale_factors):
        osize = decompositions.upsample_compute_output_size(
            input.size(), output_size, scale_factors
        )
        return torch.empty(
            (input.size(0), input.size(1), input.size(2), *osize),
            dtype=input.dtype,
            device=input.device,
        )


class InstanceNormDecompSkip(DecompSkip):
    op_callable = torch.instance_norm  # type: ignore[attr-defined]
    onnxscript_function = torchlib_core.aten_instance_norm  # type: ignore[attr-defined]
    new_op_name = "instance_norm"
    new_op_schema = (
        "(Tensor input, Tensor? weight, Tensor? bias, "
        "Tensor? running_mean, Tensor? running_var, "
        "bool use_input_stats, float momentum, float eps, "
        "bool cudnn_enabled) -> Tensor"
    )

    @classmethod
    def register(cls, export_options: torch.onnx.ExportOptions):
        if not hasattr(torch.ops, _NEW_OP_NAMESPACE) or not hasattr(
            torch.ops.onnx_export, cls.new_op_name
        ):
            cls.register_custom_op()

        torch.instance_norm = torch.ops.onnx_export.instance_norm  # type: ignore[attr-defined]
        if export_options.onnx_registry is None:
            export_options.onnx_registry = torch.onnx.OnnxRegistry()
        registry = export_options.onnx_registry
        registry.register_op(
            function=cls.onnxscript_function,
            namespace=_NEW_OP_NAMESPACE,
            op_name=cls.new_op_name,
        )

    @classmethod
    def unregister(cls):
        torch.instance_norm = cls.op_callable  # type: ignore[attr-defined]

    @classmethod
    def abstract(
        cls,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        use_input_stats: bool,
        momentum: float,
        eps: float,
        cudnn_enabled: bool,
    ):
        return torch.empty(
            input.size(),
            dtype=input.dtype,
            device=input.device,
        )


_DEFAULT_SKIP_LIST = [
    UpsampleBilinear2DDecompSkip,
    InstanceNormDecompSkip,
    UpsampleTrilinear3DDecompSkip,
]


@contextlib.contextmanager
def enable_decomposition_skips(
    export_options: torch.onnx.ExportOptions,
    skips: Sequence[type[DecompSkip]] = _DEFAULT_SKIP_LIST,
):
    """A context manager that enables the decomposition skips.

    The original operator callables that are otherwise decomposed are replaced with custom operators.
    The ONNXScript functions for exporting the custom operators are added to the ONNX registry inside export_options.
    """
    try:
        for skip in skips:
            skip.register(export_options)
        yield
    finally:
        for skip in skips:
            skip.unregister()
