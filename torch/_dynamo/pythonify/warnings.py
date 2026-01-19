"""
Warnings for unsupported features in pythonify mode.

This module provides warning utilities for the pythonify feature. When pythonify
is used with torch.compile options that are not fully supported, clear warnings
are emitted to help users understand the limitations.

The warnings are categorized by feature type:
- Backend warnings: Non-inductor backends may not generate optimal pythonify output
- Mode warnings: Some modes (reduce-overhead, etc.) have limited pythonify support
- Option warnings: Specific options may be ignored or have limited effect

Usage:
    from torch._dynamo.pythonify.warnings import check_pythonify_compatibility

    warnings = check_pythonify_compatibility(
        backend="eager",
        mode="reduce-overhead",
        options={"triton.cudagraphs": True},
    )
    for warning in warnings:
        warning.emit()
"""

from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional


class PythonifyWarningCategory(Enum):
    """
    Categories of pythonify compatibility warnings.

    This enum categorizes warnings to help users understand which aspect
    of their torch.compile configuration may have limited support.
    """

    BACKEND = "backend"
    MODE = "mode"
    OPTION = "option"
    FEATURE = "feature"
    GENERAL = "general"


@dataclass
class PythonifyWarning:
    """
    Represents a warning about pythonify compatibility.

    Attributes:
        category: The category of warning (backend, mode, option, etc.)
        feature: The specific feature or option that triggered the warning
        message: Human-readable description of the limitation
        recommendation: Suggested action to address the warning
        severity: Warning severity (info, warning, caution)
    """

    category: PythonifyWarningCategory
    feature: str
    message: str
    recommendation: Optional[str] = None
    severity: str = "warning"

    def emit(self) -> None:
        """
        Emit this warning using Python's warnings module.

        The warning is formatted to include all relevant information:
        feature name, description, and recommendation (if available).
        """
        parts = [f"[pythonify] {self.message}"]

        if self.recommendation:
            parts.append(f"Recommendation: {self.recommendation}")

        full_message = "\n".join(parts)

        if self.severity == "info":
            python_warnings.warn(full_message, stacklevel=4)
        else:
            python_warnings.warn(full_message, UserWarning, stacklevel=4)

    def __str__(self) -> str:
        """
        Return a string representation of the warning.
        """
        parts = [f"[{self.category.value}:{self.feature}] {self.message}"]
        if self.recommendation:
            parts.append(f"  Recommendation: {self.recommendation}")
        return "\n".join(parts)


UNSUPPORTED_BACKENDS = {
    "eager": (
        "The 'eager' backend does not perform compilation, so pythonify "
        "will generate minimal output without compiled kernels.",
        "Use the 'inductor' backend (default) for full pythonify support.",
    ),
    "aot_eager": (
        "The 'aot_eager' backend generates AOT graphs but does not produce "
        "optimized kernels. The pythonify output will show AOT graphs without "
        "Inductor-compiled code.",
        "Use the 'inductor' backend for optimized pythonify output.",
    ),
    "aot_eager_decomp_partition": (
        "The 'aot_eager_decomp_partition' backend generates decomposed AOT graphs "
        "without Inductor optimization.",
        "Use the 'inductor' backend for optimized pythonify output.",
    ),
    "cudagraphs": (
        "The 'cudagraphs' backend wraps eager execution in CUDA graphs. "
        "The pythonify output may not fully represent CUDA graph mechanics.",
        "Use mode='reduce-overhead' with the inductor backend instead.",
    ),
    "onnxrt": (
        "The 'onnxrt' backend uses ONNX Runtime which has a separate execution model. "
        "Pythonify output will show the ONNX export path, not native PyTorch code.",
        "Use the 'inductor' backend for native PyTorch pythonify output.",
    ),
    "tvm": (
        "The 'tvm' backend uses Apache TVM for compilation. Pythonify output will "
        "reference TVM-compiled artifacts rather than native PyTorch code.",
        "Use the 'inductor' backend for native PyTorch pythonify output.",
    ),
    "ipex": (
        "The 'ipex' backend uses Intel Extension for PyTorch. Pythonify output may "
        "not fully represent IPEX-specific optimizations.",
        "Use the 'inductor' backend for standard pythonify output.",
    ),
    "torchxla_trivial": (
        "The 'torchxla_trivial' backend targets XLA devices. Pythonify output will "
        "not include XLA-specific compilation details.",
        "Use the 'inductor' backend for CPU/CUDA pythonify output.",
    ),
    "openxla_trivial": (
        "The 'openxla_trivial' backend targets OpenXLA. Pythonify output will "
        "not include OpenXLA-specific compilation details.",
        "Use the 'inductor' backend for CPU/CUDA pythonify output.",
    ),
    "torchxla_trace_once": (
        "The 'torchxla_trace_once' backend uses XLA tracing. Pythonify output will "
        "not include XLA-specific compilation details.",
        "Use the 'inductor' backend for CPU/CUDA pythonify output.",
    ),
    "openxla_eval": (
        "The 'openxla_eval' backend targets OpenXLA evaluation. Pythonify output will "
        "not include OpenXLA-specific compilation details.",
        "Use the 'inductor' backend for CPU/CUDA pythonify output.",
    ),
}


LIMITED_SUPPORT_MODES = {
    "reduce-overhead": (
        "The 'reduce-overhead' mode uses CUDA graphs for reduced Python overhead. "
        "Pythonify can represent CUDA graph setup, but graph capture/replay "
        "mechanics are abstracted.",
        "The generated code shows CUDA graph configuration but may require "
        "runtime CUDA graph support to execute.",
    ),
    "max-autotune": (
        "The 'max-autotune' mode performs extensive kernel autotuning. "
        "The pythonify output will use the selected kernels but won't include "
        "the autotuning search process.",
        "Generated code will use autotuned kernel choices without re-autotuning.",
    ),
}


LIMITED_SUPPORT_OPTIONS = {
    "triton.cudagraphs": (
        "CUDA graphs are represented in pythonify output, but the actual graph "
        "capture and replay mechanics are abstracted.",
        "The generated code shows CUDA graph configuration; ensure CUDA graph "
        "support is available at execution time.",
    ),
    "trace.enabled": (
        "Trace output is a debugging feature. Pythonify generates code "
        "independently of trace settings.",
        None,
    ),
    "trace.graph_diagram": (
        "Graph diagrams are a debugging feature. Pythonify generates code "
        "independently of diagram settings.",
        None,
    ),
    "coordinate_descent_tuning": (
        "Coordinate descent tuning affects kernel selection. Pythonify will use "
        "the tuned kernels without the tuning search.",
        None,
    ),
    "max_autotune": (
        "Max autotune affects kernel selection. Pythonify will use the autotuned "
        "kernels without the tuning search.",
        None,
    ),
    "epilogue_fusion": (
        "Epilogue fusion is fully supported in pythonify output.",
        None,
    ),
}

UNSUPPORTED_FEATURES_WARNING_MESSAGES = {
    "distributed": (
        "Distributed training features (DDP, FSDP) are not fully represented "
        "in pythonify output. The generated code may not include distributed "
        "communication primitives.",
        "For distributed models, test the pythonify output on a single device first.",
    ),
    "nested_compile": (
        "Nested torch.compile calls within a pythonify region may produce "
        "unexpected output. Each compilation region is captured separately.",
        "Use a single torch.compile call for best pythonify results.",
    ),
    "autocast": (
        "torch.autocast contexts affect dtype selection but are not explicitly "
        "represented in pythonify output. The generated code uses the dtypes "
        "selected during tracing.",
        "Ensure autocast settings match between tracing and execution.",
    ),
}


def check_backend_compatibility(backend: str | Callable) -> list[PythonifyWarning]:
    """
    Check if a backend is fully supported for pythonify.

    Args:
        backend: The backend name or callable

    Returns:
        List of warnings for this backend
    """
    warnings_list: list[PythonifyWarning] = []

    if callable(backend):
        backend_name = getattr(backend, "__name__", str(backend))
        warnings_list.append(
            PythonifyWarning(
                category=PythonifyWarningCategory.BACKEND,
                feature=f"custom backend: {backend_name}",
                message=(
                    f"Custom backend '{backend_name}' may not be fully supported "
                    "by pythonify. The generated output depends on how the backend "
                    "integrates with Dynamo and AOT Autograd."
                ),
                recommendation=(
                    "Test the pythonify output carefully. Consider using the "
                    "'inductor' backend for guaranteed pythonify support."
                ),
                severity="warning",
            )
        )
    elif isinstance(backend, str):
        if backend in UNSUPPORTED_BACKENDS:
            message, recommendation = UNSUPPORTED_BACKENDS[backend]
            warnings_list.append(
                PythonifyWarning(
                    category=PythonifyWarningCategory.BACKEND,
                    feature=backend,
                    message=message,
                    recommendation=recommendation,
                    severity="warning",
                )
            )

    return warnings_list


def check_mode_compatibility(mode: Optional[str]) -> list[PythonifyWarning]:
    """
    Check if a mode is fully supported for pythonify.

    Args:
        mode: The mode name or None for default

    Returns:
        List of warnings for this mode
    """
    warnings_list: list[PythonifyWarning] = []

    if mode is not None and mode in LIMITED_SUPPORT_MODES:
        message, recommendation = LIMITED_SUPPORT_MODES[mode]
        warnings_list.append(
            PythonifyWarning(
                category=PythonifyWarningCategory.MODE,
                feature=mode,
                message=message,
                recommendation=recommendation,
                severity="info",
            )
        )

    return warnings_list


def check_options_compatibility(
    options: Optional[dict[str, Any]],
) -> list[PythonifyWarning]:
    """
    Check if specific options are fully supported for pythonify.

    Args:
        options: Dictionary of options or None

    Returns:
        List of warnings for these options
    """
    warnings_list: list[PythonifyWarning] = []

    if options is None:
        return warnings_list

    for option_name, option_value in options.items():
        if option_name in LIMITED_SUPPORT_OPTIONS:
            message, recommendation = LIMITED_SUPPORT_OPTIONS[option_name]
            if message:
                warnings_list.append(
                    PythonifyWarning(
                        category=PythonifyWarningCategory.OPTION,
                        feature=option_name,
                        message=message,
                        recommendation=recommendation,
                        severity="info",
                    )
                )

    return warnings_list


def check_pythonify_compatibility(
    backend: str | Callable = "inductor",
    mode: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
    fullgraph: bool = False,
    dynamic: Optional[bool] = None,
    disable: bool = False,
) -> list[PythonifyWarning]:
    """
    Check torch.compile configuration for pythonify compatibility.

    This function checks all aspects of a torch.compile configuration and
    returns warnings for any features that have limited or no pythonify support.

    Args:
        backend: The compilation backend
        mode: The compilation mode
        options: Dictionary of options
        fullgraph: Whether fullgraph mode is enabled
        dynamic: Whether dynamic shapes are enabled
        disable: Whether torch.compile is disabled

    Returns:
        List of PythonifyWarning objects for all compatibility issues found

    Example:
        warnings = check_pythonify_compatibility(
            backend="eager",
            mode="reduce-overhead",
        )
        for w in warnings:
            print(w)
            w.emit()
    """
    all_warnings: list[PythonifyWarning] = []

    if disable:
        all_warnings.append(
            PythonifyWarning(
                category=PythonifyWarningCategory.GENERAL,
                feature="disable",
                message=(
                    "torch.compile is disabled (disable=True). Pythonify will not "
                    "generate any compiled code since no compilation occurs."
                ),
                recommendation="Remove disable=True to enable compilation and pythonify.",
                severity="warning",
            )
        )
        return all_warnings

    all_warnings.extend(check_backend_compatibility(backend))
    all_warnings.extend(check_mode_compatibility(mode))
    all_warnings.extend(check_options_compatibility(options))

    return all_warnings


def emit_pythonify_warnings(
    backend: str | Callable = "inductor",
    mode: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
    fullgraph: bool = False,
    dynamic: Optional[bool] = None,
    disable: bool = False,
) -> list[PythonifyWarning]:
    """
    Check compatibility and emit warnings for pythonify.

    This is a convenience function that checks compatibility and automatically
    emits all warnings using Python's warnings module.

    Args:
        backend: The compilation backend
        mode: The compilation mode
        options: Dictionary of options
        fullgraph: Whether fullgraph mode is enabled
        dynamic: Whether dynamic shapes are enabled
        disable: Whether torch.compile is disabled

    Returns:
        List of PythonifyWarning objects that were emitted
    """
    warnings_list = check_pythonify_compatibility(
        backend=backend,
        mode=mode,
        options=options,
        fullgraph=fullgraph,
        dynamic=dynamic,
        disable=disable,
    )

    for warning in warnings_list:
        warning.emit()

    return warnings_list


def get_unsupported_backends() -> list[str]:
    """
    Get the list of backends that are not fully supported for pythonify.

    Returns:
        List of backend names that have limited or no pythonify support
    """
    return list(UNSUPPORTED_BACKENDS.keys())


def get_limited_support_modes() -> list[str]:
    """
    Get the list of modes that have limited pythonify support.

    Returns:
        List of mode names with limited pythonify support
    """
    return list(LIMITED_SUPPORT_MODES.keys())


def get_limited_support_options() -> list[str]:
    """
    Get the list of options that have limited pythonify support.

    Returns:
        List of option names with limited pythonify support
    """
    return list(LIMITED_SUPPORT_OPTIONS.keys())
