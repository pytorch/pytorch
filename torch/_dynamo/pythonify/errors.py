"""
Error handling for torch.compile pythonify feature.

This module provides custom exception classes and error utilities for the
pythonify feature. The goal is to provide clear, actionable error messages
that help users understand what went wrong and how to fix it.

The main exception class is PythonifyError, which includes:
- The compilation stage where the error occurred
- The original exception that caused the failure
- Suggested remedies when available
- Context about what was being processed when the error occurred
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PythonifyStage(Enum):
    """
    Compilation stages where pythonify errors can occur.

    This enum identifies the stage of the pythonify pipeline where an error
    occurred, helping users and developers understand the context of the failure.
    """

    INITIALIZATION = "initialization"
    ARGUMENT_EXTRACTION = "argument_extraction"
    GUARD_TRANSLATION = "guard_translation"
    AOT_AUTOGRAD = "aot_autograd"
    CUDA_GRAPH_SETUP = "cuda_graph_setup"
    IR_CONSTRUCTION = "ir_construction"
    PYTHON_CODE_GENERATION = "python_code_generation"
    BINARY_CODE_GENERATION = "binary_code_generation"
    FILE_WRITING = "file_writing"
    KERNEL_SERIALIZATION = "kernel_serialization"
    ARTIFACT_COLLECTION = "artifact_collection"
    UNKNOWN = "unknown"


class PythonifyError(Exception):
    """
    Exception raised when pythonify compilation fails.

    This exception provides detailed context about what went wrong during
    pythonify compilation, including:
    - The compilation stage where the error occurred
    - The original exception that caused the failure
    - Any context about what was being processed
    - Suggested remedies when available

    Attributes:
        stage: The compilation stage where the error occurred
        message: Human-readable description of the error
        original_exception: The underlying exception that caused the failure
        context: Additional context about what was being processed
        remedy: Suggested fix for the error (if available)

    Example:
        try:
            generate_pythonify_output()
        except PythonifyError as e:
            print(f"Pythonify failed at stage: {e.stage.value}")
            print(f"Error: {e.message}")
            if e.remedy:
                print(f"Try: {e.remedy}")
    """

    def __init__(
        self,
        message: str,
        stage: PythonifyStage = PythonifyStage.UNKNOWN,
        original_exception: Optional[Exception] = None,
        context: Optional[dict[str, Any]] = None,
        remedy: Optional[str] = None,
    ) -> None:
        """
        Initialize a PythonifyError.

        Args:
            message: Human-readable description of the error
            stage: The compilation stage where the error occurred
            original_exception: The underlying exception that caused the failure
            context: Additional context about what was being processed
            remedy: Suggested fix for the error
        """
        self.stage = stage
        self.message = message
        self.original_exception = original_exception
        self.context = context or {}
        self.remedy = remedy

        full_message = self._format_message()
        super().__init__(full_message)

    def _format_message(self) -> str:
        """
        Format the full error message with context.

        Returns:
            Formatted error message string
        """
        parts = []

        parts.append(f"Pythonify compilation failed at stage: {self.stage.value}")
        parts.append("")
        parts.append(f"Error: {self.message}")

        if self.context:
            parts.append("")
            parts.append("Context:")
            for key, value in self.context.items():
                parts.append(f"  {key}: {value}")

        if self.original_exception:
            parts.append("")
            parts.append(
                f"Caused by: {type(self.original_exception).__name__}: "
                f"{self.original_exception}"
            )

        if self.remedy:
            parts.append("")
            parts.append(f"Suggested remedy: {self.remedy}")

        return "\n".join(parts)


@dataclass
class ErrorContext:
    """
    Collects context information for error reporting.

    This class is used to accumulate context as the pythonify pipeline
    executes, so that if an error occurs, we have detailed information
    about what was being processed.

    Attributes:
        current_stage: The current pipeline stage
        model_name: Name of the model being compiled
        input_names: Names of input arguments
        current_node: The IR node currently being processed
        artifacts_processed: Number of compilation artifacts processed
        additional_info: Any additional contextual information
    """

    current_stage: PythonifyStage = PythonifyStage.UNKNOWN
    model_name: Optional[str] = None
    input_names: list[str] = field(default_factory=list)
    current_node: Optional[str] = None
    artifacts_processed: int = 0
    additional_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the context to a dictionary for error reporting.

        Returns:
            Dictionary representation of the context
        """
        result = {}
        if self.model_name:
            result["model_name"] = self.model_name
        if self.input_names:
            result["input_names"] = ", ".join(self.input_names)
        if self.current_node:
            result["current_node"] = self.current_node
        if self.artifacts_processed > 0:
            result["artifacts_processed"] = self.artifacts_processed
        result.update(self.additional_info)
        return result


def create_pythonify_error(
    stage: PythonifyStage,
    original_exception: Exception,
    context: Optional[dict[str, Any]] = None,
) -> PythonifyError:
    """
    Create a PythonifyError from an existing exception.

    This factory function examines the original exception and creates a
    PythonifyError with an appropriate message and suggested remedy based
    on common error patterns.

    Args:
        stage: The compilation stage where the error occurred
        original_exception: The exception that was caught
        context: Additional context about what was being processed

    Returns:
        A PythonifyError with detailed information
    """
    error_type = type(original_exception).__name__
    error_message = str(original_exception)

    message = f"{error_type}: {error_message}"
    remedy = None

    if isinstance(original_exception, FileNotFoundError):
        message = f"Required file not found: {error_message}"
        remedy = "Ensure all required files exist and paths are correct"

    elif isinstance(original_exception, PermissionError):
        message = f"Permission denied: {error_message}"
        remedy = (
            "Check file permissions and ensure the pythonify output "
            "directory is writable"
        )

    elif isinstance(original_exception, OSError):
        message = f"File system error: {error_message}"
        remedy = (
            "Check disk space and permissions for the pythonify output path"
        )

    elif isinstance(original_exception, TypeError):
        message = f"Type error during {stage.value}: {error_message}"
        remedy = (
            "This may indicate incompatible compilation artifacts. "
            "Try clearing the Dynamo cache with torch._dynamo.reset()"
        )

    elif isinstance(original_exception, AttributeError):
        message = f"Attribute access error during {stage.value}: {error_message}"
        if "has no attribute" in error_message:
            remedy = (
                "This may indicate a missing or incompatible compilation "
                "artifact. Ensure the model is properly traced before pythonify"
            )

    elif isinstance(original_exception, KeyError):
        message = f"Missing key during {stage.value}: {error_message}"
        remedy = (
            "A required compilation artifact is missing. "
            "Ensure the full compilation pipeline ran before pythonify"
        )

    elif isinstance(original_exception, ValueError):
        message = f"Invalid value during {stage.value}: {error_message}"
        remedy = (
            "Check that all torch.compile options are compatible with pythonify"
        )

    elif isinstance(original_exception, RuntimeError):
        message = f"Runtime error during {stage.value}: {error_message}"
        if "CUDA" in error_message or "cuda" in error_message:
            remedy = (
                "CUDA error detected. Check CUDA availability and memory. "
                "Try with cuda_graphs=False if using CUDA graphs"
            )
        else:
            remedy = (
                "Try simplifying the model or input shapes. "
                "Report this error if it persists"
            )

    elif isinstance(original_exception, ImportError):
        message = f"Import error during {stage.value}: {error_message}"
        remedy = (
            "A required module is missing. "
            "Install the missing dependency and try again"
        )

    return PythonifyError(
        message=message,
        stage=stage,
        original_exception=original_exception,
        context=context,
        remedy=remedy,
    )


def format_guard_failure_message(
    guard_type: str,
    target_name: str,
    expected_value: Any,
    actual_value: Any,
    dimension: Optional[int] = None,
) -> str:
    """
    Format a clear error message for guard failures.

    This function creates user-friendly error messages for guard failures,
    explaining what was expected vs what was received.

    Args:
        guard_type: The type of guard that failed (shape, dtype, device, etc.)
        target_name: Name of the variable being guarded
        expected_value: What the guard expected
        actual_value: What was actually received
        dimension: For shape guards, which dimension failed

    Returns:
        A formatted error message string
    """
    if guard_type.lower() == "shape":
        if dimension is not None:
            return (
                f"Shape guard failed for {target_name}:\n"
                f"  Expected {target_name}.shape[{dimension}] == {expected_value}\n"
                f"  Got {target_name}.shape[{dimension}] == {actual_value}\n"
                f"  Tip: If you want variable batch sizes, use "
                f"torch.compile(model, dynamic=True)"
            )
        else:
            return (
                f"Shape guard failed for {target_name}:\n"
                f"  Expected shape: {expected_value}\n"
                f"  Got shape: {actual_value}\n"
                f"  Tip: If you want dynamic shapes, use "
                f"torch.compile(model, dynamic=True)"
            )

    elif guard_type.lower() == "dtype":
        return (
            f"Dtype guard failed for {target_name}:\n"
            f"  Expected dtype: {expected_value}\n"
            f"  Got dtype: {actual_value}\n"
            f"  Tip: Ensure input tensors have the same dtype as during tracing"
        )

    elif guard_type.lower() == "device":
        return (
            f"Device guard failed for {target_name}:\n"
            f"  Expected device: {expected_value}\n"
            f"  Got device: {actual_value}\n"
            f"  Tip: Move tensors to the expected device before calling"
        )

    elif guard_type.lower() == "value":
        return (
            f"Value guard failed for {target_name}:\n"
            f"  Expected: {expected_value}\n"
            f"  Got: {actual_value}"
        )

    elif guard_type.lower() == "type":
        return (
            f"Type guard failed for {target_name}:\n"
            f"  Expected type: {expected_value}\n"
            f"  Got type: {actual_value}"
        )

    else:
        return (
            f"Guard failed for {target_name}:\n"
            f"  Guard type: {guard_type}\n"
            f"  Expected: {expected_value}\n"
            f"  Got: {actual_value}"
        )


def format_ir_construction_error(
    node_type: str,
    error: Exception,
    context: Optional[dict[str, Any]] = None,
) -> str:
    """
    Format an error message for IR construction failures.

    Args:
        node_type: The type of IR node being constructed
        error: The exception that occurred
        context: Additional context about the construction

    Returns:
        A formatted error message string
    """
    parts = [f"Failed to construct IR node: {node_type}"]
    parts.append(f"Error: {type(error).__name__}: {error}")

    if context:
        parts.append("Context:")
        for key, value in context.items():
            parts.append(f"  {key}: {value}")

    return "\n".join(parts)


def format_code_generation_error(
    backend: str,
    node_type: str,
    error: Exception,
) -> str:
    """
    Format an error message for code generation failures.

    Args:
        backend: The code generation backend (python/binary)
        node_type: The type of IR node being processed
        error: The exception that occurred

    Returns:
        A formatted error message string
    """
    return (
        f"Code generation failed ({backend} backend):\n"
        f"  Node type: {node_type}\n"
        f"  Error: {type(error).__name__}: {error}\n"
        f"  Tip: This may indicate an unsupported feature in the pythonify backend"
    )


def format_file_writing_error(
    file_path: str,
    error: Exception,
) -> str:
    """
    Format an error message for file writing failures.

    Args:
        file_path: The path that failed to write
        error: The exception that occurred

    Returns:
        A formatted error message string
    """
    if isinstance(error, PermissionError):
        return (
            f"Cannot write to pythonify output file:\n"
            f"  Path: {file_path}\n"
            f"  Error: Permission denied\n"
            f"  Tip: Check file permissions and ensure the directory is writable"
        )
    elif isinstance(error, FileNotFoundError):
        return (
            f"Cannot write to pythonify output file:\n"
            f"  Path: {file_path}\n"
            f"  Error: Directory does not exist\n"
            f"  Tip: Create the parent directory first"
        )
    else:
        return (
            f"Cannot write to pythonify output file:\n"
            f"  Path: {file_path}\n"
            f"  Error: {type(error).__name__}: {error}"
        )
