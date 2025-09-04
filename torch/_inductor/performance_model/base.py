from __future__ import annotations

from typing import Callable, TYPE_CHECKING


if TYPE_CHECKING:
    from ..kernel_template_choice import KernelTemplateChoice


# Type alias for performance prediction functions
PerformanceModelFunction = Callable[
    [list["KernelTemplateChoice"], str], list["KernelTemplateChoice"]
]


def predict(
    choices: list[KernelTemplateChoice], op_name: str
) -> list[KernelTemplateChoice]:
    """
    Predict the performance of a list of kernel template choices.

    This function signature defines the interface for performance model functions.
    Implementers should create functions that match this signature and register
    them with the performance model registry.

    This function analyzes each KernelTemplateChoice in the input list and
    populates the performance_prediction field with a float value representing
    the predicted performance (e.g., runtime in milliseconds, throughput, etc.).

    Args:
        choices: List of KernelTemplateChoice objects to predict performance for.
                Each choice may or may not already have a performance_prediction.
        op_name: Operation name (e.g., "mm", "bmm", "addmm") to provide context
                for performance prediction.

    Returns:
        The same list of KernelTemplateChoice objects with performance_prediction
        field populated with float values. The list should maintain the same
        order and contain the same objects (not copies).

    Note:
        - Lower performance prediction values typically indicate better performance
        - This is a template function that should not be called directly
    """
    raise NotImplementedError(
        "This is a template function. Use register_performance_model to register actual implementations."
    )
