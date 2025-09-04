"""
Performance model registry system for PyTorch Inductor.

This module provides a centralized registration system for performance model functions,
allowing registration based on hardware name, template ID, and operation name.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING, Union


if TYPE_CHECKING:
    from .base import PerformanceModelFunction

# Registry mapping: (hardware, template, op) -> function
_PERFORMANCE_MODEL_REGISTRY: dict[tuple[str, str, str], PerformanceModelFunction] = {}

log = logging.getLogger(__name__)


def register_performance_model(
    template_id: str,
    op: str,
    hardware_name: str,
) -> Any:
    """
    Decorator to register performance model functions.

    Args:
        template_id: Template identifier (e.g., "mm", "bmm", "scaled_mm")
        op: Operation name (e.g., "mm", "bmm", "scaled_mm")
        hardware_name: Hardware name ("cuda", "cpu", "xpu", etc.)

    Example:
        @register_performance_model("mm", "mm", "cuda")
        def my_mm_predictor(choices, op_name):
            for choice in choices:
                choice.performance_prediction = calculate_prediction(choice)
            return choices
    """

    def decorator(func: PerformanceModelFunction) -> PerformanceModelFunction:
        register_performance_model_fn(func, template_id, op, hardware_name)
        return func

    return decorator


def register_performance_model_fn(
    func: PerformanceModelFunction,
    template_id: str,
    op: str,
    hardware_name: str,
) -> None:
    """
    Function-based registration for performance model functions.

    Args:
        func: Performance model function to register
        template_id: Template identifier (e.g., "mm", "bmm", "scaled_mm")
        op: Operation name (e.g., "mm", "bmm", "scaled_mm")
        hardware_name: Hardware name ("cuda", "cpu", "xpu", etc.)
    """
    key = (hardware_name, template_id, op)

    # If overriding an existing registration, log it
    if key in _PERFORMANCE_MODEL_REGISTRY:
        log.debug("Replacing existing performance model for key %r", key)

    # Update registry
    _PERFORMANCE_MODEL_REGISTRY[key] = func

    log.info(
        "Registered performance model function for hardware_name=%r template_id=%r op=%r",
        hardware_name,
        template_id,
        op,
    )


def get_model_function_for_key(
    template_id: str, op: str, hardware_name: str
) -> Union[PerformanceModelFunction, None]:
    """
    Get a performance model function for a specific (hardware, template, op) key.

    Returns:
        Performance model function if found, None otherwise.
    """
    key = (hardware_name, template_id, op)
    return _PERFORMANCE_MODEL_REGISTRY.get(key)


def get_functions_for_templates(
    template_ids: list[str], op_name: str, hardware_name: str
) -> dict[PerformanceModelFunction, list[str]]:
    """
    Get performance model functions mapped to the template_ids they can handle.

    Args:
        template_ids: List of template identifiers to check
        op_name: Operation name to check for
        hardware_name: Hardware name to check for

    Returns:
        Dictionary mapping function instances to lists of template_ids
        they can handle. Only includes template_ids that have registered functions.
    """
    function_to_templates: dict[PerformanceModelFunction, list[str]] = {}

    # Group template_ids by which function handles them
    for template_id in template_ids:
        func = get_model_function_for_key(template_id, op_name, hardware_name)
        if func is not None:
            if func not in function_to_templates:
                function_to_templates[func] = []
            function_to_templates[func].append(template_id)

    return function_to_templates


def list_registered_models() -> list[tuple[str, str, str]]:
    """
    List all registered performance model keys.

    Returns:
        List of (hardware_name, template_id, op) tuples for all registered functions.
    """
    return list(_PERFORMANCE_MODEL_REGISTRY.keys())


def clear_registry() -> None:
    """
    Clear all registered performance model functions.

    This is primarily useful for testing purposes.
    """
    global _PERFORMANCE_MODEL_REGISTRY
    _PERFORMANCE_MODEL_REGISTRY.clear()
