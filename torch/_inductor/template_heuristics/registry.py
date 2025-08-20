"""
Template heuristic registry system for PyTorch Inductor.

This module provides a centralized registration system for template heuristics,
allowing automatic registration based on device type and conditional registration
for CUDA vs ROCm based on torch.version.hip.
"""

from __future__ import annotations

import contextlib
import logging
from functools import cache
from typing import Any, Optional, TYPE_CHECKING

from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Iterator


# Module-wide registry for template heuristics
_TEMPLATE_HEURISTIC_REGISTRY: dict[tuple[str, ...], type[TemplateConfigHeuristics]] = {}

log = logging.getLogger(__name__)


def register_template_heuristic(
    template_name: str,
    device_type: str,
    register: bool = True,
    op_name: Optional[str] = None,
) -> Any:
    """
    Decorator to register template heuristic classes.

    Args:
        template_name: Name of the template (e.g., "mm", "bmm", "scaled_mm")
        device_type: Device type ("cuda", "cpu", "xpu")
        register: Whether to register this heuristic. Caller should pass the condition directly.
        op_name: Name of the operator (e.g., "mm", "bmm", "scaled_mm"). This is optional
            and is only used when a template uses different heuristics for different ops

    Returns:
        Decorator function that registers the class if conditions are met.

    Example:
        @register_template_heuristic("mm", "cuda", register=torch.version.hip is None)
        class CUDAMMTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
            pass
    """

    def decorator(
        cls: type[TemplateConfigHeuristics],
    ) -> type[TemplateConfigHeuristics]:
        if register:
            key: tuple[str, ...] = (device_type, template_name)
            if op_name is not None:
                key = (device_type, template_name, op_name)
            _TEMPLATE_HEURISTIC_REGISTRY[key] = cls
            log.info(
                f"Registered template heuristic: {cls.__name__} for '{template_name=}', '{device_type=}', '{op_name=}'"  # noqa: G004
            )
        return cls

    return decorator


@cache
def get_template_heuristic(
    template_name: str, device_type: str, op_name: str
) -> TemplateConfigHeuristics:
    """
    Retrieve a template heuristic instance for the given template and device type.

    Args:
        template_name: Name of the template (e.g., "mm", "bmm", "scaled_mm")
        device_type: Device type ("cuda", "cpu", "xpu")

    Returns:
        Template heuristic instance.

    Raises:
        ValueError: If no heuristic is found for the given combination.
    """
    # First check the more specific key
    keys = [(device_type, template_name, op_name), (device_type, template_name)]

    # Look up in registry
    heuristic_class = None
    for key in keys:
        if key in _TEMPLATE_HEURISTIC_REGISTRY:
            heuristic_class = _TEMPLATE_HEURISTIC_REGISTRY[key]
            break
    if heuristic_class is None:
        raise ValueError(
            f"No template heuristic found for '{template_name=}', "
            f"'{device_type=}', '{op_name=}'. "
            f"Available combinations: {list(_TEMPLATE_HEURISTIC_REGISTRY.keys())}"
        )
    return heuristic_class()


@contextlib.contextmanager
def override_template_heuristics(
    device_type: str,
    template_op_pairs: list[tuple[str, str]],
) -> Iterator[None]:
    """
    Context manager to temporarily override template heuristics with an empty heuristic.

    This is useful for testing purposes, where we want to ensure a specific template/op pair
    is not used

    Args:
        device_type: Device type ("cuda", "cpu", "xpu")
        template_op_pairs: List of (template_name, op_name) pairs to override.
    """
    # Save original entries to restore later
    original_entries = {}
    new_keys = []
    get_template_heuristic.cache_clear()
    try:
        for template_name, op_name in template_op_pairs:
            assert op_name is not None
            key = (device_type, template_name, op_name)
            if key in _TEMPLATE_HEURISTIC_REGISTRY:
                original_entries[key] = _TEMPLATE_HEURISTIC_REGISTRY[key]
                # TemplateConfigHeuristics base class returns no entries
                # so we use it for overriding
            _TEMPLATE_HEURISTIC_REGISTRY[key] = TemplateConfigHeuristics
            new_keys.append(key)
        yield
    finally:
        # Restore original entries or remove if they didn't exist before
        for key in new_keys:
            _TEMPLATE_HEURISTIC_REGISTRY.pop(key, None)
            if key in original_entries:
                _TEMPLATE_HEURISTIC_REGISTRY[key] = original_entries[key]
        get_template_heuristic.cache_clear()
