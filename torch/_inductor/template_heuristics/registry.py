"""
Template heuristic registry system for PyTorch Inductor.

This module provides a centralized registration system for template heuristics,
allowing automatic registration based on device type and conditional registration
for CUDA vs ROCm based on torch.version.hip.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional, TYPE_CHECKING, Union

from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Iterator


# Module-wide registry for template heuristics
_TEMPLATE_HEURISTIC_REGISTRY: dict[
    tuple[Union[str, None], ...], type[TemplateConfigHeuristics]
] = {}

# Manual cache for successful lookups only (fallback instances are not cached)
_HEURISTIC_CACHE: dict[tuple[str, str, str], TemplateConfigHeuristics] = {}

log = logging.getLogger(__name__)


def register_template_heuristic(
    template_name: str,
    device_type: Union[str, None],
    register: bool = True,
    op_name: Optional[str] = None,
) -> Any:
    """
    Decorator to register template heuristic classes.

    Args:
        template_name: Name of the template (e.g., "mm", "bmm", "scaled_mm")
        device_type: Device type ("cuda", "cpu", "xpu")
            Set this to None to indicate that the heuristic is applicable to all device types.
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
            key: tuple[Union[str, None], ...] = (template_name, device_type, op_name)
            _TEMPLATE_HEURISTIC_REGISTRY[key] = cls
            log.info(
                f"Registered template heuristic: {cls.__name__} for '{template_name=}', '{device_type=}', '{op_name=}'"  # noqa: G004
            )
        return cls

    return decorator


def get_template_heuristic(
    template_name: str, device_type: str, op_name: str
) -> TemplateConfigHeuristics:
    """
    Retrieve a template heuristic instance for the given template and device type.

    Args:
        template_name: Name of the template (e.g., "mm", "bmm", "scaled_mm")
        device_type: Device type ("cuda", "cpu", "xpu")
        op_name: Name of the operator (e.g., "mm", "bmm", "scaled_mm")

    Returns:
        Template heuristic instance. If no specific heuristic is found,
        returns a fallback TemplateConfigHeuristics() instance (uncached).
    """
    # Check cache first
    cache_key = (template_name, device_type, op_name)
    if cache_key in _HEURISTIC_CACHE:
        return _HEURISTIC_CACHE[cache_key]

    heuristic_class = get_registered_heuristic_class(
        template_name, device_type, op_name
    )

    if heuristic_class is None:
        # Log error and return fallback instance (uncached)
        log.error(
            "No template heuristic found - template_name=%s, device_type=%s, op_name=%s. "
            "Available combinations: %s. Using fallback TemplateConfigHeuristics instance.",
            template_name,
            device_type,
            op_name,
            list(_TEMPLATE_HEURISTIC_REGISTRY.keys()),
        )
        return TemplateConfigHeuristics()

    # Cache successful lookup and return
    instance = heuristic_class()
    _HEURISTIC_CACHE[cache_key] = instance
    return instance


def get_registered_heuristic_class(
    template_name: str, device_type: str, op_name: str
) -> None | type[TemplateConfigHeuristics]:
    """
    Get the heuristic class registered for the given template/device/op combination.

    This is useful for creating custom heuristics that subclass the appropriate
    base class for a given template/device/op combination.

    Args:
        template_name: Name of the template (e.g., "mm", "bmm", "scaled_mm")
        device_type: Device type ("cuda", "cpu", "xpu")
        op_name: Name of the operator (e.g., "mm", "bmm", "scaled_mm")

    Returns:
        The heuristic class if found, None otherwise.
    """
    keys = [
        # everything is specified
        (template_name, device_type, op_name),
        # heuristic is valid across all devices
        (template_name, None, op_name),
        # heuristic is valid across all ops for that device
        (template_name, device_type, None),
        # heuristic is always valid for that template
        (template_name, None, None),
    ]
    for key in keys:
        if key in _TEMPLATE_HEURISTIC_REGISTRY:
            return _TEMPLATE_HEURISTIC_REGISTRY[key]

    return None


def clear_registry() -> None:
    """
    Clear all registered template heuristics.

    This is primarily useful for testing purposes to ensure a clean state.
    """
    _TEMPLATE_HEURISTIC_REGISTRY.clear()
    _HEURISTIC_CACHE.clear()


@contextlib.contextmanager
def override_template_heuristics(
    device_type: str,
    template_op_pairs: list[tuple[str, str]],
    override_heuristic_class: type[TemplateConfigHeuristics] = TemplateConfigHeuristics,
) -> Iterator[None]:
    """
    Context manager to temporarily override template heuristics.

    This is useful for testing purposes, where we want to ensure a specific template/op pair
    uses a custom heuristic or returns no entries.

    Args:
        device_type: Device type ("cuda", "cpu", "xpu")
        template_op_pairs: List of (template_name, op_name) pairs to override.
        override_heuristic_class: Heuristic class to use for the override.
            Defaults to TemplateConfigHeuristics (which returns no entries).
    """
    # Save original entries to restore later
    original_entries = {}
    new_keys = []
    _HEURISTIC_CACHE.clear()
    try:
        for template_name, op_name in template_op_pairs:
            assert op_name is not None
            key = (template_name, device_type, op_name)
            if key in _TEMPLATE_HEURISTIC_REGISTRY:
                original_entries[key] = _TEMPLATE_HEURISTIC_REGISTRY[key]
            _TEMPLATE_HEURISTIC_REGISTRY[key] = override_heuristic_class
            new_keys.append(key)
        yield
    finally:
        # Restore original entries or remove if they didn't exist before
        for key in new_keys:
            _TEMPLATE_HEURISTIC_REGISTRY.pop(key, None)
            if key in original_entries:
                _TEMPLATE_HEURISTIC_REGISTRY[key] = original_entries[key]
        _HEURISTIC_CACHE.clear()
