"""
Template heuristic registry system for PyTorch Inductor.

This module provides a centralized registration system for template heuristics,
allowing automatic registration based on device type and conditional registration
for CUDA vs ROCm based on torch.version.hip.
"""

from __future__ import annotations

import logging
from functools import cache
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from .template_heuristics import TemplateConfigHeuristics

# Module-wide registry for template heuristics
_TEMPLATE_HEURISTIC_REGISTRY: dict[tuple[str, str], type[TemplateConfigHeuristics]] = {}

log = logging.getLogger(__name__)


def register_template_heuristic(
    template_name: str, device_type: str, register: bool = True
) -> Any:
    """
    Decorator to register template heuristic classes.

    Args:
        template_name: Name of the template (e.g., "mm", "bmm", "scaled_mm")
        device_type: Device type ("cuda", "cpu", "xpu")
        register: Whether to register this heuristic. Caller should pass the condition directly.

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
            key = (device_type, template_name)
            _TEMPLATE_HEURISTIC_REGISTRY[key] = cls
            log.info(
                f"Registered template heuristic: {cls.__name__} for template='{template_name}', device='{device_type}'"  # noqa: G004
            )
        return cls

    return decorator


@cache
def get_template_heuristic(
    template_name: str, device_type: str
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
    key = (device_type, template_name)

    # Look up in registry
    if key not in _TEMPLATE_HEURISTIC_REGISTRY:
        raise ValueError(
            f"No template heuristic found for template='{template_name}', "
            f"device='{device_type}'. "
            f"Available combinations: {list(_TEMPLATE_HEURISTIC_REGISTRY.keys())}"
        )

    # Instantiate and return
    heuristic_class = _TEMPLATE_HEURISTIC_REGISTRY[key]
    return heuristic_class()
