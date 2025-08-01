"""
Template heuristic registry system for PyTorch Inductor.

This module provides a centralized registration system for template heuristics,
allowing automatic registration based on device type and conditional registration
for CUDA vs ROCm based on torch.version.hip.
"""

from __future__ import annotations

import logging
from functools import cache
from typing import Any, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from .template_heuristics import TemplateConfigHeuristics

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
