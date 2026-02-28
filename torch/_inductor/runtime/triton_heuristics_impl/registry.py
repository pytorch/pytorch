from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


log = logging.getLogger(__name__)


_TRITON_HEURISTIC_REGISTRY: dict[tuple[str, Optional[str]], Callable[..., Any]] = {}
_HEURISTIC_CACHE: dict[tuple[str, str], Callable[..., Any]] = {}


def register_triton_heuristic(
    heuristic_name: str,
    device_type: Optional[str],
    register: bool = True,
) -> Any:
    """
    Decorator to register Triton heuristic functions by name and device type.

    Args:
        heuristic_name: Heuristic kind (e.g., "pointwise", "reduction").
        device_type: Device type (e.g., "cuda", "xpu").
            Set this to None to indicate that the heuristic is applicable to all device types.
        register: Whether to register this heuristic. Caller should pass the condition directly.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if register:
            key = (heuristic_name, device_type)
            _TRITON_HEURISTIC_REGISTRY[key] = fn
            log.info(
                "Registered Triton heuristic: %s for '%s', '%s'",
                fn.__name__,
                heuristic_name,
                device_type,
            )
        return fn

    return decorator


def get_triton_heuristic(
    heuristic_name: str, device_type: str
) -> Optional[Callable[..., Any]]:
    """
    Retrieve a Triton heuristic function for the given name and device type.

    Args:
        heuristic_name: Heuristic kind (e.g., "pointwise", "reduction").
        device_type: Device type (e.g., "cuda", "xpu", "hip").

    Returns:
        The heuristic function if found, None otherwise.
    """
    cache_key = (heuristic_name, device_type)
    if cache_key in _HEURISTIC_CACHE:
        return _HEURISTIC_CACHE[cache_key]

    keys = [
        (heuristic_name, device_type),
        (heuristic_name, None),
    ]
    for key in keys:
        if key in _TRITON_HEURISTIC_REGISTRY:
            heuristic = _TRITON_HEURISTIC_REGISTRY[key]
            _HEURISTIC_CACHE[cache_key] = heuristic
            return heuristic

    return None


def get_registered_triton_heuristic(
    heuristic_name: str, device_type: str
) -> Callable[..., Any]:
    heuristic = get_triton_heuristic(heuristic_name, device_type)
    if heuristic is None:
        raise NotImplementedError(
            f"No Triton heuristic registered for '{heuristic_name}' on '{device_type}'"
        )
    return heuristic


def clear_registry() -> None:
    """Clear all registered heuristics and cached lookups."""
    _TRITON_HEURISTIC_REGISTRY.clear()
    _HEURISTIC_CACHE.clear()
