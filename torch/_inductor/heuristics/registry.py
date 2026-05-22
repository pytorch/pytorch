"""
Heuristic registry system for PyTorch Inductor.

Provides centralized registration for both:
- Template heuristics (compile-time, keyed by template_name/device_type/op_name)
- Codegen heuristics (runtime, keyed by name/device_type)

Both share one underlying registry dict and cascading fallback lookup.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator

    from .template.base import TemplateConfigHeuristics


_REGISTRY: dict[tuple[str | None, ...], Any] = {}
# Alias so tests can snapshot/restore the shared registry via the old name.
# This intentionally covers both template and codegen entries.
_TEMPLATE_HEURISTIC_REGISTRY = _REGISTRY
_CACHE: dict[tuple[str | None, ...], Any] = {}

log = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Shared internals
# -----------------------------------------------------------------


def _register(
    key: tuple[str | None, ...],
    target: Any,
) -> None:
    _REGISTRY[key] = target


def _lookup(name: str, device_type: str, op_name: str | None) -> Any | None:
    """Cascading fallback lookup: (name, device, op) -> (name, None, op) -> (name, device, None) -> (name, None, None)."""
    keys = [
        (name, device_type, op_name),
        (name, None, op_name),
        (name, device_type, None),
        (name, None, None),
    ]
    for key in keys:
        if key in _REGISTRY:
            return _REGISTRY[key]
    return None


# -----------------------------------------------------------------
# Template heuristic API
# -----------------------------------------------------------------


def register_template_heuristic(
    template_name: str,
    device_type: str | None,
    register: bool = True,
    op_name: str | None = None,
) -> Any:
    """
    Decorator to register template heuristic classes.

    Args:
        template_name: Name of the template (e.g., "mm", "bmm", "scaled_mm")
        device_type: Device type ("cuda", "cpu", "xpu").
            None means applicable to all device types.
        register: Whether to register this heuristic.
        op_name: Operator name for per-op specialization.

    Example:
        @register_template_heuristic("mm", "cuda", register=torch.version.hip is None)
        class CUDAMMTemplateConfigHeuristic(CUDAConfigHeuristic):
            pass
    """

    def decorator(
        cls: type[TemplateConfigHeuristics],
    ) -> type[TemplateConfigHeuristics]:
        if register:
            key: tuple[str | None, ...] = (template_name, device_type, op_name)
            _register(key, cls)
            log.info(
                "Registered template heuristic: %s for template_name=%s, device_type=%s, op_name=%s",
                cls.__name__,
                template_name,
                device_type,
                op_name,
            )
        return cls

    return decorator


def get_template_heuristic(
    template_name: str, device_type: str, op_name: str
) -> TemplateConfigHeuristics:
    """
    Retrieve a template heuristic instance for the given template and device type.

    Returns a cached instance. Falls back to TemplateConfigHeuristics() if not found.
    """
    cache_key: tuple[str | None, ...] = (template_name, device_type, op_name)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    heuristic_class = _lookup(template_name, device_type, op_name)

    if heuristic_class is None:
        from .template.base import TemplateConfigHeuristics as _Base

        log.error(
            "No template heuristic found - template_name=%s, device_type=%s, op_name=%s. "
            "Available: %s. Using fallback.",
            template_name,
            device_type,
            op_name,
            list(_REGISTRY.keys()),
        )
        return _Base()

    instance = heuristic_class()
    _CACHE[cache_key] = instance
    return instance


def get_registered_heuristic_class(
    template_name: str, device_type: str, op_name: str
) -> None | type[TemplateConfigHeuristics]:
    """Get the registered class (not instance) for the given template/device/op."""
    return _lookup(template_name, device_type, op_name)


# -----------------------------------------------------------------
# Codegen heuristic API
# -----------------------------------------------------------------


class CodegenConfigHeuristics:
    """Base class for codegen heuristics (pointwise, reduction, etc.)."""

    def get_configs(
        self,
        size_hints: dict[str, int],
        bs: int,
        triton_config_fn: Any,
        hinted_configs: list[Any],
        tile_hint: Any | None = None,
        inductor_meta: dict[str, Any] | None = None,
    ) -> list[Any]:
        raise NotImplementedError


def register_codegen_heuristic(
    name: str,
    device_type: str | None = None,
    register: bool = True,
) -> Any:
    """
    Decorator to register a codegen heuristic class.

    Args:
        name: Heuristic name (e.g., "pointwise", "reduction").
        device_type: Device type ("hip", "xpu", "cuda").
            None means default/fallback for all devices.
        register: Whether to actually register.

    Example:
        @register_codegen_heuristic("pointwise", "hip", register=torch.version.hip is not None)
        class ROCmPointwiseHeuristic(CodegenConfigHeuristics):
            ...
    """

    def decorator(cls: type[CodegenConfigHeuristics]) -> type[CodegenConfigHeuristics]:
        if register:
            key: tuple[str | None, ...] = (name, device_type, None)
            _register(key, cls)
            log.info(
                "Registered codegen heuristic: %s for name=%s, device_type=%s",
                cls.__name__,
                name,
                device_type,
            )
        return cls

    return decorator


def get_codegen_heuristic(name: str, device_type: str) -> CodegenConfigHeuristics:
    """
    Retrieve a codegen heuristic instance for the given name and device type.

    Returns a cached instance. Uses cascading fallback:
        (name, device_type, None) -> (name, None, None)
    """
    cache_key: tuple[str | None, ...] = (name, device_type, None)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    heuristic_class = _lookup(name, device_type, None)

    if heuristic_class is None:
        # Lazily import codegen heuristics to trigger registration
        import torch._inductor.heuristics.triton_codegen  # noqa: F401

        heuristic_class = _lookup(name, device_type, None)

    if heuristic_class is None:
        raise ValueError(
            f"No codegen heuristic found - name={name}, device_type={device_type}. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    instance = heuristic_class()
    _CACHE[cache_key] = instance
    return instance


# -----------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------


def clear_registry() -> None:
    """Clear all registered heuristics. Primarily for testing."""
    _REGISTRY.clear()
    _CACHE.clear()


@contextlib.contextmanager
def override_template_heuristics(
    device_type: str,
    template_op_pairs: list[tuple[str, str]],
    override_heuristic_class: type[TemplateConfigHeuristics] | None = None,
) -> Iterator[None]:
    """
    Context manager to temporarily override template heuristics.

    Args:
        device_type: Device type ("cuda", "cpu", "xpu")
        template_op_pairs: List of (template_name, op_name) pairs to override.
        override_heuristic_class: Heuristic class to use for the override.
    """
    if override_heuristic_class is None:
        from .template.base import TemplateConfigHeuristics as _Base

        override_heuristic_class = _Base
    original_entries: dict[tuple[str | None, ...], Any] = {}
    new_keys: list[tuple[str | None, ...]] = []
    # Clears the shared cache (both template and codegen instances).
    # Codegen instances are stateless so re-creation is cheap.
    _CACHE.clear()
    try:
        for template_name, op_name in template_op_pairs:
            assert op_name is not None
            key: tuple[str | None, ...] = (template_name, device_type, op_name)
            if key in _REGISTRY:
                original_entries[key] = _REGISTRY[key]
            _REGISTRY[key] = override_heuristic_class
            new_keys.append(key)
        yield
    finally:
        for key in new_keys:
            _REGISTRY.pop(key, None)
            if key in original_entries:
                _REGISTRY[key] = original_entries[key]
        # Same shared-cache clear on exit; see entry comment above.
        _CACHE.clear()
