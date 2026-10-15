# mypy: allow-untyped-defs
"""Experimental regional AOTInductor marking and discovery API."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch.nn as nn


_REGION_CONFIG_ATTR = "_torch_aoti_region_config"

__all__ = [
    "RegionConfig",
    "RegionResult",
    "_find_marked_regions",
    "mark_region",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class AOTIRegionError(RuntimeError):
    """Base class for all regional AOTI errors."""

    def __init__(self, message: str, result=None) -> None:
        super().__init__(message)
        self.result = result


class AOTIRegionDiscoveryError(AOTIRegionError):
    """Raised when selected regions cannot be represented safely."""


# ---------------------------------------------------------------------------
# Config / info dataclasses
# ---------------------------------------------------------------------------
class _UnsetType:
    pass


_UNSET = _UnsetType()


@dataclasses.dataclass(init=False)
class RegionConfig:
    """Artifact-related configuration for one selected region."""

    dynamic_shapes: dict[str, Any] | None | _UnsetType
    inductor_options: dict[str, Any] | None | _UnsetType
    rtol: float | None | _UnsetType
    atol: float | None | _UnsetType

    def __init__(
        self,
        dynamic_shapes=_UNSET,
        inductor_options=_UNSET,
        rtol=_UNSET,
        atol=_UNSET,
    ) -> None:
        self.dynamic_shapes = dynamic_shapes
        self.inductor_options = inductor_options
        self.rtol = rtol
        self.atol = atol


@dataclasses.dataclass
class RegionResult:
    """One marked region and everything recorded about it.

    Attributes:
        name: Path of the module in the model (``""`` for the root).
        module: The marked ``nn.Module`` instance.
        config: Effective ``RegionConfig`` (class mark, instance mark, defaults).
        aliases: Every path at which the same module instance appears.
    """

    name: str
    module: nn.Module = dataclasses.field(repr=False)
    config: RegionConfig = dataclasses.field(repr=False)
    aliases: list[str] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Marking / discovery
# ---------------------------------------------------------------------------
_CONFIG_DEFAULTS = {
    "dynamic_shapes": None,
    "inductor_options": None,
    "rtol": 1e-3,
    "atol": 1e-3,
}


def _validate_config(config: RegionConfig | None) -> RegionConfig:
    if config is None:
        return RegionConfig()
    if not isinstance(config, RegionConfig):
        raise TypeError(f"config must be RegionConfig, got {type(config)!r}")
    return config


def mark_region(module=None, *, config: RegionConfig | None = None):
    """Mark one module instance, or exactly one module class, as a region.

    A marked module nested inside another marked module is not a separate
    region: the outermost mark wins, so marking the root compiles the whole
    model as one region.
    """
    config = _validate_config(config)

    def decorate(obj):
        if isinstance(obj, type):
            if not issubclass(obj, nn.Module):
                raise TypeError("mark_region can decorate only nn.Module classes")
            # Class marks apply to exactly this class: _get_region_config reads
            # type(module).__dict__, so subclasses do not inherit them.
            setattr(obj, _REGION_CONFIG_ATTR, config)
            return obj
        if not isinstance(obj, nn.Module):
            raise TypeError(f"mark_region expects an nn.Module, got {type(obj)!r}")
        # Instance marks bypass nn.Module.__setattr__ and never touch the class.
        obj.__dict__[_REGION_CONFIG_ATTR] = config
        return obj

    return decorate if module is None else decorate(module)


def _merge_configs(
    class_config: RegionConfig | None, instance_config: RegionConfig | None
) -> RegionConfig:
    values = {}
    for field, default in _CONFIG_DEFAULTS.items():
        value = getattr(instance_config, field, _UNSET)
        if value is _UNSET:
            value = getattr(class_config, field, _UNSET)
        values[field] = default if value is _UNSET else value
    return RegionConfig(**values)


def _get_region_config(module: nn.Module) -> RegionConfig | None:
    instance_config = module.__dict__.get(_REGION_CONFIG_ATTR)
    class_config = type(module).__dict__.get(_REGION_CONFIG_ATTR)
    if not isinstance(instance_config, RegionConfig):
        instance_config = None
    if not isinstance(class_config, RegionConfig):
        class_config = None
    if instance_config is None and class_config is None:
        return None
    return _merge_configs(class_config, instance_config)


def _is_nested_under(name: str, ancestors: list[str]) -> bool:
    """True if ``name`` is a strict descendant of any name in ``ancestors``."""
    for other in ancestors:
        if other == name:
            continue
        if other == "":
            # Root is an ancestor of every non-root module.
            return True
        if name.startswith(other + "."):
            return True
    return False


def _find_marked_regions(model: nn.Module) -> list[RegionResult]:
    """Walk ``model.named_modules()`` and return only top-level marked regions.

    Nested marked regions are skipped in favor of their marked ancestor so each
    tensor is compiled at most once.
    """
    marked: dict[int, tuple[nn.Module, RegionConfig, list[str]]] = {}
    for name, mod in model.named_modules(remove_duplicate=False):
        cfg = _get_region_config(mod)
        if cfg is None:
            continue
        if id(mod) not in marked:
            marked[id(mod)] = (mod, cfg, [])
        marked[id(mod)][2].append(name)

    all_names = [name for _, _, aliases in marked.values() for name in aliases]
    regions: list[RegionResult] = []
    for mod, cfg, aliases in marked.values():
        others = [name for name in all_names if name not in aliases]
        nested = [_is_nested_under(name, others) for name in aliases]
        # Reachable both inside a marked ancestor and on its own: it would be
        # compiled twice with shared state, so reject instead of picking a side.
        if any(nested) and not all(nested):
            paths = ", ".join(repr(name or "__root__") for name in aliases)
            raise AOTIRegionDiscoveryError(
                f"Aliased selected module crosses a selected-region boundary: {paths}"
            )
        if all(nested):
            continue
        regions.append(
            RegionResult(name=aliases[0], module=mod, config=cfg, aliases=aliases)
        )
    return regions
