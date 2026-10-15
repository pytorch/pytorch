# mypy: allow-untyped-defs
"""Experimental regional AOTInductor marking and discovery API."""

from __future__ import annotations

import copy
import dataclasses
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.utils._ordered_set import OrderedSet


_REGION_CONFIG_ATTR = "_torch_aoti_region_config"

__all__ = [
    "capture_region_inputs",
    "RegionConfig",
    "RegionResult",
    "RegionStatus",
    "_find_marked_regions",
    "mark_region",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class RegionStatus(str, Enum):
    """Outcome reported for a region."""

    NOT_REACHED = "not_reached"


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


class AOTIRegionCaptureError(AOTIRegionError):
    """Raised when example execution cannot establish one region contract."""


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
        status: ``NOT_REACHED`` until capture sees a call, then ``None``.
        captured_calls: ``(args, kwargs)`` of every captured invocation.
    """

    name: str
    module: nn.Module = dataclasses.field(repr=False)
    config: RegionConfig = dataclasses.field(repr=False)
    aliases: list[str] = dataclasses.field(default_factory=list)
    status: RegionStatus | None = RegionStatus.NOT_REACHED
    captured_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = dataclasses.field(
        default_factory=list, repr=False
    )

    @property
    def captured_args(self) -> tuple[Any, ...] | None:
        return self.captured_calls[0][0] if self.captured_calls else None

    @property
    def captured_kwargs(self) -> dict[str, Any] | None:
        return self.captured_calls[0][1] if self.captured_calls else None


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


# ---------------------------------------------------------------------------
# Input capture
# ---------------------------------------------------------------------------
_IDENTITY_INSENSITIVE_TYPES = (type(None), bool, int, float, complex, str, bytes)


def _object_children(value) -> list[Any]:
    children = []
    if isinstance(value, dict):
        children.extend(child for pair in value.items() for child in pair)
    elif isinstance(value, (tuple, list, set, frozenset)):  # noqa: set_linter
        children.extend(value)
    try:
        children.extend(vars(value).values())
    except TypeError:
        pass
    children.extend(slot_value for _, slot_value in _slot_values(value))
    return children


def _object_graph_aliases(
    value,
) -> tuple[OrderedSet[int], OrderedSet[tuple[torch.device, int]]]:
    identities: OrderedSet[int] = OrderedSet()
    storages: OrderedSet[tuple[torch.device, int]] = OrderedSet()

    def visit(item) -> None:
        if type(item) in _IDENTITY_INSENSITIVE_TYPES or id(item) in identities:
            return
        identities.add(id(item))
        if isinstance(item, torch.Tensor):
            try:
                storages.add((item.device, item.untyped_storage()._cdata))
            except RuntimeError:
                pass
            return
        for child in _object_children(item):
            visit(child)

    visit(value)
    return identities, storages


def _clone_preserving_layout(
    x: torch.Tensor, storages: dict[tuple[torch.device, int], torch.Tensor]
) -> torch.Tensor:
    source = x.detach()
    is_conj = source.is_conj()
    is_neg = source.is_neg()
    if is_conj:
        source = source.conj()
    if is_neg:
        source = torch._neg_view(source)
    if x.layout == torch.strided:
        # Strides are never negative, so this many elements covers all of x.
        storage_size = 0
        if x.numel() != 0:
            storage_size = (
                x.storage_offset()
                + 1
                + sum(
                    (size - 1) * stride
                    for size, stride in zip(x.shape, x.stride(), strict=True)
                )
            )
        try:
            storage_key = (x.device, x.untyped_storage()._cdata)
            buffer = torch.empty(storage_size, dtype=x.dtype, device=x.device)
            if storage_size:
                buffer.copy_(source.as_strided((storage_size,), (1,), 0))
            y = buffer.as_strided(x.size(), x.stride(), x.storage_offset())
        except RuntimeError as e:
            raise AOTIRegionCaptureError(
                "clone_inputs=True cannot preserve the input tensor "
                "stride and storage offset"
            ) from e
        if storage_key in storages:
            raise AOTIRegionCaptureError(
                "clone_inputs=True cannot preserve aliasing between "
                "distinct tensor views that share storage"
            )
        storages[storage_key] = x
    else:
        y = source.clone()
    if is_neg:
        y = torch._neg_view(y)
    if is_conj:
        y = y.conj()
    return y


def _detach_inputs(obj, clone: bool):
    storages: dict[tuple[torch.device, int], torch.Tensor] = {}
    memo: dict[int, Any] = {}

    for x in pytree.tree_leaves(obj):
        if not isinstance(x, torch.Tensor) or id(x) in memo:
            continue
        y = _clone_preserving_layout(x, storages) if clone else x.detach()
        if isinstance(x, nn.Parameter):
            y = nn.Parameter(y, requires_grad=x.requires_grad)
        elif x.requires_grad:
            y.requires_grad_(True)
        memo[id(x)] = y

    try:
        return copy.deepcopy(obj, memo)
    except Exception as e:
        raise AOTIRegionCaptureError("Could not snapshot the input pytree") from e


def capture_region_inputs(
    model: nn.Module,
    regions: list[RegionResult],
    example_inputs,
    example_kwargs=None,
    *,
    clone_inputs: bool = False,
) -> list[RegionResult]:
    """Run one eager forward, capturing (args, kwargs) into each reached region.

    The outer model behavior is unchanged: this only observes inputs via
    forward pre-hooks and runs under ``torch.no_grad()``. Reached regions are
    left pending (``status is None``); regions that never ran are NOT_REACHED.

    Args:
        clone_inputs: copy captured tensors instead of only detaching them.
            Without it, captures share storage with the live inputs, so an
            in-place update after the region runs changes the captured values.
    """
    example_inputs = tuple(example_inputs) if example_inputs is not None else ()
    example_kwargs = dict(example_kwargs) if example_kwargs else {}

    captures: dict[int, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {}
    handles = []

    for r in regions:
        r.captured_calls = []
        r.status = RegionStatus.NOT_REACHED

    def make_hook():
        def hook(module, args, kwargs):
            module_id = id(module)
            source_tree = (args, kwargs)
            detached_args, detached_kwargs = _detach_inputs(
                (args, kwargs), clone_inputs
            )
            source_ids, source_storages = _object_graph_aliases(source_tree)
            module_ids, module_storages = _object_graph_aliases(module)
            captured_ids, captured_storages = _object_graph_aliases(
                (detached_args, detached_kwargs)
            )
            if not (source_ids & module_ids).issubset(captured_ids) or not (
                source_storages & module_storages
            ).issubset(captured_storages):
                raise AOTIRegionCaptureError(
                    "Could not snapshot invocation inputs without changing "
                    "object identity or storage relationships with region state"
                )
            captures.setdefault(module_id, []).append(
                (
                    tuple(detached_args),
                    dict(detached_kwargs),
                )
            )

        return hook

    hooked_module_ids: OrderedSet[int] = OrderedSet()
    for r in regions:
        module_id = id(r.module)
        if module_id in hooked_module_ids:
            continue
        hooked_module_ids.add(module_id)
        handles.append(
            r.module.register_forward_pre_hook(make_hook(), with_kwargs=True)
        )

    try:
        with torch.no_grad():
            model(*example_inputs, **example_kwargs)
    finally:
        for h in handles:
            h.remove()

    for r in regions:
        if id(r.module) in captures:
            r.captured_calls = list(captures[id(r.module)])
            r.status = None
        else:
            r.status = RegionStatus.NOT_REACHED
    return regions


def _resolved_slot_names(cls: type) -> tuple[str, ...]:
    slots = getattr(cls, "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    resolved = []
    class_name = cls.__name__.lstrip("_")
    for slot in slots:
        if slot in ("__dict__", "__weakref__"):
            continue
        if slot.startswith("__") and not slot.endswith("__") and class_name:
            slot = f"_{class_name}{slot}"
        resolved.append(slot)
    return tuple(resolved)


def _slot_values(value) -> list[tuple[str, Any]]:
    """(name, value) for each set ``__slots__`` entry, read without ``__getattr__``."""
    values = []
    for cls in type(value).__mro__:
        for name in _resolved_slot_names(cls):
            descriptor = cls.__dict__.get(name)
            if descriptor is None:
                continue
            try:
                values.append((name, descriptor.__get__(value, cls)))
            except AttributeError:
                pass
    return values
