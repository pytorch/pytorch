# mypy: allow-untyped-defs
"""Experimental regional AOTInductor marking and discovery API."""

from __future__ import annotations

import copy
import dataclasses
import inspect
import math
import types
from enum import Enum
from typing import Any, overload, TYPE_CHECKING, TypeVar

import torch
import torch.nn as nn
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.jit import ScriptModule
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


_REGION_CONFIG_ATTR = "_torch_aoti_region_config"

__all__ = [
    "AOTIRegionDiscoveryError",
    "AOTIRegionError",
    "capture_region_inputs",
    "RegionConfig",
    "RegionResult",
    "RegionStatus",
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


class AOTIRegionDiscoveryError(AOTIRegionError):
    """Raised when selected regions cannot be represented safely."""


class AOTIRegionCaptureError(AOTIRegionError):
    """Raised when example execution cannot establish one region contract."""


# ---------------------------------------------------------------------------
# Config / info dataclasses
# ---------------------------------------------------------------------------
class _UnsetType(Enum):
    UNSET = "unset"

    def __repr__(self) -> str:
        return "_UNSET"


_UNSET = _UnsetType.UNSET


def _convert_containers(value: Any, dict_type: type, list_type: type) -> Any:
    """Copy nested dicts, lists and tuples; leaves (e.g. ``Dim`` objects) stay shared."""
    if isinstance(value, dict):
        return dict_type(
            {
                key: _convert_containers(item, dict_type, list_type)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return list_type(
            _convert_containers(item, dict_type, list_type) for item in value
        )
    if isinstance(value, tuple):
        items = [_convert_containers(item, dict_type, list_type) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
    return value


def _freeze(value: Any) -> Any:
    """Copy nested dicts and lists into ``immutable_dict`` and ``immutable_list``."""
    return _convert_containers(value, immutable_dict, immutable_list)


def _thaw(value: Any) -> Any:
    """Copy a frozen ``RegionConfig`` value back into plain dicts and lists.

    Use it before handing ``dynamic_shapes`` to ``torch.export`` (which matches
    container types against the inputs) or ``inductor_configs`` to
    ``aoti_compile_and_package`` (which writes into the dict it receives).
    """
    return _convert_containers(value, dict, list)


@dataclasses.dataclass(kw_only=True, frozen=True)
class RegionConfig:
    """Artifact-related configuration for one selected region.

    Every field is keyword-only and starts unset. When marks are merged, an
    unset field falls back to the class mark and then to its default; an
    explicit ``None`` is a value and overrides the class mark. Values are
    validated at construction. The config is frozen all the way down: dicts and
    lists in its values are stored as ``immutable_dict`` and ``immutable_list``
    copies (leaves such as ``Dim`` objects stay shared), so neither later
    changes to the caller's dicts nor edits through the config affect it, and
    the config is hashable when its leaves are. ``torch.export`` and
    ``aoti_compile_and_package`` need plain containers, so the compile path
    hands them plain copies made by ``_thaw``.

    Attributes:
        dynamic_shapes: Dynamic-shape specs keyed by ``forward`` argument name,
            as in the dict form of ``torch.export.export(dynamic_shapes=...)``.
            Defaults to ``None``.
        inductor_configs: Inductor config overrides passed to
            ``aoti_compile_and_package``, which resolves the keys when the
            region is compiled; an unknown key fails that region's compile.
            Defaults to ``None``.
        rtol: Relative tolerance for the compiled-versus-eager accuracy check.
            Defaults to ``1e-3``.
        atol: Absolute tolerance for the compiled-versus-eager accuracy check.
            Defaults to ``1e-3``.
    """

    dynamic_shapes: dict[str, Any] | None | _UnsetType = _UNSET
    inductor_configs: dict[str, Any] | None | _UnsetType = _UNSET
    rtol: float | None | _UnsetType = _UNSET
    atol: float | None | _UnsetType = _UNSET

    def __post_init__(self) -> None:
        dynamic_shapes = self.dynamic_shapes
        if dynamic_shapes is not _UNSET and dynamic_shapes is not None:
            if not isinstance(dynamic_shapes, dict) or not all(
                isinstance(key, str) for key in dynamic_shapes
            ):
                raise TypeError(
                    "RegionConfig.dynamic_shapes must be a dict keyed by forward "
                    f"argument name, got {dynamic_shapes!r}"
                )
            object.__setattr__(self, "dynamic_shapes", _freeze(dynamic_shapes))
        inductor_configs = self.inductor_configs
        if inductor_configs is not _UNSET and inductor_configs is not None:
            if not isinstance(inductor_configs, dict) or not all(
                isinstance(key, str) for key in inductor_configs
            ):
                raise TypeError(
                    "RegionConfig.inductor_configs must be a dict keyed by config "
                    f"name, got {inductor_configs!r}"
                )
            object.__setattr__(self, "inductor_configs", _freeze(inductor_configs))
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if value is _UNSET or value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"RegionConfig.{name} must be a number, got {type(value).__name__}"
                )
            if math.isnan(value) or value < 0:
                raise ValueError(f"RegionConfig.{name} must be >= 0, got {value!r}")


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
    "inductor_configs": None,
    "rtol": 1e-3,
    "atol": 1e-3,
}


def _validate_config(config: object) -> RegionConfig:
    if config is None:
        return RegionConfig()
    if not isinstance(config, RegionConfig):
        raise TypeError(f"config must be RegionConfig, got {type(config)!r}")
    return config


_Marked = TypeVar("_Marked", bound="nn.Module | type[nn.Module]")


def _reject_script_module(cls: type) -> None:
    if issubclass(cls, ScriptModule):
        raise TypeError(
            f"mark_region does not support TorchScript modules, got {cls.__name__}; "
            "torch.export cannot export them, so mark the eager module instead"
        )


def _class_forward(cls: type) -> Any:
    """The raw ``forward`` entry in ``cls``'s MRO, without running descriptors."""
    return inspect.getattr_static(cls, "forward", None)


@overload
def mark_region(
    module: _Marked, /, *, config: RegionConfig | None = None
) -> _Marked: ...


@overload
def mark_region(
    *, config: RegionConfig | None = None
) -> Callable[[_Marked], _Marked]: ...


def mark_region(module: Any = _UNSET, /, *, config: RegionConfig | None = None) -> Any:
    """Mark an ``nn.Module`` instance or an ``nn.Module`` class as a region.

    Call it directly, ``mark_region(module_or_class, config=...)``, or use it
    as a class decorator, bare (``@mark_region``) or with arguments
    (``@mark_region(config=...)``).

    Args:
        module: The ``nn.Module`` instance or subclass to mark.
        config: Settings for the region. Unset fields fall back to the class
            mark, then to the ``RegionConfig`` defaults.

    Returns:
        The marked instance or class, or a decorator when ``module`` is
        omitted.

    Marking the same instance or class again replaces its earlier config. An
    instance mark affects only that instance. A class mark is process-wide: it
    applies to every existing and future instance and cannot be removed, so
    ``mark_region(nn.Linear)`` makes every ``nn.Linear`` a region.

    A class mark applies to every class whose ``forward`` is the marked
    class's ``forward``: the class itself, subclasses that inherit that
    ``forward``, and the classes swapped in by ``fully_shard``, ``replicate``
    and parametrizations. Marks on such classes merge field by field, nearest
    class first. A subclass that defines its own ``forward`` (even one that
    only calls ``super().forward()``) is a region only if it is marked, and a
    mark on a mixin that does not supply the resolved ``forward`` never
    applies, whatever the base order. Marking a class without a ``forward``
    raises, and so does marking an instance whose class has no ``forward``
    (such as ``nn.ModuleList``; mark its elements instead). TorchScript modules,
    scripted or traced, cannot be marked either, because ``torch.export``
    cannot export them.

    A marked module nested inside another marked module is not a separate
    region: the outermost mark wins, so marking the root makes the whole model
    one region.

    An instance mark is stored on the instance, so it travels with the module
    through ``copy.deepcopy`` and pickling. A module whose mark holds a derived
    dim such as ``2 * Dim("b")`` therefore cannot be pickled, because derived
    dims cannot.
    """
    config = _validate_config(config)

    def decorate(obj: _Marked) -> _Marked:
        if isinstance(obj, type):
            if not issubclass(obj, nn.Module):
                raise TypeError("mark_region can decorate only nn.Module classes")
            _reject_script_module(obj)
            if _class_forward(obj) is nn.Module.forward:
                raise TypeError(
                    f"mark_region needs a module with a forward, got {obj.__name__}"
                )
            setattr(obj, _REGION_CONFIG_ATTR, config)
            return obj
        if not isinstance(obj, nn.Module):
            raise TypeError(f"mark_region expects an nn.Module, got {type(obj)!r}")
        instance: nn.Module = obj
        cls = type(instance)
        _reject_script_module(cls)
        if (
            "forward" not in instance.__dict__
            and _class_forward(cls) is nn.Module.forward
        ):
            raise TypeError(
                f"mark_region needs a module with a forward, got {cls.__name__}"
            )
        # Instance marks bypass nn.Module.__setattr__ and never touch the class.
        obj.__dict__[_REGION_CONFIG_ATTR] = config
        return obj

    return decorate if module is _UNSET else decorate(module)


def _first_set_values(*configs: RegionConfig | None) -> dict[str, Any]:
    """Per field, the first value that is set across ``configs``, else unset."""
    values = {}
    for field in dataclasses.fields(RegionConfig):
        values[field.name] = next(
            (
                getattr(config, field.name)
                for config in configs
                if config is not None and getattr(config, field.name) is not _UNSET
            ),
            _UNSET,
        )
    return values


def _merge_configs(
    class_config: RegionConfig | None, instance_config: RegionConfig | None
) -> RegionConfig:
    values = _first_set_values(instance_config, class_config)
    for name, value in values.items():
        if value is _UNSET:
            values[name] = _CONFIG_DEFAULTS[name]
    return RegionConfig(**values)


def _class_region_config(cls: type) -> RegionConfig | None:
    """Class marks that apply to instances of ``cls``, merged nearest first.

    A mark applies when the marked class resolves to the same ``forward`` as
    ``cls``. Class swaps by ``fully_shard``, ``replicate`` and parametrizations
    add no ``forward``, so the original class's mark still applies.
    """
    marked: list[tuple[type, RegionConfig]] = []
    for klass in cls.__mro__:
        config = klass.__dict__.get(_REGION_CONFIG_ATTR)
        if isinstance(config, RegionConfig):
            marked.append((klass, config))
    if not marked:
        return None
    forward = _class_forward(cls)
    configs = [config for klass, config in marked if _class_forward(klass) is forward]
    if not configs:
        return None
    return RegionConfig(**_first_set_values(*configs))


def _get_region_config(module: nn.Module) -> RegionConfig | None:
    instance_config = module.__dict__.get(_REGION_CONFIG_ATTR)
    class_config = _class_region_config(type(module))
    if not isinstance(instance_config, RegionConfig):
        instance_config = None
    if instance_config is None and class_config is None:
        return None
    return _merge_configs(class_config, instance_config)


def _module_graph(
    model: nn.Module,
) -> tuple[
    dict[int, nn.Module],
    dict[int, list[tuple[str, nn.Module]]],
    dict[int, RegionConfig],
]:
    """Every module reachable from ``model``, its children, and the marked ones.

    Each module is visited once, so shared submodules and cycles are handled
    as ``named_modules()`` handles them.
    """
    modules: dict[int, nn.Module] = {}
    children: dict[int, list[tuple[str, nn.Module]]] = {}
    configs: dict[int, RegionConfig] = {}
    stack = [model]
    while stack:
        mod = stack.pop()
        if id(mod) in modules:
            continue
        modules[id(mod)] = mod
        children[id(mod)] = [
            (name, child) for name, child in mod._modules.items() if child is not None
        ]
        cfg = _get_region_config(mod)
        if cfg is not None:
            configs[id(mod)] = cfg
        stack.extend(child for _, child in reversed(children[id(mod)]))
    return modules, children, configs


def _nested_in_other_region(
    children: dict[int, list[tuple[str, nn.Module]]],
    configs: dict[int, RegionConfig],
) -> OrderedSet[int]:
    """Marked modules reachable from a marked module other than themselves."""
    # Two distinct containing regions per module are enough to tell whether one
    # differs from the module itself, and the cap keeps the walk linear.
    owners: dict[int, list[int]] = {}
    stack = [(child, owner) for owner in configs for _, child in children[owner]]
    while stack:
        mod, owner = stack.pop()
        seen = owners.setdefault(id(mod), [])
        if owner in seen or len(seen) == 2:
            continue
        seen.append(owner)
        stack.extend((child, owner) for _, child in children[id(mod)])
    return OrderedSet(
        marked
        for marked in configs
        if any(owner != marked for owner in owners.get(marked, ()))
    )


def _top_level_paths(
    model: nn.Module,
    children: dict[int, list[tuple[str, nn.Module]]],
    configs: dict[int, RegionConfig],
) -> dict[int, list[str]]:
    """Paths from ``model`` to each marked module that pass no other marked module.

    Only modules with a marked descendant are entered, so the walk costs the
    number of these paths rather than the number of paths in the whole model.
    """
    parents: dict[int, list[int]] = {}
    for parent, kids in children.items():
        for _, child in kids:
            parents.setdefault(id(child), []).append(parent)
    leads_to_marked = OrderedSet(configs)
    pending = list(configs)
    while pending:
        for parent in parents.get(pending.pop(), ()):
            if parent not in leads_to_marked:
                leads_to_marked.add(parent)
                pending.append(parent)

    paths: dict[int, list[str]] = {}
    stack: list[tuple[str, nn.Module, tuple[int, ...]]] = [("", model, ())]
    while stack:
        name, mod, on_path = stack.pop()
        if id(mod) in configs:
            paths.setdefault(id(mod), []).append(name)
            continue
        on_path = (*on_path, id(mod))
        stack.extend(
            (f"{name}.{child_name}" if name else child_name, child, on_path)
            for child_name, child in reversed(children[id(mod)])
            if id(child) in leads_to_marked and id(child) not in on_path
        )
    return paths


def _find_marked_regions(model: nn.Module) -> list[RegionResult]:
    """Return the top-level marked regions of ``model``.

    Nested marked regions are skipped in favor of their marked ancestor. A
    marked module that is both nested in another region and a top-level region
    through an alias is rejected. Sharing between regions is not otherwise
    tracked: a submodule, marked or not, that is reachable from two regions is
    part of both.

    Each module is visited once; beyond that, the walk costs the alias paths of
    the marked modules.
    """
    modules, children, configs = _module_graph(model)
    nested = _nested_in_other_region(children, configs)
    regions: list[RegionResult] = []
    for owner, aliases in _top_level_paths(model, children, configs).items():
        if owner in nested:
            paths = ", ".join(repr(name or "__root__") for name in aliases)
            raise AOTIRegionDiscoveryError(
                "Aliased selected module crosses a selected-region boundary: it is "
                f"a top-level region at {paths} and also nested in another region"
            )
        regions.append(
            RegionResult(
                name=aliases[0],
                module=modules[owner],
                config=configs[owner],
                aliases=aliases,
            )
        )
    return regions


# ---------------------------------------------------------------------------
# Input capture
# ---------------------------------------------------------------------------
_IDENTITY_INSENSITIVE_TYPES = (type(None), bool, int, float, complex, str, bytes)
_OPAQUE_TYPES = (
    types.ModuleType,
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
)


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


def _walk_object_graph(value) -> Iterator[Any]:
    """Yield every identity-sensitive object reachable from ``value`` once.

    Tensors and ``_OPAQUE_TYPES`` are yielded but not entered.
    """
    seen: OrderedSet[int] = OrderedSet()
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) in _IDENTITY_INSENSITIVE_TYPES or id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        if not isinstance(item, (torch.Tensor, *_OPAQUE_TYPES)):
            stack.extend(reversed(_object_children(item)))


def _tensor_storages(t: torch.Tensor) -> list[tuple[torch.device, int]]:
    if t.layout == torch.sparse_coo:
        parts = (t._indices(), t._values())
    elif t.layout in (torch.sparse_csr, torch.sparse_bsr):
        parts = (t.crow_indices(), t.col_indices(), t.values())
    elif t.layout in (torch.sparse_csc, torch.sparse_bsc):
        parts = (t.ccol_indices(), t.row_indices(), t.values())
    else:
        parts = (t,)
    storages = []
    for part in parts:
        try:
            storages.append((part.device, part.untyped_storage()._cdata))
        except RuntimeError:
            pass
    return storages


def _object_graph_aliases(
    value,
) -> tuple[OrderedSet[int], OrderedSet[tuple[torch.device, int]]]:
    identities: OrderedSet[int] = OrderedSet()
    storages: OrderedSet[tuple[torch.device, int]] = OrderedSet()
    for item in _walk_object_graph(value):
        identities.add(id(item))
        if isinstance(item, torch.Tensor):
            storages.update(_tensor_storages(item))
    return identities, storages


def _clone_preserving_layout(
    x: torch.Tensor, storages: dict[tuple[torch.device, int], torch.Tensor]
) -> torch.Tensor:
    if type(x).__torch_dispatch__ is not torch.Tensor.__torch_dispatch__:
        raise AOTIRegionCaptureError(
            f"clone_inputs=True does not support tensor subclass {type(x).__name__}"
        )
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
            if type(x) not in (torch.Tensor, nn.Parameter):
                y = y.as_subclass(type(x))
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


def _detach_inputs(obj, clone: bool, keep_ids: OrderedSet[int] | None = None):
    """Deep-copy ``obj`` with every reachable tensor detached (or cloned).

    Non-tensor objects whose ids are in ``keep_ids`` keep their identity.
    """
    storages: dict[tuple[torch.device, int], torch.Tensor] = {}
    memo: dict[int, Any] = {}

    for x in _walk_object_graph(obj):
        if not isinstance(x, torch.Tensor):
            if keep_ids is not None and id(x) in keep_ids:
                memo[id(x)] = x
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
    forward pre-hooks and runs under ``torch.no_grad()``. Inputs are recorded
    before the region's own forward pre-hooks run, so replaying them through
    the module applies those hooks once. Reached regions are left pending
    (``status is None``); regions that never ran are NOT_REACHED.

    Args:
        clone_inputs: copy captured tensors instead of only detaching them.
            Without it, captures share storage with the live inputs, and
            objects also held by region state stay live, so an in-place
            update after the region runs changes the captured values. With
            it, an input that shares an object or storage with region state
            is rejected.
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
            source_ids, source_storages = _object_graph_aliases(source_tree)
            module_ids, module_storages = _object_graph_aliases(module)
            shared_ids = source_ids & module_ids
            # Without cloning, captures already share tensor storage with the
            # live inputs; keep shared non-tensor objects live for the same reason.
            detached_args, detached_kwargs = _detach_inputs(
                source_tree, clone_inputs, None if clone_inputs else shared_ids
            )
            captured_ids, captured_storages = _object_graph_aliases(
                (detached_args, detached_kwargs)
            )
            if not shared_ids.issubset(captured_ids) or not (
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
            r.module.register_forward_pre_hook(
                make_hook(), prepend=True, with_kwargs=True
            )
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
