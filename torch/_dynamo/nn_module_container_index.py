"""Track mutable nn.Module state used to index builtin module containers.

The protocol has two analysis passes.  On the first pass, container subscripts
record their bytecode site and inlined translator ancestry.  If SideEffects
later observes a value-changing store to the exact selector Source, it promotes
the deepest frame shared by the lookup and mutation into a retry target and
restarts analysis.

On retry, a target in the lookup frame is raised from ``mp_subscript_impl`` so
the exact Source can be checked.  A target in an inlined ancestor is raised by
the bytecode dispatcher before re-entering the lookup helper.  The ancestor
locator distinguishes separate invocations of shared code.  The cache resolves
the selector-owning module Source at frame entry and keys on that module's exact
weak identity, so mutable list/dict roots do not prevent caching.

The resulting strategy is intentionally recursive SKIP.  If only the outer
frame were skipped, an eagerly executed helper containing the lookup could be
intercepted independently, specialize the selector, and recreate the original
recompile-limit churn.  This also makes descendants of the skipped frame eager;
callers can still compile around that frame.
"""

from __future__ import annotations

import dataclasses
import types
import weakref
from typing import Any, NoReturn, TYPE_CHECKING, TypeAlias

import torch.nn
from torch._guards import ChainedSource, Source

from . import graph_break_hints
from .exc import NNModuleContainerIndexRestartAnalysis, unimplemented, Unsupported
from .source import (
    AttrSource,
    CellContentsSource,
    DictGetItemSource,
    GetItemSource,
    GlobalSource,
    LocalCellSource,
    LocalSource,
    NNModuleSource,
)
from .types import FrameAction, FrameExecStrategy


if TYPE_CHECKING:
    from .output_graph import OutputGraph
    from .symbolic_convert import InstructionTranslatorBase
    from .variables.base import VariableTracker


FrameSite: TypeAlias = tuple[types.CodeType, int]
_MISSING = object()


BUILTIN_NN_MODULE_CONTAINER_GETITEMS = (
    torch.nn.ModuleDict.__getitem__,
    torch.nn.ModuleList.__getitem__,
    torch.nn.ParameterDict.__getitem__,
    torch.nn.ParameterList.__getitem__,
    torch.nn.Sequential.__getitem__,
)


@dataclasses.dataclass(frozen=True)
class FrameLocator:
    """How a translator frame reaches the module owning the selector."""

    name: str
    source: Source
    is_global: bool


@dataclasses.dataclass(frozen=True)
class IndexFrame:
    site: FrameSite
    locator: FrameLocator | None


@dataclasses.dataclass(frozen=True)
class InstanceDictAccessor:
    name: str


@dataclasses.dataclass(frozen=True)
class GetItemAccessor:
    container_type: type
    index: Any


CacheAccessor: TypeAlias = InstanceDictAccessor | GetItemAccessor


@dataclasses.dataclass(frozen=True)
class CacheLocator:
    """Side-effect-free path from a frame entry to a cache key object."""

    root_name: str
    is_global: bool
    accessors: tuple[CacheAccessor, ...]

    def resolve(
        self,
        f_locals: dict[str, object],
        f_globals: dict[str, object],
    ) -> object | None:
        namespace = f_globals if self.is_global else f_locals
        value = namespace.get(self.root_name, _MISSING)
        if value is _MISSING:
            return None

        for accessor in self.accessors:
            if isinstance(accessor, InstanceDictAccessor):
                try:
                    instance_dict = object.__getattribute__(value, "__dict__")
                except AttributeError:
                    return None
                if type(instance_dict) is not dict:
                    return None
                value = instance_dict.get(accessor.name, _MISSING)
                if value is _MISSING:
                    return None
            elif accessor.container_type is list and type(value) is list:
                try:
                    value = list.__getitem__(value, accessor.index)
                except (IndexError, TypeError):
                    return None
            elif accessor.container_type is tuple and type(value) is tuple:
                try:
                    value = tuple.__getitem__(value, accessor.index)
                except (IndexError, TypeError):
                    return None
            elif accessor.container_type is dict and type(value) is dict:
                try:
                    value = dict.__getitem__(value, accessor.index)
                except (KeyError, TypeError):
                    return None
            else:
                return None
        return value


@dataclasses.dataclass
class IndexCandidate:
    """One index site and its retained leaf-to-root translator ancestry.

    Translator identity distinguishes separate inline invocations of the same
    bytecode site. The per-attempt tracker is cleared in ``trace_frame``'s
    ``finally`` block so these strong references never outlive tracing.
    """

    leaf_tx: InstructionTranslatorBase
    frames: dict[InstructionTranslatorBase, IndexFrame]
    cache_locator: CacheLocator | None


@dataclasses.dataclass(frozen=True)
class IndexTarget:
    """A promoted retry site plus metadata for invocation-specific skipping."""

    source: Source
    source_aware: bool
    locator: FrameLocator | None
    cache_locator: CacheLocator | None


def is_unspecialized_nn_module_attr_source(source: Source | None) -> bool:
    if source is None or not isinstance(source, AttrSource):
        return False
    if isinstance(source, CellContentsSource):
        return False
    try:
        return source.guard_source.is_unspecialized_nn_module()
    except NotImplementedError:
        return False


def find_frame_locator_for_source(
    tx: InstructionTranslatorBase,
    source: Source,
) -> FrameLocator | None:
    source_ancestors = []
    current = source
    while isinstance(current, ChainedSource):
        current = current.base
        source_ancestors.append(current)

    entry_locals = [
        (name, variable.source)
        for name, variable in tx.symbolic_locals.items()
        if variable.source is not None
    ]
    for ancestor in source_ancestors:
        for name, local_source in entry_locals:
            if local_source == ancestor or (
                isinstance(local_source, LocalCellSource)
                and isinstance(ancestor, LocalSource)
                and ancestor.is_derefed_cell_contents
                and local_source.local_name == ancestor.local_name
            ):
                return FrameLocator(
                    name=name,
                    source=local_source,
                    is_global=False,
                )
        if isinstance(ancestor, GlobalSource) and ancestor.global_name in tx.f_globals:
            return FrameLocator(
                name=ancestor.global_name,
                source=ancestor,
                is_global=True,
            )
    return None


def frame_locator_matches(
    tx: InstructionTranslatorBase,
    locator: FrameLocator,
) -> bool:
    if locator.is_global:
        return locator.name in tx.f_globals
    local_variable = tx.symbolic_locals.get(locator.name)
    return local_variable is not None and local_variable.source == locator.source


def make_cache_locator(
    tx: InstructionTranslatorBase,
    owner_source: Source,
) -> CacheLocator | None:
    chain = []
    root_source = owner_source
    while isinstance(root_source, ChainedSource):
        chain.append(root_source)
        root_source = root_source.base
    chain.reverse()

    if isinstance(root_source, (LocalSource, LocalCellSource)):
        root_name = root_source.local_name
        is_global = False
        namespace = tx.f_locals
    elif isinstance(root_source, GlobalSource):
        root_name = root_source.global_name
        is_global = True
        namespace = tx.f_globals
    else:
        return None

    value = namespace.get(root_name, _MISSING)
    if value is _MISSING:
        return None

    try:
        weakref.ref(value)
    except TypeError:
        pass
    else:
        return CacheLocator(root_name, is_global, ())

    accessors: list[CacheAccessor] = []
    for source in chain:
        if isinstance(source, NNModuleSource):
            continue
        if isinstance(source, CellContentsSource):
            return None
        if isinstance(source, AttrSource):
            try:
                instance_dict = object.__getattribute__(value, "__dict__")
            except AttributeError:
                return None
            if type(instance_dict) is not dict or source.member not in instance_dict:
                return None
            value = dict.__getitem__(instance_dict, source.member)
            accessors.append(InstanceDictAccessor(source.member))
            continue
        if isinstance(source, DictGetItemSource):
            if type(value) is not dict or isinstance(source.index, Source):
                return None
            try:
                value = dict.__getitem__(value, source.index)
            except (KeyError, TypeError):
                return None
            accessors.append(GetItemAccessor(dict, source.index))
            continue
        if isinstance(source, GetItemSource):
            if source.index_is_slice or isinstance(source.index, Source):
                return None
            container_type = type(value)
            if container_type is list:
                getitem = list.__getitem__
            elif container_type is tuple:
                getitem = tuple.__getitem__
            elif container_type is dict:
                getitem = dict.__getitem__
            else:
                return None
            try:
                value = getitem(value, source.index)
            except (IndexError, KeyError, TypeError):
                return None
            accessors.append(GetItemAccessor(container_type, source.index))
            continue
        return None

    try:
        weakref.ref(value)
    except TypeError:
        return None
    return CacheLocator(root_name, is_global, tuple(accessors))


def frame_exec_strategy_cache_key(
    tx: InstructionTranslatorBase,
    target: IndexTarget,
) -> tuple[object | None, CacheLocator | None]:
    root_tx = tx
    while root_tx.parent is not None:
        root_tx = root_tx.parent

    locator = target.cache_locator
    if locator is None:
        return None, None
    return locator.resolve(root_tx.f_locals, root_tx.f_globals), locator


def raise_mutated_nn_module_container_index(
    tx: InstructionTranslatorBase,
    target: IndexTarget,
) -> NoReturn:
    cache_key, cache_locator = frame_exec_strategy_cache_key(tx, target)
    source = target.source
    try:
        unimplemented(
            gb_type=(
                "Unspecialized nn.Module container indexed by mutable "
                "nn.Module attribute"
            ),
            context=f"source: {source}",
            explanation="Dynamo observed an nn.Module state attribute select "
            "an item from an unspecialized nn.Module container and then mutate "
            "later in the same compiled region. Specializing the key would "
            "select a different child module or parameter as the state changes.",
            hints=[
                "Use a constant key for nn.Module container indexing.",
                *graph_break_hints.SUPPORTABLE,
            ],
            skip_frame=True,
        )
    except Unsupported as exc:
        # Recursive SKIP is required for a called helper that contains the
        # lookup but not the later mutation. Intercepting that helper on its own
        # would specialize the selector and recreate the recompile loop.
        exc.frame_exec_strategy = FrameExecStrategy(FrameAction.SKIP, FrameAction.SKIP)
        exc.frame_exec_strategy_apply_to_code = False
        if cache_key is not None:
            try:
                exc.frame_exec_strategy_cache_key_ref = weakref.ref(cache_key)
            except TypeError:
                pass
        exc.frame_exec_strategy_cache_locator = cache_locator
        raise


class NNModuleContainerIndexTracker:
    """Per-attempt candidate state for mutable module-container selectors."""

    def __init__(self) -> None:
        self.candidates: dict[
            Source,
            dict[tuple[InstructionTranslatorBase, FrameSite], IndexCandidate],
        ] = {}

    def clear(self) -> None:
        self.candidates.clear()

    def record(self, source: Source, leaf_tx: InstructionTranslatorBase) -> None:
        frames = {}
        current_tx = leaf_tx
        while True:
            instruction_offset = current_tx.current_instruction.offset
            if instruction_offset is None:
                raise AssertionError("current instruction must have an offset")
            frames[current_tx] = IndexFrame(
                site=(current_tx.f_code, instruction_offset),
                locator=find_frame_locator_for_source(current_tx, source),
            )
            if current_tx.parent is None:
                break
            current_tx = current_tx.parent

        leaf_site = frames[leaf_tx].site
        if not isinstance(source, AttrSource):
            raise AssertionError(
                "module-container selector source must be an attribute"
            )
        cache_locator = make_cache_locator(current_tx, source.base)
        self.candidates.setdefault(source, {}).setdefault(
            (leaf_tx, leaf_site), IndexCandidate(leaf_tx, frames, cache_locator)
        )

    def raise_if_selector_is_mutated(
        self,
        *,
        item: VariableTracker,
        name: str,
        value: VariableTracker,
        mutated_source: Source,
        output_graph: OutputGraph,
    ) -> None:
        index_candidates = self.candidates.get(mutated_source)
        if not index_candidates:
            return
        if not isinstance(mutated_source, AttrSource):
            raise AssertionError(
                "module-container selector source must be an attribute"
            )

        if value.is_python_constant():
            try:
                previous_value = item.tp_getattro_impl(output_graph.current_tx, name)  # type: ignore[arg-type]
            except NotImplementedError:
                pass
            else:
                if previous_value.is_python_constant():
                    previous = previous_value.as_python_constant()
                    current = value.as_python_constant()
                    if (
                        type(previous) is type(current)
                        and type(current) in (bool, int, str)
                        and previous == current
                    ):
                        return

        mutation_txs = []
        current_tx = output_graph.current_tx
        while True:
            mutation_txs.append(current_tx)
            if current_tx.parent is None:
                break
            current_tx = current_tx.parent

        index_sites: dict[FrameSite, IndexTarget] = {}
        for candidate in index_candidates.values():
            for mutation_tx in mutation_txs:
                if mutation_tx not in candidate.frames:
                    continue

                source_aware = mutation_tx is candidate.leaf_tx
                target_frame = candidate.frames[mutation_tx]
                index_sites.setdefault(
                    target_frame.site,
                    IndexTarget(
                        source=mutated_source,
                        source_aware=source_aware,
                        locator=None if source_aware else target_frame.locator,
                        cache_locator=candidate.cache_locator,
                    ),
                )
                break

        if not index_sites:
            return

        output_graph.current_tx.speculation_log.mutated_nn_module_container_index_sites.update(
            index_sites
        )
        self.clear()
        raise NNModuleContainerIndexRestartAnalysis(
            restart_reason="nn.Module container index source was mutated"
        )
