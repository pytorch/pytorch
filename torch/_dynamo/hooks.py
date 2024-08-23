import dataclasses
import functools
from typing import Any, Callable, List, Optional, Set

from torch._guards import GuardsSet

from .types import GuardFail


@dataclasses.dataclass
class Hooks:
    guard_export_fn: Optional[Callable[[GuardsSet], None]] = None
    guard_fail_fn: Optional[Callable[[GuardFail], None]] = None
    guard_filter_fn: Optional[Callable[[GuardsSet], GuardsSet]] = None
    symbolic_guard_filter_fn: Optional[Callable[[List[str]], List[str]]] = None


@functools.lru_cache(None)
def dont_skip_guards() -> Set[Callable[..., Any]]:
    # Curated set of guards that are better to be not skipped.
    from torch._dynamo.guards import GuardBuilder

    return set(
        {
            GuardBuilder.TORCH_FUNCTION_STATE,
            GuardBuilder.DEFAULT_DEVICE,
            GuardBuilder.DETERMINISTIC_ALGORITHMS,
            GuardBuilder.GRAD_MODE,
            # Removing SHAPE_ENV will remove all the symbolic shape guards
            GuardBuilder.SHAPE_ENV,
        }
    )


# Some examples of how to skip guards
def keep_tensor_guards(guards: GuardsSet) -> GuardsSet:
    from torch._dynamo.guards import GuardBuilder, guards_log

    new_guards = GuardsSet()

    for g in guards:
        if g.inner_create_fn() in dont_skip_guards():
            new_guards.add(g)
        elif g.inner_create_fn() is GuardBuilder.TENSOR_MATCH:
            new_guards.add(g)
        else:
            guards_log.debug("%s", f"Skipping guard: {g}")
    return new_guards


def skip_guards_on_nn_module_attributes(guards: GuardsSet) -> GuardsSet:
    # Skip guards on nn module attributes. For deep models, a lot of time can be spent on nn module attribute checks
    from torch._dynamo.guards import guards_log

    new_guards = GuardsSet()

    for g in guards:
        if g.source.is_unspecialized_nn_module() or g.source.is_specialized_nn_module():
            guards_log.debug("%s", f"Skipping guard: {g}")
        else:
            new_guards.add(g)
    return new_guards


def skip_all_symbolic_guards(guards: List[str]) -> List[str]:
    from torch._dynamo.guards import guards_log

    for g in guards:
        guards_log.debug("%s", f"Skipping guard: {g}")
    return []
