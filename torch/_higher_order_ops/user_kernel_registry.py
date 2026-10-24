"""Registry of user-defined-kernel higher-order operators.

A *user kernel* is a kernel the model author wrote in a Python kernel DSL (Triton,
CuTeDSL, ...) and launches from the model. Every DSL is captured by PT2 the same way:

* a process-local side table holds the kernel object and its non-graphable
  constants; the graph only carries integer indices into it;
* a *mutation* HOP ``<dsl>_kernel_wrapper_mutation(kernel_idx, constant_args_idx,
  kwargs, ...)`` launches the kernel, writes some of its tensor ``kwargs`` in place
  and returns nothing;
* a *functional* HOP ``<dsl>_kernel_wrapper_functional(..., tensors_to_clone)``
  is what functionalization records: clone the listed kwargs, launch the kernel on
  the clones, return ``{name: clone}``.

Most of the compiler support for such a pair is DSL-agnostic: post-grad reinplacing
of the functional HOP, its decomposition back into the mutation HOP, the exact-stride
layout constraint at lowering, mutation tracking, CSE / DCE / constant-folding
exclusions, and the "graph references a process-local side table" checks. Those
sites look the HOP up here instead of comparing against one DSL's HOP objects.

DSL modules register themselves on import. ``user_kernel_hops()`` imports the
built-in providers lazily so that a lookup never misses a HOP that is already in a
graph.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import torch.fx


@dataclass(frozen=True)
class UserKernelHOPs:
    """One DSL's user-kernel HOP pair and the DSL-specific hooks the compiler needs."""

    # short DSL name, e.g. "triton", "cutedsl"
    dsl: str
    # the mutation HOP (writes kwargs in place, returns None)
    mutation: Any
    # the functional HOP (returns {name: clone} for every name in tensors_to_clone)
    functional: Any
    # eager implementation of the functional HOP (same kwargs as the HOP); traced by
    # the post-grad decomposition, so it must clone exactly the keys still listed in
    # tensors_to_clone and then call the mutation HOP
    functional_dense: Callable[..., dict[str, Any]]
    # human-readable name of the user kernel behind a mutation/functional node (logs)
    kernel_name: Callable[[torch.fx.Node], str]
    # names of the kwargs a mutation node writes, given the node's kwargs and the
    # kernel kwargs with fake values (Triton: TTIR analysis; others: declared)
    mutated_arg_names: Callable[[dict[str, Any], dict[str, Any]], list[str]]


_REGISTRY: dict[str, UserKernelHOPs] = {}
_BUILTINS_LOADED = False


def register_user_kernel_hops(info: UserKernelHOPs) -> None:
    _REGISTRY[info.dsl] = info


def _load_builtin_providers() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    # Each module registers its HOP pair at import time.
    import torch._higher_order_ops.triton_kernel_wrap  # noqa: F401


def user_kernel_hops() -> tuple[UserKernelHOPs, ...]:
    _load_builtin_providers()
    return tuple(_REGISTRY.values())


def get_user_kernel_hops_by_mutation(target: object) -> UserKernelHOPs | None:
    for info in user_kernel_hops():
        if target is info.mutation:
            return info
    return None


def get_user_kernel_hops_by_functional(target: object) -> UserKernelHOPs | None:
    for info in user_kernel_hops():
        if target is info.functional:
            return info
    return None


def is_user_kernel_mutation(target: object) -> bool:
    return get_user_kernel_hops_by_mutation(target) is not None


def is_user_kernel_functional(target: object) -> bool:
    return get_user_kernel_hops_by_functional(target) is not None


def is_user_kernel_hop(target: object) -> bool:
    return is_user_kernel_mutation(target) or is_user_kernel_functional(target)


def user_kernel_functional_hops() -> tuple[Any, ...]:
    return tuple(info.functional for info in user_kernel_hops())


def user_kernel_mutation_hops() -> tuple[Any, ...]:
    return tuple(info.mutation for info in user_kernel_hops())
