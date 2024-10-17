from typing import Any, Callable, List, Optional, Type

from torch.utils._pytree import PyTree, TreeSpec


FlattenFuncSpec = Callable[[PyTree, TreeSpec], List]
FlattenFuncExactMatchSpec = Callable[[PyTree, TreeSpec], bool]


def register_pytree_flatten_spec(
    cls: Type[Any],
    flatten_fn_spec: FlattenFuncSpec,
    flatten_fn_exact_match_spec: Optional[FlattenFuncExactMatchSpec] = None,
) -> None:
    pass  # no-op


def tree_flatten_spec(
    pytree: PyTree,
    spec: TreeSpec,
    exact_structural_match: bool = False,
) -> List[Any]:
    return spec.flatten_up_to(pytree)
