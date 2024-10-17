from typing import Any, Callable, List, Optional, Type
from typing_extensions import deprecated

import torch.utils._pytree as python_pytree
from torch.utils._pytree import PyTree, TreeSpec


FlattenFuncSpec = Callable[[PyTree, TreeSpec], List]
FlattenFuncExactMatchSpec = Callable[[PyTree, TreeSpec], bool]


@deprecated(
    "torch.fx._pytree.register_pytree_flatten_spec is deprecated and it is now a no-op. "
    "Please register the flatten_with_keys function in pytree instead.",
    category=FutureWarning,
)
def register_pytree_flatten_spec(
    cls: Type[Any],
    flatten_fn_spec: FlattenFuncSpec,
    flatten_fn_exact_match_spec: Optional[FlattenFuncExactMatchSpec] = None,
) -> None:
    pass  # no-op


# The pytree may be wrapped with torch.fx.Proxy, so we cannot use `treespec.flatten_up_to(pytree)`.
def tree_flatten_spec(
    pytree: PyTree,
    spec: TreeSpec,
    exact_structural_match: bool = False,
) -> List[Any]:
    if not isinstance(spec, TreeSpec):
        assert python_pytree._cxx_pytree_exists, "C++ PyTree is not available"

        from torch.utils._cxx_pytree import PyTreeSpec

        assert isinstance(spec, PyTreeSpec), "Expected a PyTreeSpec"
        return [accessor(pytree) for accessor in spec.accessors()]

    dummy_leaf = object()
    dummy_tree = python_pytree.tree_unflatten([dummy_leaf] * spec.num_leaves, spec)
    return [
        python_pytree.key_get(pytree, key_path)
        for key_path, dummy_leaf in python_pytree.tree_leaves_with_path(dummy_tree)
    ]
