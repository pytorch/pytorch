from typing import Any, Callable, List, Optional, Type
from typing_extensions import deprecated

import torch.utils._pytree as python_pytree
from torch.utils._pytree import PyTree, TreeSpec


FlattenFuncSpec = Callable[[PyTree, TreeSpec], List]
FlattenFuncExactMatchSpec = Callable[[PyTree, TreeSpec], bool]


@deprecated(
    "torch.fx._pytree.register_pytree_flatten_spec is deprecated and it is now a no-op. "
    "Please register the class with `flatten_with_keys` function as pytree node instead.",
    category=FutureWarning,
)
def register_pytree_flatten_spec(
    cls: Type[Any],
    flatten_fn_spec: FlattenFuncSpec,
    flatten_fn_exact_match_spec: Optional[FlattenFuncExactMatchSpec] = None,
) -> None:
    # no-op, just check if the node is registered and has flatten_with_keys_fn
    handler = python_pytree.SUPPORTED_NODES.get(cls)
    if handler is None:
        raise ValueError(
            f"Unsupported node type {cls}, "
            "please consider registering it as pytree node first."
        )
    if handler.flatten_with_keys_fn is None:
        raise ValueError(
            f"Unsupported node type {cls}, "
            "please consider registering the pytree node with `flatten_with_keys` function first."
        )


# The pytree may be wrapped with torch.fx.Proxy, so we cannot use `treespec.flatten_up_to(pytree)`.
# Use the key path API to index into the pytree instead.
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

    # FX `tracer.create_arg(x)` and Dynamo does not support `dummy_leaf = object()`
    # as a sentinel value. Use None here.
    dummy_leaf = None
    dummy_tree = python_pytree.tree_unflatten([dummy_leaf] * spec.num_leaves, spec)
    return [
        python_pytree.key_get(pytree, key_path)
        for key_path, _ in python_pytree.tree_leaves_with_path(dummy_tree)
    ]
