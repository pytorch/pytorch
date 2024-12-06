"""
Python polyfills for torch.utils.pytree
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch.utils._pytree as python_pytree

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from torch.utils._cxx_pytree import PyTree


__all__: list[str] = []


if python_pytree._cxx_pytree_exists:
    import optree
    import optree._C

    import torch.utils._cxx_pytree as cxx_pytree

    @substitute_in_graph(
        optree._C.is_dict_insertion_ordered,
        can_constant_fold_through=True,
    )
    def _(*args: Any, **kwargs: Any) -> bool:
        # In namespace 'torch', the dictionary is always traversed in insertion order.
        # This function returns True.
        raise ValueError(
            "Should not be called directly "
            "because the original function will be called in the constant fold path."
        )

    __name = ""
    for __name in (
        "is_namedtuple",
        "is_namedtuple_class",
        "is_namedtuple_instance",
        "is_structseq",
        "is_structseq_class",
        "is_structseq_instance",
        "namedtuple_fields",
        "structseq_fields",
    ):
        __func = getattr(optree, __name)
        substitute_in_graph(__func, can_constant_fold_through=True)(
            __func.__python_implementation__
        )
        del __func
    del __name

    @substitute_in_graph(cxx_pytree.tree_is_leaf, can_constant_fold_through=True)
    def tree_is_leaf(
        tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> bool:
        if tree is None or (is_leaf is not None and is_leaf(tree)):
            return True
        if optree.register_pytree_node.get(type(tree), namespace="torch") is None:  # type: ignore[attr-defined]
            return True
        return False

    @substitute_in_graph(cxx_pytree.tree_iter, can_constant_fold_through=False)
    def tree_iter(
        tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> Iterable[Any]:
        stack = [tree]
        while stack:
            node = stack.pop()
            if tree_is_leaf(node, is_leaf=is_leaf):
                yield node
                continue

            children, *_ = optree.tree_flatten_one_level(
                node,
                is_leaf=is_leaf,
                none_is_leaf=True,
                namespace="torch",
            )
            stack.extend(reversed(children))

    __all__ += ["tree_iter"]

    @substitute_in_graph(cxx_pytree.tree_leaves, can_constant_fold_through=True)
    def tree_leaves(
        tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> list[Any]:
        return list(tree_iter(tree, is_leaf=is_leaf))

    __all__ += ["tree_leaves"]
