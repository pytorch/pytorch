"""
Python polyfills for torch.utils.pytree
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, TYPE_CHECKING
from typing_extensions import TypeIs

import torch.utils._pytree as python_pytree
from torch.utils._pytree import BUILTIN_TYPES

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    import builtins
    from typing_extensions import Self

    from torch.utils._cxx_pytree import PyTree


__all__: list[str] = []


if python_pytree._cxx_pytree_exists:
    import optree

    import torch.utils._cxx_pytree as cxx_pytree

    @substitute_in_graph(
        optree._C.is_dict_insertion_ordered,
        can_constant_fold_through=True,
    )
    def _(*args: Any, **kwargs: Any) -> Literal[True]:
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

    @substitute_in_graph(cxx_pytree.tree_iter, can_constant_fold_through=False)
    def tree_iter(
        tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> Iterable[Any]:
        stack = [tree]
        while stack:
            node = stack.pop()
            if node is None or (is_leaf is not None and is_leaf(node)):
                yield node
                continue
            if optree.register_pytree_node.get(type(node), namespace="torch") is None:  # type: ignore[attr-defined]
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

    class _Asterisk(str):
        def __new__(cls) -> Self:
            return super().__new__(cls, "*")

        def __repr__(self) -> str:
            return "*"  # no quotes

    _asterisk = _Asterisk()
    del _Asterisk

    @dataclass(frozen=True)
    class PyTreeSpec:
        """Analog for :class:`optree.PyTreeSpec` in Python."""

        _children: tuple[PyTreeSpec, ...]
        _type: builtins.type | None
        _metadata: Any
        _entries: tuple[Any, ...]
        _unflatten_func: Callable[[Any | None, Iterable[PyTree]], PyTree] | None

        num_nodes: int = field(init=False)
        num_leaves: int = field(init=False)
        num_children: int = field(init=False)
        none_is_leaf: Literal[True] = field(init=False)
        namespace: Literal["torch"] = field(init=False)

        def __post_init__(self) -> None:
            if self._type is None:
                assert len(self._children) == 0
                assert self._metadata is None
                assert self._entries == ()
                assert self._unflatten_func is None
                num_nodes = 1
                num_leaves = 1
                num_children = 0
            else:
                assert callable(self._unflatten_func)
                num_nodes = sum((spec.num_nodes for spec in self._children), start=1)
                num_leaves = sum(spec.num_leaves for spec in self._children)
                num_children = len(self._children)

            object.__setattr__(self, "num_nodes", num_nodes)
            object.__setattr__(self, "num_leaves", num_leaves)
            object.__setattr__(self, "num_children", num_children)
            object.__setattr__(self, "none_is_leaf", True)
            object.__setattr__(self, "namespace", "torch")

        def __repr__(self) -> str:
            def helper(treespec: PyTreeSpec) -> str:
                if treespec.is_leaf():
                    assert treespec.type is None
                    return _asterisk

                assert treespec.type is not None
                assert callable(treespec._unflatten_func)
                children_representations = [
                    helper(subspec) for subspec in treespec._children
                ]
                if (
                    treespec.type in BUILTIN_TYPES
                    or optree.is_namedtuple_class(treespec.type)
                    or optree.is_structseq_class(treespec.type)
                ):
                    return treespec._unflatten_func(
                        treespec._metadata,
                        children_representations,
                    )
                return (
                    f"CustomTreeNode({treespec.type.__name__}[{treespec._metadata!r}], "
                    f"[{', '.join(children_representations)}])"
                )

            return (
                f"PyTreeSpec({helper(self)}, NoneIsLeaf, namespace={self.namespace!r})"
            )

        def __len__(self) -> int:
            return self.num_leaves

        @property
        def type(self) -> builtins.type | None:
            return self._type

        def is_leaf(self) -> bool:
            return self.num_nodes == 1 and self.num_leaves == 1

        def children(self) -> list[PyTreeSpec]:
            return list(self._children)

        def child(self, index: int) -> PyTreeSpec:
            return self._children[index]

        def entries(self) -> list[Any]:
            return list(self._entries)

        def entry(self, index: int) -> Any:
            return self._entries[index]

        def unflatten(self, leaves: Iterable[Any]) -> PyTree:
            if not isinstance(leaves, (list, tuple)):
                leaves = list(leaves)
            if len(leaves) != self.num_leaves:
                raise ValueError(
                    f"treespec.unflatten(leaves): `leaves` has length {len(leaves)} "
                    f"but the spec refers to a pytree that holds {self.num_leaves} "
                    f"items ({self}).",
                )
            if self.is_leaf():
                return leaves[0]

            # Recursively unflatten the children
            start = 0
            end = 0
            subtrees = []
            for subspec in self._children:
                end += subspec.num_leaves
                subtrees.append(subspec.unflatten(leaves[start:end]))
                start = end

            assert callable(self._unflatten_func)
            return self._unflatten_func(self._metadata, subtrees)

    _LEAF_SPEC = PyTreeSpec((), None, None, (), None)

    def _is_pytreespec_instance(obj: Any, /) -> TypeIs[PyTreeSpec]:
        return isinstance(obj, PyTreeSpec)

    @substitute_in_graph(  # type: ignore[arg-type]
        cxx_pytree.tree_flatten,
        # We need to disable constant folding here because we want the function to reference the
        # PyTreeSpec class defined above, not the one in the C++ module.
        can_constant_fold_through=False,
    )
    def tree_flatten(
        tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> tuple[list[Any], PyTreeSpec]:
        def helper(node: PyTree, leaves: list[Any]) -> PyTreeSpec:
            if node is None or (is_leaf is not None and is_leaf(node)):
                leaves.append(node)
                return _LEAF_SPEC

            node_type = type(node)
            if optree.register_pytree_node.get(node_type, namespace="torch") is None:  # type: ignore[attr-defined]
                leaves.append(node)
                return _LEAF_SPEC

            (
                children,
                metadata,
                entries,
                unflatten_func,
            ) = optree.tree_flatten_one_level(
                node,
                is_leaf=is_leaf,
                none_is_leaf=True,
                namespace="torch",
            )

            # Recursively flatten the children
            subspecs = tuple(helper(child, leaves) for child in children)
            return PyTreeSpec(subspecs, node_type, metadata, entries, unflatten_func)  # type: ignore[arg-type]

        leaves: list[Any] = []
        treespec = helper(tree, leaves)
        return leaves, treespec

    __all__ += ["tree_flatten"]

    @substitute_in_graph(  # type: ignore[arg-type]
        cxx_pytree.tree_structure,
        # We need to disable constant folding here because we want the function to reference the
        # PyTreeSpec class defined above, not the one in the C++ module.
        can_constant_fold_through=False,
    )
    def tree_structure(
        tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> PyTreeSpec:
        return tree_flatten(tree, is_leaf=is_leaf)[1]  # type: ignore[return-value]

    __all__ += ["tree_structure"]

    @substitute_in_graph(  # type: ignore[arg-type]
        cxx_pytree.tree_unflatten,
        # We need to disable constant folding here because we want the function to reference the
        # PyTreeSpec class defined above, not the one in the C++ module.
        can_constant_fold_through=False,
    )
    def tree_unflatten(leaves: Iterable[Any], treespec: PyTreeSpec) -> PyTree:
        if not _is_pytreespec_instance(treespec):
            raise TypeError(
                f"tree_unflatten(leaves, treespec): Expected `treespec` to be instance of "
                f"PyTreeSpec but got item of type {type(treespec)}."
            )
        return treespec.unflatten(leaves)

    __all__ += ["tree_unflatten"]
