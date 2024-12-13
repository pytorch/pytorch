"""
Python polyfills for torch.utils.pytree
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, TYPE_CHECKING
from typing_extensions import TypeIs

import torch.utils._pytree as python_pytree
from torch.utils._pytree import BUILTIN_TYPES, STANDARD_DICT_TYPES

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    import builtins
    from typing_extensions import Self

    from torch.utils._cxx_pytree import PyTree


__all__: list[str] = []


if python_pytree._cxx_pytree_dynamo_traceable:
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

        def flatten_up_to(self, tree: PyTree) -> list[PyTree]:
            def helper(
                treespec: PyTreeSpec,
                node: PyTree,
                subtrees: list[PyTree],
            ) -> None:
                if treespec.is_leaf():
                    subtrees.append(node)
                    return

                node_type = type(node)
                if treespec.type not in BUILTIN_TYPES:
                    # Always require custom node types to match exactly
                    if node_type != treespec.type:
                        raise ValueError(
                            f"Type mismatch; "
                            f"expected {treespec.type!r}, but got {node_type!r}.",
                        )

                    children, metadata, *_ = optree.tree_flatten_one_level(
                        node,
                        none_is_leaf=True,
                        namespace="torch",
                    )
                    if len(children) != treespec.num_children:
                        raise ValueError(
                            f"Node arity mismatch; "
                            f"expected {treespec.num_children}, but got {len(children)}.",
                        )
                    if metadata != treespec._metadata:
                        raise ValueError(
                            f"Node context mismatch for custom node type {treespec.type!r}.",
                        )
                else:
                    # For builtin dictionary types, we allow some flexibility
                    # Otherwise, we require exact matches
                    both_standard_dict = (
                        treespec.type in STANDARD_DICT_TYPES
                        and node_type in STANDARD_DICT_TYPES
                    )
                    if not both_standard_dict and node_type != treespec.type:
                        raise ValueError(
                            f"Node type mismatch; "
                            f"expected {treespec.type!r}, but got {node_type!r}.",
                        )
                    if len(node) != treespec.num_children:
                        raise ValueError(
                            f"Node arity mismatch; "
                            f"expected {treespec.num_children}, but got {len(node)}.",
                        )

                    if both_standard_dict:
                        # dictionary types are compatible with each other
                        expected_keys = treespec.entries()
                        got_key_set = set(node)
                        expected_key_set = set(expected_keys)
                        if got_key_set != expected_key_set:
                            missing_keys = expected_key_set.difference(got_key_set)
                            extra_keys = got_key_set.difference(expected_key_set)
                            message = ""
                            if missing_keys:
                                message += f"; missing key(s): {missing_keys}"
                            if extra_keys:
                                message += f"; extra key(s): {extra_keys}"
                            raise ValueError(f"Node keys mismatch{message}.")
                        children = [node[key] for key in expected_keys]
                    else:
                        # node_type is treespec.type
                        children, metadata, *_ = optree.tree_flatten_one_level(
                            node,
                            none_is_leaf=True,
                            namespace="torch",
                        )
                        if (
                            node_type
                            is not deque  # ignore mismatch of `maxlen` for deque
                        ) and metadata != treespec._metadata:
                            raise ValueError(
                                f"Node metadata mismatch for node type {treespec.type!r}; "
                                f"expected {treespec._metadata!r}, but got {metadata!r}.",  # namedtuple type mismatch
                            )

                for subtree, subspec in zip(children, treespec._children):
                    helper(subspec, subtree, subtrees)

            subtrees: list[PyTree] = []
            helper(self, tree, subtrees)
            return subtrees

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
        cxx_pytree.treespec_leaf,
        # We need to disable constant folding here because we want the function to reference the
        # PyTreeSpec class defined above, not the one in the C++ module.
        can_constant_fold_through=False,
    )
    def treespec_leaf() -> PyTreeSpec:
        return _LEAF_SPEC

    @substitute_in_graph(  # type: ignore[arg-type]
        cxx_pytree.treespec_tuple,
        # We need to disable constant folding here because we want the function to reference the
        # PyTreeSpec class defined above, not the one in the C++ module.
        can_constant_fold_through=False,
    )
    def treespec_tuple(iterable: Iterable[PyTreeSpec] = (), /) -> PyTreeSpec:
        children = tuple(iterable)
        if any(not _is_pytreespec_instance(child) for child in children):
            raise ValueError(f"Expected a tuple of PyTreeSpecs, got: {children!r}.")
        handler = optree.register_pytree_node.get(tuple, namespace="torch")  # type: ignore[attr-defined]
        return PyTreeSpec(
            tuple(children),
            tuple,
            None,
            tuple(range(len(children))),
            handler.unflatten_func,
        )

    @substitute_in_graph(  # type: ignore[arg-type]
        cxx_pytree.treespec_dict,
        # We need to disable constant folding here because we want the function to reference the
        # PyTreeSpec class defined above, not the one in the C++ module.
        can_constant_fold_through=False,
    )
    def treespec_dict(
        mapping: Mapping[Any, PyTreeSpec] | Iterable[tuple[Any, PyTreeSpec]] = (),
        /,
        **kwargs: PyTreeSpec,
    ) -> PyTreeSpec:
        dct = dict(mapping, **kwargs)
        if any(not _is_pytreespec_instance(child) for child in dct.values()):
            raise ValueError(f"Expected a dictionary of TreeSpecs, got: {dct!r}.")

        (
            children,
            metadata,
            entries,
            unflatten_func,
        ) = optree.tree_flatten_one_level(  # type: ignore[assignment,var-annotated]
            dct,  # type: ignore[arg-type]
            none_is_leaf=True,
            namespace="torch",
        )
        return PyTreeSpec(tuple(children), dict, metadata, entries, unflatten_func)  # type: ignore[arg-type]

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
            if tree_is_leaf(node, is_leaf=is_leaf):
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
            return PyTreeSpec(subspecs, type(node), metadata, entries, unflatten_func)  # type: ignore[arg-type]

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

    @substitute_in_graph(cxx_pytree.tree_map, can_constant_fold_through=True)
    def tree_map(
        func: Callable[..., Any],
        tree: PyTree,
        *rests: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> PyTree:
        leaves, treespec = tree_flatten(tree, is_leaf=is_leaf)
        flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
        return treespec.unflatten(map(func, *flat_args))

    __all__ += ["tree_map"]

    @substitute_in_graph(cxx_pytree.tree_map_, can_constant_fold_through=True)
    def tree_map_(
        func: Callable[..., Any],
        tree: PyTree,
        *rests: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> PyTree:
        leaves, treespec = tree_flatten(tree, is_leaf=is_leaf)
        flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
        deque(map(func, *flat_args), maxlen=0)  # consume and exhaust the iterable
        return tree

    __all__ += ["tree_map_"]
