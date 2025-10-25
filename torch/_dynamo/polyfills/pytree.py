"""
Python polyfills for torch.utils.pytree
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import TypeIs

import torch.utils._pytree as python_pytree
from torch.utils._pytree import BUILTIN_TYPES, STANDARD_DICT_TYPES

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    import builtins
    from collections.abc import Callable, Iterable
    from typing_extensions import Self


__all__: list[str] = []


_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


if python_pytree._cxx_pytree_dynamo_traceable:
    import optree
    import optree._C
    import optree.utils

    import torch.utils._cxx_pytree as cxx_pytree  # noqa: F401

    if TYPE_CHECKING:
        from torch.utils._cxx_pytree import PyTree

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
        globals()[__name] = substitute_in_graph(__func, can_constant_fold_through=True)(
            __func.__python_implementation__
        )
        __all__ += [__name]  # noqa: PLE0604
        del __func
    del __name

    @substitute_in_graph(optree.tree_is_leaf, can_constant_fold_through=True)
    def tree_is_leaf(
        tree: PyTree,
        /,
        is_leaf: Callable[[PyTree], bool] | None = None,
        *,
        none_is_leaf: bool = False,
        namespace: str = "",
    ) -> bool:
        if (tree is None and none_is_leaf) or (is_leaf is not None and is_leaf(tree)):
            return True
        if optree.register_pytree_node.get(type(tree), namespace=namespace) is None:  # type: ignore[attr-defined]
            return True
        return False

    @substitute_in_graph(optree.tree_iter, can_constant_fold_through=False)
    def tree_iter(
        tree: PyTree,
        /,
        is_leaf: Callable[[PyTree], bool] | None = None,
        *,
        none_is_leaf: bool = False,
        namespace: str = "",
    ) -> Iterable[Any]:
        stack = [tree]
        while stack:
            node = stack.pop()
            if tree_is_leaf(
                node,
                is_leaf=is_leaf,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            ):
                yield node
                continue

            children, *_ = optree.tree_flatten_one_level(
                node,
                is_leaf=is_leaf,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )
            stack.extend(reversed(children))

    __all__ += ["tree_iter"]

    @substitute_in_graph(optree.tree_leaves, can_constant_fold_through=True)
    def tree_leaves(
        tree: PyTree,
        /,
        is_leaf: Callable[[PyTree], bool] | None = None,
        *,
        none_is_leaf: bool = False,
        namespace: str = "",
    ) -> list[Any]:
        return list(
            tree_iter(
                tree,
                is_leaf=is_leaf,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )
        )

    __all__ += ["tree_leaves"]

    class _Asterisk(str):
        __slots__ = ()

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
        none_is_leaf: bool
        namespace: str

        num_nodes: int = field(init=False)
        num_leaves: int = field(init=False)
        num_children: int = field(init=False)

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
                    or (treespec.type is type(None) and not self.none_is_leaf)
                    or optree.is_namedtuple_class(treespec.type)
                    or optree.is_structseq_class(treespec.type)
                ):
                    # pyrefly: ignore  # bad-return
                    return treespec._unflatten_func(
                        treespec._metadata,
                        children_representations,
                    )
                return (
                    f"CustomTreeNode({treespec.type.__name__}[{treespec._metadata!r}], "
                    f"[{', '.join(children_representations)}])"
                )

            inner = [
                str(helper(self)),
                *(["NoneIsLeaf"] if self.none_is_leaf else []),
                f"namespace={self.namespace!r}",
            ]
            return f"PyTreeSpec({', '.join(inner)})"

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
                        none_is_leaf=self.none_is_leaf,
                        namespace=self.namespace,
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
                            none_is_leaf=self.none_is_leaf,
                            namespace=self.namespace,
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

    def _is_pytreespec_instance(obj: Any, /) -> TypeIs[PyTreeSpec]:
        return isinstance(obj, PyTreeSpec)

    @substitute_in_graph(  # type: ignore[arg-type]
        optree.tree_flatten,
        # We need to disable constant folding here because we want the function to reference the
        # PyTreeSpec class defined above, not the one in the C++ module.
        can_constant_fold_through=False,
    )
    def tree_flatten(
        tree: PyTree,
        /,
        is_leaf: Callable[[PyTree], bool] | None = None,
        *,
        none_is_leaf: bool = False,
        namespace: str = "",
    ) -> tuple[list[Any], PyTreeSpec]:
        def helper(node: PyTree, leaves: list[Any]) -> PyTreeSpec:
            if tree_is_leaf(
                node,
                is_leaf=is_leaf,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            ):
                leaves.append(node)
                return PyTreeSpec(
                    (),
                    None,
                    None,
                    (),
                    None,
                    none_is_leaf=none_is_leaf,
                    namespace=namespace,
                )

            (
                children,
                metadata,
                entries,
                unflatten_func,
            ) = optree.tree_flatten_one_level(
                node,
                is_leaf=is_leaf,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )

            # Recursively flatten the children
            subspecs = tuple(helper(child, leaves) for child in children)
            return PyTreeSpec(
                subspecs,
                type(node),
                metadata,
                entries,
                unflatten_func,
                none_is_leaf=none_is_leaf,
                namespace=namespace,
            )  # type: ignore[arg-type]

        leaves: list[Any] = []
        treespec = helper(tree, leaves)
        return leaves, treespec

    __all__ += ["tree_flatten"]

    @substitute_in_graph(  # type: ignore[arg-type]
        optree.tree_structure,
        # We need to disable constant folding here because we want the function to reference the
        # PyTreeSpec class defined above, not the one in the C++ module.
        can_constant_fold_through=False,
    )
    def tree_structure(
        tree: PyTree,
        /,
        is_leaf: Callable[[PyTree], bool] | None = None,
        *,
        none_is_leaf: bool = False,
        namespace: str = "",
    ) -> PyTreeSpec:
        return tree_flatten(  # type: ignore[return-value]
            tree,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )[1]

    __all__ += ["tree_structure"]

    @substitute_in_graph(  # type: ignore[arg-type]
        optree.tree_unflatten,
        # We need to disable constant folding here because we want the function to reference the
        # PyTreeSpec class defined above, not the one in the C++ module.
        can_constant_fold_through=False,
    )
    def tree_unflatten(treespec: PyTreeSpec, leaves: Iterable[Any]) -> PyTree:
        if not _is_pytreespec_instance(treespec):
            raise TypeError(
                f"tree_unflatten(leaves, treespec): Expected `treespec` to be instance of "
                f"PyTreeSpec but got item of type {type(treespec)}."
            )
        return treespec.unflatten(leaves)

    __all__ += ["tree_unflatten"]

    @substitute_in_graph(optree.tree_map, can_constant_fold_through=True)
    def tree_map(
        func: Callable[..., Any],
        tree: PyTree,
        /,
        *rests: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
        none_is_leaf: bool = False,
        namespace: str = "",
    ) -> PyTree:
        leaves, treespec = tree_flatten(
            tree,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
        return treespec.unflatten(map(func, *flat_args))

    __all__ += ["tree_map"]

    @substitute_in_graph(optree.tree_map_, can_constant_fold_through=True)
    def tree_map_(
        func: Callable[..., Any],
        tree: PyTree,
        /,
        *rests: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
        none_is_leaf: bool = False,
        namespace: str = "",
    ) -> PyTree:
        leaves, treespec = tree_flatten(
            tree,
            is_leaf=is_leaf,
            none_is_leaf=none_is_leaf,
            namespace=namespace,
        )
        flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
        deque(map(func, *flat_args), maxlen=0)  # consume and exhaust the iterable
        return tree

    __all__ += ["tree_map_"]

    _none_registration = optree.register_pytree_node.get(type(None))
    assert _none_registration is not None

    @substitute_in_graph(  # type: ignore[arg-type]
        _none_registration.unflatten_func,
        can_constant_fold_through=True,
        skip_signature_check=True,
    )
    def none_unflatten(_: None, children: Iterable[_T], /) -> None:
        if len(list(children)) != 0:
            raise ValueError("Expected no children.")
        return None

    with optree.dict_insertion_ordered(False, namespace="torch"):
        _dict_registration = optree.register_pytree_node.get(dict)
        assert _dict_registration is not None

    @substitute_in_graph(  # type: ignore[arg-type]
        _dict_registration.flatten_func,
        can_constant_fold_through=True,
        skip_signature_check=True,
    )
    def dict_flatten(
        dct: dict[_KT, _VT], /
    ) -> tuple[list[_VT], tuple[list[_KT], list[_KT]], tuple[_KT, ...]]:
        sorted_keys = optree.utils.total_order_sorted(dct)
        values = [dct[key] for key in sorted_keys]
        original_keys = list(dct)
        return values, (original_keys, sorted_keys), tuple(sorted_keys)

    @substitute_in_graph(  # type: ignore[arg-type]
        _dict_registration.unflatten_func,
        can_constant_fold_through=True,
        skip_signature_check=True,
    )
    def dict_unflatten(
        metadata: tuple[list[_KT], list[_KT]],
        values: Iterable[_VT],
        /,
    ) -> dict[_KT, _VT]:
        original_keys, sorted_keys = metadata
        d = dict.fromkeys(original_keys)
        d.update(zip(sorted_keys, values))
        return d  # type: ignore[return-value]
