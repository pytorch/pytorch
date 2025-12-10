# Copyright 2022-2025 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Registry for custom pytree node types."""

# pylint: disable=too-many-lines

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import sys
from collections import OrderedDict, defaultdict, deque, namedtuple
from operator import itemgetter, methodcaller
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, NamedTuple, TypeVar, overload

import optree._C as _C
from optree.accessors import (
    AutoEntry,
    MappingEntry,
    NamedTupleEntry,
    PyTreeEntry,
    SequenceEntry,
    StructSequenceEntry,
)
from optree.typing import (
    Children,
    MetaData,
    PyTreeKind,
    StructSequence,
    T,
    is_namedtuple_class,
    is_structseq_class,
)
from optree.utils import safe_zip, total_order_sorted, unzip2


if TYPE_CHECKING:
    import builtins
    from collections.abc import Collection, Generator, Iterable

    from optree.typing import KT, VT, CustomTreeNode, FlattenFunc, UnflattenFunc

    # pylint: disable-next=invalid-name
    CustomTreeNodeType = TypeVar('CustomTreeNodeType', bound=type[CustomTreeNode])


__all__ = [
    'register_pytree_node',
    'register_pytree_node_class',
    'unregister_pytree_node',
    'dict_insertion_ordered',
]


SLOTS = {'slots': True} if sys.version_info >= (3, 10) else {}  # Python 3.10+


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True, **SLOTS)
class PyTreeNodeRegistryEntry(Generic[T]):
    """A dataclass that stores the information of a pytree node type."""

    type: builtins.type[Collection[T]]
    flatten_func: FlattenFunc[T]
    unflatten_func: UnflattenFunc[T]

    if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
        _: dataclasses.KW_ONLY  # Python 3.10+

    path_entry_type: builtins.type[PyTreeEntry] = AutoEntry
    kind: PyTreeKind = PyTreeKind.CUSTOM
    namespace: str = ''


del SLOTS


# pylint: disable-next=missing-class-docstring,too-few-public-methods
class GlobalNamespace:  # pragma: no cover
    __slots__: ClassVar[tuple[()]] = ()

    def __repr__(self, /) -> str:
        return '<GLOBAL NAMESPACE>'


__GLOBAL_NAMESPACE: str = GlobalNamespace()  # type: ignore[assignment]
__REGISTRY_LOCK: Lock = Lock()
del GlobalNamespace


if TYPE_CHECKING:
    from typing_extensions import ParamSpec  # Python 3.10+

    _P = ParamSpec('_P')
    _T = TypeVar('_T')
    _GetP = ParamSpec('_GetP')
    _GetT = TypeVar('_GetT')

    class _CallableWithGet(Generic[_P, _T, _GetP, _GetT]):
        def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
            raise NotImplementedError

        # pylint: disable-next=missing-function-docstring
        def get(self, /, *args: _GetP.args, **kwargs: _GetP.kwargs) -> _GetT:
            raise NotImplementedError


def _add_get(
    get: Callable[_GetP, _GetT],
    /,
) -> Callable[
    [Callable[_P, _T]],
    _CallableWithGet[_P, _T, _GetP, _GetT],
]:
    def decorator(func: Callable[_P, _T], /) -> _CallableWithGet[_P, _T, _GetP, _GetT]:
        func.get = get  # type: ignore[attr-defined]
        return func  # type: ignore[return-value]

    return decorator


@overload
def pytree_node_registry_get(
    cls: type,
    /,
    *,
    namespace: str = '',
) -> PyTreeNodeRegistryEntry | None: ...


@overload
def pytree_node_registry_get(
    cls: None = None,
    /,
    *,
    namespace: str = '',
) -> dict[type, PyTreeNodeRegistryEntry]: ...


# pylint: disable-next=too-many-return-statements,too-many-branches
def pytree_node_registry_get(  # noqa: C901
    cls: type | None = None,
    /,
    *,
    namespace: str = '',
) -> dict[type, PyTreeNodeRegistryEntry] | PyTreeNodeRegistryEntry | None:
    """Lookup the pytree node registry.

    >>> register_pytree_node.get()  # doctest: +IGNORE_WHITESPACE,ELLIPSIS
    {
        <class 'NoneType'>: PyTreeNodeRegistryEntry(
            type=<class 'NoneType'>,
            flatten_func=<function ...>,
            unflatten_func=<function ...>,
            path_entry_type=<class 'optree.PyTreeEntry'>,
            kind=<PyTreeKind.NONE: 2>,
            namespace=''
        ),
        <class 'tuple'>: PyTreeNodeRegistryEntry(
            type=<class 'tuple'>,
            flatten_func=<function ...>,
            unflatten_func=<function ...>,
            path_entry_type=<class 'optree.SequenceEntry'>,
            kind=<PyTreeKind.TUPLE: 3>,
            namespace=''
        ),
        <class 'list'>: PyTreeNodeRegistryEntry(
            type=<class 'list'>,
            flatten_func=<function ...>,
            unflatten_func=<function ...>,
            path_entry_type=<class 'optree.SequenceEntry'>,
            kind=<PyTreeKind.LIST: 4>,
            namespace=''
        ),
        ...
    }
    >>> register_pytree_node.get(defaultdict)  # doctest: +IGNORE_WHITESPACE,ELLIPSIS
    PyTreeNodeRegistryEntry(
        type=<class 'collections.defaultdict'>,
        flatten_func=<function ...>,
        unflatten_func=<function ...>,
        path_entry_type=<class 'optree.MappingEntry'>,
        kind=<PyTreeKind.DEFAULTDICT: 8>,
        namespace=''
    )
    >>> register_pytree_node.get(frozenset)  # frozenset is considered as a leaf node
    None

    Args:
        cls (type or None, optional): The class of the pytree node to retrieve. If not provided, all
            the registered pytree nodes in the namespace are returned.
        namespace (str, optional): The namespace of the registry to retrieve. If not provided, the
            global namespace is used.

    Returns:
        If the ``cls`` is not provided, a dictionary of all the registered pytree nodes in the
        namespace is returned. If the ``cls`` is provided, the corresponding registry entry is
        returned if the ``cls`` is registered as a pytree node. Otherwise, :data:`None` is returned,
        i.e., the ``cls`` is represented as a leaf node.
    """
    if namespace is __GLOBAL_NAMESPACE:
        namespace = ''
    if (
        cls is not None
        and cls is not namedtuple  # noqa: PYI024
        and not inspect.isclass(cls)
    ):
        raise TypeError(f'Expected a class or None, got {cls!r}.')  # pragma: !=3.9 cover
    if not isinstance(namespace, str):
        raise TypeError(  # pragma: !=3.9 cover
            f'The namespace must be a string, got {namespace!r}.',
        )

    if cls is None:
        namespaces = frozenset({namespace, ''})
        with __REGISTRY_LOCK:
            registry = {
                handler.type: handler
                for handler in _NODETYPE_REGISTRY.values()
                if handler.namespace in namespaces
            }
        if _C.is_dict_insertion_ordered(namespace):
            registry[dict] = _DICT_INSERTION_ORDERED_REGISTRY_ENTRY
            registry[defaultdict] = _DEFAULTDICT_INSERTION_ORDERED_REGISTRY_ENTRY
        return registry

    if namespace != '':
        handler = _NODETYPE_REGISTRY.get((namespace, cls))
        if handler is not None:
            return handler

    if _C.is_dict_insertion_ordered(namespace):
        if cls is dict:
            return _DICT_INSERTION_ORDERED_REGISTRY_ENTRY
        if cls is defaultdict:
            return _DEFAULTDICT_INSERTION_ORDERED_REGISTRY_ENTRY

    handler = _NODETYPE_REGISTRY.get(cls)
    if handler is not None:
        return handler
    if is_structseq_class(cls):
        return _NODETYPE_REGISTRY.get(StructSequence)
    if is_namedtuple_class(cls):
        return _NODETYPE_REGISTRY.get(namedtuple)  # type: ignore[call-overload] # noqa: PYI024
    return None


@_add_get(pytree_node_registry_get)
def register_pytree_node(
    cls: type[Collection[T]],
    /,
    flatten_func: FlattenFunc[T],
    unflatten_func: UnflattenFunc[T],
    *,
    path_entry_type: type[PyTreeEntry] = AutoEntry,
    namespace: str,
) -> type[Collection[T]]:
    """Extend the set of types that are considered internal nodes in pytrees.

    See also :func:`register_pytree_node_class` and :func:`unregister_pytree_node`.

    The ``namespace`` argument is used to avoid collisions that occur when different libraries
    register the same Python type with different behaviors. It is recommended to add a unique prefix
    to the namespace to avoid conflicts with other libraries. Namespaces can also be used to specify
    the same class in different namespaces for different use cases.

    .. warning::
        For safety reasons, a ``namespace`` must be specified while registering a custom type. It is
        used to isolate the behavior of flattening and unflattening a pytree node type. This is to
        prevent accidental collisions between different libraries that may register the same type.

    Args:
        cls (type): A Python type to treat as an internal pytree node.
        flatten_func (callable): A function to be used during flattening, taking an instance of ``cls``
            and returning a triple or optionally a pair, with (1) an iterable for the children to be
            flattened recursively, and (2) some hashable metadata to be stored in the treespec and
            to be passed to the ``unflatten_func``, and (3) (optional) an iterable for the tree path
            entries to the corresponding children. If the entries are not provided or given by
            :data:`None`, then `range(len(children))` will be used.
        unflatten_func (callable): A function taking two arguments: the metadata that was returned
            by ``flatten_func`` and stored in the treespec, and the unflattened children. The
            function should return an instance of ``cls``.
        path_entry_type (type, optional): The type of the path entry to be used in the treespec.
            (default: :class:`AutoEntry`)
        namespace (str): A non-empty string that uniquely identifies the namespace of the type registry.
            This is used to isolate the registry from other modules that might register a different
            custom behavior for the same type.

    Returns:
        The same type as the input ``cls``.

    Raises:
        TypeError: If the input type is not a class.
        TypeError: If the path entry class is not a subclass of :class:`PyTreeEntry`.
        TypeError: If the namespace is not a string.
        ValueError: If the namespace is an empty string.
        ValueError: If the type is already registered in the registry.

    .. versionadded:: 0.12.0
        The ``path_entry_type`` argument to specify the path entry type used in
        :meth:`PyTreeSpec.accessors` and :func:`tree_flatten_with_accessor`.
        If not provided, :class:`AutoEntry` will be used.

    Examples:
        >>> # Registry a Python type with lambda functions
        >>> register_pytree_node(
        ...     set,
        ...     lambda s: (sorted(s), None, None),
        ...     lambda _, children: set(children),
        ...     namespace='set',
        ... )
        <class 'set'>

        >>> # Register a Python type into a namespace
        >>> import torch
        >>> register_pytree_node(
        ...     torch.Tensor,
        ...     flatten_func=lambda tensor: (
        ...         (tensor.cpu().detach().numpy(),),
        ...         {'dtype': tensor.dtype, 'device': tensor.device, 'requires_grad': tensor.requires_grad},
        ...     ),
        ...     unflatten_func=lambda metadata, children: torch.tensor(children[0], **metadata),
        ...     namespace='torch2numpy',
        ... )
        <class 'torch.Tensor'>

        >>> # doctest: +SKIP
        >>> tree = {'weight': torch.ones(size=(1, 2)).cuda(), 'bias': torch.zeros(size=(2,))}
        >>> tree
        {'weight': tensor([[1., 1.]], device='cuda:0'), 'bias': tensor([0., 0.])}

        >>> # Flatten without specifying the namespace
        >>> tree_flatten(tree)  # `torch.Tensor`s are leaf nodes
        ([tensor([0., 0.]), tensor([[1., 1.]], device='cuda:0')], PyTreeSpec({'bias': *, 'weight': *}))

        >>> # Flatten with the namespace
        >>> tree_flatten(tree, namespace='torch2numpy')
        (
            [array([0., 0.], dtype=float32), array([[1., 1.]], dtype=float32)],
            PyTreeSpec(
                {
                    'bias': CustomTreeNode(Tensor[{'dtype': torch.float32, 'device': device(type='cpu'), 'requires_grad': False}], [*]),
                    'weight': CustomTreeNode(Tensor[{'dtype': torch.float32, 'device': device(type='cuda', index=0), 'requires_grad': False}], [*])
                },
                namespace='torch2numpy'
            )
        )

        >>> # Register the same type with a different namespace for different behaviors
        >>> def tensor2flatparam(tensor):
        ...     return [torch.nn.Parameter(tensor.reshape(-1))], tensor.shape, None
        ...
        ... def flatparam2tensor(metadata, children):
        ...     return children[0].reshape(metadata)
        ...
        ... register_pytree_node(
        ...     torch.Tensor,
        ...     flatten_func=tensor2flatparam,
        ...     unflatten_func=flatparam2tensor,
        ...     namespace='tensor2flatparam',
        ... )
        <class 'torch.Tensor'>

        >>> # Flatten with the new namespace
        >>> tree_flatten(tree, namespace='tensor2flatparam')
        (
            [
                Parameter containing: tensor([0., 0.], requires_grad=True),
                Parameter containing: tensor([1., 1.], device='cuda:0', requires_grad=True)
            ],
            PyTreeSpec(
                {
                    'bias': CustomTreeNode(Tensor[torch.Size([2])], [*]),
                    'weight': CustomTreeNode(Tensor[torch.Size([1, 2])], [*])
                },
                namespace='tensor2flatparam'
            )
        )
    """  # pylint: disable=line-too-long
    if not inspect.isclass(cls):
        raise TypeError(f'Expected a class, got {cls!r}.')
    if not (inspect.isclass(path_entry_type) and issubclass(path_entry_type, PyTreeEntry)):
        raise TypeError(f'Expected a subclass of PyTreeEntry, got {path_entry_type!r}.')
    if namespace is not __GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace!r}.')
    if namespace == '':
        raise ValueError('The namespace cannot be an empty string.')

    registration_key: type | tuple[str, type]
    if namespace is __GLOBAL_NAMESPACE:
        registration_key = cls
        namespace = ''
    else:
        registration_key = (namespace, cls)

    with __REGISTRY_LOCK:
        _C.register_node(
            cls,
            flatten_func,
            unflatten_func,
            path_entry_type,
            namespace,
        )
        _NODETYPE_REGISTRY[registration_key] = PyTreeNodeRegistryEntry(
            cls,
            flatten_func,
            unflatten_func,
            path_entry_type=path_entry_type,
            namespace=namespace,
        )
    return cls


del pytree_node_registry_get, _add_get


@overload
def register_pytree_node_class(
    cls: str | None = None,
    /,
    *,
    path_entry_type: type[PyTreeEntry] | None = None,
    namespace: str | None = None,
) -> Callable[[CustomTreeNodeType], CustomTreeNodeType]: ...


@overload
def register_pytree_node_class(
    cls: CustomTreeNodeType,
    /,
    *,
    path_entry_type: type[PyTreeEntry] | None,
    namespace: str,
) -> CustomTreeNodeType: ...


# pylint: disable-next=too-many-branches
def register_pytree_node_class(  # noqa: C901
    cls: CustomTreeNodeType | str | None = None,
    /,
    *,
    path_entry_type: type[PyTreeEntry] | None = None,
    namespace: str | None = None,
) -> CustomTreeNodeType | Callable[[CustomTreeNodeType], CustomTreeNodeType]:
    """Extend the set of types that are considered internal nodes in pytrees.

    See also :func:`register_pytree_node` and :func:`unregister_pytree_node`.

    The ``namespace`` argument is used to avoid collisions that occur when different libraries
    register the same Python type with different behaviors. It is recommended to add a unique prefix
    to the namespace to avoid conflicts with other libraries. Namespaces can also be used to specify
    the same class in different namespaces for different use cases.

    .. warning::
        For safety reasons, a ``namespace`` must be specified while registering a custom type. It is
        used to isolate the behavior of flattening and unflattening a pytree node type. This is to
        prevent accidental collisions between different libraries that may register the same type.

    Args:
        cls (type, optional): A Python type to treat as an internal pytree node.
        path_entry_type (type, optional): The type of the path entry to be used in the treespec.
            (default: :class:`AutoEntry`)
        namespace (str, optional): A non-empty string that uniquely identifies the namespace of the
            type registry. This is used to isolate the registry from other modules that might
            register a different custom behavior for the same type.

    Returns:
        The same type as the input ``cls`` if the argument presents. Otherwise, return a decorator
        function that registers the class as a pytree node.

    Raises:
        TypeError: If the path entry class is not a subclass of :class:`PyTreeEntry`.
        TypeError: If the namespace is not a string.
        TypeError: If the class does not define the required method pairs.
        ValueError: If the namespace is an empty string.
        ValueError: If the type is already registered in the registry.

    .. versionadded:: 0.12.0
        The ``TREE_PATH_ENTRY_TYPE`` class variable to specify the path entry type used in
        :meth:`PyTreeSpec.accessors` and :func:`tree_flatten_with_accessor`.
        If not provided, :class:`AutoEntry` will be used.

    .. versionadded:: 0.18.0
        Previously, this function looked for methods named ``tree_flatten`` and ``tree_unflatten``
        for the given class. Since version 0.18.0, it prefers methods named ``__tree_flatten__``
        and ``__tree_unflatten__`` instead. The old method names are still supported for
        backward compatibility, but it is recommended to use the new method names.
        The method resolution follows this priority:
        1. If both ``__tree_flatten__`` and ``__tree_unflatten__`` are defined, use them directly.
        2. If both ``tree_flatten`` and ``tree_unflatten`` are defined, wrap them as dunder methods.
        3. If neither complete pair is available, raise a :exc:`TypeError` suggesting the new method names.

    This function is a thin wrapper around :func:`register_pytree_node`, and provides a
    class-oriented interface:

    .. code-block:: python

        @register_pytree_node_class(namespace='foo')
        class Special:
            TREE_PATH_ENTRY_TYPE = GetAttrEntry

            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __tree_flatten__(self):
                return ((self.x, self.y), None, ('x', 'y'))

            @classmethod
            def __tree_unflatten__(cls, metadata, children):
                return cls(*children)

        @register_pytree_node_class('mylist')
        class MyList(UserList):
            TREE_PATH_ENTRY_TYPE = SequenceEntry

            def __tree_flatten__(self):
                return self.data, None, None

            @classmethod
            def __tree_unflatten__(cls, metadata, children):
                return cls(*children)

        # Legacy style (still supported but not recommended)
        @register_pytree_node_class(namespace='legacy')
        class LegacyStyleMyList(UserList):
            def tree_flatten(self):
                # Implementation automatically wrapped as __tree_flatten__
                return self.data, None, None

            @classmethod
            def tree_unflatten(cls, metadata, children):
                # Implementation automatically wrapped as __tree_unflatten__
                return cls(*children)
    """
    if cls is __GLOBAL_NAMESPACE or isinstance(cls, str):
        if namespace is not None:
            raise ValueError('Cannot specify `namespace` when the first argument is a string.')
        if cls == '':
            raise ValueError('The namespace cannot be an empty string.')
        cls, namespace = None, cls

    if namespace is None:
        raise ValueError('Must specify `namespace` when the first argument is a class.')
    if namespace is not __GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace!r}')
    if namespace == '':
        raise ValueError('The namespace cannot be an empty string.')

    if cls is None:

        def decorator(cls: CustomTreeNodeType, /) -> CustomTreeNodeType:
            return register_pytree_node_class(
                cls,
                path_entry_type=path_entry_type,
                namespace=namespace,
            )

        return decorator

    if not inspect.isclass(cls):
        raise TypeError(f'Expected a class, got {cls!r}.')
    if path_entry_type is None:
        path_entry_type = getattr(cls, 'TREE_PATH_ENTRY_TYPE', AutoEntry)
    if not (inspect.isclass(path_entry_type) and issubclass(path_entry_type, PyTreeEntry)):
        raise TypeError(f'Expected a subclass of PyTreeEntry, got {path_entry_type!r}.')

    # Check for dunder-styled methods first (preferred since 0.18.0)
    if not all(
        callable(getattr(cls, method, None))
        for method in ('__tree_flatten__', '__tree_unflatten__')
    ):
        # Check for old-styled methods (backward compatibility)
        if not all(
            callable(getattr(cls, method, None)) for method in ('tree_flatten', 'tree_unflatten')
        ):
            raise TypeError(
                f'{cls!r} must define both `__tree_flatten__` and `__tree_unflatten__` methods '
                'for registration as a pytree node.',
            )

        # Add dunder-styled wrapper methods to the class
        # pylint: disable=no-member

        @functools.wraps(cls.tree_flatten)
        def __tree_flatten__(  # noqa: N807
            self: CustomTreeNode[T],
            /,
        ) -> tuple[Children[T], MetaData] | tuple[Children[T], MetaData, Iterable[Any] | None]:
            return self.tree_flatten()  # type: ignore[attr-defined]

        @classmethod  # type: ignore[misc]
        @functools.wraps(getattr(cls.tree_unflatten, '__func__', cls.tree_unflatten))
        def __tree_unflatten__(  # noqa: N807
            cls: type[CustomTreeNode[T]],
            metadata: MetaData,
            children: Children[T],
            /,
        ) -> CustomTreeNode[T]:
            return cls.tree_unflatten(metadata, children)  # type: ignore[attr-defined]

        # pylint: enable=no-member

        cls.__tree_flatten__ = __tree_flatten__
        cls.__tree_unflatten__ = __tree_unflatten__

    register_pytree_node(
        cls,
        methodcaller('__tree_flatten__'),
        cls.__tree_unflatten__,
        path_entry_type=path_entry_type,
        namespace=namespace,
    )
    return cls


def unregister_pytree_node(cls: type, /, *, namespace: str) -> PyTreeNodeRegistryEntry:
    """Remove a type from the pytree node registry.

    See also :func:`register_pytree_node` and :func:`register_pytree_node_class`.

    This function is the inverse operation of function :func:`register_pytree_node`.

    Args:
        cls (type): A Python type to remove from the pytree node registry.
        namespace (str): The namespace of the pytree node registry to remove the type from.

    Returns:
        The removed registry entry.

    Raises:
        TypeError: If the input type is not a class.
        TypeError: If the namespace is not a string.
        ValueError: If the namespace is an empty string.
        ValueError: If the type is a built-in type that cannot be unregistered.
        ValueError: If the type is not found in the registry.

    Examples:
        >>> # Register a Python type with lambda functions
        >>> register_pytree_node(
        ...     set,
        ...     lambda s: (sorted(s), None, None),
        ...     lambda _, children: set(children),
        ...     namespace='temp',
        ... )
        <class 'set'>

        >>> # Unregister the Python type
        >>> unregister_pytree_node(set, namespace='temp')
    """
    if not inspect.isclass(cls):
        raise TypeError(f'Expected a class, got {cls!r}.')
    if namespace is not __GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace!r}.')
    if namespace == '':
        raise ValueError('The namespace cannot be an empty string.')

    registration_key: type | tuple[str, type]
    if namespace is __GLOBAL_NAMESPACE:
        registration_key = cls
        namespace = ''
    else:
        registration_key = (namespace, cls)

    with __REGISTRY_LOCK:
        _C.unregister_node(cls, namespace)
        return _NODETYPE_REGISTRY.pop(registration_key)


@contextlib.contextmanager
def dict_insertion_ordered(mode: bool, /, *, namespace: str) -> Generator[None]:
    """Context manager to temporarily set the dictionary sorting mode.

    This context manager is used to temporarily set the dictionary sorting mode for a specific
    namespace. The dictionary sorting mode is used to determine whether the keys of a dictionary
    should be sorted or keeping the insertion order when flattening a pytree.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [1, 2, 3, 4, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    )
    >>> with dict_insertion_ordered(True, namespace='some-namespace'):  # doctest: +IGNORE_WHITESPACE
    ...     tree_flatten(tree, namespace='some-namespace')
    (
        [2, 3, 4, 1, 5],
        PyTreeSpec({'b': (*, [*, *]), 'a': *, 'c': None, 'd': *}, namespace='some-namespace')
    )

    .. warning::
        The dictionary sorting mode is a global setting and is **not thread-safe**. It is
        recommended to use this context manager in a single-threaded environment.

    Args:
        mode (bool): The dictionary sorting mode to set.
        namespace (str): The namespace to set the dictionary sorting mode for.
    """
    if namespace is not __GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace!r}.')
    if namespace == '':
        raise ValueError('The namespace cannot be an empty string.')
    if namespace is __GLOBAL_NAMESPACE:
        namespace = ''

    with __REGISTRY_LOCK:
        prev = _C.is_dict_insertion_ordered(namespace, inherit_global_namespace=False)
        _C.set_dict_insertion_ordered(bool(mode), namespace)

    try:
        yield
    finally:
        with __REGISTRY_LOCK:
            _C.set_dict_insertion_ordered(prev, namespace)


def _sorted_items(items: Iterable[tuple[KT, VT]], /) -> list[tuple[KT, VT]]:
    return total_order_sorted(items, key=itemgetter(0))


def _none_flatten(_: None, /) -> tuple[tuple[()], None]:
    return (), None


def _none_unflatten(_: None, children: Iterable[Any], /) -> None:
    sentinel = object()
    if next(iter(children), sentinel) is not sentinel:
        raise ValueError('Expected no children.')


def _tuple_flatten(tup: tuple[T, ...], /) -> tuple[tuple[T, ...], None]:
    return tup, None


def _tuple_unflatten(_: None, children: Iterable[T], /) -> tuple[T, ...]:
    return tuple(children)


def _list_flatten(lst: list[T], /) -> tuple[list[T], None]:
    return lst, None


def _list_unflatten(_: None, children: Iterable[T], /) -> list[T]:
    return list(children)


def _dict_flatten(dct: dict[KT, VT], /) -> tuple[tuple[VT, ...], list[KT], tuple[KT, ...]]:
    keys, values = unzip2(_sorted_items(dct.items()))
    return values, list(keys), keys


def _dict_unflatten(keys: list[KT], values: Iterable[VT], /) -> dict[KT, VT]:
    return dict(safe_zip(keys, values))


def _dict_insertion_ordered_flatten(
    dct: dict[KT, VT],
    /,
) -> tuple[
    tuple[VT, ...],
    list[KT],
    tuple[KT, ...],
]:
    keys, values = unzip2(dct.items())
    return values, list(keys), keys


def _dict_insertion_ordered_unflatten(keys: list[KT], values: Iterable[VT], /) -> dict[KT, VT]:
    return dict(safe_zip(keys, values))


def _ordereddict_flatten(
    dct: OrderedDict[KT, VT],
    /,
) -> tuple[
    tuple[VT, ...],
    list[KT],
    tuple[KT, ...],
]:
    keys, values = unzip2(dct.items())
    return values, list(keys), keys


def _ordereddict_unflatten(keys: list[KT], values: Iterable[VT], /) -> OrderedDict[KT, VT]:
    return OrderedDict(safe_zip(keys, values))


def _defaultdict_flatten(
    dct: defaultdict[KT, VT],
    /,
) -> tuple[
    tuple[VT, ...],
    tuple[Callable[[], VT] | None, list[KT]],
    tuple[KT, ...],
]:
    values, dict_metadata, entries = _dict_flatten(dct)
    return values, (dct.default_factory, dict_metadata), entries


def _defaultdict_unflatten(
    metadata: tuple[Callable[[], VT], list[KT]],
    values: Iterable[VT],
    /,
) -> defaultdict[KT, VT]:
    default_factory, dict_metadata = metadata
    return defaultdict(default_factory, _dict_unflatten(dict_metadata, values))


def _defaultdict_insertion_ordered_flatten(
    dct: defaultdict[KT, VT],
    /,
) -> tuple[
    tuple[VT, ...],
    tuple[Callable[[], VT] | None, list[KT]],
    tuple[KT, ...],
]:
    values, dict_metadata, entries = _dict_insertion_ordered_flatten(dct)
    return values, (dct.default_factory, dict_metadata), entries


def _defaultdict_insertion_ordered_unflatten(
    metadata: tuple[Callable[[], VT], list[KT]],
    values: Iterable[VT],
    /,
) -> defaultdict[KT, VT]:
    default_factory, dict_metadata = metadata
    return defaultdict(default_factory, _dict_insertion_ordered_unflatten(dict_metadata, values))


def _deque_flatten(deq: deque[T], /) -> tuple[deque[T], int | None]:
    return deq, deq.maxlen


def _deque_unflatten(maxlen: int | None, children: Iterable[T], /) -> deque[T]:
    return deque(children, maxlen=maxlen)


def _namedtuple_flatten(tup: NamedTuple[T], /) -> tuple[tuple[T, ...], type[NamedTuple[T]]]:  # type: ignore[type-arg]
    return tup, type(tup)


# pylint: disable-next=line-too-long
def _namedtuple_unflatten(cls: type[NamedTuple[T]], children: Iterable[T], /) -> NamedTuple[T]:  # type: ignore[type-arg]
    return cls(*children)  # type: ignore[call-overload]


def _structseq_flatten(seq: StructSequence[T], /) -> tuple[tuple[T, ...], type[StructSequence[T]]]:
    return seq, type(seq)


def _structseq_unflatten(
    cls: type[StructSequence[T]],
    children: Iterable[T],
    /,
) -> StructSequence[T]:
    return cls(children)


_NODETYPE_REGISTRY: dict[type | tuple[str, type], PyTreeNodeRegistryEntry] = {
    type(None): PyTreeNodeRegistryEntry(
        type(None),  # type: ignore[arg-type]
        _none_flatten,
        _none_unflatten,
        path_entry_type=PyTreeEntry,
        kind=PyTreeKind.NONE,
    ),
    tuple: PyTreeNodeRegistryEntry(
        tuple,
        _tuple_flatten,
        _tuple_unflatten,
        path_entry_type=SequenceEntry,
        kind=PyTreeKind.TUPLE,
    ),
    list: PyTreeNodeRegistryEntry(
        list,
        _list_flatten,
        _list_unflatten,
        path_entry_type=SequenceEntry,
        kind=PyTreeKind.LIST,
    ),
    dict: PyTreeNodeRegistryEntry(
        dict,
        _dict_flatten,
        _dict_unflatten,
        path_entry_type=MappingEntry,
        kind=PyTreeKind.DICT,
    ),
    namedtuple: PyTreeNodeRegistryEntry(  # type: ignore[dict-item] # noqa: PYI024
        namedtuple,  # type: ignore[arg-type] # noqa: PYI024
        _namedtuple_flatten,
        _namedtuple_unflatten,
        path_entry_type=NamedTupleEntry,
        kind=PyTreeKind.NAMEDTUPLE,
    ),
    OrderedDict: PyTreeNodeRegistryEntry(
        OrderedDict,
        _ordereddict_flatten,
        _ordereddict_unflatten,
        path_entry_type=MappingEntry,
        kind=PyTreeKind.ORDEREDDICT,
    ),
    defaultdict: PyTreeNodeRegistryEntry(
        defaultdict,
        _defaultdict_flatten,
        _defaultdict_unflatten,
        path_entry_type=MappingEntry,
        kind=PyTreeKind.DEFAULTDICT,
    ),
    deque: PyTreeNodeRegistryEntry(
        deque,
        _deque_flatten,
        _deque_unflatten,
        path_entry_type=SequenceEntry,
        kind=PyTreeKind.DEQUE,
    ),
    StructSequence: PyTreeNodeRegistryEntry(
        StructSequence,
        _structseq_flatten,
        _structseq_unflatten,
        path_entry_type=StructSequenceEntry,
        kind=PyTreeKind.STRUCTSEQUENCE,
    ),
}


_DICT_INSERTION_ORDERED_REGISTRY_ENTRY = PyTreeNodeRegistryEntry(
    dict,
    _dict_insertion_ordered_flatten,
    _dict_insertion_ordered_unflatten,
    path_entry_type=MappingEntry,
    kind=PyTreeKind.DICT,
)
_DEFAULTDICT_INSERTION_ORDERED_REGISTRY_ENTRY = PyTreeNodeRegistryEntry(
    defaultdict,
    _defaultdict_insertion_ordered_flatten,
    _defaultdict_insertion_ordered_unflatten,
    path_entry_type=MappingEntry,
    kind=PyTreeKind.DEFAULTDICT,
)
