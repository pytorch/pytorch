# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
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
"""PyTree integration with :mod:`dataclasses`.

This module implements PyTree integration with :mod:`dataclasses` by redefining the :func:`field`,
:func:`dataclass`, and :func:`make_dataclass` functions. Other APIs are re-exported from the
original :mod:`dataclasses` module.

The PyTree integration allows dataclasses to be flattened and unflattened recursively. The fields
are stored in a special attribute named ``__optree_dataclass_fields__`` in the dataclass.

>>> import math
... import optree
...
>>> @optree.dataclasses.dataclass(namespace='my_module')
... class Point:
...     x: float
...     y: float
...     z: float = 0.0
...     norm: float = optree.dataclasses.field(init=False, pytree_node=False)
...
...     def __post_init__(self) -> None:
...         self.norm = math.hypot(self.x, self.y, self.z)
...
>>> point = Point(2.0, 6.0, 3.0)
>>> point
Point(x=2.0, y=6.0, z=3.0, norm=7.0)
>>> # Flatten without specifying the namespace
>>> optree.tree_flatten(point)  # `Point`s are leaf nodes
([Point(x=2.0, y=6.0, z=3.0, norm=7.0)], PyTreeSpec(*))
>>> # Flatten with the namespace
>>> accessors, leaves, treespec = optree.tree_flatten_with_accessor(point, namespace='my_module')
>>> accessors, leaves, treespec  # doctest: +IGNORE_WHITESPACE,ELLIPSIS
(
    [
        PyTreeAccessor(*.x, (DataclassEntry(field='x', type=<class '...Point'>),)),
        PyTreeAccessor(*.y, (DataclassEntry(field='y', type=<class '...Point'>),)),
        PyTreeAccessor(*.z, (DataclassEntry(field='z', type=<class '...Point'>),))
    ],
    [2.0, 6.0, 3.0],
    PyTreeSpec(CustomTreeNode(Point[()], [*, *, *]), namespace='my_module')
)
>>> point == optree.tree_unflatten(treespec, leaves)
True
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import sys
import types
from dataclasses import *  # noqa: F401,F403,RUF100 # pylint: disable=wildcard-import,unused-wildcard-import
from typing import Any, Callable, Iterable, TypeVar, overload
from typing_extensions import Literal  # Python 3.8+
from typing_extensions import dataclass_transform  # Python 3.11+

from optree.accessor import DataclassEntry
from optree.registry import register_pytree_node


# Redefine `field`, `dataclasses`, and `make_dataclasses`.
# The remaining APIs are re-exported from the original package.
__all__ = [*dataclasses.__all__]


_FIELDS = '__optree_dataclass_fields__'
_PYTREE_NODE_DEFAULT: bool = True


def field(  # type: ignore[no-redef] # pylint: disable=function-redefined,too-many-arguments
    *,
    default: Any = dataclasses.MISSING,
    default_factory: Any = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    hash: bool | None = None,  # pylint: disable=redefined-builtin
    compare: bool = True,
    metadata: dict[str, Any] | None = None,
    kw_only: bool | Literal[dataclasses.MISSING] = dataclasses.MISSING,  # type: ignore[valid-type] # Python 3.10+
    pytree_node: bool | None = None,
) -> dataclasses.Field:
    """Field factory for :func:`dataclass`.

    This factory function is used to define the fields in a dataclass. It is similar to the field
    factory :func:`dataclasses.field`, but with an additional ``pytree_node`` parameter. If
    ``pytree_node`` is :data:`True` (default), the field will be considered a child node in the
    PyTree structure which can be recursively flattened and unflattened. Otherwise, the field will
    be considered as PyTree metadata.

    Setting ``pytree_node`` in the field factory is equivalent to setting a key ``'pytree_node'`` in
    ``metadata`` in the original field factory. The ``pytree_node`` value can be accessed using
    ``field.metadata['pytree_node']``. If ``pytree_node`` is :data:`None`, the value
    ``metadata.get('pytree_node', True)`` will be used.

    .. note::
        If a field is considered a child node, it must be included in the argument list of the
        :meth:`__init__` method, i.e., passes ``init=True`` in the field factory.

    Args:
        pytree_node (bool or None, optional): Whether the field is a PyTree node.
        **kwargs (optional): Optional keyword arguments passed to :func:`dataclasses.field`.

    Returns:
        dataclasses.Field: The field defined using the provided arguments with
        ``field.metadata['pytree_node']`` set.
    """
    metadata = (metadata or {}).copy()
    if pytree_node is None:
        pytree_node = metadata.get('pytree_node', _PYTREE_NODE_DEFAULT)
    metadata['pytree_node'] = pytree_node

    kwargs = {
        'default': default,
        'default_factory': default_factory,
        'init': init,
        'repr': repr,
        'hash': hash,
        'compare': compare,
        'metadata': metadata,
    }

    if sys.version_info >= (3, 10):
        kwargs['kw_only'] = kw_only
    elif kw_only is not dataclasses.MISSING:
        raise TypeError("field() got an unexpected keyword argument 'kw_only'")

    if not init and pytree_node:
        raise TypeError(
            '`pytree_node=True` is not allowed for non-init fields. '
            f'Please explicitly set `{__name__}.field(init=False, pytree_node=False)`.',
        )

    return dataclasses.field(**kwargs)  # pylint: disable=invalid-field-call


_T = TypeVar('_T')
_U = TypeVar('_U')
_TypeT = TypeVar('_TypeT', bound=type)


@overload  # type: ignore[no-redef]
@dataclass_transform(field_specifiers=(field,))
def dataclass(  # pylint: disable=too-many-arguments
    cls: None,
    *,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,  # Python 3.10+
    kw_only: bool = False,  # Python 3.10+
    slots: bool = False,  # Python 3.10+
    weakref_slot: bool = False,  # Python 3.11+
    namespace: str,
) -> Callable[[_TypeT], _TypeT]:
    """Dataclass decorator with PyTree integration."""


@overload
@dataclass_transform(field_specifiers=(field,))
def dataclass(  # pylint: disable=too-many-arguments
    cls: _TypeT,
    *,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,  # Python 3.10+
    kw_only: bool = False,  # Python 3.10+
    slots: bool = False,  # Python 3.10+
    weakref_slot: bool = False,  # Python 3.11+
    namespace: str,
) -> _TypeT: ...


@dataclass_transform(field_specifiers=(field,))
def dataclass(  # noqa: C901 # pylint: disable=function-redefined,too-many-arguments,too-many-locals,too-many-branches
    cls: _TypeT | None = None,
    *,
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,  # Python 3.10+
    kw_only: bool = False,  # Python 3.10+
    slots: bool = False,  # Python 3.10+
    weakref_slot: bool = False,  # Python 3.11+
    namespace: str,
) -> _TypeT | Callable[[_TypeT], _TypeT]:
    """Dataclass decorator with PyTree integration.

    Args:
        cls (type or None, optional): The class to decorate. If :data:`None`, return a decorator.
        namespace (str): The registry namespace used for the PyTree registration.
        **kwargs (optional): Optional keyword arguments passed to :func:`dataclasses.dataclass`.

    Returns:
        type or callable: The decorated class with PyTree integration or decorator function.
    """
    # pylint: disable-next=import-outside-toplevel
    from optree.registry import __GLOBAL_NAMESPACE as GLOBAL_NAMESPACE

    kwargs = {
        'init': init,
        'repr': repr,
        'eq': eq,
        'order': order,
        'unsafe_hash': unsafe_hash,
        'frozen': frozen,
    }

    if sys.version_info >= (3, 10):
        kwargs['match_args'] = match_args
        kwargs['kw_only'] = kw_only
        kwargs['slots'] = slots
    elif match_args is not True:
        raise TypeError("dataclass() got an unexpected keyword argument 'match_args'")
    elif kw_only is not False:
        raise TypeError("dataclass() got an unexpected keyword argument 'kw_only'")
    elif slots is not False:
        raise TypeError("dataclass() got an unexpected keyword argument 'slots'")

    if sys.version_info >= (3, 11):
        kwargs['weakref_slot'] = weakref_slot
    elif weakref_slot is not False:
        raise TypeError("dataclass() got an unexpected keyword argument 'weakref_slot'")

    if cls is None:

        def decorator(cls: _TypeT) -> _TypeT:
            return dataclass(cls, namespace=namespace, **kwargs)  # type: ignore[call-overload]

        return decorator

    if not inspect.isclass(cls):
        raise TypeError(f'@{__name__}.dataclass() can only be used with classes, not {cls!r}.')
    if _FIELDS in cls.__dict__:
        raise TypeError(
            f'@{__name__}.dataclass() cannot be applied to {cls.__name__} more than once.',
        )
    if namespace is not GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace!r}.')
    if namespace == '':
        raise ValueError('The namespace cannot be an empty string.')

    cls = dataclasses.dataclass(cls, **kwargs)  # type: ignore[assignment]

    children_fields = {}
    metadata_fields = {}
    for f in dataclasses.fields(cls):
        if f.metadata.get('pytree_node', _PYTREE_NODE_DEFAULT):
            if not f.init:
                raise TypeError(
                    f'PyTree node field {f.name!r} must be included in `__init__()`. '
                    f'Or you can explicitly set `{__name__}.field(init=False, pytree_node=False)`.',
                )
            children_fields[f.name] = f
        elif f.init:
            metadata_fields[f.name] = f

    children_field_names = tuple(children_fields)
    children_fields = types.MappingProxyType(children_fields)
    metadata_fields = types.MappingProxyType(metadata_fields)
    setattr(cls, _FIELDS, (children_fields, metadata_fields))

    def flatten_func(
        obj: _T,
    ) -> tuple[
        tuple[_U, ...],
        tuple[tuple[str, Any], ...],
        tuple[str, ...],
    ]:
        children = tuple(getattr(obj, name) for name in children_field_names)
        metadata = tuple((name, getattr(obj, name)) for name in metadata_fields)
        return children, metadata, children_field_names

    def unflatten_func(metadata: tuple[tuple[str, Any], ...], children: tuple[_U, ...]) -> _T:  # type: ignore[type-var]
        kwargs = dict(zip(children_field_names, children))
        kwargs.update(metadata)
        return cls(**kwargs)

    return register_pytree_node(  # type: ignore[return-value]
        cls,
        flatten_func,
        unflatten_func,  # type: ignore[arg-type]
        path_entry_type=DataclassEntry,
        namespace=namespace,
    )


# pylint: disable-next=function-redefined,too-many-arguments,too-many-locals,too-many-branches
def make_dataclass(  # type: ignore[no-redef] # noqa: C901
    cls_name: str,
    # pylint: disable-next=redefined-outer-name
    fields: Iterable[str | tuple[str, Any] | tuple[str, Any, Any]],
    *,
    bases: tuple[type, ...] = (),
    ns: dict[str, Any] | None = None,  # redirect to `namespace` to `dataclasses.make_dataclass()`
    init: bool = True,
    repr: bool = True,  # pylint: disable=redefined-builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,  # Python 3.10+
    kw_only: bool = False,  # Python 3.10+
    slots: bool = False,  # Python 3.10+
    weakref_slot: bool = False,  # Python 3.11+
    module: str | None = None,  # Python 3.12+
    namespace: str,  # the PyTree registration namespace
) -> type:
    """Make a new dynamically created dataclass with PyTree integration.

    The dataclass name will be ``cls_name``. ``fields`` is an iterable of either (name), (name, type),
    or (name, type, Field) objects. If type is omitted, use the string :data:`typing.Any`. Field
    objects are created by the equivalent of calling :func:`field` (name, type [, Field-info]).

    The ``namespace`` parameter is the PyTree registration namespace which should be a string. The
    ``namespace`` in the original :func:`dataclasses.make_dataclass` function is renamed to ``ns``
    to avoid conflicts.

    The remaining parameters are passed to :func:`dataclasses.make_dataclass`.
    See :func:`dataclasses.make_dataclass` for more information.

    Args:
        cls_name: The name of the dataclass.
        fields (Iterable[str | tuple[str, Any] | tuple[str, Any, Any]]): An iterable of either
            (name), (name, type), or (name, type, Field) objects.
        namespace (str): The registry namespace used for the PyTree registration.
        ns (dict or None, optional): The namespace used in dynamic type creation.
            See :func:`dataclasses.make_dataclass` and the builtin :func:`type` function for more
            information.
        **kwargs (optional): Optional keyword arguments passed to :func:`dataclasses.make_dataclass`.

    Returns:
        type: The dynamically created dataclass with PyTree integration.
    """
    # pylint: disable-next=import-outside-toplevel
    from optree.registry import __GLOBAL_NAMESPACE as GLOBAL_NAMESPACE

    if isinstance(namespace, dict) or namespace is None:  # type: ignore[unreachable]
        if ns is GLOBAL_NAMESPACE or isinstance(ns, str):  # type: ignore[unreachable]
            ns, namespace = namespace, ns
        elif ns is None:
            raise TypeError("make_dataclass() missing 1 required keyword-only argument: 'ns'")
    if namespace is not GLOBAL_NAMESPACE and not isinstance(namespace, str):
        raise TypeError(f'The namespace must be a string, got {namespace!r}.')
    if namespace == '':
        raise ValueError('The namespace cannot be an empty string.')

    dataclass_kwargs = {
        'init': init,
        'repr': repr,
        'eq': eq,
        'order': order,
        'unsafe_hash': unsafe_hash,
        'frozen': frozen,
    }
    make_dataclass_kwargs = {
        'bases': bases,
        'namespace': ns,
    }

    if sys.version_info >= (3, 10):
        dataclass_kwargs['match_args'] = match_args
        dataclass_kwargs['kw_only'] = kw_only
        dataclass_kwargs['slots'] = slots
    elif match_args is not True:
        raise TypeError("make_dataclass() got an unexpected keyword argument 'match_args'")
    elif kw_only is not False:
        raise TypeError("make_dataclass() got an unexpected keyword argument 'kw_only'")
    elif slots is not False:
        raise TypeError("make_dataclass() got an unexpected keyword argument 'slots'")

    if sys.version_info >= (3, 11):
        dataclass_kwargs['weakref_slot'] = weakref_slot
    elif weakref_slot is not False:
        raise TypeError("make_dataclass() got an unexpected keyword argument 'weakref_slot'")

    if sys.version_info >= (3, 12):
        if module is None:
            try:
                # pylint: disable-next=protected-access
                module = sys._getframemodulename(1) or '__main__'  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover
                with contextlib.suppress(AttributeError, ValueError):
                    # pylint: disable-next=protected-access
                    module = sys._getframe(1).f_globals.get('__name__', '__main__')
        make_dataclass_kwargs['module'] = module
    elif module is not None:
        raise TypeError("make_dataclass() got an unexpected keyword argument'module'")

    cls = dataclasses.make_dataclass(
        cls_name,
        fields=fields,
        **dataclass_kwargs,  # type: ignore[arg-type]
        **make_dataclass_kwargs,  # type: ignore[arg-type]
    )
    dataclass_kwargs.pop('slots', None)  # already defined in `make_dataclass()`
    dataclass_kwargs.pop('weakref_slot', None)  # already used in `make_dataclass()`
    return dataclass(cls, **dataclass_kwargs, namespace=namespace)  # type: ignore[call-overload]
