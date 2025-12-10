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
r"""Utilities for working with ``PyTree``\s.

The :mod:`optree.pytree` namespace contains aliases of ``optree.tree_*`` utilities.

>>> import optree.pytree as pytree
>>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
>>> leaves, treespec = pytree.flatten(tree)
>>> leaves, treespec  # doctest: +IGNORE_WHITESPACE
(
    [1, 2, 3, 4, 5],
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
)
>>> tree == pytree.unflatten(treespec, leaves)
True

.. versionadded:: 0.14.1
"""

from __future__ import annotations

import functools as _functools
import inspect as _inspect
import sys as _sys
from builtins import all as _all
from types import ModuleType as _ModuleType
from typing import TYPE_CHECKING as _TYPE_CHECKING

import optree.dataclasses as dataclasses
import optree.functools as functools
from optree.accessors import PyTreeEntry
from optree.ops import tree_accessors as accessors
from optree.ops import tree_all as all  # pylint: disable=redefined-builtin
from optree.ops import tree_any as any  # pylint: disable=redefined-builtin
from optree.ops import tree_broadcast_common as broadcast_common
from optree.ops import tree_broadcast_map as broadcast_map
from optree.ops import tree_broadcast_map_with_accessor as broadcast_map_with_accessor
from optree.ops import tree_broadcast_map_with_path as broadcast_map_with_path
from optree.ops import tree_broadcast_prefix as broadcast_prefix
from optree.ops import tree_flatten as flatten
from optree.ops import tree_flatten_one_level as flatten_one_level
from optree.ops import tree_flatten_with_accessor as flatten_with_accessor
from optree.ops import tree_flatten_with_path as flatten_with_path
from optree.ops import tree_is_leaf as is_leaf
from optree.ops import tree_iter as iter  # pylint: disable=redefined-builtin
from optree.ops import tree_leaves as leaves
from optree.ops import tree_map as map  # pylint: disable=redefined-builtin
from optree.ops import tree_map_ as map_
from optree.ops import tree_map_with_accessor as map_with_accessor
from optree.ops import tree_map_with_accessor_ as map_with_accessor_
from optree.ops import tree_map_with_path as map_with_path
from optree.ops import tree_map_with_path_ as map_with_path_
from optree.ops import tree_max as max  # pylint: disable=redefined-builtin
from optree.ops import tree_min as min  # pylint: disable=redefined-builtin
from optree.ops import tree_partition as partition
from optree.ops import tree_paths as paths
from optree.ops import tree_reduce as reduce
from optree.ops import tree_replace_nones as replace_nones
from optree.ops import tree_structure as structure
from optree.ops import tree_sum as sum  # pylint: disable=redefined-builtin
from optree.ops import tree_transpose as transpose
from optree.ops import tree_transpose_map as transpose_map
from optree.ops import tree_transpose_map_with_accessor as transpose_map_with_accessor
from optree.ops import tree_transpose_map_with_path as transpose_map_with_path
from optree.ops import tree_unflatten as unflatten
from optree.registry import dict_insertion_ordered
from optree.registry import register_pytree_node as register_node
from optree.registry import register_pytree_node_class as register_node_class
from optree.registry import unregister_pytree_node as unregister_node
from optree.typing import PyTreeKind, PyTreeSpec
from optree.version import __version__ as __version__  # pylint: disable=useless-import-alias


__all__ = [
    'reexport',
    'PyTreeSpec',
    'PyTreeKind',
    'PyTreeEntry',
    'flatten',
    'flatten_with_path',
    'flatten_with_accessor',
    'unflatten',
    'iter',
    'leaves',
    'structure',
    'paths',
    'accessors',
    'is_leaf',
    'map',
    'map_',
    'map_with_path',
    'map_with_path_',
    'map_with_accessor',
    'map_with_accessor_',
    'replace_nones',
    'partition',
    'transpose',
    'transpose_map',
    'transpose_map_with_path',
    'transpose_map_with_accessor',
    'broadcast_prefix',
    'broadcast_common',
    'broadcast_map',
    'broadcast_map_with_path',
    'broadcast_map_with_accessor',
    'reduce',
    'sum',
    'max',
    'min',
    'all',
    'any',
    'flatten_one_level',
    'register_node',
    'register_node_class',
    'unregister_node',
    'dict_insertion_ordered',
]


if _TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any, TypeVar  # pylint: disable=ungrouped-imports
    from typing_extensions import ParamSpec  # Python 3.10+

    _P = ParamSpec('_P')
    _T = TypeVar('_T')


class ReexportedModule(_ModuleType):
    """A module that re-exports APIs from another module."""

    __doc__: str

    def __init__(
        self,
        /,
        name: str,
        *,
        namespace: str,
        original: _ModuleType,
        doc: str | None = None,
        __all__: Iterable[str] | None = None,
        __dir__: Iterable[str] | None = None,
        extra_members: dict[str, Any] | None = None,
    ) -> None:
        doc = doc or (
            f'Re-exports :mod:`{original.__name__}` as :mod:`{name}` '
            f'with namespace :const:`{namespace!r}`.'
        )
        super().__init__(name, doc)

        if __all__ is None:  # pragma: no branch
            __all__ = {n for n in original.__all__ if n != 'reexport'}
        __all__ = set(__all__)
        if __dir__ is None:  # pragma: no branch
            __dir__ = {n for n in original.__dir__() if not n.startswith('_') and n != 'reexport'}
        __dir__ = set(__dir__).intersection(__all__)

        if extra_members:
            for key, value in extra_members.items():
                setattr(self, key, value)
            __dir__.update(extra_members)

        self.__namespace = namespace
        self.__original = original
        self.__all_set = __all__
        self.__all = sorted(__all__)
        self.__dir = sorted(__dir__)

    @property
    def __all__(self, /) -> list[str]:
        """Return the list of attributes available in this module."""
        return self.__all

    def __dir__(self, /) -> list[str]:
        """Return the list of attributes available in this module."""
        return self.__dir.copy()

    def __getattr__(self, name: str, /) -> Any:
        """Get an attribute from the re-exported module."""
        if name in self.__all_set:
            attr = getattr(self.__original, name)
            if _inspect.isfunction(attr):
                attr = self.__reexport__(attr)
            setattr(self, name, attr)
            return attr
        raise AttributeError(f'module {self.__name__!r} has no attribute {name!r}')

    def __reexport__(self, func: Callable[_P, _T], /) -> Callable[_P, _T]:
        """Re-export a function with the default namespace."""
        sig = _inspect.signature(func)
        if 'namespace' not in sig.parameters:

            @_functools.wraps(func)
            def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
                return func(*args, **kwargs)
        else:

            @_functools.wraps(func)
            def wrapped(  # type: ignore[valid-type]
                *args: _P.args,
                namespace: str = self.__namespace,
                **kwargs: _P.kwargs,
            ) -> _T:
                return func(*args, namespace=namespace, **kwargs)  # type: ignore[arg-type]

            if func.__doc__:  # pragma: no branch
                wrapped.__doc__ = func.__doc__.replace(
                    "(default: :const:`''`, i.e., the global namespace)",
                    f'(default: :const:`{self.__namespace!r}`)',
                )
            wrapped.__signature__ = sig.replace(  # type: ignore[attr-defined]
                parameters=[
                    p if p.name != 'namespace' else p.replace(default=self.__namespace)
                    for p in sig.parameters.values()
                ],
            )

        if callable(getattr(func, 'get', None)):
            wrapped.get = self.__reexport__(func.get)  # type: ignore[attr-defined]

        return wrapped


if _TYPE_CHECKING:
    # pylint: disable-next=missing-class-docstring,too-few-public-methods
    class ReexportedPyTreeModule(ReexportedModule):
        __version__: str
        functools: _ModuleType
        dataclasses: _ModuleType

        PyTreeSpec: type[PyTreeSpec] = PyTreeSpec
        PyTreeKind: type[PyTreeKind] = PyTreeKind
        PyTreeEntry: type[PyTreeEntry] = PyTreeEntry
        flatten = staticmethod(flatten)
        flatten_with_path = staticmethod(flatten_with_path)
        flatten_with_accessor = staticmethod(flatten_with_accessor)
        unflatten = staticmethod(unflatten)
        iter = staticmethod(iter)
        leaves = staticmethod(leaves)
        structure = staticmethod(structure)
        paths = staticmethod(paths)
        accessors = staticmethod(accessors)
        is_leaf = staticmethod(is_leaf)
        map = staticmethod(map)
        map_ = staticmethod(map_)
        map_with_path = staticmethod(map_with_path)
        map_with_path_ = staticmethod(map_with_path_)
        map_with_accessor = staticmethod(map_with_accessor)
        map_with_accessor_ = staticmethod(map_with_accessor_)
        replace_nones = staticmethod(replace_nones)
        partition = staticmethod(partition)
        transpose = staticmethod(transpose)
        transpose_map = staticmethod(transpose_map)
        transpose_map_with_path = staticmethod(transpose_map_with_path)
        transpose_map_with_accessor = staticmethod(transpose_map_with_accessor)
        broadcast_prefix = staticmethod(broadcast_prefix)
        broadcast_common = staticmethod(broadcast_common)
        broadcast_map = staticmethod(broadcast_map)
        broadcast_map_with_path = staticmethod(broadcast_map_with_path)
        broadcast_map_with_accessor = staticmethod(broadcast_map_with_accessor)
        reduce = staticmethod(reduce)
        sum = staticmethod(sum)
        max = staticmethod(max)
        min = staticmethod(min)
        all = staticmethod(all)
        any = staticmethod(any)
        flatten_one_level = staticmethod(flatten_one_level)
        register_node = staticmethod(register_node)
        register_node_class = staticmethod(register_node_class)
        unregister_node = staticmethod(unregister_node)
        dict_insertion_ordered = staticmethod(dict_insertion_ordered)

    def reexport(*, namespace: str, module: str | None = None) -> ReexportedPyTreeModule:
        """Re-export a pytree utility module with the given namespace as default."""
        raise NotImplementedError('reexport() is not available in type checking mode')

else:

    def reexport(*, namespace: str, module: str | None = None) -> _ModuleType:  # type: ignore[misc]
        """Re-export a pytree utility module with the given namespace as default.

        >>> import optree
        >>> pytree = optree.pytree.reexport(namespace='my-pkg', module='my_pkg.pytree')
        >>> pytree.flatten({'a': 1, 'b': 2})
        ([1, 2], PyTreeSpec({'a': *, 'b': *}))

        This function is useful for downstream libraries that want to re-export the pytree utilities
        with their own namespace:

        .. code-block:: python

            # foo/__init__.py
            import optree
            pytree = optree.pytree.reexport(namespace='foo')
            del optree

            # foo/bar.py
            from foo import pytree

            @pytree.dataclasses.dataclass
            class Bar:
                a: int
                b: float

            # User code
            In [1]: import foo

            In [2]: foo.pytree.flatten({'a': 1, 'b': 2, 'c': foo.bar.Bar(3, 4.0)}))
            Out[2]:
            (
                [1, 2, 3, 4.0],
                PyTreeSpec({'a': *, 'b': *, 'c': CustomTreeNode(Bar[()], [*, *])}, namespace='foo')
            )

            In [3]: foo.pytree.functools.reduce(lambda x, y: x * y, {'a': 1, 'b': 2, 'c': foo.bar.Bar(3, 4.0)}))
            Out[3]: 24.0

        .. versionadded:: 0.16.0

        Args:
            namespace (str): The namespace to use in the re-exported module.
            module (str, optional): The name of the re-exported module.
                If not provided, defaults to ``<caller_module>.pytree``. The caller module is determined
                by inspecting the stack frame.

        Returns:
            The re-exported module.
        """
        # pylint: disable-next=import-outside-toplevel
        from optree.registry import __GLOBAL_NAMESPACE as GLOBAL_NAMESPACE

        if namespace is GLOBAL_NAMESPACE:
            namespace = ''
        elif not isinstance(namespace, str):
            raise TypeError(f'The namespace must be a string, got {namespace!r}.')

        if module is None:
            try:
                # pylint: disable-next=protected-access
                caller_module = _sys._getframemodulename(1) or '__main__'  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover
                try:
                    # pylint: disable-next=protected-access
                    caller_module = _sys._getframe(1).f_globals.get('__name__', '__main__')
                except (AttributeError, ValueError):
                    caller_module = '__main__'
            module = f'{caller_module}.pytree'
        if not module or not _all(part.isidentifier() for part in module.split('.')):
            raise ValueError(f'invalid module name: {module!r}')

        for module_name in (module, f'{module}.dataclasses', f'{module}.functools'):
            if module_name in _sys.modules:
                raise ValueError(f'module {module_name!r} already exists')

        reexported_dataclasses = ReexportedModule(
            f'{module}.dataclasses',
            namespace=namespace,
            original=dataclasses,
        )
        reexported_functools = ReexportedModule(
            f'{module}.functools',
            namespace=namespace,
            original=functools,
        )
        mod: ReexportedPyTreeModule = ReexportedModule(  # type: ignore[assignment]
            module,
            namespace=namespace,
            original=_sys.modules[__name__],
            extra_members={
                '__version__': __version__,
                'dataclasses': reexported_dataclasses,
                'functools': reexported_functools,
            },
        )
        _sys.modules[module] = mod
        _sys.modules[f'{module}.dataclasses'] = reexported_dataclasses
        _sys.modules[f'{module}.functools'] = reexported_functools
        return mod
