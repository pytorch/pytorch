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
"""PyTree integration with :mod:`functools`."""

from __future__ import annotations

import functools
from typing import Any, Callable, ClassVar
from typing_extensions import Self  # Python 3.11+
from typing_extensions import deprecated  # Python 3.13+

from optree import registry
from optree.accessor import GetAttrEntry, PyTreeEntry
from optree.ops import tree_reduce as reduce
from optree.typing import CustomTreeNode, T


__all__ = [
    'partial',
    'reduce',
]


class _HashablePartialShim:
    """Object that delegates :meth:`__call__`, :meth:`__eq__`, and :meth:`__hash__` to another object."""

    __slots__: ClassVar[tuple[str, ...]] = ('partial_func', 'func', 'args', 'keywords')

    func: Callable[..., Any]
    args: tuple[Any, ...]
    keywords: dict[str, Any]

    def __init__(self, partial_func: functools.partial) -> None:
        self.partial_func: functools.partial = partial_func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.partial_func(*args, **kwargs)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _HashablePartialShim):
            return self.partial_func == other.partial_func
        return self.partial_func == other

    def __hash__(self) -> int:
        return hash(self.partial_func)

    def __repr__(self) -> str:
        return repr(self.partial_func)


# pylint: disable-next=protected-access
@registry.register_pytree_node_class(namespace=registry.__GLOBAL_NAMESPACE)
class partial(  # noqa: N801 # pylint: disable=invalid-name,too-few-public-methods
    functools.partial,
    CustomTreeNode[T],
):
    """A version of :func:`functools.partial` that works in pytrees.

    Use it for partial function evaluation in a way that is compatible with transformations,
    e.g., ``partial(func, *args, **kwargs)``.

    (You need to explicitly opt-in to this behavior because we did not want to give
    :func:`functools.partial` different semantics than normal function closures.)

    For example, here is a basic usage of :class:`partial` in a manner similar to
    :func:`functools.partial`:

    >>> import operator
    >>> import torch
    >>> add_one = partial(operator.add, torch.ones(()))
    >>> add_one(torch.tensor([[1, 2], [3, 4]]))
    tensor([[2., 3.],
            [4., 5.]])

    Pytree compatibility means that the resulting partial function can be passed as an argument
    within tree-map functions, which is not possible with a standard :func:`functools.partial`
    function:

    >>> def call_func_on_cuda(f, *args, **kwargs):
    ...     f, args, kwargs = tree_map(lambda t: t.cuda(), (f, args, kwargs))
    ...     return f(*args, **kwargs)
    ...
    >>> # doctest: +SKIP
    >>> tree_map(lambda t: t.cuda(), add_one)
    optree.functools.partial(<built-in function add>, tensor(1., device='cuda:0'))
    >>> call_func_on_cuda(add_one, torch.tensor([[1, 2], [3, 4]]))
    tensor([[2., 3.],
            [4., 5.]], device='cuda:0')

    Passing zero arguments to :class:`partial` effectively wraps the original function, making it a
    valid argument in tree-map functions:

    >>> # doctest: +SKIP
    >>> call_func_on_cuda(partial(torch.add), torch.tensor(1), torch.tensor(2))
    tensor(3, device='cuda:0')

    Had we passed :func:`operator.add` to ``call_func_on_cuda`` directly, it would have resulted in
    a :class:`TypeError` or :class:`AttributeError`.
    """

    __slots__: ClassVar[tuple[()]] = ()

    func: Callable[..., Any]
    args: tuple[T, ...]
    keywords: dict[str, T]

    TREE_PATH_ENTRY_TYPE: ClassVar[type[PyTreeEntry]] = GetAttrEntry

    def __new__(cls, func: Callable[..., Any], *args: T, **keywords: T) -> Self:
        """Create a new :class:`partial` instance."""
        # In Python 3.10+, if func is itself a functools.partial instance, functools.partial.__new__
        # would merge the arguments of this partial instance with the arguments of the func. We box
        # func in a class that does not (yet) have a `func` attribute to defeat this optimization,
        # since we care exactly which arguments are considered part of the pytree.
        if isinstance(func, functools.partial):
            original_func = func
            func = _HashablePartialShim(original_func)
            assert not hasattr(func, 'func'), 'shimmed function should not have a `func` attribute'
            out = super().__new__(cls, func, *args, **keywords)
            func.func = original_func.func
            func.args = original_func.args
            func.keywords = original_func.keywords
            return out

        return super().__new__(cls, func, *args, **keywords)

    def __repr__(self) -> str:
        """Return a string representation of the :class:`partial` instance."""
        args = [repr(self.func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f'{k}={v!r}' for (k, v) in self.keywords.items())
        return f'{self.__class__.__module__}.{self.__class__.__qualname__}({", ".join(args)})'

    def tree_flatten(self) -> tuple[  # type: ignore[override]
        tuple[tuple[T, ...], dict[str, T]],
        Callable[..., Any],
        tuple[str, str],
    ]:
        """Flatten the :class:`partial` instance to children and metadata."""
        return (self.args, self.keywords), self.func, ('args', 'keywords')

    @classmethod
    def tree_unflatten(  # type: ignore[override]
        cls,
        metadata: Callable[..., Any],
        children: tuple[tuple[T, ...], dict[str, T]],
    ) -> Self:
        """Unflatten the children and metadata into a :class:`partial` instance."""
        args, keywords = children
        return cls(metadata, *args, **keywords)


# pylint: disable-next=protected-access
@registry.register_pytree_node_class(namespace=registry.__GLOBAL_NAMESPACE)
@deprecated(
    'The class `optree.Partial` is deprecated and will be removed in a future version. '
    'Please use `optree.functools.partial` instead.',
    category=FutureWarning,
)
class Partial(partial):
    """Deprecated alias for :class:`partial`."""

    __slots__: ClassVar[tuple[()]] = ()
