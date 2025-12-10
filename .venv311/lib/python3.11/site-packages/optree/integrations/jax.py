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
# This file is modified from:
# https://github.com/google/jax/blob/jax-v0.4.20/jax/_src/flatten_util.py
# ==============================================================================
# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Integration with JAX."""

# pragma: jax cover file
# pylint: disable=import-error

from __future__ import annotations

import contextlib
import itertools
import warnings
from operator import itemgetter
from types import FunctionType
from typing import Any, Callable
from typing_extensions import TypeAlias  # Python 3.10+

import jax.numpy as jnp
from jax import Array, lax
from jax._src import dtypes
from jax.typing import ArrayLike

from optree.ops import tree_flatten, tree_unflatten
from optree.typing import PyTreeSpec, PyTreeTypeVar
from optree.utils import safe_zip, total_order_sorted


__all__ = ['ArrayLikeTree', 'ArrayTree', 'tree_ravel']


# pylint: disable-next=invalid-name
ArrayLikeTree: TypeAlias = PyTreeTypeVar('ArrayLikeTree', ArrayLike)  # type: ignore[valid-type]
# pylint: disable-next=invalid-name
ArrayTree: TypeAlias = PyTreeTypeVar('ArrayTree', Array)  # type: ignore[valid-type]


# Vendor from https://github.com/google/jax/blob/jax-v0.4.20/jax/_src/util.py
class HashablePartial:  # pragma: no cover
    """A hashable version of :class:`functools.partial`."""

    func: FunctionType
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, func: FunctionType | HashablePartial, /, *args: Any, **kwargs: Any) -> None:
        """Construct a :class:`HashablePartial` instance."""
        if not callable(func):
            raise TypeError(f'Expected a callable, got {func!r}.')

        if isinstance(func, HashablePartial):
            self.func = func.func
            self.args = func.args + args
            self.kwargs = {**func.kwargs, **kwargs}
        elif isinstance(func, FunctionType):
            self.func = func  # type: ignore[assignment]
            self.args = args
            self.kwargs = kwargs
        else:
            raise TypeError(f'Expected a function, got {func!r}.')

    def __eq__(self, other: object, /) -> bool:
        return (
            type(other) is HashablePartial  # pylint: disable=unidiomatic-typecheck
            and self.func.__code__ == other.func.__code__
            and (self.args, self.kwargs) == (other.args, other.kwargs)
        )

    def __hash__(self, /) -> int:
        return hash(
            (
                self.func.__code__,
                self.args,
                tuple(total_order_sorted(self.kwargs.items(), key=itemgetter(0))),
            ),
        )

    def __call__(self, /, *args: Any, **kwargs: Any) -> Any:
        kwargs = {**self.kwargs, **kwargs}
        return self.func(*self.args, *args, **kwargs)


with contextlib.suppress(ImportError):  # pragma: no cover
    # pylint: disable-next=ungrouped-imports
    from jax._src.util import HashablePartial  # type: ignore[no-redef] # noqa: F811,RUF100


def tree_ravel(
    tree: ArrayLikeTree,
    /,
    is_leaf: Callable[[Any], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[Array, Callable[[Array], ArrayTree]]:
    r"""Ravel (flatten) a pytree of arrays down to a 1D array.

    >>> tree = {
    ...     'layer1': {
    ...         'weight': jnp.arange(0, 6, dtype=jnp.float32).reshape((2, 3)),
    ...         'bias': jnp.arange(6, 8, dtype=jnp.float32).reshape((2,)),
    ...     },
    ...     'layer2': {
    ...         'weight': jnp.arange(8, 10, dtype=jnp.float32).reshape((1, 2)),
    ...         'bias': jnp.arange(10, 11, dtype=jnp.float32).reshape((1,)),
    ...     },
    ... }
    >>> tree  # doctest: +IGNORE_WHITESPACE
    {
        'layer1': {
            'weight': Array([[0., 1., 2.],
                             [3., 4., 5.]], dtype=float32),
            'bias': Array([6., 7.], dtype=float32)
        },
        'layer2': {
            'weight': Array([[8., 9.]], dtype=float32),
            'bias': Array([10.], dtype=float32)
        }
    }
    >>> flat, unravel_func = tree_ravel(tree)
    >>> flat
    Array([ 6.,  7.,  0.,  1.,  2.,  3.,  4.,  5., 10.,  8.,  9.], dtype=float32)
    >>> unravel_func(flat)  # doctest: +IGNORE_WHITESPACE
    {
        'layer1': {
            'weight': Array([[0., 1., 2.],
                             [3., 4., 5.]], dtype=float32),
            'bias': Array([6., 7.], dtype=float32)
        },
        'layer2': {
            'weight': Array([[8., 9.]], dtype=float32),
            'bias': Array([10.], dtype=float32)
        }
    }

    Args:
        tree (pytree): a pytree of arrays and scalars to ravel.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A pair ``(array, unravel_func)`` where the first element is a 1D array representing the
        flattened and concatenated leaf values, with ``dtype`` determined by promoting the
        ``dtype``\s of leaf values, and the second element is a callable for unflattening a 1D array
        of the same length back to a pytree of the same structure as the input ``tree``. If the
        input pytree is empty (i.e. has no leaves) then as a convention a 1D empty array of the
        default dtype is returned in the first component of the output.
    """
    leaves, treespec = tree_flatten(
        tree,
        is_leaf=is_leaf,
        none_is_leaf=none_is_leaf,
        namespace=namespace,
    )
    flat, unravel_flat = _ravel_leaves(leaves)
    return flat, HashablePartial(_tree_unravel, treespec, unravel_flat)  # type: ignore[arg-type]


ravel_pytree = tree_ravel


def _tree_unravel(
    treespec: PyTreeSpec,
    unravel_flat: Callable[[Array], list[ArrayLike]],
    flat: Array,
    /,
) -> ArrayTree:
    return tree_unflatten(treespec, unravel_flat(flat))


def _ravel_leaves(
    leaves: list[ArrayLike],
    /,
) -> tuple[
    Array,
    Callable[[Array], list[ArrayLike]],
]:
    if not leaves:
        return (jnp.zeros(0), _unravel_empty)

    from_dtypes = tuple(dtypes.dtype(leaf) for leaf in leaves)
    to_dtype = dtypes.result_type(*from_dtypes)
    sizes = tuple(jnp.size(leaf) for leaf in leaves)
    shapes = tuple(jnp.shape(leaf) for leaf in leaves)
    indices = tuple(itertools.accumulate(sizes))

    if all(dt == to_dtype for dt in from_dtypes):
        # Skip any dtype conversion, resulting in a dtype-polymorphic `unravel`.
        # See https://github.com/google/jax/issues/7809.
        raveled = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
        return (
            raveled,
            HashablePartial(_unravel_leaves_single_dtype, indices, shapes),  # type: ignore[arg-type]
        )

    # When there is more than one distinct input dtype, we perform type conversions and produce a
    # dtype-specific unravel function.
    raveled = jnp.concatenate(
        [jnp.ravel(lax.convert_element_type(leaf, to_dtype)) for leaf in leaves],
    )
    return (
        raveled,
        HashablePartial(_unravel_leaves, indices, shapes, from_dtypes, to_dtype),  # type: ignore[arg-type]
    )


def _unravel_empty(flat: Array, /) -> list[ArrayLike]:
    if jnp.shape(flat) != (0,):
        raise ValueError(
            f'The unravel function expected an array of shape {(0,)}, got shape {jnp.shape(flat)}.',
        )

    return []


def _unravel_leaves_single_dtype(
    indices: tuple[int, ...],
    shapes: tuple[tuple[int, ...], ...],
    flat: Array,
    /,
) -> list[Array]:
    if jnp.shape(flat) != (indices[-1],):
        raise ValueError(
            f'The unravel function expected an array of shape {(indices[-1],)}, '
            f'got shape {jnp.shape(flat)}.',
        )

    chunks = jnp.split(flat, indices[:-1])
    return [chunk.reshape(shape) for chunk, shape in safe_zip(chunks, shapes)]


def _unravel_leaves(
    indices: tuple[int, ...],
    shapes: tuple[tuple[int, ...], ...],
    from_dtypes: tuple[jnp.dtype, ...],
    to_dtype: jnp.dtype,
    flat: Array,
    /,
) -> list[Array]:
    if jnp.shape(flat) != (indices[-1],):
        raise ValueError(
            f'The unravel function expected an array of shape {(indices[-1],)}, '
            f'got shape {jnp.shape(flat)}.',
        )
    array_dtype = dtypes.dtype(flat)
    if array_dtype != to_dtype:
        raise ValueError(
            f'The unravel function expected an array of dtype {to_dtype}, got dtype {array_dtype}.',
        )

    chunks = jnp.split(flat, indices[:-1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # ignore complex-to-real cast warning
        return [
            lax.convert_element_type(chunk.reshape(shape), dtype)
            for chunk, shape, dtype in safe_zip(chunks, shapes, from_dtypes)
        ]
