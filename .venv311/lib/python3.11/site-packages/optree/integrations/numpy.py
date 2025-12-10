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
"""Integration with NumPy."""

# pragma: numpy cover file
# pylint: disable=import-error

from __future__ import annotations

import functools
import itertools
import warnings
from typing import Any, Callable
from typing_extensions import TypeAlias  # Python 3.10+

import numpy as np
from numpy.typing import ArrayLike

from optree.ops import tree_flatten, tree_unflatten
from optree.typing import PyTreeSpec, PyTreeTypeVar
from optree.utils import safe_zip


__all__ = ['ArrayLikeTree', 'ArrayTree', 'tree_ravel']


# pylint: disable-next=invalid-name
ArrayLikeTree: TypeAlias = PyTreeTypeVar('ArrayLikeTree', ArrayLike)  # type: ignore[valid-type]
# pylint: disable-next=invalid-name
ArrayTree: TypeAlias = PyTreeTypeVar('ArrayTree', np.ndarray)  # type: ignore[valid-type]


def tree_ravel(
    tree: ArrayLikeTree,
    /,
    is_leaf: Callable[[Any], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[np.ndarray, Callable[[np.ndarray], ArrayTree]]:
    r"""Ravel (flatten) a pytree of arrays down to a 1D array.

    >>> tree = {
    ...     'layer1': {
    ...         'weight': np.arange(0, 6, dtype=np.float32).reshape((2, 3)),
    ...         'bias': np.arange(6, 8, dtype=np.float32).reshape((2,)),
    ...     },
    ...     'layer2': {
    ...         'weight': np.arange(8, 10, dtype=np.float32).reshape((1, 2)),
    ...         'bias': np.arange(10, 11, dtype=np.float32).reshape((1,)),
    ...     },
    ... }
    >>> tree  # doctest: +IGNORE_WHITESPACE
    {
        'layer1': {
            'weight': array([[0., 1., 2.],
                             [3., 4., 5.]], dtype=float32),
            'bias': array([6., 7.], dtype=float32)
        },
        'layer2': {
            'weight': array([[8., 9.]], dtype=float32),
            'bias': array([10.], dtype=float32)
        }
    }
    >>> flat, unravel_func = tree_ravel(tree)
    >>> flat
    array([ 6.,  7.,  0.,  1.,  2.,  3.,  4.,  5., 10.,  8.,  9.], dtype=float32)
    >>> unravel_func(flat)  # doctest: +IGNORE_WHITESPACE
    {
        'layer1': {
            'weight': array([[0., 1., 2.],
                             [3., 4., 5.]], dtype=float32),
            'bias': array([6., 7.], dtype=float32)
        },
        'layer2': {
            'weight': array([[8., 9.]], dtype=float32),
            'bias': array([10.], dtype=float32)
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
    return flat, functools.partial(_tree_unravel, treespec, unravel_flat)


ravel_pytree = tree_ravel


def _tree_unravel(
    treespec: PyTreeSpec,
    unravel_flat: Callable[[np.ndarray], list[np.ndarray]],
    flat: np.ndarray,
    /,
) -> ArrayTree:
    return tree_unflatten(treespec, unravel_flat(flat))


def _ravel_leaves(
    leaves: list[np.ndarray],
    /,
) -> tuple[
    np.ndarray,
    Callable[[np.ndarray], list[np.ndarray]],
]:
    if not leaves:
        return (np.zeros(0), _unravel_empty)

    from_dtypes = tuple(np.result_type(leaf) for leaf in leaves)
    to_dtype = np.result_type(*leaves)
    sizes = tuple(np.size(leaf) for leaf in leaves)
    shapes = tuple(np.shape(leaf) for leaf in leaves)
    indices = tuple(itertools.accumulate(sizes))

    if all(dt == to_dtype for dt in from_dtypes):
        # Skip any dtype conversion, resulting in a dtype-polymorphic `unravel`.
        raveled = np.concatenate([np.ravel(leaf) for leaf in leaves])
        return (
            raveled,
            functools.partial(_unravel_leaves_single_dtype, indices, shapes),
        )

    # When there is more than one distinct input dtype, we perform type conversions and produce a
    # dtype-specific unravel function.
    raveled = np.concatenate([np.ravel(leaf).astype(to_dtype) for leaf in leaves])
    return (
        raveled,
        functools.partial(_unravel_leaves, indices, shapes, from_dtypes, to_dtype),
    )


def _unravel_empty(flat: np.ndarray, /) -> list[np.ndarray]:
    if np.shape(flat) != (0,):
        raise ValueError(
            f'The unravel function expected an array of shape {(0,)}, got shape {np.shape(flat)}.',
        )
    return []


def _unravel_leaves_single_dtype(
    indices: tuple[int, ...],
    shapes: tuple[tuple[int, ...], ...],
    flat: np.ndarray,
    /,
) -> list[np.ndarray]:
    if np.shape(flat) != (indices[-1],):
        raise ValueError(
            f'The unravel function expected an array of shape {(indices[-1],)}, '
            f'got shape {np.shape(flat)}.',
        )

    chunks = np.split(flat, indices[:-1])
    return [chunk.reshape(shape) for chunk, shape in safe_zip(chunks, shapes)]


def _unravel_leaves(
    indices: tuple[int, ...],
    shapes: tuple[tuple[int, ...], ...],
    from_dtypes: tuple[np.dtype, ...],
    to_dtype: np.dtype,
    flat: np.ndarray,
    /,
) -> list[np.ndarray]:
    if np.shape(flat) != (indices[-1],):
        raise ValueError(
            f'The unravel function expected an array of shape {(indices[-1],)}, '
            f'got shape {np.shape(flat)}.',
        )
    array_dtype = np.result_type(flat)
    if array_dtype != to_dtype:
        raise ValueError(
            f'The unravel function expected an array of dtype {to_dtype}, got dtype {array_dtype}.',
        )

    chunks = np.split(flat, indices[:-1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # ignore complex-to-real cast warning
        return [
            chunk.reshape(shape).astype(dtype)
            for chunk, shape, dtype in safe_zip(chunks, shapes, from_dtypes)
        ]
