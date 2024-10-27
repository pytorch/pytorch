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
"""Integration with PyTorch."""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable
from typing_extensions import TypeAlias  # Python 3.10+

import torch  # pylint: disable=import-error

from optree.ops import tree_flatten, tree_unflatten
from optree.typing import PyTreeSpec, PyTreeTypeVar
from optree.utils import safe_zip


__all__ = ['TensorTree', 'tree_ravel']


TensorTree: TypeAlias = PyTreeTypeVar('TensorTree', torch.Tensor)  # type: ignore[valid-type]


def tree_ravel(
    tree: TensorTree,
    is_leaf: Callable[[Any], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[torch.Tensor, Callable[[torch.Tensor], TensorTree]]:
    r"""Ravel (flatten) a pytree of tensors down to a 1D tensor.

    >>> tree = {
    ...     'layer1': {
    ...         'weight': torch.arange(0, 6, dtype=torch.float64).reshape((2, 3)),
    ...         'bias': torch.arange(6, 8, dtype=torch.float64).reshape((2,)),
    ...     },
    ...     'layer2': {
    ...         'weight': torch.arange(8, 10, dtype=torch.float64).reshape((1, 2)),
    ...         'bias': torch.arange(10, 11, dtype=torch.float64).reshape((1,)),
    ...     },
    ... }
    >>> tree  # doctest: +IGNORE_WHITESPACE
    {
        'layer1': {
            'weight': tensor([[0., 1., 2.],
                              [3., 4., 5.]], dtype=torch.float64),
            'bias': tensor([6., 7.], dtype=torch.float64)
        },
        'layer2': {
            'weight': tensor([[8., 9.]], dtype=torch.float64),
            'bias': tensor([10.], dtype=torch.float64)
        }
    }
    >>> flat, unravel_func = tree_ravel(tree)
    >>> flat
    tensor([ 6.,  7.,  0.,  1.,  2.,  3.,  4.,  5., 10.,  8.,  9.], dtype=torch.float64)
    >>> unravel_func(flat)  # doctest: +IGNORE_WHITESPACE
    {
        'layer1': {
            'weight': tensor([[0., 1., 2.],
                              [3., 4., 5.]], dtype=torch.float64),
            'bias': tensor([6., 7.], dtype=torch.float64)
        },
        'layer2': {
            'weight': tensor([[8., 9.]], dtype=torch.float64),
            'bias': tensor([10.], dtype=torch.float64)
        }
    }

    Args:
        tree (pytree): a pytree of tensors to ravel.
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
        A pair ``(tensor, unravel_func)`` where the first element is a 1D tensor representing the
        flattened and concatenated leaf values, with ``dtype`` determined by promoting the
        ``dtype``\s of leaf values, and the second element is a callable for unflattening a 1D tensor
        of the same length back to a pytree of the same structure as the input ``tree``. If the
        input pytree is empty (i.e. has no leaves) then as a convention a 1D empty tensor of the
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
    unravel_flat: Callable[[torch.Tensor], list[torch.Tensor]],
    flat: torch.Tensor,
) -> TensorTree:
    return tree_unflatten(treespec, unravel_flat(flat))


def _ravel_leaves(
    leaves: list[torch.Tensor],
) -> tuple[torch.Tensor, Callable[[torch.Tensor], list[torch.Tensor]]]:
    if not leaves:
        return (torch.zeros(0), _unravel_empty)
    if not all(torch.is_tensor(leaf) for leaf in leaves):
        raise ValueError('All leaves must be tensors.')

    from_dtypes = tuple(leaf.dtype for leaf in leaves)
    to_dtype = from_dtypes[0]
    for from_dtype in from_dtypes[1:]:
        to_dtype = torch.promote_types(to_dtype, from_dtype)
    sizes = tuple(leaf.numel() for leaf in leaves)
    shapes = tuple(leaf.shape for leaf in leaves)

    if all(dt == to_dtype for dt in from_dtypes):
        # Skip any dtype conversion, resulting in a dtype-polymorphic `unravel`.
        raveled = torch.cat([torch.ravel(leaf) for leaf in leaves])
        return (
            raveled,
            functools.partial(_unravel_leaves_single_dtype, sizes, shapes),
        )

    # When there is more than one distinct input dtype, we perform type conversions and produce a
    # dtype-specific unravel function.
    raveled = torch.cat([torch.ravel(leaf).to(to_dtype) for leaf in leaves])
    return (
        raveled,
        functools.partial(_unravel_leaves, sizes, shapes, from_dtypes, to_dtype),
    )


def _unravel_empty(flat: torch.Tensor) -> list[torch.Tensor]:
    if not torch.is_tensor(flat):
        raise ValueError(f'Expected a tensor to unravel, got {type(flat)!r}.')
    if flat.shape != (0,):
        raise ValueError(
            f'The unravel function expected a tensor of shape {(0,)}, got shape {flat.shape}.',
        )
    return []


def _unravel_leaves_single_dtype(
    sizes: tuple[int, ...],
    shapes: tuple[tuple[int, ...], ...],
    flat: torch.Tensor,
) -> list[torch.Tensor]:
    if not torch.is_tensor(flat):
        raise ValueError(f'Expected a tensor to unravel, got {type(flat)!r}.')
    if flat.shape != (sum(sizes),):
        raise ValueError(
            f'The unravel function expected a tensor of shape {(sum(sizes),)}, '
            f'got shape {flat.shape}.',
        )

    chunks = torch.split(flat, list(sizes))
    return [chunk.reshape(shape) for chunk, shape in safe_zip(chunks, shapes)]


def _unravel_leaves(
    sizes: tuple[int, ...],
    shapes: tuple[tuple[int, ...], ...],
    from_dtypes: tuple[torch.dtype, ...],
    to_dtype: torch.dtype,
    flat: torch.Tensor,
) -> list[torch.Tensor]:
    if not torch.is_tensor(flat):
        raise ValueError(f'Expected a tensor to unravel, got {type(flat)!r}.')
    if flat.shape != (sum(sizes),):
        raise ValueError(
            f'The unravel function expected a tensor of shape {(sum(sizes),)}, '
            f'got shape {flat.shape}.',
        )
    if flat.dtype != to_dtype:
        raise ValueError(
            f'The unravel function expected a tensor of dtype {to_dtype}, got dtype {flat.dtype}.',
        )

    chunks = torch.split(flat, list(sizes))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # ignore complex-to-real cast warning
        return [
            chunk.reshape(shape).to(dtype)
            for chunk, shape, dtype in safe_zip(chunks, shapes, from_dtypes)
        ]
