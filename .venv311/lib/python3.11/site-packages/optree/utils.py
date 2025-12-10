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
"""Utility functions for OpTree."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable, overload


if TYPE_CHECKING:
    from optree.typing import S, T, U


def total_order_sorted(
    iterable: Iterable[T],
    /,
    *,
    key: Callable[[T], Any] | None = None,
    reverse: bool = False,
) -> list[T]:
    """Sort an iterable in a total order.

    This is useful for sorting objects that are not comparable, e.g., dictionaries with different
    types of keys.
    """
    sequence = list(iterable)

    try:
        # Sort directly if possible
        return sorted(sequence, key=key, reverse=reverse)  # type: ignore[type-var,arg-type]
    except TypeError:
        if key is None:

            def key_fn(x: T) -> tuple[str, Any]:
                return (f'{x.__class__.__module__}.{x.__class__.__qualname__}', x)

        else:

            def key_fn(x: T) -> tuple[str, Any]:
                y = key(x)
                return (f'{y.__class__.__module__}.{y.__class__.__qualname__}', y)

        try:
            # Add `{obj.__class__.__module__}.{obj.__class__.__qualname__}` to the key order to make
            # it sortable between different types (e.g., `int` vs. `str`)
            return sorted(sequence, key=key_fn, reverse=reverse)
        except TypeError:  # cannot sort the keys (e.g., user-defined types)
            return sequence  # fallback to original order


@overload
def safe_zip(
    iter1: Iterable[T],
    /,
) -> zip[tuple[T]]: ...


@overload
def safe_zip(
    iter1: Iterable[T],
    iter2: Iterable[S],
    /,
) -> zip[tuple[T, S]]: ...


@overload
def safe_zip(
    iter1: Iterable[T],
    iter2: Iterable[S],
    iter3: Iterable[U],
    /,
) -> zip[tuple[T, S, U]]: ...


@overload
def safe_zip(
    iter1: Iterable[Any],
    iter2: Iterable[Any],
    iter3: Iterable[Any],
    iter4: Iterable[Any],
    /,
    *iters: Iterable[Any],
) -> zip[tuple[Any, ...]]: ...


def safe_zip(*args: Iterable[Any]) -> zip[tuple[Any, ...]]:
    """Strict zip that requires all arguments to be the same length."""
    seqs = [arg if isinstance(arg, Sequence) else list(arg) for arg in args]
    if len(set(map(len, seqs))) > 1:
        raise ValueError(f'length mismatch: {list(map(len, seqs))}')
    return zip(*seqs)


def unzip2(xys: Iterable[tuple[T, S]], /) -> tuple[tuple[T, ...], tuple[S, ...]]:
    """Unzip sequence of length-2 tuples into two tuples."""
    # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
    # is too permissive about inputs, and does not guarantee a length-2 output.
    # For example, for empty dict: tuple(zip(*{}.items())) -> ()
    xs = []
    ys = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return tuple(xs), tuple(ys)
