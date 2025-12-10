# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NoReturn

from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.strategies._internal.strategies import (
    Ex,
    RecurT,
    SampledFromStrategy,
    SearchStrategy,
    T,
    is_hashable,
)
from hypothesis.strategies._internal.utils import cacheable, defines_strategy
from hypothesis.utils.conventions import UniqueIdentifier

if TYPE_CHECKING:
    from typing_extensions import Never


class JustStrategy(SampledFromStrategy[Ex]):
    """A strategy which always returns a single fixed value.

    It's implemented as a length-one SampledFromStrategy so that all our
    special-case logic for filtering and sets applies also to just(x).

    The important difference from a SampledFromStrategy with only one
    element to choose is that JustStrategy *never* touches the underlying
    choice sequence, i.e. drawing neither reads from nor writes to `data`.
    This is a reasonably important optimisation (or semantic distinction!)
    for both JustStrategy and SampledFromStrategy.
    """

    @property
    def value(self) -> Ex:
        return self.elements[0]

    def __repr__(self) -> str:
        suffix = "".join(
            f".{name}({get_pretty_function_description(f)})"
            for name, f in self._transformations
        )
        if self.value is None:
            return "none()" + suffix
        return f"just({get_pretty_function_description(self.value)}){suffix}"

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        return is_hashable(self.value)

    def do_filtered_draw(self, data: ConjectureData) -> Ex | UniqueIdentifier:
        # The parent class's `do_draw` implementation delegates directly to
        # `do_filtered_draw`, which we can greatly simplify in this case since
        # we have exactly one value. (This also avoids drawing any data.)
        return self._transform(self.value)


@defines_strategy(eager=True)
def just(value: T) -> SearchStrategy[T]:
    """Return a strategy which only generates ``value``.

    Note: ``value`` is not copied. Be wary of using mutable values.

    If ``value`` is the result of a callable, you can use
    :func:`builds(callable) <hypothesis.strategies.builds>` instead
    of ``just(callable())`` to get a fresh value each time.

    Examples from this strategy do not shrink (because there is only one).
    """
    return JustStrategy([value])


@defines_strategy(force_reusable_values=True)
def none() -> SearchStrategy[None]:
    """Return a strategy which only generates None.

    Examples from this strategy do not shrink (because there is only
    one).
    """
    return just(None)


class Nothing(SearchStrategy["Never"]):
    def calc_is_empty(self, recur: RecurT) -> bool:
        return True

    def do_draw(self, data: ConjectureData) -> NoReturn:
        # This method should never be called because draw() will mark the
        # data as invalid immediately because is_empty is True.
        raise NotImplementedError("This should never happen")

    def calc_has_reusable_values(self, recur: RecurT) -> bool:
        return True

    def __repr__(self) -> str:
        return "nothing()"

    def map(self, pack: Callable[[Any], Any]) -> SearchStrategy["Never"]:
        return self

    def filter(self, condition: Callable[[Any], Any]) -> "SearchStrategy[Never]":
        return self

    def flatmap(
        self, expand: Callable[[Any], "SearchStrategy[Any]"]
    ) -> "SearchStrategy[Never]":
        return self


NOTHING = Nothing()


@cacheable
@defines_strategy(eager=True)
def nothing() -> SearchStrategy["Never"]:
    """This strategy never successfully draws a value and will always reject on
    an attempt to draw.

    Examples from this strategy do not shrink (because there are none).
    """
    return NOTHING


class BooleansStrategy(SearchStrategy[bool]):
    def do_draw(self, data: ConjectureData) -> bool:
        return data.draw_boolean()

    def __repr__(self) -> str:
        return "booleans()"
