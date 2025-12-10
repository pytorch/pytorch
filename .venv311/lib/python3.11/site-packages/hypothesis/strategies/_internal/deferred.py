# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import inspect
from collections.abc import Callable, Sequence

from hypothesis.configuration import check_sideeffect_during_initialization
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.strategies._internal.strategies import (
    Ex,
    RecurT,
    SearchStrategy,
    check_strategy,
)


class DeferredStrategy(SearchStrategy[Ex]):
    """A strategy which may be used before it is fully defined."""

    def __init__(self, definition: Callable[[], SearchStrategy[Ex]]):
        super().__init__()
        self.__wrapped_strategy: SearchStrategy[Ex] | None = None
        self.__in_repr: bool = False
        self.__definition: Callable[[], SearchStrategy[Ex]] | None = definition

    @property
    def wrapped_strategy(self) -> SearchStrategy[Ex]:
        # we assign this before entering the condition to avoid a race condition
        # under threading. See issue #4523.
        definition = self.__definition
        if self.__wrapped_strategy is None:
            check_sideeffect_during_initialization("deferred evaluation of {!r}", self)

            if not inspect.isfunction(definition):
                raise InvalidArgument(
                    f"Expected definition to be a function but got {definition!r} "
                    f"of type {type(definition).__name__} instead."
                )
            result = definition()
            if result is self:
                raise InvalidArgument("Cannot define a deferred strategy to be itself")
            check_strategy(result, "definition()")
            self.__wrapped_strategy = result
            self.__definition = None
        return self.__wrapped_strategy

    @property
    def branches(self) -> Sequence[SearchStrategy[Ex]]:
        return self.wrapped_strategy.branches

    def calc_label(self) -> int:
        """Deferred strategies don't have a calculated label, because we would
        end up having to calculate the fixed point of some hash function in
        order to calculate it when they recursively refer to themself!

        The label for the wrapped strategy will still appear because it
        will be passed to draw.
        """
        # This is actually the same as the parent class implementation, but we
        # include it explicitly here in order to document that this is a
        # deliberate decision.
        return self.class_label

    def calc_is_empty(self, recur: RecurT) -> bool:
        return recur(self.wrapped_strategy)

    def calc_has_reusable_values(self, recur: RecurT) -> bool:
        return recur(self.wrapped_strategy)

    def __repr__(self) -> str:
        if self.__wrapped_strategy is not None:
            if self.__in_repr:
                return f"(deferred@{id(self)!r})"
            try:
                self.__in_repr = True
                return repr(self.__wrapped_strategy)
            finally:
                self.__in_repr = False
        else:
            description = get_pretty_function_description(self.__definition)
            return f"deferred({description})"

    def do_draw(self, data: ConjectureData) -> Ex:
        return data.draw(self.wrapped_strategy)
