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

from hypothesis.configuration import check_sideeffect_during_initialization
from hypothesis.errors import InvalidArgument
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.strategies._internal.strategies import SearchStrategy, check_strategy


class DeferredStrategy(SearchStrategy):
    """A strategy which may be used before it is fully defined."""

    def __init__(self, definition):
        super().__init__()
        self.__wrapped_strategy = None
        self.__in_repr = False
        self.__definition = definition

    @property
    def wrapped_strategy(self):
        if self.__wrapped_strategy is None:
            check_sideeffect_during_initialization("deferred evaluation of {!r}", self)

            if not inspect.isfunction(self.__definition):
                raise InvalidArgument(
                    f"Expected definition to be a function but got {self.__definition!r} "
                    f"of type {type(self.__definition).__name__} instead."
                )
            result = self.__definition()
            if result is self:
                raise InvalidArgument("Cannot define a deferred strategy to be itself")
            check_strategy(result, "definition()")
            self.__wrapped_strategy = result
            self.__definition = None
        return self.__wrapped_strategy

    @property
    def branches(self):
        return self.wrapped_strategy.branches

    @property
    def supports_find(self):
        return self.wrapped_strategy.supports_find

    def calc_label(self):
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

    def calc_is_empty(self, recur):
        return recur(self.wrapped_strategy)

    def calc_has_reusable_values(self, recur):
        return recur(self.wrapped_strategy)

    def __repr__(self):
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

    def do_draw(self, data):
        return data.draw(self.wrapped_strategy)
