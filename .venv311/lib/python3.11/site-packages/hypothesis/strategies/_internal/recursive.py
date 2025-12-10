# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import threading
from contextlib import contextmanager

from hypothesis.errors import InvalidArgument
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.internal.validation import check_type
from hypothesis.strategies._internal.strategies import (
    OneOfStrategy,
    SearchStrategy,
    check_strategy,
)


class LimitReached(BaseException):
    pass


class LimitedStrategy(SearchStrategy):
    def __init__(self, strategy):
        super().__init__()
        self.base_strategy = strategy
        self._threadlocal = threading.local()

    @property
    def marker(self):
        return getattr(self._threadlocal, "marker", 0)

    @marker.setter
    def marker(self, value):
        self._threadlocal.marker = value

    @property
    def currently_capped(self):
        return getattr(self._threadlocal, "currently_capped", False)

    @currently_capped.setter
    def currently_capped(self, value):
        self._threadlocal.currently_capped = value

    def __repr__(self) -> str:
        return f"LimitedStrategy({self.base_strategy!r})"

    def do_validate(self) -> None:
        self.base_strategy.validate()

    def do_draw(self, data):
        assert self.currently_capped
        if self.marker <= 0:
            raise LimitReached
        self.marker -= 1
        return data.draw(self.base_strategy)

    @contextmanager
    def capped(self, max_templates):
        try:
            was_capped = self.currently_capped
            self.currently_capped = True
            self.marker = max_templates
            yield
        finally:
            self.currently_capped = was_capped


class RecursiveStrategy(SearchStrategy):
    def __init__(self, base, extend, max_leaves):
        super().__init__()
        self.max_leaves = max_leaves
        self.base = base
        self.limited_base = LimitedStrategy(base)
        self.extend = extend

        strategies = [self.limited_base, self.extend(self.limited_base)]
        while 2 ** (len(strategies) - 1) <= max_leaves:
            strategies.append(extend(OneOfStrategy(tuple(strategies))))
        self.strategy = OneOfStrategy(strategies)

    def __repr__(self) -> str:
        if not hasattr(self, "_cached_repr"):
            self._cached_repr = "recursive(%r, %s, max_leaves=%d)" % (
                self.base,
                get_pretty_function_description(self.extend),
                self.max_leaves,
            )
        return self._cached_repr

    def do_validate(self) -> None:
        check_strategy(self.base, "base")
        extended = self.extend(self.limited_base)
        check_strategy(extended, f"extend({self.limited_base!r})")
        self.limited_base.validate()
        extended.validate()
        check_type(int, self.max_leaves, "max_leaves")
        if self.max_leaves <= 0:
            raise InvalidArgument(
                f"max_leaves={self.max_leaves!r} must be greater than zero"
            )

    def do_draw(self, data):
        count = 0
        while True:
            try:
                with self.limited_base.capped(self.max_leaves):
                    return data.draw(self.strategy)
            except LimitReached:
                if count == 0:
                    msg = f"Draw for {self!r} exceeded max_leaves and had to be retried"
                    data.events[msg] = ""
                count += 1
