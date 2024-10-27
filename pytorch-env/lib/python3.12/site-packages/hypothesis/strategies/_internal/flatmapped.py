# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.strategies._internal.strategies import SearchStrategy, check_strategy


class FlatMapStrategy(SearchStrategy):
    def __init__(self, strategy, expand):
        super().__init__()
        self.flatmapped_strategy = strategy
        self.expand = expand

    def calc_is_empty(self, recur):
        return recur(self.flatmapped_strategy)

    def __repr__(self):
        if not hasattr(self, "_cached_repr"):
            self._cached_repr = f"{self.flatmapped_strategy!r}.flatmap({get_pretty_function_description(self.expand)})"
        return self._cached_repr

    def do_draw(self, data):
        source = data.draw(self.flatmapped_strategy)
        expanded_source = self.expand(source)
        check_strategy(expanded_source)
        return data.draw(expanded_source)

    @property
    def branches(self):
        return [
            FlatMapStrategy(strategy=strategy, expand=self.expand)
            for strategy in self.flatmapped_strategy.branches
        ]
