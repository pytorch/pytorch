# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from hypothesis.strategies._internal import SearchStrategy

SHARED_STRATEGY_ATTRIBUTE = "_hypothesis_shared_strategies"


class SharedStrategy(SearchStrategy):
    def __init__(self, base, key=None):
        self.key = key
        self.base = base

    @property
    def supports_find(self):
        return self.base.supports_find

    def __repr__(self):
        if self.key is not None:
            return f"shared({self.base!r}, key={self.key!r})"
        else:
            return f"shared({self.base!r})"

    def do_draw(self, data):
        if not hasattr(data, SHARED_STRATEGY_ATTRIBUTE):
            setattr(data, SHARED_STRATEGY_ATTRIBUTE, {})
        sharing = getattr(data, SHARED_STRATEGY_ATTRIBUTE)
        key = self.key or self
        if key not in sharing:
            sharing[key] = data.draw(self.base)
        return sharing[key]
