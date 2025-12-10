# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import warnings
from collections.abc import Hashable
from typing import Any

from hypothesis.errors import HypothesisWarning
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.strategies._internal import SearchStrategy
from hypothesis.strategies._internal.strategies import Ex


class SharedStrategy(SearchStrategy[Ex]):
    def __init__(self, base: SearchStrategy[Ex], key: Hashable | None = None):
        super().__init__()
        self.key = key
        self.base = base

    def __repr__(self) -> str:
        if self.key is not None:
            return f"shared({self.base!r}, key={self.key!r})"
        else:
            return f"shared({self.base!r})"

    def calc_label(self) -> int:
        return self.base.calc_label()

    # Ideally would be -> Ex, but key collisions with different-typed values are
    # possible. See https://github.com/HypothesisWorks/hypothesis/issues/4301.
    def do_draw(self, data: ConjectureData) -> Any:
        key = self.key or self
        if key not in data._shared_strategy_draws:
            drawn = data.draw(self.base)
            data._shared_strategy_draws[key] = (drawn, self)
        else:
            drawn, other = data._shared_strategy_draws[key]

            # Check that the strategies shared under this key are equivalent
            if self.label != other.label:
                warnings.warn(
                    f"Different strategies are shared under {key=}. This"
                    " risks drawing values that are not valid examples for the strategy,"
                    " or that have a narrower range than expected."
                    f" Conflicting strategies: ({self!r}, {other!r}).",
                    HypothesisWarning,
                    stacklevel=1,
                )
        return drawn
