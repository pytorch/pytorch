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
from typing import Generic, TypeVar

from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.conjecture.utils import (
    calc_label_from_callable,
    combine_labels,
)
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.strategies._internal.strategies import (
    RecurT,
    SearchStrategy,
    check_strategy,
)

MappedFrom = TypeVar("MappedFrom")
MappedTo = TypeVar("MappedTo")


class FlatMapStrategy(SearchStrategy[MappedTo], Generic[MappedFrom, MappedTo]):
    def __init__(
        self,
        base: SearchStrategy[MappedFrom],
        expand: Callable[[MappedFrom], SearchStrategy[MappedTo]],
    ):
        super().__init__()
        self.base = base
        self.expand = expand

    def calc_is_empty(self, recur: RecurT) -> bool:
        return recur(self.base)

    def calc_label(self) -> int:
        return combine_labels(
            self.class_label,
            self.base.label,
            calc_label_from_callable(self.expand),
        )

    def __repr__(self) -> str:
        if not hasattr(self, "_cached_repr"):
            self._cached_repr = (
                f"{self.base!r}.flatmap({get_pretty_function_description(self.expand)})"
            )
        return self._cached_repr

    def do_draw(self, data: ConjectureData) -> MappedTo:
        base = data.draw(self.base)
        expanded = self.expand(base)
        check_strategy(expanded)
        return data.draw(expanded)

    @property
    def branches(self) -> list[SearchStrategy[MappedTo]]:
        return [
            FlatMapStrategy(strategy, expand=self.expand)
            for strategy in self.base.branches
        ]
