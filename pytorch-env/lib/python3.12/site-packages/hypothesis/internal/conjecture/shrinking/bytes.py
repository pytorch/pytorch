# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from hypothesis.internal.conjecture.shrinking.collection import Collection
from hypothesis.internal.conjecture.shrinking.integer import Integer


class Bytes(Collection):
    def __init__(self, initial, predicate, **kwargs):
        super().__init__(
            # implicit conversion from bytes to list of integers here
            list(initial),
            lambda val: predicate(bytes(val)),
            ElementShrinker=Integer,
            **kwargs,
        )
