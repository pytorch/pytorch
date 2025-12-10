# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.


class UniqueIdentifier:
    """A factory for sentinel objects with nice reprs."""

    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def __repr__(self) -> str:
        return self.identifier


infer = ...
not_set = UniqueIdentifier("not_set")
