# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from types import MethodType


def is_hypothesis_test(test: object) -> bool:
    if isinstance(test, MethodType):
        return is_hypothesis_test(test.__func__)
    return getattr(test, "is_hypothesis_test", False)
