# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Hypothesis is a library for writing unit tests which are parametrized by
some source of data.

It verifies your code against a wide range of input and minimizes any
failing examples it finds.
"""

import _hypothesis_globals

from hypothesis._settings import HealthCheck, Phase, Verbosity, settings
from hypothesis.control import (
    assume,
    currently_in_test_context,
    event,
    note,
    reject,
    target,
)
from hypothesis.core import example, find, given, reproduce_failure, seed
from hypothesis.entry_points import run
from hypothesis.internal.detection import is_hypothesis_test
from hypothesis.internal.entropy import register_random
from hypothesis.utils.conventions import infer
from hypothesis.version import __version__, __version_info__

__all__ = [
    "HealthCheck",
    "Phase",
    "Verbosity",
    "__version__",
    "__version_info__",
    "assume",
    "currently_in_test_context",
    "event",
    "example",
    "find",
    "given",
    "infer",
    "is_hypothesis_test",
    "note",
    "register_random",
    "reject",
    "reproduce_failure",
    "seed",
    "settings",
    "target",
]

run()
del run

_hypothesis_globals.in_initialization -= 1
del _hypothesis_globals
