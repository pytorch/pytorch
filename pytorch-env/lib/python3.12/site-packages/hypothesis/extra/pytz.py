# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
----------------
hypothesis[pytz]
----------------

This module provides :pypi:`pytz` timezones.

You can use this strategy to make
:py:func:`hypothesis.strategies.datetimes` and
:py:func:`hypothesis.strategies.times` produce timezone-aware values.
"""

import datetime as dt

import pytz
from pytz.tzfile import StaticTzInfo  # type: ignore  # considered private by typeshed

from hypothesis import strategies as st
from hypothesis.strategies._internal.utils import cacheable, defines_strategy

__all__ = ["timezones"]


@cacheable
@defines_strategy()
def timezones() -> st.SearchStrategy[dt.tzinfo]:
    """Any timezone in the Olsen database, as a pytz tzinfo object.

    This strategy minimises to UTC, or the smallest possible fixed
    offset, and is designed for use with
    :py:func:`hypothesis.strategies.datetimes`.
    """
    all_timezones = [pytz.timezone(tz) for tz in pytz.all_timezones]
    # Some timezones have always had a constant offset from UTC.  This makes
    # them simpler than timezones with daylight savings, and the smaller the
    # absolute offset the simpler they are.  Of course, UTC is even simpler!
    static: list = [pytz.UTC]
    static += sorted(
        (t for t in all_timezones if isinstance(t, StaticTzInfo)),
        key=lambda tz: abs(tz.utcoffset(dt.datetime(2000, 1, 1))),
    )
    # Timezones which have changed UTC offset; best ordered by name.
    dynamic = [tz for tz in all_timezones if tz not in static]
    return st.sampled_from(static + dynamic)
