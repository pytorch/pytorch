# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import datetime as dt
import operator as op
import zoneinfo
from calendar import monthrange
from functools import lru_cache, partial
from importlib import resources
from pathlib import Path
from typing import Optional

from hypothesis.errors import InvalidArgument
from hypothesis.internal.validation import check_type, check_valid_interval
from hypothesis.strategies._internal.core import sampled_from
from hypothesis.strategies._internal.misc import just, none, nothing
from hypothesis.strategies._internal.strategies import SearchStrategy
from hypothesis.strategies._internal.utils import defines_strategy

DATENAMES = ("year", "month", "day")
TIMENAMES = ("hour", "minute", "second", "microsecond")


def is_pytz_timezone(tz):
    if not isinstance(tz, dt.tzinfo):
        return False
    module = type(tz).__module__
    return module == "pytz" or module.startswith("pytz.")


def replace_tzinfo(value, timezone):
    if is_pytz_timezone(timezone):
        # Pytz timezones are a little complicated, and using the .replace method
        # can cause some weird issues, so we use their special "localize" instead.
        #
        # We use the fold attribute as a convenient boolean for is_dst, even though
        # they're semantically distinct.  For ambiguous or imaginary hours, fold says
        # whether you should use the offset that applies before the gap (fold=0) or
        # the offset that applies after the gap (fold=1). is_dst says whether you
        # should choose the side that is "DST" or "STD" (STD->STD or DST->DST
        # transitions are unclear as you might expect).
        #
        # WARNING: this is INCORRECT for timezones with negative DST offsets such as
        #       "Europe/Dublin", but it's unclear what we could do instead beyond
        #       documenting the problem and recommending use of `dateutil` instead.
        return timezone.localize(value, is_dst=not value.fold)
    return value.replace(tzinfo=timezone)


def datetime_does_not_exist(value):
    """This function tests whether the given datetime can be round-tripped to and
    from UTC.  It is an exact inverse of (and very similar to) the dateutil method
    https://dateutil.readthedocs.io/en/stable/tz.html#dateutil.tz.datetime_exists
    """
    # Naive datetimes cannot be imaginary, but we need this special case because
    # chaining .astimezone() ends with *the system local timezone*, not None.
    # See bug report in https://github.com/HypothesisWorks/hypothesis/issues/2662
    if value.tzinfo is None:
        return False
    try:
        # Does the naive portion of the datetime change when round-tripped to
        # UTC?  If so, or if this overflows, we say that it does not exist.
        roundtrip = value.astimezone(dt.timezone.utc).astimezone(value.tzinfo)
    except OverflowError:
        # Overflows at datetime.min or datetime.max boundary condition.
        # Rejecting these is acceptable, because timezones are close to
        # meaningless before ~1900 and subject to a lot of change by
        # 9999, so it should be a very small fraction of possible values.
        return True

    if (
        value.tzinfo is not roundtrip.tzinfo
        and value.utcoffset() != roundtrip.utcoffset()
    ):
        # This only ever occurs during imaginary (i.e. nonexistent) datetimes,
        # and only for pytz timezones which do not follow PEP-495 semantics.
        # (may exclude a few other edge cases, but you should use zoneinfo anyway)
        return True

    assert value.tzinfo is roundtrip.tzinfo, "so only the naive portions are compared"
    return value != roundtrip


def draw_capped_multipart(
    data, min_value, max_value, duration_names=DATENAMES + TIMENAMES
):
    assert isinstance(min_value, (dt.date, dt.time, dt.datetime))
    assert type(min_value) == type(max_value)
    assert min_value <= max_value
    result = {}
    cap_low, cap_high = True, True
    for name in duration_names:
        low = getattr(min_value if cap_low else dt.datetime.min, name)
        high = getattr(max_value if cap_high else dt.datetime.max, name)
        if name == "day" and not cap_high:
            _, high = monthrange(**result)
        if name == "year":
            val = data.draw_integer(low, high, shrink_towards=2000)
        else:
            val = data.draw_integer(low, high)
        result[name] = val
        cap_low = cap_low and val == low
        cap_high = cap_high and val == high
    if hasattr(min_value, "fold"):
        # The `fold` attribute is ignored in comparison of naive datetimes.
        # In tz-aware datetimes it would require *very* invasive changes to
        # the logic above, and be very sensitive to the specific timezone
        # (at the cost of efficient shrinking and mutation), so at least for
        # now we stick with the status quo and generate it independently.
        result["fold"] = data.draw_integer(0, 1)
    return result


class DatetimeStrategy(SearchStrategy):
    def __init__(self, min_value, max_value, timezones_strat, allow_imaginary):
        assert isinstance(min_value, dt.datetime)
        assert isinstance(max_value, dt.datetime)
        assert min_value.tzinfo is None
        assert max_value.tzinfo is None
        assert min_value <= max_value
        assert isinstance(timezones_strat, SearchStrategy)
        assert isinstance(allow_imaginary, bool)
        self.min_value = min_value
        self.max_value = max_value
        self.tz_strat = timezones_strat
        self.allow_imaginary = allow_imaginary

    def do_draw(self, data):
        # We start by drawing a timezone, and an initial datetime.
        tz = data.draw(self.tz_strat)
        result = self.draw_naive_datetime_and_combine(data, tz)

        # TODO: with some probability, systematically search for one of
        #   - an imaginary time (if allowed),
        #   - a time within 24hrs of a leap second (if there any are within bounds),
        #   - other subtle, little-known, or nasty issues as described in
        #     https://github.com/HypothesisWorks/hypothesis/issues/69

        # If we happened to end up with a disallowed imaginary time, reject it.
        if (not self.allow_imaginary) and datetime_does_not_exist(result):
            data.mark_invalid(f"{result} does not exist (usually a DST transition)")
        return result

    def draw_naive_datetime_and_combine(self, data, tz):
        result = draw_capped_multipart(data, self.min_value, self.max_value)
        try:
            return replace_tzinfo(dt.datetime(**result), timezone=tz)
        except (ValueError, OverflowError):
            data.mark_invalid(
                f"Failed to draw a datetime between {self.min_value!r} and "
                f"{self.max_value!r} with timezone from {self.tz_strat!r}."
            )


@defines_strategy(force_reusable_values=True)
def datetimes(
    min_value: dt.datetime = dt.datetime.min,
    max_value: dt.datetime = dt.datetime.max,
    *,
    timezones: SearchStrategy[Optional[dt.tzinfo]] = none(),
    allow_imaginary: bool = True,
) -> SearchStrategy[dt.datetime]:
    """datetimes(min_value=datetime.datetime.min, max_value=datetime.datetime.max, *, timezones=none(), allow_imaginary=True)

    A strategy for generating datetimes, which may be timezone-aware.

    This strategy works by drawing a naive datetime between ``min_value``
    and ``max_value``, which must both be naive (have no timezone).

    ``timezones`` must be a strategy that generates either ``None``, for naive
    datetimes, or :class:`~python:datetime.tzinfo` objects for 'aware' datetimes.
    You can construct your own, though we recommend using one of these built-in
    strategies:

    * with the standard library: :func:`hypothesis.strategies.timezones`;
    * with :pypi:`dateutil <python-dateutil>`:
      :func:`hypothesis.extra.dateutil.timezones`; or
    * with :pypi:`pytz`: :func:`hypothesis.extra.pytz.timezones`.

    You may pass ``allow_imaginary=False`` to filter out "imaginary" datetimes
    which did not (or will not) occur due to daylight savings, leap seconds,
    timezone and calendar adjustments, etc.  Imaginary datetimes are allowed
    by default, because malformed timestamps are a common source of bugs.

    Examples from this strategy shrink towards midnight on January 1st 2000,
    local time.
    """
    # Why must bounds be naive?  In principle, we could also write a strategy
    # that took aware bounds, but the API and validation is much harder.
    # If you want to generate datetimes between two particular moments in
    # time I suggest (a) just filtering out-of-bounds values; (b) if bounds
    # are very close, draw a value and subtract its UTC offset, handling
    # overflows and nonexistent times; or (c) do something customised to
    # handle datetimes in e.g. a four-microsecond span which is not
    # representable in UTC.  Handling (d), all of the above, leads to a much
    # more complex API for all users and a useful feature for very few.
    check_type(bool, allow_imaginary, "allow_imaginary")
    check_type(dt.datetime, min_value, "min_value")
    check_type(dt.datetime, max_value, "max_value")
    if min_value.tzinfo is not None:
        raise InvalidArgument(f"{min_value=} must not have tzinfo")
    if max_value.tzinfo is not None:
        raise InvalidArgument(f"{max_value=} must not have tzinfo")
    check_valid_interval(min_value, max_value, "min_value", "max_value")
    if not isinstance(timezones, SearchStrategy):
        raise InvalidArgument(
            f"{timezones=} must be a SearchStrategy that can "
            "provide tzinfo for datetimes (either None or dt.tzinfo objects)"
        )
    return DatetimeStrategy(min_value, max_value, timezones, allow_imaginary)


class TimeStrategy(SearchStrategy):
    def __init__(self, min_value, max_value, timezones_strat):
        self.min_value = min_value
        self.max_value = max_value
        self.tz_strat = timezones_strat

    def do_draw(self, data):
        result = draw_capped_multipart(data, self.min_value, self.max_value, TIMENAMES)
        tz = data.draw(self.tz_strat)
        return dt.time(**result, tzinfo=tz)


@defines_strategy(force_reusable_values=True)
def times(
    min_value: dt.time = dt.time.min,
    max_value: dt.time = dt.time.max,
    *,
    timezones: SearchStrategy[Optional[dt.tzinfo]] = none(),
) -> SearchStrategy[dt.time]:
    """times(min_value=datetime.time.min, max_value=datetime.time.max, *, timezones=none())

    A strategy for times between ``min_value`` and ``max_value``.

    The ``timezones`` argument is handled as for :py:func:`datetimes`.

    Examples from this strategy shrink towards midnight, with the timezone
    component shrinking as for the strategy that provided it.
    """
    check_type(dt.time, min_value, "min_value")
    check_type(dt.time, max_value, "max_value")
    if min_value.tzinfo is not None:
        raise InvalidArgument(f"{min_value=} must not have tzinfo")
    if max_value.tzinfo is not None:
        raise InvalidArgument(f"{max_value=} must not have tzinfo")
    check_valid_interval(min_value, max_value, "min_value", "max_value")
    return TimeStrategy(min_value, max_value, timezones)


class DateStrategy(SearchStrategy):
    def __init__(self, min_value, max_value):
        assert isinstance(min_value, dt.date)
        assert isinstance(max_value, dt.date)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def do_draw(self, data):
        return dt.date(
            **draw_capped_multipart(data, self.min_value, self.max_value, DATENAMES)
        )

    def filter(self, condition):
        if (
            isinstance(condition, partial)
            and len(args := condition.args) == 1
            and not condition.keywords
            and isinstance(arg := args[0], dt.date)
            and condition.func in (op.lt, op.le, op.eq, op.ge, op.gt)
        ):
            try:
                arg += dt.timedelta(days={op.lt: 1, op.gt: -1}.get(condition.func, 0))
            except OverflowError:  # gt date.max, or lt date.min
                return nothing()
            lo, hi = {
                # We're talking about op(arg, x) - the reverse of our usual intuition!
                op.lt: (arg, self.max_value),  # lambda x: arg < x
                op.le: (arg, self.max_value),  # lambda x: arg <= x
                op.eq: (arg, arg),  #            lambda x: arg == x
                op.ge: (self.min_value, arg),  # lambda x: arg >= x
                op.gt: (self.min_value, arg),  # lambda x: arg > x
            }[condition.func]
            lo = max(lo, self.min_value)
            hi = min(hi, self.max_value)
            print(lo, hi)
            if hi < lo:
                return nothing()
            if lo <= self.min_value and self.max_value <= hi:
                return self
            return dates(lo, hi)

        return super().filter(condition)


@defines_strategy(force_reusable_values=True)
def dates(
    min_value: dt.date = dt.date.min, max_value: dt.date = dt.date.max
) -> SearchStrategy[dt.date]:
    """dates(min_value=datetime.date.min, max_value=datetime.date.max)

    A strategy for dates between ``min_value`` and ``max_value``.

    Examples from this strategy shrink towards January 1st 2000.
    """
    check_type(dt.date, min_value, "min_value")
    check_type(dt.date, max_value, "max_value")
    check_valid_interval(min_value, max_value, "min_value", "max_value")
    if min_value == max_value:
        return just(min_value)
    return DateStrategy(min_value, max_value)


class TimedeltaStrategy(SearchStrategy):
    def __init__(self, min_value, max_value):
        assert isinstance(min_value, dt.timedelta)
        assert isinstance(max_value, dt.timedelta)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def do_draw(self, data):
        result = {}
        low_bound = True
        high_bound = True
        for name in ("days", "seconds", "microseconds"):
            low = getattr(self.min_value if low_bound else dt.timedelta.min, name)
            high = getattr(self.max_value if high_bound else dt.timedelta.max, name)
            val = data.draw_integer(low, high)
            result[name] = val
            low_bound = low_bound and val == low
            high_bound = high_bound and val == high
        return dt.timedelta(**result)


@defines_strategy(force_reusable_values=True)
def timedeltas(
    min_value: dt.timedelta = dt.timedelta.min,
    max_value: dt.timedelta = dt.timedelta.max,
) -> SearchStrategy[dt.timedelta]:
    """timedeltas(min_value=datetime.timedelta.min, max_value=datetime.timedelta.max)

    A strategy for timedeltas between ``min_value`` and ``max_value``.

    Examples from this strategy shrink towards zero.
    """
    check_type(dt.timedelta, min_value, "min_value")
    check_type(dt.timedelta, max_value, "max_value")
    check_valid_interval(min_value, max_value, "min_value", "max_value")
    if min_value == max_value:
        return just(min_value)
    return TimedeltaStrategy(min_value=min_value, max_value=max_value)


@lru_cache(maxsize=None)
def _valid_key_cacheable(tzpath, key):
    assert isinstance(tzpath, tuple)  # zoneinfo changed, better update this function!
    for root in tzpath:
        if Path(root).joinpath(key).exists():  # pragma: no branch
            # No branch because most systems only have one TZPATH component.
            return True
    else:  # pragma: no cover
        # This branch is only taken for names which are known to zoneinfo
        # but not present on the filesystem, i.e. on Windows with tzdata,
        # and so is never executed by our coverage tests.
        *package_loc, resource_name = key.split("/")
        package = "tzdata.zoneinfo." + ".".join(package_loc)
        try:
            return (resources.files(package) / resource_name).exists()
        except ModuleNotFoundError:
            return False


@defines_strategy(force_reusable_values=True)
def timezone_keys(
    *,
    # allow_alias: bool = True,
    # allow_deprecated: bool = True,
    allow_prefix: bool = True,
) -> SearchStrategy[str]:
    """A strategy for :wikipedia:`IANA timezone names <List_of_tz_database_time_zones>`.

    As well as timezone names like ``"UTC"``, ``"Australia/Sydney"``, or
    ``"America/New_York"``, this strategy can generate:

    - Aliases such as ``"Antarctica/McMurdo"``, which links to ``"Pacific/Auckland"``.
    - Deprecated names such as ``"Antarctica/South_Pole"``, which *also* links to
      ``"Pacific/Auckland"``.  Note that most but
      not all deprecated timezone names are also aliases.
    - Timezone names with the ``"posix/"`` or ``"right/"`` prefixes, unless
      ``allow_prefix=False``.

    These strings are provided separately from Tzinfo objects - such as ZoneInfo
    instances from the timezones() strategy - to facilitate testing of timezone
    logic without needing workarounds to access non-canonical names.

    .. note::

        `The tzdata package is required on Windows
        <https://docs.python.org/3/library/zoneinfo.html#data-sources>`__.
        ``pip install hypothesis[zoneinfo]`` installs it, if and only if needed.

    On Windows, you may need to access IANA timezone data via the :pypi:`tzdata`
    package.  For non-IANA timezones, such as Windows-native names or GNU TZ
    strings, we recommend using :func:`~hypothesis.strategies.sampled_from` with
    the :pypi:`dateutil <python-dateutil>` package, e.g.
    :meth:`dateutil:dateutil.tz.tzwin.list`.
    """
    # check_type(bool, allow_alias, "allow_alias")
    # check_type(bool, allow_deprecated, "allow_deprecated")
    check_type(bool, allow_prefix, "allow_prefix")

    available_timezones = ("UTC", *sorted(zoneinfo.available_timezones()))

    # TODO: filter out alias and deprecated names if disallowed

    # When prefixes are allowed, we first choose a key and then flatmap to get our
    # choice with one of the available prefixes.  That in turn means that we need
    # some logic to determine which prefixes are available for a given key:

    def valid_key(key):
        return key == "UTC" or _valid_key_cacheable(zoneinfo.TZPATH, key)

    # TODO: work out how to place a higher priority on "weird" timezones
    # For details see https://github.com/HypothesisWorks/hypothesis/issues/2414
    strategy = sampled_from([key for key in available_timezones if valid_key(key)])

    if not allow_prefix:
        return strategy

    def sample_with_prefixes(zone):
        keys_with_prefixes = (zone, f"posix/{zone}", f"right/{zone}")
        return sampled_from([key for key in keys_with_prefixes if valid_key(key)])

    return strategy.flatmap(sample_with_prefixes)


@defines_strategy(force_reusable_values=True)
def timezones(*, no_cache: bool = False) -> SearchStrategy["zoneinfo.ZoneInfo"]:
    """A strategy for :class:`python:zoneinfo.ZoneInfo` objects.

    If ``no_cache=True``, the generated instances are constructed using
    :meth:`ZoneInfo.no_cache <python:zoneinfo.ZoneInfo.no_cache>` instead
    of the usual constructor.  This may change the semantics of your datetimes
    in surprising ways, so only use it if you know that you need to!

    .. note::

        `The tzdata package is required on Windows
        <https://docs.python.org/3/library/zoneinfo.html#data-sources>`__.
        ``pip install hypothesis[zoneinfo]`` installs it, if and only if needed.
    """
    check_type(bool, no_cache, "no_cache")
    return timezone_keys().map(
        zoneinfo.ZoneInfo.no_cache if no_cache else zoneinfo.ZoneInfo
    )
