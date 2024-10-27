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
from hypothesis.strategies._internal.collections import tuples
from hypothesis.strategies._internal.core import (
    DataObject,
    DrawFn,
    binary,
    booleans,
    builds,
    characters,
    complex_numbers,
    composite,
    data,
    decimals,
    deferred,
    dictionaries,
    emails,
    fixed_dictionaries,
    fractions,
    from_regex,
    from_type,
    frozensets,
    functions,
    iterables,
    lists,
    permutations,
    random_module,
    randoms,
    recursive,
    register_type_strategy,
    runner,
    sampled_from,
    sets,
    shared,
    slices,
    text,
    uuids,
)
from hypothesis.strategies._internal.datetime import (
    dates,
    datetimes,
    timedeltas,
    times,
    timezone_keys,
    timezones,
)
from hypothesis.strategies._internal.ipaddress import ip_addresses
from hypothesis.strategies._internal.misc import just, none, nothing
from hypothesis.strategies._internal.numbers import floats, integers
from hypothesis.strategies._internal.strategies import one_of
from hypothesis.strategies._internal.utils import _strategies

# The implementation of all of these lives in `_strategies.py`, but we
# re-export them via this module to avoid exposing implementation details
# to over-zealous tab completion in editors that do not respect __all__.


__all__ = [
    "binary",
    "booleans",
    "builds",
    "characters",
    "complex_numbers",
    "composite",
    "data",
    "DataObject",
    "dates",
    "datetimes",
    "decimals",
    "deferred",
    "dictionaries",
    "DrawFn",
    "emails",
    "fixed_dictionaries",
    "floats",
    "fractions",
    "from_regex",
    "from_type",
    "frozensets",
    "functions",
    "integers",
    "ip_addresses",
    "iterables",
    "just",
    "lists",
    "none",
    "nothing",
    "one_of",
    "permutations",
    "random_module",
    "randoms",
    "recursive",
    "register_type_strategy",
    "runner",
    "sampled_from",
    "sets",
    "shared",
    "slices",
    "text",
    "timedeltas",
    "times",
    "timezone_keys",
    "timezones",
    "tuples",
    "uuids",
    "SearchStrategy",
]


def _check_exports(_public):
    assert set(__all__) == _public, (set(__all__) - _public, _public - set(__all__))

    # Verify that all exported strategy functions were registered with
    # @declares_strategy.

    existing_strategies = set(_strategies) - {"_maybe_nil_uuids"}

    exported_strategies = set(__all__) - {
        "DataObject",
        "DrawFn",
        "SearchStrategy",
        "composite",
        "register_type_strategy",
    }
    assert existing_strategies == exported_strategies, (
        existing_strategies - exported_strategies,
        exported_strategies - existing_strategies,
    )


_check_exports({n for n in dir() if n[0] not in "_@"})
del _check_exports
