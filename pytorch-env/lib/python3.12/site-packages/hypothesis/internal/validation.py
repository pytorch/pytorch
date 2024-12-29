# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import decimal
import math
from numbers import Rational, Real

from hypothesis.errors import InvalidArgument
from hypothesis.internal.coverage import check_function


@check_function
def check_type(typ, arg, name):
    if not isinstance(arg, typ):
        if isinstance(typ, tuple):
            assert len(typ) >= 2, "Use bare type instead of len-1 tuple"
            typ_string = "one of " + ", ".join(t.__name__ for t in typ)
        else:
            typ_string = typ.__name__

            if typ_string == "SearchStrategy":
                from hypothesis.strategies import SearchStrategy

                # Use hypothesis.strategies._internal.strategies.check_strategy
                # instead, as it has some helpful "did you mean..." logic.
                assert typ is not SearchStrategy, "use check_strategy instead"

        raise InvalidArgument(
            f"Expected {typ_string} but got {name}={arg!r} (type={type(arg).__name__})"
        )


@check_function
def check_valid_integer(value, name):
    """Checks that value is either unspecified, or a valid integer.

    Otherwise raises InvalidArgument.
    """
    if value is None:
        return
    check_type(int, value, name)


@check_function
def check_valid_bound(value, name):
    """Checks that value is either unspecified, or a valid interval bound.

    Otherwise raises InvalidArgument.
    """
    if value is None or isinstance(value, (int, Rational)):
        return
    if not isinstance(value, (Real, decimal.Decimal)):
        raise InvalidArgument(f"{name}={value!r} must be a real number.")
    if math.isnan(value):
        raise InvalidArgument(f"Invalid end point {name}={value!r}")


@check_function
def check_valid_magnitude(value, name):
    """Checks that value is either unspecified, or a non-negative valid
    interval bound.

    Otherwise raises InvalidArgument.
    """
    check_valid_bound(value, name)
    if value is not None and value < 0:
        raise InvalidArgument(f"{name}={value!r} must not be negative.")
    elif value is None and name == "min_magnitude":
        raise InvalidArgument("Use min_magnitude=0 or omit the argument entirely.")


@check_function
def try_convert(typ, value, name):
    if value is None:
        return None
    if isinstance(value, typ):
        return value
    try:
        return typ(value)
    except (TypeError, ValueError, ArithmeticError) as err:
        raise InvalidArgument(
            f"Cannot convert {name}={value!r} of type "
            f"{type(value).__name__} to type {typ.__name__}"
        ) from err


@check_function
def check_valid_size(value, name):
    """Checks that value is either unspecified, or a valid non-negative size
    expressed as an integer.

    Otherwise raises InvalidArgument.
    """
    if value is None and name not in ("min_size", "size"):
        return
    check_type(int, value, name)
    if value < 0:
        raise InvalidArgument(f"Invalid size {name}={value!r} < 0")


@check_function
def check_valid_interval(lower_bound, upper_bound, lower_name, upper_name):
    """Checks that lower_bound and upper_bound are either unspecified, or they
    define a valid interval on the number line.

    Otherwise raises InvalidArgument.
    """
    if lower_bound is None or upper_bound is None:
        return
    if upper_bound < lower_bound:
        raise InvalidArgument(
            f"Cannot have {upper_name}={upper_bound!r} < {lower_name}={lower_bound!r}"
        )


@check_function
def check_valid_sizes(min_size, max_size):
    check_valid_size(min_size, "min_size")
    check_valid_size(max_size, "max_size")
    check_valid_interval(min_size, max_size, "min_size", "max_size")
