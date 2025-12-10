# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import dataclasses
import sys
from collections.abc import Callable
from functools import partial
from typing import Literal, TypeAlias, TypeVar
from weakref import WeakValueDictionary

from hypothesis.errors import InvalidArgument
from hypothesis.internal.cache import LRUReusedCache
from hypothesis.internal.floats import clamp, float_to_int
from hypothesis.internal.reflection import proxies
from hypothesis.vendor.pretty import pretty

T = TypeVar("T")
ValueKey: TypeAlias = tuple[type, object]
# (fn, args, kwargs)
StrategyCacheKey: TypeAlias = tuple[
    object, tuple[ValueKey, ...], frozenset[tuple[str, ValueKey]]
]

_all_strategies: WeakValueDictionary[str, Callable] = WeakValueDictionary()
# note: LRUReusedCache is already thread-local internally
_STRATEGY_CACHE = LRUReusedCache[StrategyCacheKey, object](1024)


def _value_key(value: object) -> ValueKey:
    if isinstance(value, float):
        return (float, float_to_int(value))
    return (type(value), value)


def clear_strategy_cache() -> None:
    _STRATEGY_CACHE.clear()


def cacheable(fn: T) -> T:
    from hypothesis.control import _current_build_context
    from hypothesis.strategies._internal.strategies import SearchStrategy

    @proxies(fn)
    def cached_strategy(*args, **kwargs):
        context = _current_build_context.value
        if context is not None and context.data.provider.avoid_realization:
            return fn(*args, **kwargs)

        try:
            kwargs_cache_key = {(k, _value_key(v)) for k, v in kwargs.items()}
        except TypeError:
            return fn(*args, **kwargs)

        cache_key = (
            fn,
            tuple(_value_key(v) for v in args),
            frozenset(kwargs_cache_key),
        )
        try:
            return _STRATEGY_CACHE[cache_key]
        except KeyError:
            pass
        except TypeError:
            return fn(*args, **kwargs)

        result = fn(*args, **kwargs)
        if not isinstance(result, SearchStrategy) or result.is_cacheable:
            _STRATEGY_CACHE[cache_key] = result
        return result

    # note that calling this clears the full _STRATEGY_CACHE for all strategies,
    # not just the cache for this strategy.
    cached_strategy.__clear_cache = clear_strategy_cache  # type: ignore
    return cached_strategy


def defines_strategy(
    *,
    force_reusable_values: bool = False,
    eager: bool | Literal["try"] = False,
) -> Callable[[T], T]:
    """
    Each standard strategy function provided to users by Hypothesis should be
    decorated with @defines_strategy. This registers the strategy with _all_strategies,
    which is used in our own test suite to check that e.g. we document all strategies
    in sphinx.

    If you're reading this and are the author of a third-party strategy library:
    don't worry, third-party strategies don't need to be decorated with
    @defines_strategy. This function is internal to Hypothesis and not intended
    for outside use.

    Parameters
    ----------
    force_reusable_values : bool
        If ``True``, strategies returned from the strategy function will have
        ``.has_reusable_values == True`` set, even if it uses maps/filters or
        non-reusable strategies internally. This tells our numpy/pandas strategies
        that they can implicitly use such strategies as background values.
    eager : bool | "try"
        If ``True``, strategies returned by the strategy function are returned
        as-is, and not wrapped in LazyStrategy.

        If "try", we first attempt to call the strategy function and return the
        resulting strategy. If this throws an exception, we treat it the same as
        ``eager = False``, by returning the strategy function wrapped in a
        LazyStrategy.
    """

    if eager is not False and force_reusable_values:  # pragma: no cover
        # We could support eager + force_reusable_values with a suitable wrapper,
        # but there are currently no callers that request this combination.
        raise InvalidArgument(
            f"Passing both eager={eager} and force_reusable_values=True is "
            "currently not supported"
        )

    def decorator(strategy_definition):
        _all_strategies[strategy_definition.__name__] = strategy_definition

        if eager is True:
            return strategy_definition

        @proxies(strategy_definition)
        def accept(*args, **kwargs):
            from hypothesis.strategies._internal.lazy import LazyStrategy

            if eager == "try":
                # Why not try this unconditionally?  Because we'd end up with very
                # deep nesting of recursive strategies - better to be lazy unless we
                # *know* that eager evaluation is the right choice.
                try:
                    return strategy_definition(*args, **kwargs)
                except Exception:
                    # If invoking the strategy definition raises an exception,
                    # wrap that up in a LazyStrategy so it happens again later.
                    pass
            result = LazyStrategy(strategy_definition, args, kwargs)
            if force_reusable_values:
                # Setting `force_has_reusable_values` here causes the recursive
                # property code to set `.has_reusable_values == True`.
                result.force_has_reusable_values = True
                assert result.has_reusable_values
            return result

        accept.is_hypothesis_strategy_function = True
        return accept

    return decorator


def _to_jsonable(obj: object, *, avoid_realization: bool, seen: set[int]) -> object:
    if isinstance(obj, (str, int, float, bool, type(None))):
        # We convert integers of 2**63 to floats, to avoid crashing external
        # utilities with a 64 bit integer cap (notable, sqlite). See
        # https://github.com/HypothesisWorks/hypothesis/pull/3797#discussion_r1413425110
        # and https://github.com/simonw/sqlite-utils/issues/605.
        if isinstance(obj, int) and not isinstance(obj, bool) and abs(obj) >= 2**63:
            # Silently clamp very large ints to max_float, to avoid OverflowError when
            # casting to float.  (but avoid adding more constraints to symbolic values)
            if avoid_realization:
                return "<symbolic>"
            obj = clamp(-sys.float_info.max, obj, sys.float_info.max)
            return float(obj)
        return obj
    if avoid_realization:
        return "<symbolic>"

    obj_id = id(obj)
    if obj_id in seen:
        return pretty(obj, cycle=True)

    recur = partial(
        _to_jsonable, avoid_realization=avoid_realization, seen=seen | {obj_id}
    )
    if isinstance(obj, (list, tuple, set, frozenset)):
        if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
            return recur(obj._asdict())  # treat namedtuples as dicts
        return [recur(x) for x in obj]
    if isinstance(obj, dict):
        return {
            k if isinstance(k, str) else pretty(k): recur(v) for k, v in obj.items()
        }

    # Hey, might as well try calling a .to_json() method - it works for Pandas!
    # We try this before the below general-purpose handlers to give folks a
    # chance to control this behavior on their custom classes.
    try:
        return recur(obj.to_json())  # type: ignore
    except Exception:
        pass

    # Special handling for dataclasses, attrs, and pydantic classes
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Avoid dataclasses.asdict here to ensure that inner to_json overrides
        # can get called as well
        return {
            field.name: recur(getattr(obj, field.name))
            for field in dataclasses.fields(obj)
        }
    if (attr := sys.modules.get("attr")) is not None and attr.has(type(obj)):
        return recur(attr.asdict(obj, recurse=False))
    if (pyd := sys.modules.get("pydantic")) and isinstance(obj, pyd.BaseModel):
        return recur(obj.model_dump())

    # If all else fails, we'll just pretty-print as a string.
    return pretty(obj)


def to_jsonable(obj: object, *, avoid_realization: bool) -> object:
    """Recursively convert an object to json-encodable form.

    This is not intended to round-trip, but rather provide an analysis-ready
    format for observability.  To avoid side affects, we pretty-print all but
    known types.
    """
    return _to_jsonable(obj, avoid_realization=avoid_realization, seen=set())
