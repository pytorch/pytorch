# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math
import sys
from collections.abc import Collection, Iterable, Sequence
from typing import Any

from hypothesis import (
    HealthCheck,
    assume,
    note,
    settings as Settings,
    strategies as st,
)
from hypothesis.errors import BackendCannotProceed
from hypothesis.internal.compat import batched
from hypothesis.internal.conjecture.choice import (
    ChoiceTypeT,
    choice_permitted,
)
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.conjecture.providers import (
    COLLECTION_DEFAULT_MAX_SIZE,
    HypothesisProvider,
    PrimitiveProvider,
    with_register_backend,
)
from hypothesis.internal.floats import SMALLEST_SUBNORMAL, sign_aware_lte
from hypothesis.internal.intervalsets import IntervalSet
from hypothesis.stateful import RuleBasedStateMachine, initialize, precondition, rule
from hypothesis.strategies import DrawFn, SearchStrategy
from hypothesis.strategies._internal.strings import OneCharStringStrategy, TextStrategy


def build_intervals(intervals: list[int]) -> list[tuple[int, int]]:
    if len(intervals) % 2:
        intervals = intervals[:-1]
    intervals.sort()
    return list(batched(intervals, 2, strict=True))


def interval_lists(
    *, min_codepoint: int = 0, max_codepoint: int = sys.maxunicode, min_size: int = 0
) -> SearchStrategy[Iterable[Sequence[int]]]:
    return (
        st.lists(
            st.integers(min_codepoint, max_codepoint),
            unique=True,
            min_size=min_size * 2,
        )
        .map(sorted)
        .map(build_intervals)
    )


def intervals(
    *, min_codepoint: int = 0, max_codepoint: int = sys.maxunicode, min_size: int = 0
) -> SearchStrategy[IntervalSet]:
    return st.builds(
        IntervalSet,
        interval_lists(
            min_codepoint=min_codepoint, max_codepoint=max_codepoint, min_size=min_size
        ),
    )


@st.composite
def integer_weights(
    draw: DrawFn, min_value: int | None = None, max_value: int | None = None
) -> dict[int, float]:
    # Sampler doesn't play well with super small floats, so exclude them
    weights = draw(
        st.dictionaries(
            st.integers(min_value=min_value, max_value=max_value),
            st.floats(0.001, 1),
            min_size=1,
            max_size=255,
        )
    )
    # invalid to have a weighting that disallows all possibilities
    assume(sum(weights.values()) != 0)
    # re-normalize probabilities to sum to some arbitrary target < 1
    target = draw(st.floats(0.001, 0.999))
    factor = target / sum(weights.values())
    weights = {k: v * factor for k, v in weights.items()}
    # float rounding error can cause this to fail.
    assume(0.001 <= sum(weights.values()) <= 0.999)
    return weights


@st.composite
def integer_constraints(
    draw,
    *,
    use_min_value=None,
    use_max_value=None,
    use_shrink_towards=None,
    use_weights=None,
    use_forced=False,
):
    min_value = None
    max_value = None
    shrink_towards = 0
    weights = None

    if use_min_value is None:
        use_min_value = draw(st.booleans())
    if use_max_value is None:
        use_max_value = draw(st.booleans())
    use_shrink_towards = draw(st.booleans())
    if use_weights is None:
        use_weights = (
            draw(st.booleans()) if (use_min_value and use_max_value) else False
        )

    # Invariants:
    # (1) min_value <= forced <= max_value
    # (2) sum(weights.values()) < 1
    # (3) len(weights) <= 255

    if use_shrink_towards:
        shrink_towards = draw(st.integers())

    forced = draw(st.integers()) if use_forced else None
    if use_weights:
        assert use_max_value
        assert use_min_value

        min_value = draw(st.integers(max_value=forced))
        min_val = max(min_value, forced) if forced is not None else min_value
        max_value = draw(st.integers(min_value=min_val))

        weights = draw(integer_weights(min_value, max_value))
    else:
        if use_min_value:
            min_value = draw(st.integers(max_value=forced))
        if use_max_value:
            min_vals = []
            if min_value is not None:
                min_vals.append(min_value)
            if forced is not None:
                min_vals.append(forced)
            min_val = max(min_vals) if min_vals else None
            max_value = draw(st.integers(min_value=min_val))

    if forced is not None:
        assume((forced - shrink_towards).bit_length() < 128)

    return {
        "min_value": min_value,
        "max_value": max_value,
        "shrink_towards": shrink_towards,
        "weights": weights,
        "forced": forced,
    }


@st.composite
def _collection_constraints(
    draw: DrawFn,
    *,
    forced: Any | None,
    use_min_size: bool | None = None,
    use_max_size: bool | None = None,
) -> dict[str, int]:
    min_size = 0
    max_size = COLLECTION_DEFAULT_MAX_SIZE
    # collections are quite expensive in entropy. cap to avoid overruns.
    cap = 50

    if use_min_size is None:
        use_min_size = draw(st.booleans())
    if use_max_size is None:
        use_max_size = draw(st.booleans())

    if use_min_size:
        min_size = draw(
            st.integers(0, min(len(forced), cap) if forced is not None else cap)
        )

    if use_max_size:
        max_size = draw(
            st.integers(
                min_value=min_size if forced is None else max(min_size, len(forced))
            )
        )
        if forced is None:
            # cap to some reasonable max size to avoid overruns.
            max_size = min(max_size, min_size + 100)

    return {"min_size": min_size, "max_size": max_size}


@st.composite
def string_constraints(
    draw: DrawFn,
    *,
    use_min_size: bool | None = None,
    use_max_size: bool | None = None,
    use_forced: bool = False,
) -> Any:
    interval_set = draw(intervals())
    forced = (
        draw(TextStrategy(OneCharStringStrategy(interval_set))) if use_forced else None
    )
    constraints = draw(
        _collection_constraints(
            forced=forced, use_min_size=use_min_size, use_max_size=use_max_size
        )
    )
    # if the intervalset is empty, then the min size must be zero, because the
    # only valid value is the empty string.
    if len(interval_set) == 0:
        constraints["min_size"] = 0

    return {"intervals": interval_set, "forced": forced, **constraints}


@st.composite
def bytes_constraints(
    draw: DrawFn,
    *,
    use_min_size: bool | None = None,
    use_max_size: bool | None = None,
    use_forced: bool = False,
) -> Any:
    forced = draw(st.binary()) if use_forced else None

    constraints = draw(
        _collection_constraints(
            forced=forced, use_min_size=use_min_size, use_max_size=use_max_size
        )
    )
    return {"forced": forced, **constraints}


@st.composite
def float_constraints(
    draw,
    *,
    use_min_value=None,
    use_max_value=None,
    use_forced=False,
):
    if use_min_value is None:
        use_min_value = draw(st.booleans())
    if use_max_value is None:
        use_max_value = draw(st.booleans())

    forced = draw(st.floats()) if use_forced else None
    pivot = forced if (use_forced and not math.isnan(forced)) else None
    min_value = -math.inf
    max_value = math.inf
    smallest_nonzero_magnitude = SMALLEST_SUBNORMAL
    allow_nan = True if (use_forced and math.isnan(forced)) else draw(st.booleans())

    if use_min_value:
        min_value = draw(st.floats(max_value=pivot, allow_nan=False))

    if use_max_value:
        if pivot is None:
            min_val = min_value
        else:
            min_val = pivot if sign_aware_lte(min_value, pivot) else min_value
        max_value = draw(st.floats(min_value=min_val, allow_nan=False))

    largest_magnitude = max(abs(min_value), abs(max_value))
    # can't force something smaller than our smallest magnitude.
    if pivot is not None and pivot != 0.0:
        largest_magnitude = min(largest_magnitude, pivot)

    # avoid drawing from an empty range
    if largest_magnitude > 0:
        smallest_nonzero_magnitude = draw(
            st.floats(
                min_value=0,
                # smallest_nonzero_magnitude breaks internal clamper invariants if
                # it is allowed to be larger than the magnitude of {min, max}_value.
                #
                # Let's also be reasonable here; smallest_nonzero_magnitude is used
                # for subnormals, so we will never provide a number above 1 in practice.
                max_value=min(largest_magnitude, 1.0),
                exclude_min=True,
            )
        )

    assert sign_aware_lte(min_value, max_value)
    return {
        "min_value": min_value,
        "max_value": max_value,
        "forced": forced,
        "allow_nan": allow_nan,
        "smallest_nonzero_magnitude": smallest_nonzero_magnitude,
    }


@st.composite
def boolean_constraints(draw: DrawFn, *, use_forced: bool = False) -> Any:
    forced = draw(st.booleans()) if use_forced else None
    # avoid invalid forced combinations
    p = draw(st.floats(0, 1, exclude_min=forced is True, exclude_max=forced is False))

    return {"p": p, "forced": forced}


def constraints_strategy(choice_type, strategy_constraints=None, *, use_forced=False):
    strategy = {
        "boolean": boolean_constraints,
        "integer": integer_constraints,
        "float": float_constraints,
        "bytes": bytes_constraints,
        "string": string_constraints,
    }[choice_type]
    if strategy_constraints is None:
        strategy_constraints = {}
    return strategy(**strategy_constraints.get(choice_type, {}), use_forced=use_forced)


def choice_types_constraints(strategy_constraints=None, *, use_forced=False):
    options: list[ChoiceTypeT] = ["boolean", "integer", "float", "bytes", "string"]
    return st.one_of(
        st.tuples(
            st.just(name),
            constraints_strategy(name, strategy_constraints, use_forced=use_forced),
        )
        for name in options
    )


def run_conformance_test(
    Provider: type[PrimitiveProvider],
    *,
    context_manager_exceptions: Collection[type[BaseException]] = (),
    settings: Settings | None = None,
    _realize_objects: SearchStrategy[Any] = (
        st.from_type(object) | st.from_type(type).flatmap(st.from_type)
    ),
) -> None:
    """
    Test that the given ``Provider`` class conforms to the |PrimitiveProvider|
    interface.

    For instance, this tests that ``Provider`` does not return out of bounds
    choices from any of the ``draw_*`` methods, or violate other invariants
    which Hypothesis depends on.

    This function is intended to be called at test-time, not at runtime. It is
    provided by Hypothesis to make it easy for third-party backend authors to
    test their provider. Backend authors wishing to test their provider should
    include a test similar to the following in their test suite:

    .. code-block:: python

        from hypothesis.internal.conjecture.provider_conformance import run_conformance_test

        def test_conformance():
            run_conformance_test(MyProvider)

    If your provider can raise control flow exceptions inside one of the five
    ``draw_*`` methods that are handled by your provider's
    ``per_test_case_context_manager``, pass a list of these exceptions types to
    ``context_manager_exceptions``. Otherwise, ``run_conformance_test`` will
    treat those exceptions as fatal errors.
    """

    class CopiesRealizationProvider(HypothesisProvider):
        avoid_realization = Provider.avoid_realization

    with with_register_backend("copies_realization", CopiesRealizationProvider):

        @Settings(
            settings,
            suppress_health_check=[HealthCheck.too_slow],
            backend="copies_realization",
        )
        class ProviderConformanceTest(RuleBasedStateMachine):
            def __init__(self):
                super().__init__()

            @initialize(random=st.randoms())
            def setup(self, random):
                if Provider.lifetime == "test_case":
                    data = ConjectureData(random=random, provider=Provider)
                    self.provider = data.provider
                else:
                    self.provider = Provider(None)

                self.context_manager = self.provider.per_test_case_context_manager()
                self.context_manager.__enter__()
                self.frozen = False

            def _draw(self, choice_type, constraints):
                del constraints["forced"]
                draw_func = getattr(self.provider, f"draw_{choice_type}")

                try:
                    choice = draw_func(**constraints)
                    note(f"drew {choice_type} {choice}")
                    expected_type = {
                        "integer": int,
                        "float": float,
                        "bytes": bytes,
                        "string": str,
                        "boolean": bool,
                    }[choice_type]
                    assert isinstance(choice, expected_type)
                    assert choice_permitted(choice, constraints)
                except context_manager_exceptions as e:
                    note(
                        f"caught exception {type(e)} in context_manager_exceptions: {e}"
                    )
                    try:
                        self.context_manager.__exit__(type(e), e, None)
                    except BackendCannotProceed:
                        self.frozen = True
                        return None

                return choice

            @precondition(lambda self: not self.frozen)
            @rule(constraints=integer_constraints())
            def draw_integer(self, constraints):
                self._draw("integer", constraints)

            @precondition(lambda self: not self.frozen)
            @rule(constraints=float_constraints())
            def draw_float(self, constraints):
                self._draw("float", constraints)

            @precondition(lambda self: not self.frozen)
            @rule(constraints=bytes_constraints())
            def draw_bytes(self, constraints):
                self._draw("bytes", constraints)

            @precondition(lambda self: not self.frozen)
            @rule(constraints=string_constraints())
            def draw_string(self, constraints):
                self._draw("string", constraints)

            @precondition(lambda self: not self.frozen)
            @rule(constraints=boolean_constraints())
            def draw_boolean(self, constraints):
                self._draw("boolean", constraints)

            @precondition(lambda self: not self.frozen)
            @rule(label=st.integers())
            def span_start(self, label):
                self.provider.span_start(label)

            @precondition(lambda self: not self.frozen)
            @rule(discard=st.booleans())
            def span_end(self, discard):
                self.provider.span_end(discard)

            @precondition(lambda self: not self.frozen)
            @rule()
            def freeze(self):
                # phase-transition, mimicking data.freeze() at the end of a test case.
                self.frozen = True
                self.context_manager.__exit__(None, None, None)

            @precondition(lambda self: self.frozen)
            @rule(value=_realize_objects)
            def realize(self, value):
                # filter out nans and weirder things
                try:
                    assume(value == value)
                except Exception:
                    # e.g. value = Decimal('-sNaN')
                    assume(False)

                # if `value` is non-symbolic, the provider should return it as-is.
                assert self.provider.realize(value) == value

            @precondition(lambda self: self.frozen)
            @rule()
            def observe_test_case(self):
                observations = self.provider.observe_test_case()
                assert isinstance(observations, dict)

            @precondition(lambda self: self.frozen)
            @rule(lifetime=st.sampled_from(["test_function", "test_case"]))
            def observe_information_messages(self, lifetime):
                observations = self.provider.observe_information_messages(
                    lifetime=lifetime
                )
                for observation in observations:
                    assert isinstance(observation, dict)

            def teardown(self):
                if not self.frozen:
                    self.context_manager.__exit__(None, None, None)

        ProviderConformanceTest.TestCase().runTest()
