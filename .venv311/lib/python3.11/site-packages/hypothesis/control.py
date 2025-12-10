# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import inspect
import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any, Literal, NoReturn, Optional, overload
from weakref import WeakKeyDictionary

from hypothesis import Verbosity, settings
from hypothesis._settings import note_deprecation
from hypothesis.errors import InvalidArgument, UnsatisfiedAssumption
from hypothesis.internal.compat import BaseExceptionGroup
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.observability import observability_enabled
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.internal.validation import check_type
from hypothesis.reporting import report, verbose_report
from hypothesis.utils.dynamicvariables import DynamicVariable
from hypothesis.vendor.pretty import IDKey, PrettyPrintFunction, pretty


def _calling_function_location(what: str, frame: Any) -> str:
    where = frame.f_back
    return f"{what}() in {where.f_code.co_name} (line {where.f_lineno})"


def reject() -> NoReturn:
    if _current_build_context.value is None:
        note_deprecation(
            "Using `reject` outside a property-based test is deprecated",
            since="2023-09-25",
            has_codemod=False,
        )
    where = _calling_function_location("reject", inspect.currentframe())
    if currently_in_test_context():
        counts = current_build_context().data._observability_predicates[where]
        counts.update_count(condition=False)
    raise UnsatisfiedAssumption(where)


@overload
def assume(condition: Literal[False] | None) -> NoReturn: ...
@overload
def assume(condition: object) -> Literal[True]: ...


def assume(condition: object) -> Literal[True]:
    """Calling ``assume`` is like an :ref:`assert <python:assert>` that marks
    the example as bad, rather than failing the test.

    This allows you to specify properties that you *assume* will be
    true, and let Hypothesis try to avoid similar examples in future.
    """
    if _current_build_context.value is None:
        note_deprecation(
            "Using `assume` outside a property-based test is deprecated",
            since="2023-09-25",
            has_codemod=False,
        )
    if observability_enabled() or not condition:
        where = _calling_function_location("assume", inspect.currentframe())
        if observability_enabled() and currently_in_test_context():
            counts = current_build_context().data._observability_predicates[where]
            counts.update_count(condition=bool(condition))
        if not condition:
            raise UnsatisfiedAssumption(f"failed to satisfy {where}")
    return True


_current_build_context = DynamicVariable[Optional["BuildContext"]](None)


def currently_in_test_context() -> bool:
    """Return ``True`` if the calling code is currently running inside an
    |@given| or :ref:`stateful <stateful>` test, and ``False`` otherwise.

    This is useful for third-party integrations and assertion helpers which
    may be called from either traditional or property-based tests, and can only
    use e.g. |assume| or |target| in the latter case.
    """
    return _current_build_context.value is not None


def current_build_context() -> "BuildContext":
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument("No build context registered")
    return context


@contextmanager
def deprecate_random_in_strategy(fmt, *args):
    from hypothesis.internal import entropy

    state_before = random.getstate()
    yield
    state_after = random.getstate()
    if (
        # there is a threading race condition here with deterministic_PRNG. Say
        # we have two threads 1 and 2. We start in global random state A, and
        # deterministic_PRNG sets to global random state B (which is constant across
        # threads since we seed to 0 unconditionally). Then we might have state
        # transitions:
        #
        #  [1]        [2]
        # A -> B                           deterministic_PRNG().__enter__
        #            B ->B                 deterministic_PRNG().__enter__
        #            state_before = B      deprecate_random_in_strategy.__enter__
        # B -> A                           deterministic_PRNG().__exit__
        #            state_after  = A      deprecate_random_in_strategy.__exit__
        #
        # where state_before != state_after because a different thread has reset
        # the global random state.
        #
        # To fix this, we track the known random states set by deterministic_PRNG,
        # and will not note a deprecation if it matches one of those.
        state_after != state_before
        and hash(state_after) not in entropy._known_random_state_hashes
    ):
        note_deprecation(
            "Do not use the `random` module inside strategies; instead "
            "consider `st.randoms()`, `st.sampled_from()`, etc.  " + fmt.format(*args),
            since="2024-02-05",
            has_codemod=False,
            stacklevel=1,
        )


class BuildContext:
    def __init__(
        self,
        data: ConjectureData,
        *,
        is_final: bool = False,
        wrapped_test: Callable,
    ) -> None:
        self.data = data
        self.tasks: list[Callable[[], Any]] = []
        self.is_final = is_final
        self.wrapped_test = wrapped_test

        # Use defaultdict(list) here to handle the possibility of having multiple
        # functions registered for the same object (due to caching, small ints, etc).
        # The printer will discard duplicates which return different representations.
        self.known_object_printers: dict[IDKey, list[PrettyPrintFunction]] = (
            defaultdict(list)
        )

    def record_call(
        self,
        obj: object,
        func: object,
        *,
        args: Sequence[object],
        kwargs: dict[str, object],
    ) -> None:
        self.known_object_printers[IDKey(obj)].append(
            # _func=func prevents mypy from inferring lambda type. Would need
            # paramspec I think - not worth it.
            lambda obj, p, cycle, *, _func=func: p.maybe_repr_known_object_as_call(  # type: ignore
                obj, cycle, get_pretty_function_description(_func), args, kwargs
            )
        )

    def prep_args_kwargs_from_strategies(self, kwarg_strategies):
        arg_labels = {}
        kwargs = {}
        for k, s in kwarg_strategies.items():
            start_idx = len(self.data.nodes)
            with deprecate_random_in_strategy("from {}={!r}", k, s):
                obj = self.data.draw(s, observe_as=f"generate:{k}")
            end_idx = len(self.data.nodes)
            kwargs[k] = obj

            # This high up the stack, we can't see or really do much with the conjecture
            # Example objects - not least because they're only materialized after the
            # test case is completed.  Instead, we'll stash the (start_idx, end_idx)
            # pair on our data object for the ConjectureRunner engine to deal with, and
            # pass a dict of such out so that the pretty-printer knows where to place
            # the which-parts-matter comments later.
            if start_idx != end_idx:
                arg_labels[k] = (start_idx, end_idx)
                self.data.arg_slices.add((start_idx, end_idx))

        return kwargs, arg_labels

    def __enter__(self):
        self.assign_variable = _current_build_context.with_value(self)
        self.assign_variable.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.assign_variable.__exit__(exc_type, exc_value, tb)
        errors = []
        for task in self.tasks:
            try:
                task()
            except BaseException as err:
                errors.append(err)
        if errors:
            if len(errors) == 1:
                raise errors[0] from exc_value
            raise BaseExceptionGroup("Cleanup failed", errors) from exc_value


def cleanup(teardown):
    """Register a function to be called when the current test has finished
    executing. Any exceptions thrown in teardown will be printed but not
    rethrown.

    Inside a test this isn't very interesting, because you can just use
    a finally block, but note that you can use this inside map, flatmap,
    etc. in order to e.g. insist that a value is closed at the end.
    """
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument("Cannot register cleanup outside of build context")
    context.tasks.append(teardown)


def should_note():
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument("Cannot make notes outside of a test")
    return context.is_final or settings.default.verbosity >= Verbosity.verbose


def note(value: object) -> None:
    """Report this value for the minimal failing example."""
    if should_note():
        if not isinstance(value, str):
            value = pretty(value)
        report(value)


def event(value: str, payload: str | int | float = "") -> None:
    """Record an event that occurred during this test. Statistics on the number of test
    runs with each event will be reported at the end if you run Hypothesis in
    statistics reporting mode.

    Event values should be strings or convertible to them.  If an optional
    payload is given, it will be included in the string for :ref:`statistics`.
    """
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument("Cannot record events outside of a test")

    avoid_realization = context.data.provider.avoid_realization
    payload = _event_to_string(
        payload, allowed_types=(str, int, float), avoid_realization=avoid_realization
    )
    value = _event_to_string(value, avoid_realization=avoid_realization)
    context.data.events[value] = payload


_events_to_strings: WeakKeyDictionary = WeakKeyDictionary()


def _event_to_string(event, *, allowed_types=str, avoid_realization):
    if isinstance(event, allowed_types):
        return event

    # _events_to_strings is a cache which persists across iterations, causing
    # problems for symbolic backends. see
    # https://github.com/pschanely/hypothesis-crosshair/issues/41
    if avoid_realization:
        return str(event)

    try:
        return _events_to_strings[event]
    except (KeyError, TypeError):
        pass

    result = str(event)
    try:
        _events_to_strings[event] = result
    except TypeError:
        pass
    return result


def target(observation: int | float, *, label: str = "") -> int | float:
    """Calling this function with an ``int`` or ``float`` observation gives it feedback
    with which to guide our search for inputs that will cause an error, in
    addition to all the usual heuristics.  Observations must always be finite.

    Hypothesis will try to maximize the observed value over several examples;
    almost any metric will work so long as it makes sense to increase it.
    For example, ``-abs(error)`` is a metric that increases as ``error``
    approaches zero.

    Example metrics:

    - Number of elements in a collection, or tasks in a queue
    - Mean or maximum runtime of a task (or both, if you use ``label``)
    - Compression ratio for data (perhaps per-algorithm or per-level)
    - Number of steps taken by a state machine

    The optional ``label`` argument can be used to distinguish between
    and therefore separately optimise distinct observations, such as the
    mean and standard deviation of a dataset.  It is an error to call
    ``target()`` with any label more than once per test case.

    .. note::
        The more examples you run, the better this technique works.

        As a rule of thumb, the targeting effect is noticeable above
        :obj:`max_examples=1000 <hypothesis.settings.max_examples>`,
        and immediately obvious by around ten thousand examples
        *per label* used by your test.

    :ref:`statistics` include the best score seen for each label,
    which can help avoid `the threshold problem
    <https://hypothesis.works/articles/threshold-problem/>`__ when the minimal
    example shrinks right down to the threshold of failure (:issue:`2180`).
    """
    check_type((int, float), observation, "observation")
    if not math.isfinite(observation):
        raise InvalidArgument(f"{observation=} must be a finite float.")
    check_type(str, label, "label")

    context = _current_build_context.value
    if context is None:
        raise InvalidArgument(
            "Calling target() outside of a test is invalid.  "
            "Consider guarding this call with `if currently_in_test_context(): ...`"
        )
    elif context.data.provider.avoid_realization:
        # We could in principle realize this in the engine, but it seems more
        # efficient to have our alternative backend optimize it for us.
        # See e.g. https://github.com/pschanely/hypothesis-crosshair/issues/3
        return observation  # pragma: no cover
    verbose_report(f"Saw target({observation!r}, {label=})")

    if label in context.data.target_observations:
        raise InvalidArgument(
            f"Calling target({observation!r}, {label=}) would overwrite "
            f"target({context.data.target_observations[label]!r}, {label=})"
        )
    else:
        context.data.target_observations[label] = observation

    return observation
