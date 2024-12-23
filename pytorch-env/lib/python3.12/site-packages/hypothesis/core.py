# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""This module provides the core primitives of Hypothesis, such as given."""

import base64
import contextlib
import datetime
import inspect
import io
import math
import sys
import time
import traceback
import types
import unittest
import warnings
import zlib
from collections import defaultdict
from collections.abc import Coroutine, Hashable
from functools import partial
from random import Random
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Optional,
    TypeVar,
    Union,
    overload,
)
from unittest import TestCase

import attr

from hypothesis import strategies as st
from hypothesis._settings import (
    HealthCheck,
    Phase,
    Verbosity,
    local_settings,
    settings as Settings,
)
from hypothesis.control import BuildContext
from hypothesis.errors import (
    BackendCannotProceed,
    DeadlineExceeded,
    DidNotReproduce,
    FailedHealthCheck,
    FlakyFailure,
    FlakyReplay,
    Found,
    HypothesisException,
    HypothesisWarning,
    InvalidArgument,
    NoSuchExample,
    StopTest,
    Unsatisfiable,
    UnsatisfiedAssumption,
)
from hypothesis.internal.compat import (
    PYPY,
    BaseExceptionGroup,
    add_note,
    bad_django_TestCase,
    get_type_hints,
    int_from_bytes,
)
from hypothesis.internal.conjecture.data import (
    ConjectureData,
    PrimitiveProvider,
    Status,
)
from hypothesis.internal.conjecture.engine import BUFFER_SIZE, ConjectureRunner
from hypothesis.internal.conjecture.junkdrawer import (
    ensure_free_stackframes,
    gc_cumulative_time,
)
from hypothesis.internal.conjecture.shrinker import sort_key
from hypothesis.internal.entropy import deterministic_PRNG
from hypothesis.internal.escalation import (
    InterestingOrigin,
    current_pytest_item,
    format_exception,
    get_trimmed_traceback,
    is_hypothesis_file,
)
from hypothesis.internal.healthcheck import fail_health_check
from hypothesis.internal.observability import (
    OBSERVABILITY_COLLECT_COVERAGE,
    TESTCASE_CALLBACKS,
    _system_metadata,
    deliver_json_blob,
    make_testcase,
)
from hypothesis.internal.reflection import (
    convert_positional_arguments,
    define_function_signature,
    function_digest,
    get_pretty_function_description,
    get_signature,
    impersonate,
    is_mock,
    nicerepr,
    proxies,
    repr_call,
)
from hypothesis.internal.scrutineer import (
    MONITORING_TOOL_ID,
    Trace,
    Tracer,
    explanatory_lines,
    tractable_coverage_report,
)
from hypothesis.internal.validation import check_type
from hypothesis.reporting import (
    current_verbosity,
    report,
    verbose_report,
    with_reporter,
)
from hypothesis.statistics import describe_statistics, describe_targets, note_statistics
from hypothesis.strategies._internal.misc import NOTHING
from hypothesis.strategies._internal.strategies import (
    Ex,
    SearchStrategy,
    check_strategy,
)
from hypothesis.strategies._internal.utils import to_jsonable
from hypothesis.vendor.pretty import RepresentationPrinter
from hypothesis.version import __version__

if sys.version_info >= (3, 10):
    from types import EllipsisType as EllipsisType
elif TYPE_CHECKING:
    from builtins import ellipsis as EllipsisType
else:  # pragma: no cover
    EllipsisType = type(Ellipsis)


TestFunc = TypeVar("TestFunc", bound=Callable)


running_under_pytest = False
pytest_shows_exceptiongroups = True
global_force_seed = None
_hypothesis_global_random = None


@attr.s()
class Example:
    args = attr.ib()
    kwargs = attr.ib()
    # Plus two optional arguments for .xfail()
    raises = attr.ib(default=None)
    reason = attr.ib(default=None)


class example:
    """A decorator which ensures a specific example is always tested."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and kwargs:
            raise InvalidArgument(
                "Cannot mix positional and keyword arguments for examples"
            )
        if not (args or kwargs):
            raise InvalidArgument("An example must provide at least one argument")

        self.hypothesis_explicit_examples: list[Example] = []
        self._this_example = Example(tuple(args), kwargs)

    def __call__(self, test: TestFunc) -> TestFunc:
        if not hasattr(test, "hypothesis_explicit_examples"):
            test.hypothesis_explicit_examples = self.hypothesis_explicit_examples  # type: ignore
        test.hypothesis_explicit_examples.append(self._this_example)  # type: ignore
        return test

    def xfail(
        self,
        condition: bool = True,  # noqa: FBT002
        *,
        reason: str = "",
        raises: Union[
            type[BaseException], tuple[type[BaseException], ...]
        ] = BaseException,
    ) -> "example":
        """Mark this example as an expected failure, similarly to
        :obj:`pytest.mark.xfail(strict=True) <pytest.mark.xfail>`.

        Expected-failing examples allow you to check that your test does fail on
        some examples, and therefore build confidence that *passing* tests are
        because your code is working, not because the test is missing something.

        .. code-block:: python

            @example(...).xfail()
            @example(...).xfail(reason="Prices must be non-negative")
            @example(...).xfail(raises=(KeyError, ValueError))
            @example(...).xfail(sys.version_info[:2] >= (3, 12), reason="needs py 3.12")
            @example(...).xfail(condition=sys.platform != "linux", raises=OSError)
            def test(x):
                pass

        .. note::

            Expected-failing examples are handled separately from those generated
            by strategies, so you should usually ensure that there is no overlap.

            .. code-block:: python

                @example(x=1, y=0).xfail(raises=ZeroDivisionError)
                @given(x=st.just(1), y=st.integers())  # Missing `.filter(bool)`!
                def test_fraction(x, y):
                    # This test will try the explicit example and see it fail as
                    # expected, then go on to generate more examples from the
                    # strategy.  If we happen to generate y=0, the test will fail
                    # because only the explicit example is treated as xfailing.
                    x / y
        """
        check_type(bool, condition, "condition")
        check_type(str, reason, "reason")
        if not (
            isinstance(raises, type) and issubclass(raises, BaseException)
        ) and not (
            isinstance(raises, tuple)
            and raises  # () -> expected to fail with no error, which is impossible
            and all(
                isinstance(r, type) and issubclass(r, BaseException) for r in raises
            )
        ):
            raise InvalidArgument(
                f"{raises=} must be an exception type or tuple of exception types"
            )
        if condition:
            self._this_example = attr.evolve(
                self._this_example, raises=raises, reason=reason
            )
        return self

    def via(self, whence: str, /) -> "example":
        """Attach a machine-readable label noting whence this example came.

        The idea is that tools will be able to add ``@example()`` cases for you, e.g.
        to maintain a high-coverage set of explicit examples, but also *remove* them
        if they become redundant - without ever deleting manually-added examples:

        .. code-block:: python

            # You can choose to annotate examples, or not, as you prefer
            @example(...)
            @example(...).via("regression test for issue #42")

            # The `hy-` prefix is reserved for automated tooling
            @example(...).via("hy-failing")
            @example(...).via("hy-coverage")
            @example(...).via("hy-target-$label")
            def test(x):
                pass
        """
        if not isinstance(whence, str):
            raise InvalidArgument(".via() must be passed a string")
        # This is deliberately a no-op at runtime; the tools operate on source code.
        return self


def seed(seed: Hashable) -> Callable[[TestFunc], TestFunc]:
    """seed: Start the test execution from a specific seed.

    May be any hashable object. No exact meaning for seed is provided
    other than that for a fixed seed value Hypothesis will try the same
    actions (insofar as it can given external sources of non-
    determinism. e.g. timing and hash randomization).

    Overrides the derandomize setting, which is designed to enable
    deterministic builds rather than reproducing observed failures.

    """

    def accept(test):
        test._hypothesis_internal_use_seed = seed
        current_settings = getattr(test, "_hypothesis_internal_use_settings", None)
        test._hypothesis_internal_use_settings = Settings(
            current_settings, database=None
        )
        return test

    return accept


def reproduce_failure(version: str, blob: bytes) -> Callable[[TestFunc], TestFunc]:
    """Run the example that corresponds to this data blob in order to reproduce
    a failure.

    A test with this decorator *always* runs only one example and always fails.
    If the provided example does not cause a failure, or is in some way invalid
    for this test, then this will fail with a DidNotReproduce error.

    This decorator is not intended to be a permanent addition to your test
    suite. It's simply some code you can add to ease reproduction of a problem
    in the event that you don't have access to the test database. Because of
    this, *no* compatibility guarantees are made between different versions of
    Hypothesis - its API may change arbitrarily from version to version.
    """

    def accept(test):
        test._hypothesis_internal_use_reproduce_failure = (version, blob)
        return test

    return accept


def encode_failure(buffer):
    buffer = bytes(buffer)
    compressed = zlib.compress(buffer)
    if len(compressed) < len(buffer):
        buffer = b"\1" + compressed
    else:
        buffer = b"\0" + buffer
    return base64.b64encode(buffer)


def decode_failure(blob):
    try:
        buffer = base64.b64decode(blob)
    except Exception:
        raise InvalidArgument(f"Invalid base64 encoded string: {blob!r}") from None
    prefix = buffer[:1]
    if prefix == b"\0":
        return buffer[1:]
    elif prefix == b"\1":
        try:
            return zlib.decompress(buffer[1:])
        except zlib.error as err:
            raise InvalidArgument(
                f"Invalid zlib compression for blob {blob!r}"
            ) from err
    else:
        raise InvalidArgument(
            f"Could not decode blob {blob!r}: Invalid start byte {prefix!r}"
        )


def _invalid(message, *, exc=InvalidArgument, test, given_kwargs):
    @impersonate(test)
    def wrapped_test(*arguments, **kwargs):  # pragma: no cover  # coverage limitation
        raise exc(message)

    wrapped_test.is_hypothesis_test = True
    wrapped_test.hypothesis = HypothesisHandle(
        inner_test=test,
        get_fuzz_target=wrapped_test,
        given_kwargs=given_kwargs,
    )
    return wrapped_test


def is_invalid_test(test, original_sig, given_arguments, given_kwargs):
    """Check the arguments to ``@given`` for basic usage constraints.

    Most errors are not raised immediately; instead we return a dummy test
    function that will raise the appropriate error if it is actually called.
    When the user runs a subset of tests (e.g via ``pytest -k``), errors will
    only be reported for tests that actually ran.
    """
    invalid = partial(_invalid, test=test, given_kwargs=given_kwargs)

    if not (given_arguments or given_kwargs):
        return invalid("given must be called with at least one argument")

    params = list(original_sig.parameters.values())
    pos_params = [p for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
    kwonly_params = [p for p in params if p.kind is p.KEYWORD_ONLY]
    if given_arguments and params != pos_params:
        return invalid(
            "positional arguments to @given are not supported with varargs, "
            "varkeywords, positional-only, or keyword-only arguments"
        )

    if len(given_arguments) > len(pos_params):
        return invalid(
            f"Too many positional arguments for {test.__name__}() were passed to "
            f"@given - expected at most {len(pos_params)} "
            f"arguments, but got {len(given_arguments)} {given_arguments!r}"
        )

    if ... in given_arguments:
        return invalid(
            "... was passed as a positional argument to @given, but may only be "
            "passed as a keyword argument or as the sole argument of @given"
        )

    if given_arguments and given_kwargs:
        return invalid("cannot mix positional and keyword arguments to @given")
    extra_kwargs = [
        k for k in given_kwargs if k not in {p.name for p in pos_params + kwonly_params}
    ]
    if extra_kwargs and (params == [] or params[-1].kind is not params[-1].VAR_KEYWORD):
        arg = extra_kwargs[0]
        return invalid(
            f"{test.__name__}() got an unexpected keyword argument {arg!r}, "
            f"from `{arg}={given_kwargs[arg]!r}` in @given"
        )
    if any(p.default is not p.empty for p in params):
        return invalid("Cannot apply @given to a function with defaults.")

    # This case would raise Unsatisfiable *anyway*, but by detecting it here we can
    # provide a much more helpful error message for people e.g. using the Ghostwriter.
    empty = [
        f"{s!r} (arg {idx})" for idx, s in enumerate(given_arguments) if s is NOTHING
    ] + [f"{name}={s!r}" for name, s in given_kwargs.items() if s is NOTHING]
    if empty:
        strats = "strategies" if len(empty) > 1 else "strategy"
        return invalid(
            f"Cannot generate examples from empty {strats}: " + ", ".join(empty),
            exc=Unsatisfiable,
        )


def execute_explicit_examples(state, wrapped_test, arguments, kwargs, original_sig):
    assert isinstance(state, StateForActualGivenExecution)
    posargs = [
        p.name
        for p in original_sig.parameters.values()
        if p.kind is p.POSITIONAL_OR_KEYWORD
    ]

    for example in reversed(getattr(wrapped_test, "hypothesis_explicit_examples", ())):
        assert isinstance(example, Example)
        # All of this validation is to check that @example() got "the same" arguments
        # as @given, i.e. corresponding to the same parameters, even though they might
        # be any mixture of positional and keyword arguments.
        if example.args:
            assert not example.kwargs
            if any(
                p.kind is p.POSITIONAL_ONLY for p in original_sig.parameters.values()
            ):
                raise InvalidArgument(
                    "Cannot pass positional arguments to @example() when decorating "
                    "a test function which has positional-only parameters."
                )
            if len(example.args) > len(posargs):
                raise InvalidArgument(
                    "example has too many arguments for test. Expected at most "
                    f"{len(posargs)} but got {len(example.args)}"
                )
            example_kwargs = dict(zip(posargs[-len(example.args) :], example.args))
        else:
            example_kwargs = dict(example.kwargs)
        given_kws = ", ".join(
            repr(k) for k in sorted(wrapped_test.hypothesis._given_kwargs)
        )
        example_kws = ", ".join(repr(k) for k in sorted(example_kwargs))
        if given_kws != example_kws:
            raise InvalidArgument(
                f"Inconsistent args: @given() got strategies for {given_kws}, "
                f"but @example() got arguments for {example_kws}"
            ) from None

        # This is certainly true because the example_kwargs exactly match the params
        # reserved by @given(), which are then remove from the function signature.
        assert set(example_kwargs).isdisjoint(kwargs)
        example_kwargs.update(kwargs)

        if Phase.explicit not in state.settings.phases:
            continue

        with local_settings(state.settings):
            fragments_reported = []
            empty_data = ConjectureData.for_buffer(b"")
            try:
                bits = ", ".join(nicerepr(x) for x in arguments) + ", ".join(
                    f"{k}={nicerepr(v)}" for k, v in example_kwargs.items()
                )
                execute_example = partial(
                    state.execute_once,
                    empty_data,
                    is_final=True,
                    print_example=True,
                    example_kwargs=example_kwargs,
                )
                with with_reporter(fragments_reported.append):
                    if example.raises is None:
                        execute_example()
                    else:
                        # @example(...).xfail(...)

                        try:
                            execute_example()
                        except failure_exceptions_to_catch() as err:
                            if not isinstance(err, example.raises):
                                raise
                            # Save a string form of this example; we'll warn if it's
                            # ever generated by the strategy (which can't be xfailed)
                            state.xfail_example_reprs.add(
                                repr_call(state.test, arguments, example_kwargs)
                            )
                        except example.raises as err:
                            # We'd usually check this as early as possible, but it's
                            # possible for failure_exceptions_to_catch() to grow when
                            # e.g. pytest is imported between import- and test-time.
                            raise InvalidArgument(
                                f"@example({bits}) raised an expected {err!r}, "
                                "but Hypothesis does not treat this as a test failure"
                            ) from err
                        else:
                            # Unexpectedly passing; always raise an error in this case.
                            reason = f" because {example.reason}" * bool(example.reason)
                            if example.raises is BaseException:
                                name = "exception"  # special-case no raises= arg
                            elif not isinstance(example.raises, tuple):
                                name = example.raises.__name__
                            elif len(example.raises) == 1:
                                name = example.raises[0].__name__
                            else:
                                name = (
                                    ", ".join(ex.__name__ for ex in example.raises[:-1])
                                    + f", or {example.raises[-1].__name__}"
                                )
                            vowel = name.upper()[0] in "AEIOU"
                            raise AssertionError(
                                f"Expected a{'n' * vowel} {name} from @example({bits})"
                                f"{reason}, but no exception was raised."
                            )
            except UnsatisfiedAssumption:
                # Odd though it seems, we deliberately support explicit examples that
                # are then rejected by a call to `assume()`.  As well as iterative
                # development, this is rather useful to replay Hypothesis' part of
                # a saved failure when other arguments are supplied by e.g. pytest.
                # See https://github.com/HypothesisWorks/hypothesis/issues/2125
                with contextlib.suppress(StopTest):
                    empty_data.conclude_test(Status.INVALID)
            except BaseException as err:
                # In order to support reporting of multiple failing examples, we yield
                # each of the (report text, error) pairs we find back to the top-level
                # runner.  This also ensures that user-facing stack traces have as few
                # frames of Hypothesis internals as possible.
                err = err.with_traceback(get_trimmed_traceback())

                # One user error - whether misunderstanding or typo - we've seen a few
                # times is to pass strategies to @example() where values are expected.
                # Checking is easy, and false-positives not much of a problem, so:
                if isinstance(err, failure_exceptions_to_catch()) and any(
                    isinstance(arg, SearchStrategy)
                    for arg in example.args + tuple(example.kwargs.values())
                ):
                    new = HypothesisWarning(
                        "The @example() decorator expects to be passed values, but "
                        "you passed strategies instead.  See https://hypothesis."
                        "readthedocs.io/en/latest/reproducing.html for details."
                    )
                    new.__cause__ = err
                    err = new

                with contextlib.suppress(StopTest):
                    empty_data.conclude_test(Status.INVALID)
                yield (fragments_reported, err)
                if (
                    state.settings.report_multiple_bugs
                    and pytest_shows_exceptiongroups
                    and isinstance(err, failure_exceptions_to_catch())
                    and not isinstance(err, skip_exceptions_to_reraise())
                ):
                    continue
                break
            finally:
                if fragments_reported:
                    assert fragments_reported[0].startswith("Falsifying example")
                    fragments_reported[0] = fragments_reported[0].replace(
                        "Falsifying example", "Falsifying explicit example", 1
                    )

                tc = make_testcase(
                    start_timestamp=state._start_timestamp,
                    test_name_or_nodeid=state.test_identifier,
                    data=empty_data,
                    how_generated="explicit example",
                    string_repr=state._string_repr,
                    timing=state._timing_features,
                )
                deliver_json_blob(tc)

            if fragments_reported:
                verbose_report(fragments_reported[0].replace("Falsifying", "Trying", 1))
                for f in fragments_reported[1:]:
                    verbose_report(f)


def get_random_for_wrapped_test(test, wrapped_test):
    settings = wrapped_test._hypothesis_internal_use_settings
    wrapped_test._hypothesis_internal_use_generated_seed = None

    if wrapped_test._hypothesis_internal_use_seed is not None:
        return Random(wrapped_test._hypothesis_internal_use_seed)
    elif settings.derandomize:
        return Random(int_from_bytes(function_digest(test)))
    elif global_force_seed is not None:
        return Random(global_force_seed)
    else:
        global _hypothesis_global_random
        if _hypothesis_global_random is None:  # pragma: no cover
            _hypothesis_global_random = Random()
        seed = _hypothesis_global_random.getrandbits(128)
        wrapped_test._hypothesis_internal_use_generated_seed = seed
        return Random(seed)


@attr.s
class Stuff:
    selfy: Any = attr.ib(default=None)
    args: tuple = attr.ib(factory=tuple)
    kwargs: dict = attr.ib(factory=dict)
    given_kwargs: dict = attr.ib(factory=dict)


def process_arguments_to_given(wrapped_test, arguments, kwargs, given_kwargs, params):
    selfy = None
    arguments, kwargs = convert_positional_arguments(wrapped_test, arguments, kwargs)

    # If the test function is a method of some kind, the bound object
    # will be the first named argument if there are any, otherwise the
    # first vararg (if any).
    posargs = [p.name for p in params.values() if p.kind is p.POSITIONAL_OR_KEYWORD]
    if posargs:
        selfy = kwargs.get(posargs[0])
    elif arguments:
        selfy = arguments[0]

    # Ensure that we don't mistake mocks for self here.
    # This can cause the mock to be used as the test runner.
    if is_mock(selfy):
        selfy = None

    arguments = tuple(arguments)

    with ensure_free_stackframes():
        for k, s in given_kwargs.items():
            check_strategy(s, name=k)
            s.validate()

    stuff = Stuff(selfy=selfy, args=arguments, kwargs=kwargs, given_kwargs=given_kwargs)

    return arguments, kwargs, stuff


def skip_exceptions_to_reraise():
    """Return a tuple of exceptions meaning 'skip this test', to re-raise.

    This is intended to cover most common test runners; if you would
    like another to be added please open an issue or pull request adding
    it to this function and to tests/cover/test_lazy_import.py
    """
    # This is a set because nose may simply re-export unittest.SkipTest
    exceptions = set()
    # We use this sys.modules trick to avoid importing libraries -
    # you can't be an instance of a type from an unimported module!
    # This is fast enough that we don't need to cache the result,
    # and more importantly it avoids possible side-effects :-)
    if "unittest" in sys.modules:
        exceptions.add(sys.modules["unittest"].SkipTest)
    if "unittest2" in sys.modules:
        exceptions.add(sys.modules["unittest2"].SkipTest)
    if "nose" in sys.modules:
        exceptions.add(sys.modules["nose"].SkipTest)
    if "_pytest" in sys.modules:
        exceptions.add(sys.modules["_pytest"].outcomes.Skipped)
    return tuple(sorted(exceptions, key=str))


def failure_exceptions_to_catch():
    """Return a tuple of exceptions meaning 'this test has failed', to catch.

    This is intended to cover most common test runners; if you would
    like another to be added please open an issue or pull request.
    """
    # While SystemExit and GeneratorExit are instances of BaseException, we also
    # expect them to be deterministic - unlike KeyboardInterrupt - and so we treat
    # them as standard exceptions, check for flakiness, etc.
    # See https://github.com/HypothesisWorks/hypothesis/issues/2223 for details.
    exceptions = [Exception, SystemExit, GeneratorExit]
    if "_pytest" in sys.modules:
        exceptions.append(sys.modules["_pytest"].outcomes.Failed)
    return tuple(exceptions)


def new_given_signature(original_sig, given_kwargs):
    """Make an updated signature for the wrapped test."""
    return original_sig.replace(
        parameters=[
            p
            for p in original_sig.parameters.values()
            if not (
                p.name in given_kwargs
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            )
        ],
        return_annotation=None,
    )


def default_executor(data, function):
    return function(data)


def get_executor(runner):
    try:
        execute_example = runner.execute_example
    except AttributeError:
        pass
    else:
        return lambda data, function: execute_example(partial(function, data))

    if hasattr(runner, "setup_example") or hasattr(runner, "teardown_example"):
        setup = getattr(runner, "setup_example", None) or (lambda: None)
        teardown = getattr(runner, "teardown_example", None) or (lambda ex: None)

        def execute(data, function):
            token = None
            try:
                token = setup()
                return function(data)
            finally:
                teardown(token)

        return execute

    return default_executor


class StateForActualGivenExecution:
    def __init__(self, stuff, test, settings, random, wrapped_test):
        self.test_runner = get_executor(stuff.selfy)
        self.stuff = stuff
        self.settings = settings
        self.last_exception = None
        self.falsifying_examples = ()
        self.random = random
        self.ever_executed = False

        self.is_find = getattr(wrapped_test, "_hypothesis_internal_is_find", False)
        self.wrapped_test = wrapped_test
        self.xfail_example_reprs = set()

        self.test = test

        self.print_given_args = getattr(
            wrapped_test, "_hypothesis_internal_print_given_args", True
        )

        self.files_to_propagate = set()
        self.failed_normally = False
        self.failed_due_to_deadline = False

        self.explain_traces = defaultdict(set)
        self._start_timestamp = time.time()
        self._string_repr = ""
        self._timing_features = {}

    @property
    def test_identifier(self):
        return getattr(
            current_pytest_item.value, "nodeid", None
        ) or get_pretty_function_description(self.wrapped_test)

    def _should_trace(self):
        _trace_obs = TESTCASE_CALLBACKS and OBSERVABILITY_COLLECT_COVERAGE
        _trace_failure = (
            self.failed_normally
            and not self.failed_due_to_deadline
            and {Phase.shrink, Phase.explain}.issubset(self.settings.phases)
        )
        return _trace_obs or _trace_failure

    def execute_once(
        self,
        data,
        *,
        print_example=False,
        is_final=False,
        expected_failure=None,
        example_kwargs=None,
    ):
        """Run the test function once, using ``data`` as input.

        If the test raises an exception, it will propagate through to the
        caller of this method. Depending on its type, this could represent
        an ordinary test failure, or a fatal error, or a control exception.

        If this method returns normally, the test might have passed, or
        it might have placed ``data`` in an unsuccessful state and then
        swallowed the corresponding control exception.
        """

        self.ever_executed = True
        data.is_find = self.is_find

        self._string_repr = ""
        text_repr = None
        if self.settings.deadline is None and not TESTCASE_CALLBACKS:

            @proxies(self.test)
            def test(*args, **kwargs):
                with ensure_free_stackframes():
                    return self.test(*args, **kwargs)

        else:

            @proxies(self.test)
            def test(*args, **kwargs):
                arg_drawtime = math.fsum(data.draw_times.values())
                arg_stateful = math.fsum(data._stateful_run_times.values())
                arg_gctime = gc_cumulative_time()
                start = time.perf_counter()
                try:
                    with ensure_free_stackframes():
                        result = self.test(*args, **kwargs)
                finally:
                    finish = time.perf_counter()
                    in_drawtime = math.fsum(data.draw_times.values()) - arg_drawtime
                    in_stateful = (
                        math.fsum(data._stateful_run_times.values()) - arg_stateful
                    )
                    in_gctime = gc_cumulative_time() - arg_gctime
                    runtime = finish - start - in_drawtime - in_stateful - in_gctime
                    self._timing_features = {
                        "execute:test": runtime,
                        "overall:gc": in_gctime,
                        **data.draw_times,
                        **data._stateful_run_times,
                    }

                if (current_deadline := self.settings.deadline) is not None:
                    if not is_final:
                        current_deadline = (current_deadline // 4) * 5
                    if runtime >= current_deadline.total_seconds():
                        raise DeadlineExceeded(
                            datetime.timedelta(seconds=runtime), self.settings.deadline
                        )
                return result

        def run(data):
            # Set up dynamic context needed by a single test run.
            if self.stuff.selfy is not None:
                data.hypothesis_runner = self.stuff.selfy
            # Generate all arguments to the test function.
            args = self.stuff.args
            kwargs = dict(self.stuff.kwargs)
            if example_kwargs is None:
                kw, argslices = context.prep_args_kwargs_from_strategies(
                    self.stuff.given_kwargs
                )
            else:
                kw = example_kwargs
                argslices = {}
            kwargs.update(kw)
            if expected_failure is not None:
                nonlocal text_repr
                text_repr = repr_call(test, args, kwargs)
                if text_repr in self.xfail_example_reprs:
                    warnings.warn(
                        f"We generated {text_repr}, which seems identical "
                        "to one of your `@example(...).xfail()` cases.  "
                        "Revise the strategy to avoid this overlap?",
                        HypothesisWarning,
                        # Checked in test_generating_xfailed_examples_warns!
                        stacklevel=6,
                    )

            if print_example or current_verbosity() >= Verbosity.verbose:
                printer = RepresentationPrinter(context=context)
                if print_example:
                    printer.text("Falsifying example:")
                else:
                    printer.text("Trying example:")

                if self.print_given_args:
                    printer.text(" ")
                    printer.repr_call(
                        test.__name__,
                        args,
                        kwargs,
                        force_split=True,
                        arg_slices=argslices,
                        leading_comment=(
                            "# " + context.data.slice_comments[(0, 0)]
                            if (0, 0) in context.data.slice_comments
                            else None
                        ),
                    )
                report(printer.getvalue())

            if TESTCASE_CALLBACKS:
                printer = RepresentationPrinter(context=context)
                printer.repr_call(
                    test.__name__,
                    args,
                    kwargs,
                    force_split=True,
                    arg_slices=argslices,
                    leading_comment=(
                        "# " + context.data.slice_comments[(0, 0)]
                        if (0, 0) in context.data.slice_comments
                        else None
                    ),
                )
                self._string_repr = printer.getvalue()
                data._observability_arguments = {
                    **dict(enumerate(map(to_jsonable, args))),
                    **{k: to_jsonable(v) for k, v in kwargs.items()},
                }

            try:
                return test(*args, **kwargs)
            except TypeError as e:
                # If we sampled from a sequence of strategies, AND failed with a
                # TypeError, *AND that exception mentions SearchStrategy*, add a note:
                if "SearchStrategy" in str(e) and hasattr(
                    data, "_sampled_from_all_strategies_elements_message"
                ):
                    msg, format_arg = data._sampled_from_all_strategies_elements_message
                    add_note(e, msg.format(format_arg))
                raise
            finally:
                if parts := getattr(data, "_stateful_repr_parts", None):
                    self._string_repr = "\n".join(parts)

        # self.test_runner can include the execute_example method, or setup/teardown
        # _example, so it's important to get the PRNG and build context in place first.
        with local_settings(self.settings):
            with deterministic_PRNG():
                with BuildContext(data, is_final=is_final) as context:
                    # providers may throw in per_case_context_fn, and we'd like
                    # `result` to still be set in these cases.
                    result = None
                    with data.provider.per_test_case_context_manager():
                        # Run the test function once, via the executor hook.
                        # In most cases this will delegate straight to `run(data)`.
                        result = self.test_runner(data, run)

        # If a failure was expected, it should have been raised already, so
        # instead raise an appropriate diagnostic error.
        if expected_failure is not None:
            exception, traceback = expected_failure
            if isinstance(exception, DeadlineExceeded) and (
                runtime_secs := math.fsum(
                    v
                    for k, v in self._timing_features.items()
                    if k.startswith("execute:")
                )
            ):
                report(
                    "Unreliable test timings! On an initial run, this "
                    "test took %.2fms, which exceeded the deadline of "
                    "%.2fms, but on a subsequent run it took %.2f ms, "
                    "which did not. If you expect this sort of "
                    "variability in your test timings, consider turning "
                    "deadlines off for this test by setting deadline=None."
                    % (
                        exception.runtime.total_seconds() * 1000,
                        self.settings.deadline.total_seconds() * 1000,
                        runtime_secs * 1000,
                    )
                )
            else:
                report("Failed to reproduce exception. Expected: \n" + traceback)
            raise FlakyFailure(
                f"Hypothesis {text_repr} produces unreliable results: "
                "Falsified on the first call but did not on a subsequent one",
                [exception],
            )
        return result

    def _flaky_replay_to_failure(
        self, err: FlakyReplay, context: BaseException
    ) -> FlakyFailure:
        interesting_examples = [
            self._runner.interesting_examples[io]
            for io in err._interesting_origins
            if io
        ]
        exceptions = [
            ie.extra_information._expected_exception for ie in interesting_examples
        ]
        exceptions.append(context)  # the offending assume (or whatever)
        return FlakyFailure(err.reason, exceptions)

    def _execute_once_for_engine(self, data: ConjectureData) -> None:
        """Wrapper around ``execute_once`` that intercepts test failure
        exceptions and single-test control exceptions, and turns them into
        appropriate method calls to `data` instead.

        This allows the engine to assume that any exception other than
        ``StopTest`` must be a fatal error, and should stop the entire engine.
        """
        trace: Trace = set()
        try:
            if self._should_trace() and Tracer.can_trace():  # pragma: no cover
                # This is in fact covered by our *non-coverage* tests, but due to the
                # settrace() contention *not* by our coverage tests.  Ah well.
                with Tracer() as tracer:
                    try:
                        result = self.execute_once(data)
                        if data.status == Status.VALID:
                            self.explain_traces[None].add(frozenset(tracer.branches))
                    finally:
                        trace = tracer.branches
            else:
                result = self.execute_once(data)
            if result is not None:
                fail_health_check(
                    self.settings,
                    "Tests run under @given should return None, but "
                    f"{self.test.__name__} returned {result!r} instead.",
                    HealthCheck.return_value,
                )
        except UnsatisfiedAssumption as e:
            # An "assume" check failed, so instead we inform the engine that
            # this test run was invalid.
            try:
                data.mark_invalid(e.reason)
            except FlakyReplay as err:
                # This was unexpected, meaning that the assume was flaky.
                # Report it as such.
                raise self._flaky_replay_to_failure(err, e) from None
        except (StopTest, BackendCannotProceed):
            # The engine knows how to handle this control exception, so it's
            # OK to re-raise it.
            raise
        except (
            FailedHealthCheck,
            *skip_exceptions_to_reraise(),
        ):
            # These are fatal errors or control exceptions that should stop the
            # engine, so we re-raise them.
            raise
        except failure_exceptions_to_catch() as e:
            # If an unhandled (i.e., non-Hypothesis) error was raised by
            # Hypothesis-internal code, re-raise it as a fatal error instead
            # of treating it as a test failure.
            filepath = traceback.extract_tb(e.__traceback__)[-1][0]
            if is_hypothesis_file(filepath) and not isinstance(e, HypothesisException):
                raise

            if data.frozen:
                # This can happen if an error occurred in a finally
                # block somewhere, suppressing our original StopTest.
                # We raise a new one here to resume normal operation.
                raise StopTest(data.testcounter) from e
            else:
                # The test failed by raising an exception, so we inform the
                # engine that this test run was interesting. This is the normal
                # path for test runs that fail.
                tb = get_trimmed_traceback()
                info = data.extra_information
                info._expected_traceback = format_exception(e, tb)  # type: ignore
                info._expected_exception = e  # type: ignore
                verbose_report(info._expected_traceback)  # type: ignore

                self.failed_normally = True

                interesting_origin = InterestingOrigin.from_exception(e)
                if trace:  # pragma: no cover
                    # Trace collection is explicitly disabled under coverage.
                    self.explain_traces[interesting_origin].add(frozenset(trace))
                if interesting_origin[0] == DeadlineExceeded:
                    self.failed_due_to_deadline = True
                    self.explain_traces.clear()
                data.mark_interesting(interesting_origin)  # type: ignore  # mypy bug?
        finally:
            # Conditional here so we can save some time constructing the payload; in
            # other cases (without coverage) it's cheap enough to do that regardless.
            if TESTCASE_CALLBACKS:
                if runner := getattr(self, "_runner", None):
                    phase = runner._current_phase
                else:  # pragma: no cover  # in case of messing with internals
                    if self.failed_normally or self.failed_due_to_deadline:
                        phase = "shrink"
                    else:
                        phase = "unknown"
                backend_desc = f", using backend={self.settings.backend!r}" * (
                    self.settings.backend != "hypothesis"
                    and not getattr(runner, "_switch_to_hypothesis_provider", False)
                )
                try:
                    data._observability_args = data.provider.realize(
                        data._observability_args
                    )
                    self._string_repr = data.provider.realize(self._string_repr)
                except BackendCannotProceed:
                    data._observability_args = {}
                    self._string_repr = "<backend failed to realize symbolic arguments>"

                tc = make_testcase(
                    start_timestamp=self._start_timestamp,
                    test_name_or_nodeid=self.test_identifier,
                    data=data,
                    how_generated=f"during {phase} phase{backend_desc}",
                    string_repr=self._string_repr,
                    arguments=data._observability_args,
                    timing=self._timing_features,
                    coverage=tractable_coverage_report(trace) or None,
                    phase=phase,
                    backend_metadata=data.provider.observe_test_case(),
                )
                deliver_json_blob(tc)
                for msg in data.provider.observe_information_messages(
                    lifetime="test_case"
                ):
                    self._deliver_information_message(**msg)
            self._timing_features = {}

    def _deliver_information_message(
        self, *, type: str, title: str, content: Union[str, dict]
    ) -> None:
        deliver_json_blob(
            {
                "type": type,
                "run_start": self._start_timestamp,
                "property": self.test_identifier,
                "title": title,
                "content": content,
            }
        )

    def run_engine(self):
        """Run the test function many times, on database input and generated
        input, using the Conjecture engine.
        """
        # Tell pytest to omit the body of this function from tracebacks
        __tracebackhide__ = True
        try:
            database_key = self.wrapped_test._hypothesis_internal_database_key
        except AttributeError:
            if global_force_seed is None:
                database_key = function_digest(self.test)
            else:
                database_key = None

        runner = self._runner = ConjectureRunner(
            self._execute_once_for_engine,
            settings=self.settings,
            random=self.random,
            database_key=database_key,
        )
        # Use the Conjecture engine to run the test function many times
        # on different inputs.
        runner.run()
        note_statistics(runner.statistics)
        if TESTCASE_CALLBACKS:
            self._deliver_information_message(
                type="info",
                title="Hypothesis Statistics",
                content=describe_statistics(runner.statistics),
            )
            for msg in (
                p if isinstance(p := runner.provider, PrimitiveProvider) else p(None)
            ).observe_information_messages(lifetime="test_function"):
                self._deliver_information_message(**msg)

        if runner.call_count == 0:
            return
        if runner.interesting_examples:
            self.falsifying_examples = sorted(
                runner.interesting_examples.values(),
                key=lambda d: sort_key(d.buffer),
                reverse=True,
            )
        else:
            if runner.valid_examples == 0:
                rep = get_pretty_function_description(self.test)
                raise Unsatisfiable(f"Unable to satisfy assumptions of {rep}")

        # If we have not traced executions, warn about that now (but only when
        # we'd expect to do so reliably, i.e. on CPython>=3.12)
        if (
            sys.version_info[:2] >= (3, 12)
            and not PYPY
            and self._should_trace()
            and not Tracer.can_trace()
        ):  # pragma: no cover
            # actually covered by our tests, but only on >= 3.12
            warnings.warn(
                "avoiding tracing test function because tool id "
                f"{MONITORING_TOOL_ID} is already taken by tool "
                f"{sys.monitoring.get_tool(MONITORING_TOOL_ID)}.",
                HypothesisWarning,
                stacklevel=3,
            )

        if not self.falsifying_examples:
            return
        elif not (self.settings.report_multiple_bugs and pytest_shows_exceptiongroups):
            # Pretend that we only found one failure, by discarding the others.
            del self.falsifying_examples[:-1]

        # The engine found one or more failures, so we need to reproduce and
        # report them.

        errors_to_report = []

        report_lines = describe_targets(runner.best_observed_targets)
        if report_lines:
            report_lines.append("")

        explanations = explanatory_lines(self.explain_traces, self.settings)
        for falsifying_example in self.falsifying_examples:
            info = falsifying_example.extra_information
            fragments = []

            ran_example = runner.new_conjecture_data_for_buffer(
                falsifying_example.buffer
            )
            ran_example.slice_comments = falsifying_example.slice_comments
            tb = None
            origin = None
            assert info._expected_exception is not None
            try:
                with with_reporter(fragments.append):
                    self.execute_once(
                        ran_example,
                        print_example=not self.is_find,
                        is_final=True,
                        expected_failure=(
                            info._expected_exception,
                            info._expected_traceback,
                        ),
                    )
            except StopTest as e:
                # Link the expected exception from the first run. Not sure
                # how to access the current exception, if it failed
                # differently on this run. In fact, in the only known
                # reproducer, the StopTest is caused by OVERRUN before the
                # test is even executed. Possibly because all initial examples
                # failed until the final non-traced replay, and something was
                # exhausted? Possibly a FIXME, but sufficiently weird to
                # ignore for now.
                err = FlakyFailure(
                    "Inconsistent results: An example failed on the "
                    "first run but now succeeds (or fails with another "
                    "error, or is for some reason not runnable).",
                    [info._expected_exception or e],  # (note: e is a BaseException)
                )
                errors_to_report.append((fragments, err))
            except UnsatisfiedAssumption as e:  # pragma: no cover  # ironically flaky
                err = FlakyFailure(
                    "Unreliable assumption: An example which satisfied "
                    "assumptions on the first run now fails it.",
                    [e],
                )
                errors_to_report.append((fragments, err))
            except BaseException as e:
                # If we have anything for explain-mode, this is the time to report.
                fragments.extend(explanations[falsifying_example.interesting_origin])
                errors_to_report.append(
                    (fragments, e.with_traceback(get_trimmed_traceback()))
                )
                tb = format_exception(e, get_trimmed_traceback(e))
                origin = InterestingOrigin.from_exception(e)
            else:
                # execute_once() will always raise either the expected error, or Flaky.
                raise NotImplementedError("This should be unreachable")
            finally:
                # log our observability line for the final failing example
                tc = {
                    "type": "test_case",
                    "run_start": self._start_timestamp,
                    "property": self.test_identifier,
                    "status": "passed" if sys.exc_info()[0] else "failed",
                    "status_reason": str(origin or "unexpected/flaky pass"),
                    "representation": self._string_repr,
                    "arguments": ran_example._observability_args,
                    "how_generated": "minimal failing example",
                    "features": {
                        **{
                            f"target:{k}".strip(":"): v
                            for k, v in ran_example.target_observations.items()
                        },
                        **ran_example.events,
                    },
                    "timing": self._timing_features,
                    "coverage": None,  # Not recorded when we're replaying the MFE
                    "metadata": {
                        "traceback": tb,
                        "predicates": ran_example._observability_predicates,
                        **_system_metadata(),
                    },
                }
                deliver_json_blob(tc)
                # Whether or not replay actually raised the exception again, we want
                # to print the reproduce_failure decorator for the failing example.
                if self.settings.print_blob:
                    fragments.append(
                        "\nYou can reproduce this example by temporarily adding "
                        "@reproduce_failure(%r, %r) as a decorator on your test case"
                        % (__version__, encode_failure(falsifying_example.buffer))
                    )
                # Mostly useful for ``find`` and ensuring that objects that
                # hold on to a reference to ``data`` know that it's now been
                # finished and they can't draw more data from it.
                ran_example.freeze()  # pragma: no branch
                # No branch is possible here because we never have an active exception.
        _raise_to_user(
            errors_to_report,
            self.settings,
            report_lines,
            verified_by=runner._verified_by,
        )


def _raise_to_user(
    errors_to_report, settings, target_lines, trailer="", verified_by=None
):
    """Helper function for attaching notes and grouping multiple errors."""
    failing_prefix = "Falsifying example: "
    ls = []
    for fragments, err in errors_to_report:
        for note in fragments:
            add_note(err, note)
            if note.startswith(failing_prefix):
                ls.append(note.removeprefix(failing_prefix))
    if current_pytest_item.value:
        current_pytest_item.value._hypothesis_failing_examples = ls

    if len(errors_to_report) == 1:
        _, the_error_hypothesis_found = errors_to_report[0]
    else:
        assert errors_to_report
        the_error_hypothesis_found = BaseExceptionGroup(
            f"Hypothesis found {len(errors_to_report)} distinct failures{trailer}.",
            [e for _, e in errors_to_report],
        )

    if settings.verbosity >= Verbosity.normal:
        for line in target_lines:
            add_note(the_error_hypothesis_found, line)

    if verified_by:
        msg = f"backend={verified_by!r} claimed to verify this test passes - please send them a bug report!"
        add_note(err, msg)

    raise the_error_hypothesis_found


@contextlib.contextmanager
def fake_subTest(self, msg=None, **__):
    """Monkeypatch for `unittest.TestCase.subTest` during `@given`.

    If we don't patch this out, each failing example is reported as a
    separate failing test by the unittest test runner, which is
    obviously incorrect. We therefore replace it for the duration with
    this version.
    """
    warnings.warn(
        "subTest per-example reporting interacts badly with Hypothesis "
        "trying hundreds of examples, so we disable it for the duration of "
        "any test that uses `@given`.",
        HypothesisWarning,
        stacklevel=2,
    )
    yield


@attr.s()
class HypothesisHandle:
    """This object is provided as the .hypothesis attribute on @given tests.

    Downstream users can reassign its attributes to insert custom logic into
    the execution of each case, for example by converting an async into a
    sync function.

    This must be an attribute of an attribute, because reassignment of a
    first-level attribute would not be visible to Hypothesis if the function
    had been decorated before the assignment.

    See https://github.com/HypothesisWorks/hypothesis/issues/1257 for more
    information.
    """

    inner_test = attr.ib()
    _get_fuzz_target = attr.ib()
    _given_kwargs = attr.ib()

    @property
    def fuzz_one_input(
        self,
    ) -> Callable[[Union[bytes, bytearray, memoryview, BinaryIO]], Optional[bytes]]:
        """Run the test as a fuzz target, driven with the `buffer` of bytes.

        Returns None if buffer invalid for the strategy, canonical pruned
        bytes if the buffer was valid, and leaves raised exceptions alone.
        """
        # Note: most users, if they care about fuzzer performance, will access the
        # property and assign it to a local variable to move the attribute lookup
        # outside their fuzzing loop / before the fork point.  We cache it anyway,
        # so that naive or unusual use-cases get the best possible performance too.
        try:
            return self.__cached_target  # type: ignore
        except AttributeError:
            self.__cached_target = self._get_fuzz_target()
            return self.__cached_target


@overload
def given(
    _: EllipsisType, /
) -> Callable[
    [Callable[..., Optional[Coroutine[Any, Any, None]]]], Callable[[], None]
]:  # pragma: no cover
    ...


@overload
def given(
    *_given_arguments: SearchStrategy[Any],
) -> Callable[
    [Callable[..., Optional[Coroutine[Any, Any, None]]]], Callable[..., None]
]:  # pragma: no cover
    ...


@overload
def given(
    **_given_kwargs: Union[SearchStrategy[Any], EllipsisType],
) -> Callable[
    [Callable[..., Optional[Coroutine[Any, Any, None]]]], Callable[..., None]
]:  # pragma: no cover
    ...


def given(
    *_given_arguments: Union[SearchStrategy[Any], EllipsisType],
    **_given_kwargs: Union[SearchStrategy[Any], EllipsisType],
) -> Callable[
    [Callable[..., Optional[Coroutine[Any, Any, None]]]], Callable[..., None]
]:
    """A decorator for turning a test function that accepts arguments into a
    randomized test.

    This is the main entry point to Hypothesis.
    """

    def run_test_as_given(test):
        if inspect.isclass(test):
            # Provide a meaningful error to users, instead of exceptions from
            # internals that assume we're dealing with a function.
            raise InvalidArgument("@given cannot be applied to a class.")
        given_arguments = tuple(_given_arguments)
        given_kwargs = dict(_given_kwargs)

        original_sig = get_signature(test)
        if given_arguments == (Ellipsis,) and not given_kwargs:
            # user indicated that they want to infer all arguments
            given_kwargs = {
                p.name: Ellipsis
                for p in original_sig.parameters.values()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            }
            given_arguments = ()

        check_invalid = is_invalid_test(
            test, original_sig, given_arguments, given_kwargs
        )

        # If the argument check found problems, return a dummy test function
        # that will raise an error if it is actually called.
        if check_invalid is not None:
            return check_invalid

        # Because the argument check succeeded, we can convert @given's
        # positional arguments into keyword arguments for simplicity.
        if given_arguments:
            assert not given_kwargs
            posargs = [
                p.name
                for p in original_sig.parameters.values()
                if p.kind is p.POSITIONAL_OR_KEYWORD
            ]
            given_kwargs = dict(list(zip(posargs[::-1], given_arguments[::-1]))[::-1])
        # These have been converted, so delete them to prevent accidental use.
        del given_arguments

        new_signature = new_given_signature(original_sig, given_kwargs)

        # Use type information to convert "infer" arguments into appropriate strategies.
        if ... in given_kwargs.values():
            hints = get_type_hints(test)
        for name in [name for name, value in given_kwargs.items() if value is ...]:
            if name not in hints:
                return _invalid(
                    f"passed {name}=... for {test.__name__}, but {name} has "
                    "no type annotation",
                    test=test,
                    given_kwargs=given_kwargs,
                )
            given_kwargs[name] = st.from_type(hints[name])

        prev_self = Unset = object()

        @impersonate(test)
        @define_function_signature(test.__name__, test.__doc__, new_signature)
        def wrapped_test(*arguments, **kwargs):
            # Tell pytest to omit the body of this function from tracebacks
            __tracebackhide__ = True

            test = wrapped_test.hypothesis.inner_test

            if getattr(test, "is_hypothesis_test", False):
                raise InvalidArgument(
                    f"You have applied @given to the test {test.__name__} more than "
                    "once, which wraps the test several times and is extremely slow. "
                    "A similar effect can be gained by combining the arguments "
                    "of the two calls to given. For example, instead of "
                    "@given(booleans()) @given(integers()), you could write "
                    "@given(booleans(), integers())"
                )

            settings = wrapped_test._hypothesis_internal_use_settings

            random = get_random_for_wrapped_test(test, wrapped_test)

            arguments, kwargs, stuff = process_arguments_to_given(
                wrapped_test, arguments, kwargs, given_kwargs, new_signature.parameters
            )

            if (
                inspect.iscoroutinefunction(test)
                and get_executor(stuff.selfy) is default_executor
            ):
                # See https://github.com/HypothesisWorks/hypothesis/issues/3054
                # If our custom executor doesn't handle coroutines, or we return an
                # awaitable from a non-async-def function, we just rely on the
                # return_value health check.  This catches most user errors though.
                raise InvalidArgument(
                    "Hypothesis doesn't know how to run async test functions like "
                    f"{test.__name__}.  You'll need to write a custom executor, "
                    "or use a library like pytest-asyncio or pytest-trio which can "
                    "handle the translation for you.\n    See https://hypothesis."
                    "readthedocs.io/en/latest/details.html#custom-function-execution"
                )

            runner = stuff.selfy
            if isinstance(stuff.selfy, TestCase) and test.__name__ in dir(TestCase):
                msg = (
                    f"You have applied @given to the method {test.__name__}, which is "
                    "used by the unittest runner but is not itself a test."
                    "  This is not useful in any way."
                )
                fail_health_check(settings, msg, HealthCheck.not_a_test_method)
            if bad_django_TestCase(runner):  # pragma: no cover
                # Covered by the Django tests, but not the pytest coverage task
                raise InvalidArgument(
                    "You have applied @given to a method on "
                    f"{type(runner).__qualname__}, but this "
                    "class does not inherit from the supported versions in "
                    "`hypothesis.extra.django`.  Use the Hypothesis variants "
                    "to ensure that each example is run in a separate "
                    "database transaction."
                )
            if settings.database is not None:
                nonlocal prev_self
                # Check selfy really is self (not e.g. a mock) before we health-check
                cur_self = (
                    stuff.selfy
                    if getattr(type(stuff.selfy), test.__name__, None) is wrapped_test
                    else None
                )
                if prev_self is Unset:
                    prev_self = cur_self
                elif cur_self is not prev_self:
                    msg = (
                        f"The method {test.__qualname__} was called from multiple "
                        "different executors. This may lead to flaky tests and "
                        "nonreproducible errors when replaying from database."
                    )
                    fail_health_check(settings, msg, HealthCheck.differing_executors)

            state = StateForActualGivenExecution(
                stuff, test, settings, random, wrapped_test
            )

            reproduce_failure = wrapped_test._hypothesis_internal_use_reproduce_failure

            # If there was a @reproduce_failure decorator, use it to reproduce
            # the error (or complain that we couldn't). Either way, this will
            # always raise some kind of error.
            if reproduce_failure is not None:
                expected_version, failure = reproduce_failure
                if expected_version != __version__:
                    raise InvalidArgument(
                        "Attempting to reproduce a failure from a different "
                        "version of Hypothesis. This failure is from %s, but "
                        "you are currently running %r. Please change your "
                        "Hypothesis version to a matching one."
                        % (expected_version, __version__)
                    )
                try:
                    state.execute_once(
                        ConjectureData.for_buffer(decode_failure(failure)),
                        print_example=True,
                        is_final=True,
                    )
                    raise DidNotReproduce(
                        "Expected the test to raise an error, but it "
                        "completed successfully."
                    )
                except StopTest:
                    raise DidNotReproduce(
                        "The shape of the test data has changed in some way "
                        "from where this blob was defined. Are you sure "
                        "you're running the same test?"
                    ) from None
                except UnsatisfiedAssumption:
                    raise DidNotReproduce(
                        "The test data failed to satisfy an assumption in the "
                        "test. Have you added it since this blob was generated?"
                    ) from None

            # There was no @reproduce_failure, so start by running any explicit
            # examples from @example decorators.
            errors = list(
                execute_explicit_examples(
                    state, wrapped_test, arguments, kwargs, original_sig
                )
            )
            if errors:
                # If we're not going to report multiple bugs, we would have
                # stopped running explicit examples at the first failure.
                assert len(errors) == 1 or state.settings.report_multiple_bugs

                # If an explicit example raised a 'skip' exception, ensure it's never
                # wrapped up in an exception group.  Because we break out of the loop
                # immediately on finding a skip, if present it's always the last error.
                if isinstance(errors[-1][1], skip_exceptions_to_reraise()):
                    # Covered by `test_issue_3453_regression`, just in a subprocess.
                    del errors[:-1]  # pragma: no cover

                _raise_to_user(errors, state.settings, [], " in explicit examples")

            # If there were any explicit examples, they all ran successfully.
            # The next step is to use the Conjecture engine to run the test on
            # many different inputs.

            ran_explicit_examples = Phase.explicit in state.settings.phases and getattr(
                wrapped_test, "hypothesis_explicit_examples", ()
            )
            SKIP_BECAUSE_NO_EXAMPLES = unittest.SkipTest(
                "Hypothesis has been told to run no examples for this test."
            )
            if not (
                Phase.reuse in settings.phases or Phase.generate in settings.phases
            ):
                if not ran_explicit_examples:
                    raise SKIP_BECAUSE_NO_EXAMPLES
                return

            try:
                if isinstance(runner, TestCase) and hasattr(runner, "subTest"):
                    subTest = runner.subTest
                    try:
                        runner.subTest = types.MethodType(fake_subTest, runner)
                        state.run_engine()
                    finally:
                        runner.subTest = subTest
                else:
                    state.run_engine()
            except BaseException as e:
                # The exception caught here should either be an actual test
                # failure (or BaseExceptionGroup), or some kind of fatal error
                # that caused the engine to stop.

                generated_seed = wrapped_test._hypothesis_internal_use_generated_seed
                with local_settings(settings):
                    if not (state.failed_normally or generated_seed is None):
                        if running_under_pytest:
                            report(
                                f"You can add @seed({generated_seed}) to this test or "
                                f"run pytest with --hypothesis-seed={generated_seed} "
                                "to reproduce this failure."
                            )
                        else:
                            report(
                                f"You can add @seed({generated_seed}) to this test to "
                                "reproduce this failure."
                            )
                    # The dance here is to avoid showing users long tracebacks
                    # full of Hypothesis internals they don't care about.
                    # We have to do this inline, to avoid adding another
                    # internal stack frame just when we've removed the rest.
                    #
                    # Using a variable for our trimmed error ensures that the line
                    # which will actually appear in tracebacks is as clear as
                    # possible - "raise the_error_hypothesis_found".
                    the_error_hypothesis_found = e.with_traceback(
                        None
                        if isinstance(e, BaseExceptionGroup)
                        else get_trimmed_traceback()
                    )
                    raise the_error_hypothesis_found

            if not (ran_explicit_examples or state.ever_executed):
                raise SKIP_BECAUSE_NO_EXAMPLES

        def _get_fuzz_target() -> (
            Callable[[Union[bytes, bytearray, memoryview, BinaryIO]], Optional[bytes]]
        ):
            # Because fuzzing interfaces are very performance-sensitive, we use a
            # somewhat more complicated structure here.  `_get_fuzz_target()` is
            # called by the `HypothesisHandle.fuzz_one_input` property, allowing
            # us to defer our collection of the settings, random instance, and
            # reassignable `inner_test` (etc) until `fuzz_one_input` is accessed.
            #
            # We then share the performance cost of setting up `state` between
            # many invocations of the target.  We explicitly force `deadline=None`
            # for performance reasons, saving ~40% the runtime of an empty test.
            test = wrapped_test.hypothesis.inner_test
            settings = Settings(
                parent=wrapped_test._hypothesis_internal_use_settings, deadline=None
            )
            random = get_random_for_wrapped_test(test, wrapped_test)
            _args, _kwargs, stuff = process_arguments_to_given(
                wrapped_test, (), {}, given_kwargs, new_signature.parameters
            )
            assert not _args
            assert not _kwargs
            state = StateForActualGivenExecution(
                stuff, test, settings, random, wrapped_test
            )
            digest = function_digest(test)
            # We track the minimal-so-far example for each distinct origin, so
            # that we track log-n instead of n examples for long runs.  In particular
            # it means that we saturate for common errors in long runs instead of
            # storing huge volumes of low-value data.
            minimal_failures: dict = {}

            def fuzz_one_input(
                buffer: Union[bytes, bytearray, memoryview, BinaryIO]
            ) -> Optional[bytes]:
                # This inner part is all that the fuzzer will actually run,
                # so we keep it as small and as fast as possible.
                if isinstance(buffer, io.IOBase):
                    buffer = buffer.read(BUFFER_SIZE)
                assert isinstance(buffer, (bytes, bytearray, memoryview))
                data = ConjectureData.for_buffer(buffer)
                try:
                    state.execute_once(data)
                except (StopTest, UnsatisfiedAssumption):
                    return None
                except BaseException:
                    buffer = bytes(data.buffer)
                    known = minimal_failures.get(data.interesting_origin)
                    if settings.database is not None and (
                        known is None or sort_key(buffer) <= sort_key(known)
                    ):
                        settings.database.save(digest, buffer)
                        minimal_failures[data.interesting_origin] = buffer
                    raise
                return bytes(data.buffer)

            fuzz_one_input.__doc__ = HypothesisHandle.fuzz_one_input.__doc__
            return fuzz_one_input

        # After having created the decorated test function, we need to copy
        # over some attributes to make the switch as seamless as possible.

        for attrib in dir(test):
            if not (attrib.startswith("_") or hasattr(wrapped_test, attrib)):
                setattr(wrapped_test, attrib, getattr(test, attrib))
        wrapped_test.is_hypothesis_test = True
        if hasattr(test, "_hypothesis_internal_settings_applied"):
            # Used to check if @settings is applied twice.
            wrapped_test._hypothesis_internal_settings_applied = True
        wrapped_test._hypothesis_internal_use_seed = getattr(
            test, "_hypothesis_internal_use_seed", None
        )
        wrapped_test._hypothesis_internal_use_settings = (
            getattr(test, "_hypothesis_internal_use_settings", None) or Settings.default
        )
        wrapped_test._hypothesis_internal_use_reproduce_failure = getattr(
            test, "_hypothesis_internal_use_reproduce_failure", None
        )
        wrapped_test.hypothesis = HypothesisHandle(test, _get_fuzz_target, given_kwargs)
        return wrapped_test

    return run_test_as_given


def find(
    specifier: SearchStrategy[Ex],
    condition: Callable[[Any], bool],
    *,
    settings: Optional[Settings] = None,
    random: Optional[Random] = None,
    database_key: Optional[bytes] = None,
) -> Ex:
    """Returns the minimal example from the given strategy ``specifier`` that
    matches the predicate function ``condition``."""
    if settings is None:
        settings = Settings(max_examples=2000)
    settings = Settings(
        settings, suppress_health_check=list(HealthCheck), report_multiple_bugs=False
    )

    if database_key is None and settings.database is not None:
        # Note: The database key is not guaranteed to be unique. If not, replaying
        # of database examples may fail to reproduce due to being replayed on the
        # wrong condition.
        database_key = function_digest(condition)

    if not isinstance(specifier, SearchStrategy):
        raise InvalidArgument(
            f"Expected SearchStrategy but got {specifier!r} of "
            f"type {type(specifier).__name__}"
        )
    specifier.validate()

    last: list[Ex] = []

    @settings
    @given(specifier)
    def test(v):
        if condition(v):
            last[:] = [v]
            raise Found

    if random is not None:
        test = seed(random.getrandbits(64))(test)

    # Aliasing as Any avoids mypy errors (attr-defined) when accessing and
    # setting custom attributes on the decorated function or class.
    _test: Any = test
    _test._hypothesis_internal_is_find = True
    _test._hypothesis_internal_database_key = database_key

    try:
        test()
    except Found:
        return last[0]

    raise NoSuchExample(get_pretty_function_description(condition))
