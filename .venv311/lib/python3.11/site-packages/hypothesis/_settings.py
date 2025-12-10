# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The settings module configures runtime options for Hypothesis.

Either an explicit settings object can be used or the default object on
this module can be modified.
"""

import contextlib
import datetime
import inspect
import os
import warnings
from collections.abc import Collection, Generator, Sequence
from enum import Enum, EnumMeta, unique
from functools import total_ordering
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Optional,
    TypeVar,
)

from hypothesis.errors import (
    HypothesisDeprecationWarning,
    InvalidArgument,
)
from hypothesis.internal.conjecture.providers import AVAILABLE_PROVIDERS
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.internal.validation import check_type, try_convert
from hypothesis.utils.conventions import not_set
from hypothesis.utils.dynamicvariables import DynamicVariable

if TYPE_CHECKING:
    from hypothesis.database import ExampleDatabase

__all__ = ["settings"]

T = TypeVar("T")
all_settings: list[str] = [
    "max_examples",
    "derandomize",
    "database",
    "verbosity",
    "phases",
    "stateful_step_count",
    "report_multiple_bugs",
    "suppress_health_check",
    "deadline",
    "print_blob",
    "backend",
]


@unique
@total_ordering
class Verbosity(Enum):
    """Options for the |settings.verbosity| argument to |@settings|."""

    quiet = "quiet"
    """
    Hypothesis will not print any output, not even the final falsifying example.
    """

    normal = "normal"
    """
    Standard verbosity. Hypothesis will print the falsifying example, alongside
    any notes made with |note| (only for the falsfying example).
    """

    verbose = "verbose"
    """
    Increased verbosity. In addition to everything in |Verbosity.normal|, Hypothesis
    will:

    * Print each test case as it tries it
    * Print any notes made with |note| for each test case
    * Print each shrinking attempt
    * Print all explicit failing examples when using |@example|, instead of only
      the simplest one
    """

    debug = "debug"
    """
    Even more verbosity. Useful for debugging Hypothesis internals. You probably
    don't want this.
    """

    @classmethod
    def _missing_(cls, value):
        # deprecation pathway for integer values. Can be removed in Hypothesis 7.
        if isinstance(value, int) and not isinstance(value, bool):
            int_to_name = {0: "quiet", 1: "normal", 2: "verbose", 3: "debug"}
            if value in int_to_name:
                note_deprecation(
                    f"Passing Verbosity({value}) as an integer is deprecated. "
                    "Hypothesis now treats Verbosity values as strings, not integers. "
                    f"Use Verbosity.{int_to_name[value]} instead.",
                    since="2025-11-05",
                    has_codemod=False,
                    stacklevel=2,
                )
                return cls(int_to_name[value])
        return None

    def __repr__(self) -> str:
        return f"Verbosity.{self.name}"

    @staticmethod
    def _int_value(value: "Verbosity") -> int:
        # we would just map Verbosity keys, except it's not hashable
        mapping = {
            Verbosity.quiet.name: 0,
            Verbosity.normal.name: 1,
            Verbosity.verbose.name: 2,
            Verbosity.debug.name: 3,
        }
        # make sure we don't forget any new verbosity members
        assert list(mapping.keys()) == [verbosity.name for verbosity in Verbosity]
        return mapping[value.name]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Verbosity):
            return super().__eq__(other)
        return Verbosity._int_value(self) == other

    def __gt__(self, other: Any) -> bool:
        value1 = Verbosity._int_value(self)
        value2 = Verbosity._int_value(other) if isinstance(other, Verbosity) else other
        return value1 > value2


@unique
class Phase(Enum):
    """Options for the |settings.phases| argument to |@settings|."""

    explicit = "explicit"
    """
    Controls whether explicit examples are run.
    """

    reuse = "reuse"
    """
    Controls whether previous examples will be reused.
    """

    generate = "generate"
    """
    Controls whether new examples will be generated.
    """

    target = "target"
    """
    Controls whether examples will be mutated for targeting.
    """

    shrink = "shrink"
    """
    Controls whether examples will be shrunk.
    """

    explain = "explain"
    """
    Controls whether Hypothesis attempts to explain test failures.

    The explain phase has two parts, each of which is best-effort - if Hypothesis
    can't find a useful explanation, we'll just print the minimal failing example.
    """

    @classmethod
    def _missing_(cls, value):
        # deprecation pathway for integer values. Can be removed in Hypothesis 7.
        if isinstance(value, int) and not isinstance(value, bool):
            int_to_name = {
                0: "explicit",
                1: "reuse",
                2: "generate",
                3: "target",
                4: "shrink",
                5: "explain",
            }
            if value in int_to_name:
                note_deprecation(
                    f"Passing Phase({value}) as an integer is deprecated. "
                    "Hypothesis now treats Phase values as strings, not integers. "
                    f"Use Phase.{int_to_name[value]} instead.",
                    since="2025-11-05",
                    has_codemod=False,
                    stacklevel=2,
                )
                return cls(int_to_name[value])
        return None

    def __repr__(self) -> str:
        return f"Phase.{self.name}"


class HealthCheckMeta(EnumMeta):
    def __iter__(self):
        deprecated = (HealthCheck.return_value, HealthCheck.not_a_test_method)
        return iter(x for x in super().__iter__() if x not in deprecated)


@unique
class HealthCheck(Enum, metaclass=HealthCheckMeta):
    """
    A |HealthCheck| is proactively raised by Hypothesis when Hypothesis detects
    that your test has performance problems, which may result in less rigorous
    testing than you expect. For example, if your test takes a long time to generate
    inputs, or filters away too many  inputs using |assume| or |filter|, Hypothesis
    will raise a corresponding health check.

    A health check is a proactive warning, not an error. We encourage suppressing
    health checks where you have evaluated they will not pose a problem, or where
    you have evaluated that fixing the underlying issue is not worthwhile.

    With the exception of |HealthCheck.function_scoped_fixture| and
    |HealthCheck.differing_executors|, all health checks warn about performance
    problems, not correctness errors.

    Disabling health checks
    -----------------------

    Health checks can be disabled by |settings.suppress_health_check|. To suppress
    all health checks, you can pass ``suppress_health_check=list(HealthCheck)``.

    .. seealso::

        See also the :doc:`/how-to/suppress-healthchecks` how-to.

    Correctness health checks
    -------------------------

    Some health checks report potential correctness errors, rather than performance
    problems.

    * |HealthCheck.function_scoped_fixture| indicates that a function-scoped
      pytest fixture is used by an |@given| test. Many Hypothesis users expect
      function-scoped fixtures to reset once per input, but they actually reset once
      per test. We proactively raise |HealthCheck.function_scoped_fixture| to
      ensure you have considered this case.
    * |HealthCheck.differing_executors| indicates that the same |@given| test has
      been executed multiple times with multiple distinct executors.

    We recommend treating these particular health checks with more care, as
    suppressing them may result in an unsound test.
    """

    @classmethod
    def _missing_(cls, value):
        # deprecation pathway for integer values. Can be removed in Hypothesis 7.
        if isinstance(value, int) and not isinstance(value, bool):
            int_to_name = {
                1: "data_too_large",
                2: "filter_too_much",
                3: "too_slow",
                5: "return_value",
                7: "large_base_example",
                8: "not_a_test_method",
                9: "function_scoped_fixture",
                10: "differing_executors",
                11: "nested_given",
            }
            if value in int_to_name:
                note_deprecation(
                    f"Passing HealthCheck({value}) as an integer is deprecated. "
                    "Hypothesis now treats HealthCheck values as strings, not integers. "
                    f"Use HealthCheck.{int_to_name[value]} instead.",
                    since="2025-11-05",
                    has_codemod=False,
                    stacklevel=2,
                )
                return cls(int_to_name[value])
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    @classmethod
    def all(cls) -> list["HealthCheck"]:
        # Skipping of deprecated attributes is handled in HealthCheckMeta.__iter__
        note_deprecation(
            "`HealthCheck.all()` is deprecated; use `list(HealthCheck)` instead.",
            since="2023-04-16",
            has_codemod=True,
            stacklevel=1,
        )
        return list(HealthCheck)

    data_too_large = "data_too_large"
    """Checks if too many examples are aborted for being too large.

    This is measured by the number of random choices that Hypothesis makes
    in order to generate something, not the size of the generated object.
    For example, choosing a 100MB object from a predefined list would take
    only a few bits, while generating 10KB of JSON from scratch might trigger
    this health check.
    """

    filter_too_much = "filter_too_much"
    """Check for when the test is filtering out too many examples, either
    through use of |assume| or |.filter|, or occasionally for Hypothesis
    internal reasons."""

    too_slow = "too_slow"
    """
    Check for when input generation is very slow. Since Hypothesis generates 100
    (by default) inputs per test execution, a slowdown in generating each input
    can result in very slow tests overall.
    """

    return_value = "return_value"
    """Deprecated; we always error if a test returns a non-None value."""

    large_base_example = "large_base_example"
    """
    Checks if the smallest natural input to your test is very large. This makes
    it difficult for Hypothesis to generate good inputs, especially when trying to
    shrink failing inputs.
    """

    not_a_test_method = "not_a_test_method"
    """Deprecated; we always error if |@given| is applied
    to a method defined by :class:`python:unittest.TestCase` (i.e. not a test)."""

    function_scoped_fixture = "function_scoped_fixture"
    """Checks if |@given| has been applied to a test
    with a pytest function-scoped fixture. Function-scoped fixtures run once
    for the whole function, not once per example, and this is usually not what
    you want.

    Because of this limitation, tests that need to set up or reset
    state for every example need to do so manually within the test itself,
    typically using an appropriate context manager.

    Suppress this health check only in the rare case that you are using a
    function-scoped fixture that does not need to be reset between individual
    examples, but for some reason you cannot use a wider fixture scope
    (e.g. session scope, module scope, class scope).

    This check requires the :ref:`Hypothesis pytest plugin<pytest-plugin>`,
    which is enabled by default when running Hypothesis inside pytest."""

    differing_executors = "differing_executors"
    """Checks if |@given| has been applied to a test
    which is executed by different :ref:`executors<custom-function-execution>`.
    If your test function is defined as a method on a class, that class will be
    your executor, and subclasses executing an inherited test is a common way
    for things to go wrong.

    The correct fix is often to bring the executor instance under the control
    of hypothesis by explicit parametrization over, or sampling from,
    subclasses, or to refactor so that |@given| is
    specified on leaf subclasses."""

    nested_given = "nested_given"
    """Checks if |@given| is used inside another
    |@given|. This results in quadratic generation and
    shrinking behavior, and can usually be expressed more cleanly by using
    :func:`~hypothesis.strategies.data` to replace the inner
    |@given|.

    Nesting @given can be appropriate if you set appropriate limits for the
    quadratic behavior and cannot easily reexpress the inner function with
    :func:`~hypothesis.strategies.data`. To suppress this health check, set
    ``suppress_health_check=[HealthCheck.nested_given]`` on the outer
    |@given|. Setting it on the inner
    |@given| has no effect. If you have more than one
    level of nesting, add a suppression for this health check to every
    |@given| except the innermost one.
    """


class duration(datetime.timedelta):
    """A timedelta specifically measured in milliseconds."""

    def __repr__(self) -> str:
        ms = self.total_seconds() * 1000
        return f"timedelta(milliseconds={int(ms) if ms == int(ms) else ms!r})"


# see https://adamj.eu/tech/2020/03/09/detect-if-your-tests-are-running-on-ci
# initially from https://github.com/tox-dev/tox/blob/e911788a/src/tox/util/ci.py
_CI_VARS = {
    "CI": None,  # various, including GitHub Actions, Travis CI, and AppVeyor
    # see https://github.com/tox-dev/tox/issues/3442
    "__TOX_ENVIRONMENT_VARIABLE_ORIGINAL_CI": None,
    "TF_BUILD": "true",  # Azure Pipelines
    "bamboo.buildKey": None,  # Bamboo
    "BUILDKITE": "true",  # Buildkite
    "CIRCLECI": "true",  # Circle CI
    "CIRRUS_CI": "true",  # Cirrus CI
    "CODEBUILD_BUILD_ID": None,  # CodeBuild
    "GITHUB_ACTIONS": "true",  # GitHub Actions
    "GITLAB_CI": None,  # GitLab CI
    "HEROKU_TEST_RUN_ID": None,  # Heroku CI
    "TEAMCITY_VERSION": None,  # TeamCity
}


def is_in_ci() -> bool:
    return any(
        key in os.environ and (value is None or os.environ[key] == value)
        for key, value in _CI_VARS.items()
    )


default_variable = DynamicVariable[Optional["settings"]](None)


def _validate_choices(name: str, value: T, *, choices: Sequence[object]) -> T:
    if value not in choices:
        msg = f"Invalid {name}, {value!r}. Valid choices: {choices!r}"
        raise InvalidArgument(msg)
    return value


def _validate_enum_value(cls: Any, value: object, *, name: str) -> Any:
    try:
        return cls(value)
    except ValueError:
        raise InvalidArgument(
            f"{name}={value} is not a valid value. The options "
            f"are: {', '.join(repr(m.name) for m in cls)}"
        ) from None


def _validate_max_examples(max_examples: int) -> int:
    check_type(int, max_examples, name="max_examples")
    if max_examples < 1:
        raise InvalidArgument(
            f"max_examples={max_examples!r} must be at least one. If you want "
            "to disable generation entirely, use phases=[Phase.explicit] instead."
        )
    return max_examples


def _validate_database(
    database: Optional["ExampleDatabase"],
) -> Optional["ExampleDatabase"]:
    from hypothesis.database import ExampleDatabase

    if database is None or isinstance(database, ExampleDatabase):
        return database
    raise InvalidArgument(
        "Arguments to the database setting must be None or an instance of "
        "ExampleDatabase. Use one of the database classes in "
        "hypothesis.database"
    )


def _validate_phases(phases: Collection[Phase]) -> Sequence[Phase]:
    phases = try_convert(tuple, phases, "phases")
    phases = tuple(
        _validate_enum_value(Phase, phase, name="phases") for phase in phases
    )
    # sort by definition order
    return tuple(phase for phase in list(Phase) if phase in phases)


def _validate_stateful_step_count(stateful_step_count: int) -> int:
    check_type(int, stateful_step_count, name="stateful_step_count")
    if stateful_step_count < 1:
        raise InvalidArgument(
            f"stateful_step_count={stateful_step_count!r} must be at least one."
        )
    return stateful_step_count


def _validate_suppress_health_check(suppressions: object) -> tuple[HealthCheck, ...]:
    suppressions = try_convert(tuple, suppressions, "suppress_health_check")
    for health_check in suppressions:
        if health_check in (HealthCheck.return_value, HealthCheck.not_a_test_method):
            note_deprecation(
                f"The {health_check.name} health check is deprecated, because this is always an error.",
                since="2023-03-15",
                has_codemod=False,
                stacklevel=2,
            )
    return tuple(
        _validate_enum_value(HealthCheck, health_check, name="suppress_health_check")
        for health_check in suppressions
    )


def _validate_deadline(
    deadline: int | float | datetime.timedelta | None,
) -> duration | None:
    if deadline is None:
        return deadline
    invalid_deadline_error = InvalidArgument(
        f"deadline={deadline!r} (type {type(deadline).__name__}) must be a timedelta object, "
        "an integer or float number of milliseconds, or None to disable the "
        "per-test-case deadline."
    )
    if isinstance(deadline, (int, float)):
        if isinstance(deadline, bool):
            raise invalid_deadline_error
        try:
            deadline = duration(milliseconds=deadline)
        except OverflowError:
            raise InvalidArgument(
                f"deadline={deadline!r} is invalid, because it is too large to represent "
                "as a timedelta. Use deadline=None to disable deadlines."
            ) from None
    if isinstance(deadline, datetime.timedelta):
        if deadline <= datetime.timedelta(0):
            raise InvalidArgument(
                f"deadline={deadline!r} is invalid, because it is impossible to meet a "
                "deadline <= 0. Use deadline=None to disable deadlines."
            )
        return duration(seconds=deadline.total_seconds())
    raise invalid_deadline_error


def _validate_backend(backend: str) -> str:
    if backend not in AVAILABLE_PROVIDERS:
        if backend == "crosshair":  # pragma: no cover
            install = '`pip install "hypothesis[crosshair]"` and try again.'
            raise InvalidArgument(f"backend={backend!r} is not available.  {install}")
        raise InvalidArgument(
            f"backend={backend!r} is not available - maybe you need to install a plugin?"
            f"\n    Installed backends: {sorted(AVAILABLE_PROVIDERS)!r}"
        )
    return backend


class settingsMeta(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def default(cls) -> Optional["settings"]:
        v = default_variable.value
        if v is not None:
            return v
        if getattr(settings, "_current_profile", None) is not None:
            assert settings._current_profile is not None
            settings.load_profile(settings._current_profile)
            assert default_variable.value is not None
        return default_variable.value

    def __setattr__(cls, name: str, value: object) -> None:
        if name == "default":
            raise AttributeError(
                "Cannot assign to the property settings.default - "
                "consider using settings.load_profile instead."
            )
        elif not name.startswith("_"):
            raise AttributeError(
                f"Cannot assign hypothesis.settings.{name}={value!r} - the settings "
                "class is immutable.  You can change the global default "
                "settings with settings.load_profile, or use @settings(...) "
                "to decorate your test instead."
            )
        super().__setattr__(name, value)

    def __repr__(cls):
        return "hypothesis.settings"


class settings(metaclass=settingsMeta):
    """
    A settings object controls the following aspects of test behavior:
    |~settings.max_examples|, |~settings.derandomize|, |~settings.database|,
    |~settings.verbosity|, |~settings.phases|, |~settings.stateful_step_count|,
    |~settings.report_multiple_bugs|, |~settings.suppress_health_check|,
    |~settings.deadline|, |~settings.print_blob|, and |~settings.backend|.

    A settings object can be applied as a decorator to a test function, in which
    case that test function will use those settings. A test may only have one
    settings object applied to it. A settings object can also be passed to
    |settings.register_profile| or as a parent to another |settings|.

    Attribute inheritance
    ---------------------

    Settings objects are immutable once created. When a settings object is created,
    it uses the value specified for each attribute. Any attribute which is
    not specified will inherit from its value in the ``parent`` settings object.
    If ``parent`` is not passed, any attributes which are not specified will inherit
    from the current settings profile instead.

    For instance, ``settings(max_examples=10)`` will have a ``max_examples`` of ``10``,
    and the value of all other attributes will be equal to its value in the
    current settings profile.

    Changes made from activating a new settings profile with |settings.load_profile|
    will be reflected in settings objects created after the profile was loaded,
    but not in existing settings objects.

    .. _builtin-profiles:

    Built-in profiles
    -----------------

    While you can register additional profiles with |settings.register_profile|,
    Hypothesis comes with two built-in profiles: ``default`` and ``ci``.

    By default, the ``default`` profile is active. If the ``CI`` environment
    variable is set to any value, the ``ci`` profile is active by default. Hypothesis
    also automatically detects various vendor-specific CI environment variables.

    The attributes of the currently active settings profile can be retrieved with
    ``settings()`` (so ``settings().max_examples`` is the currently active default
    for |settings.max_examples|).

    The settings attributes for the built-in profiles are as follows:

    .. code-block:: python

        default = settings.register_profile(
            "default",
            max_examples=100,
            derandomize=False,
            database=not_set,  # see settings.database for the default database
            verbosity=Verbosity.normal,
            phases=tuple(Phase),
            stateful_step_count=50,
            report_multiple_bugs=True,
            suppress_health_check=(),
            deadline=duration(milliseconds=200),
            print_blob=False,
            backend="hypothesis",
        )

        ci = settings.register_profile(
            "ci",
            parent=default,
            derandomize=True,
            deadline=None,
            database=None,
            print_blob=True,
            suppress_health_check=[HealthCheck.too_slow],
        )

    You can replace either of the built-in profiles with |settings.register_profile|:

    .. code-block:: python

        # run more examples in CI
        settings.register_profile(
            "ci",
            settings.get_profile("ci"),
            max_examples=1000,
        )
    """

    _profiles: ClassVar[dict[str, "settings"]] = {}
    _current_profile: ClassVar[str | None] = None

    def __init__(
        self,
        parent: Optional["settings"] = None,
        *,
        # This looks pretty strange, but there's good reason: we want Mypy to detect
        # bad calls downstream, but not to freak out about the `= not_set` part even
        # though it's not semantically valid to pass that as an argument value.
        # The intended use is "like **kwargs, but more tractable for tooling".
        max_examples: int = not_set,  # type: ignore
        derandomize: bool = not_set,  # type: ignore
        database: Optional["ExampleDatabase"] = not_set,  # type: ignore
        verbosity: "Verbosity" = not_set,  # type: ignore
        phases: Collection["Phase"] = not_set,  # type: ignore
        stateful_step_count: int = not_set,  # type: ignore
        report_multiple_bugs: bool = not_set,  # type: ignore
        suppress_health_check: Collection["HealthCheck"] = not_set,  # type: ignore
        deadline: int | float | datetime.timedelta | None = not_set,  # type: ignore
        print_blob: bool = not_set,  # type: ignore
        backend: str = not_set,  # type: ignore
    ) -> None:
        self._in_definition = True

        if parent is not None:
            check_type(settings, parent, "parent")
        if derandomize not in (not_set, False):
            if database not in (not_set, None):  # type: ignore
                raise InvalidArgument(
                    "derandomize=True implies database=None, so passing "
                    f"{database=} too is invalid."
                )
            database = None

        # fallback is None if we're creating the default settings object, and
        # the parent (or default settings object) otherwise
        self._fallback = parent or settings.default
        self._max_examples = (
            self._fallback.max_examples  # type: ignore
            if max_examples is not_set  # type: ignore
            else _validate_max_examples(max_examples)
        )
        self._derandomize = (
            self._fallback.derandomize  # type: ignore
            if derandomize is not_set  # type: ignore
            else _validate_choices("derandomize", derandomize, choices=[True, False])
        )
        if database is not not_set:  # type: ignore
            database = _validate_database(database)
        self._database = database
        self._cached_database = None
        self._verbosity = (
            self._fallback.verbosity  # type: ignore
            if verbosity is not_set  # type: ignore
            else _validate_enum_value(Verbosity, verbosity, name="verbosity")
        )
        self._phases = (
            self._fallback.phases  # type: ignore
            if phases is not_set  # type: ignore
            else _validate_phases(phases)
        )
        self._stateful_step_count = (
            self._fallback.stateful_step_count  # type: ignore
            if stateful_step_count is not_set  # type: ignore
            else _validate_stateful_step_count(stateful_step_count)
        )
        self._report_multiple_bugs = (
            self._fallback.report_multiple_bugs  # type: ignore
            if report_multiple_bugs is not_set  # type: ignore
            else _validate_choices(
                "report_multiple_bugs", report_multiple_bugs, choices=[True, False]
            )
        )
        self._suppress_health_check = (
            self._fallback.suppress_health_check  # type: ignore
            if suppress_health_check is not_set  # type: ignore
            else _validate_suppress_health_check(suppress_health_check)
        )
        self._deadline = (
            self._fallback.deadline  # type: ignore
            if deadline is not_set  # type: ignore
            else _validate_deadline(deadline)
        )
        self._print_blob = (
            self._fallback.print_blob  # type: ignore
            if print_blob is not_set  # type: ignore
            else _validate_choices("print_blob", print_blob, choices=[True, False])
        )
        self._backend = (
            self._fallback.backend  # type: ignore
            if backend is not_set  # type: ignore
            else _validate_backend(backend)
        )

        self._in_definition = False

    @property
    def max_examples(self):
        """
        Once this many satisfying examples have been considered without finding any
        counter-example, Hypothesis will stop looking.

        Note that we might call your test function fewer times if we find a bug early
        or can tell that we've exhausted the search space; or more if we discard some
        examples due to use of .filter(), assume(), or a few other things that can
        prevent the test case from completing successfully.

        The default value is chosen to suit a workflow where the test will be part of
        a suite that is regularly executed locally or on a CI server, balancing total
        running time against the chance of missing a bug.

        If you are writing one-off tests, running tens of thousands of examples is
        quite reasonable as Hypothesis may miss uncommon bugs with default settings.
        For very complex code, we have observed Hypothesis finding novel bugs after
        *several million* examples while testing :pypi:`SymPy <sympy>`.
        If you are running more than 100k examples for a test, consider using our
        :ref:`integration for coverage-guided fuzzing <fuzz_one_input>` - it really
        shines when given minutes or hours to run.

        The default max examples is ``100``.
        """
        return self._max_examples

    @property
    def derandomize(self):
        """
        If True, seed Hypothesis' random number generator using a hash of the test
        function, so that every run will test the same set of examples until you
        update Hypothesis, Python, or the test function.

        This allows you to `check for regressions and look for bugs
        <https://blog.nelhage.com/post/two-kinds-of-testing/>`__ using separate
        settings profiles - for example running
        quick deterministic tests on every commit, and a longer non-deterministic
        nightly testing run.

        The default is ``False``. If running on CI, the default is ``True`` instead.
        """
        return self._derandomize

    @property
    def database(self):
        """
        An instance of |ExampleDatabase| that will be used to save examples to
        and load previous examples from.

        If not set, a |DirectoryBasedExampleDatabase| is created in the current
        working directory under ``.hypothesis/examples``. If this location is
        unusable, e.g. due to the lack of read or write permissions, Hypothesis
        will emit a warning and fall back to an |InMemoryExampleDatabase|.

        If ``None``, no storage will be used.

        See the :ref:`database documentation <database>` for a list of database
        classes, and how to define custom database classes.
        """
        from hypothesis.database import _db_for_path

        # settings.database has two conflicting requirements:
        # * The default settings should respect changes to set_hypothesis_home_dir
        #   in-between accesses
        # * `s.database is s.database` should be true, except for the default settings
        #
        # We therefore cache s.database for everything except the default settings,
        # which always recomputes dynamically.
        if self._fallback is None:
            # if self._fallback is None, we are the default settings, at which point
            # we should recompute the database dynamically
            assert self._database is not_set
            return _db_for_path(not_set)

        # otherwise, we cache the database
        if self._cached_database is None:
            self._cached_database = (
                self._fallback.database if self._database is not_set else self._database
            )
        return self._cached_database

    @property
    def verbosity(self):
        """
        Control the verbosity level of Hypothesis messages.

        To see what's going on while Hypothesis runs your tests, you can turn
        up the verbosity setting.

        .. code-block:: pycon

            >>> from hypothesis import settings, Verbosity
            >>> from hypothesis.strategies import lists, integers
            >>> @given(lists(integers()))
            ... @settings(verbosity=Verbosity.verbose)
            ... def f(x):
            ...     assert not any(x)
            ... f()
            Trying example: []
            Falsifying example: [-1198601713, -67, 116, -29578]
            Shrunk example to [-1198601713]
            Shrunk example to [-128]
            Shrunk example to [32]
            Shrunk example to [1]
            [1]

        The four levels are |Verbosity.quiet|, |Verbosity.normal|,
        |Verbosity.verbose|, and |Verbosity.debug|. |Verbosity.normal| is the
        default. For |Verbosity.quiet|, Hypothesis will not print anything out,
        not even the final falsifying example. |Verbosity.debug| is basically
        |Verbosity.verbose| but a bit more so. You probably don't want it.

        Verbosity can be passed either as a |Verbosity| enum value, or as the
        corresponding string value, or as the corresponding integer value. For
        example:

        .. code-block:: python

            # these three are equivalent
            settings(verbosity=Verbosity.verbose)
            settings(verbosity="verbose")

        If you are using :pypi:`pytest`, you may also need to :doc:`disable
        output capturing for passing tests <pytest:how-to/capture-stdout-stderr>`
        to see verbose output as tests run.
        """
        return self._verbosity

    @property
    def phases(self):
        """
        Control which phases should be run.

        Hypothesis divides tests into logically distinct phases.

        - |Phase.explicit|: Running explicit examples from |@example|.
        - |Phase.reuse|: Running examples from the database which previously failed.
        - |Phase.generate|: Generating new random examples.
        - |Phase.target|: Mutating examples for :ref:`targeted property-based
          testing <targeted>`. Requires |Phase.generate|.
        - |Phase.shrink|: Shrinking failing examples.
        - |Phase.explain|: Attempting to explain why a failure occurred.
          Requires |Phase.shrink|.

        The phases argument accepts a collection with any subset of these. E.g.
        ``settings(phases=[Phase.generate, Phase.shrink])`` will generate new examples
        and shrink them, but will not run explicit examples or reuse previous failures,
        while ``settings(phases=[Phase.explicit])`` will only run explicit examples
        from |@example|.

        Phases can be passed either as a |Phase| enum value, or as the corresponding
        string value. For example:

        .. code-block:: python

            # these two are equivalent
            settings(phases=[Phase.explicit])
            settings(phases=["explicit"])

        Following the first failure, Hypothesis will (usually, depending on
        which |Phase| is enabled) track which lines of code are always run on
        failing but never on passing inputs. On 3.12+, this uses
        :mod:`sys.monitoring`, while 3.11 and earlier uses :func:`python:sys.settrace`.
        For python 3.11 and earlier, we therefore automatically disable the explain
        phase on PyPy, or if you are using :pypi:`coverage` or a debugger. If
        there are no clearly suspicious lines of code, :pep:`we refuse the
        temptation to guess <20>`.

        After shrinking to a minimal failing example, Hypothesis will try to find
        parts of the example -- e.g. separate args to |@given|
        -- which can vary freely without changing the result
        of that minimal failing example. If the automated experiments run without
        finding a passing variation, we leave a comment in the final report:

        .. code-block:: python

            test_x_divided_by_y(
                x=0,  # or any other generated value
                y=0,
            )

        Just remember that the *lack* of an explanation sometimes just means that
        Hypothesis couldn't efficiently find one, not that no explanation (or
        simpler failing example) exists.
        """

        return self._phases

    @property
    def stateful_step_count(self):
        """
        The maximum number of times to call an additional |@rule| method in
        :ref:`stateful testing <stateful>` before we give up on finding a bug.

        Note that this setting is effectively multiplicative with max_examples,
        as each example will run for a maximum of ``stateful_step_count`` steps.

        The default stateful step count is ``50``.
        """
        return self._stateful_step_count

    @property
    def report_multiple_bugs(self):
        """
        Because Hypothesis runs the test many times, it can sometimes find multiple
        bugs in a single run.  Reporting all of them at once is usually very useful,
        but replacing the exceptions can occasionally clash with debuggers.
        If disabled, only the exception with the smallest minimal example is raised.

        The default value is ``True``.
        """
        return self._report_multiple_bugs

    @property
    def suppress_health_check(self):
        """
        Suppress the given |HealthCheck| exceptions. Those health checks will not
        be raised by Hypothesis. To suppress all health checks, you can pass
        ``suppress_health_check=list(HealthCheck)``.

        Health checks can be passed either as a |HealthCheck| enum value, or as
        the corresponding string value. For example:

        .. code-block:: python

            # these two are equivalent
            settings(suppress_health_check=[HealthCheck.filter_too_much])
            settings(suppress_health_check=["filter_too_much"])

        Health checks are proactive warnings, not correctness errors, so we
        encourage suppressing health checks where you have evaluated they will
        not pose a problem, or where you have evaluated that fixing the underlying
        issue is not worthwhile.

        .. seealso::

            See also the :doc:`/how-to/suppress-healthchecks` how-to.
        """
        return self._suppress_health_check

    @property
    def deadline(self):
        """
        The maximum allowed duration of an individual test case, in milliseconds.
        You can pass an integer, float, or timedelta. If ``None``, the deadline
        is disabled entirely.

        We treat the deadline as a soft limit in some cases, where that would
        avoid flakiness due to timing variability.

        The default deadline is 200 milliseconds. If running on CI, the default is
        ``None`` instead.
        """
        return self._deadline

    @property
    def print_blob(self):
        """
        If set to ``True``, Hypothesis will print code for failing examples that
        can be used with |@reproduce_failure| to reproduce the failing example.

        The default value is ``False``. If running on CI, the default is ``True`` instead.
        """
        return self._print_blob

    @property
    def backend(self):
        """
        .. warning::

            EXPERIMENTAL AND UNSTABLE - see :ref:`alternative-backends`.

        The importable name of a backend which Hypothesis should use to generate
        primitive types. We support heuristic-random, solver-based, and fuzzing-based
        backends.
        """
        return self._backend

    def __call__(self, test: T) -> T:
        """Make the settings object (self) an attribute of the test.

        The settings are later discovered by looking them up on the test itself.
        """
        # Aliasing as Any avoids mypy errors (attr-defined) when accessing and
        # setting custom attributes on the decorated function or class.
        _test: Any = test

        # Using the alias here avoids a mypy error (return-value) later when
        # ``test`` is returned, because this check results in type refinement.
        if not callable(_test):
            raise InvalidArgument(
                "settings objects can be called as a decorator with @given, "
                f"but decorated {test=} is not callable."
            )
        if inspect.isclass(test):
            from hypothesis.stateful import RuleBasedStateMachine

            if issubclass(_test, RuleBasedStateMachine):
                attr_name = "_hypothesis_internal_settings_applied"
                if getattr(test, attr_name, False):
                    raise InvalidArgument(
                        "Applying the @settings decorator twice would "
                        "overwrite the first version; merge their arguments "
                        "instead."
                    )
                setattr(test, attr_name, True)
                _test.TestCase.settings = self
                return test
            else:
                raise InvalidArgument(
                    "@settings(...) can only be used as a decorator on "
                    "functions, or on subclasses of RuleBasedStateMachine."
                )
        if hasattr(_test, "_hypothesis_internal_settings_applied"):
            # Can't use _hypothesis_internal_use_settings as an indicator that
            # @settings was applied, because @given also assigns that attribute.
            descr = get_pretty_function_description(test)
            raise InvalidArgument(
                f"{descr} has already been decorated with a settings object.\n"
                f"    Previous:  {_test._hypothesis_internal_use_settings!r}\n"
                f"    This:  {self!r}"
            )

        _test._hypothesis_internal_use_settings = self
        _test._hypothesis_internal_settings_applied = True
        return test

    def __setattr__(self, name: str, value: object) -> None:
        if not name.startswith("_") and not self._in_definition:
            raise AttributeError("settings objects are immutable")
        return super().__setattr__(name, value)

    def __repr__(self) -> str:
        bits = sorted(
            f"{name}={getattr(self, name)!r}"
            for name in all_settings
            if (name != "backend" or len(AVAILABLE_PROVIDERS) > 1)  # experimental
        )
        return "settings({})".format(", ".join(bits))

    def show_changed(self) -> str:
        bits = []
        for name in all_settings:
            value = getattr(self, name)
            if value != getattr(default, name):
                bits.append(f"{name}={value!r}")
        return ", ".join(sorted(bits, key=len))

    @staticmethod
    def register_profile(
        name: str,
        parent: Optional["settings"] = None,
        **kwargs: Any,
    ) -> None:
        """
        Register a settings object as a settings profile, under the name ``name``.
        The ``parent`` and ``kwargs`` arguments to this method are as for
        |settings|.

        If a settings profile already exists under ``name``, it will be overwritten.
        Registering a profile with the same name as the currently active profile
        will cause those changes to take effect in the active profile immediately,
        and do not require reloading the profile.

        Registered settings profiles can be retrieved later by name with
        |settings.get_profile|.
        """
        check_type(str, name, "name")

        if (
            default_variable.value
            and settings._current_profile
            and default_variable.value != settings._profiles[settings._current_profile]
        ):
            note_deprecation(
                "Cannot register a settings profile when the current settings differ "
                "from the current profile (usually due to an @settings decorator). "
                "Register profiles at module level instead.",
                since="2025-11-15",
                has_codemod=False,
            )

        # if we just pass the parent and no kwargs, like
        #   settings.register_profile(settings(max_examples=10))
        # then optimize out the pointless intermediate settings object which
        # would just forward everything to the parent.
        settings._profiles[name] = (
            parent
            if parent is not None and not kwargs
            else settings(parent=parent, **kwargs)
        )
        if settings._current_profile == name:
            settings.load_profile(name)

    @staticmethod
    def get_profile(name: str) -> "settings":
        """
        Returns the settings profile registered under ``name``. If no settings
        profile is registered under ``name``, raises |InvalidArgument|.
        """
        check_type(str, name, "name")
        try:
            return settings._profiles[name]
        except KeyError:
            raise InvalidArgument(f"Profile {name!r} is not registered") from None

    @staticmethod
    def load_profile(name: str) -> None:
        """
        Makes the settings profile registered under ``name`` the active profile.

        If no settings profile is registered under ``name``, raises |InvalidArgument|.
        """
        check_type(str, name, "name")
        settings._current_profile = name
        default_variable.value = settings.get_profile(name)

    @staticmethod
    def get_current_profile_name() -> str:
        """
        The name of the current settings profile. For example:

        .. code-block:: python

            >>> settings.load_profile("myprofile")
            >>> settings.get_current_profile_name()
            'myprofile'
        """
        assert settings._current_profile is not None
        return settings._current_profile


@contextlib.contextmanager
def local_settings(s: settings) -> Generator[settings, None, None]:
    with default_variable.with_value(s):
        yield s


def note_deprecation(
    message: str, *, since: str, has_codemod: bool, stacklevel: int = 0
) -> None:
    if since != "RELEASEDAY":
        date = datetime.date.fromisoformat(since)
        assert datetime.date(2021, 1, 1) <= date
    if has_codemod:
        message += (
            "\n    The `hypothesis codemod` command-line tool can automatically "
            "refactor your code to fix this warning."
        )
    warnings.warn(HypothesisDeprecationWarning(message), stacklevel=2 + stacklevel)


default = settings(
    max_examples=100,
    derandomize=False,
    database=not_set,  # type: ignore
    verbosity=Verbosity.normal,
    phases=tuple(Phase),
    stateful_step_count=50,
    report_multiple_bugs=True,
    suppress_health_check=(),
    deadline=duration(milliseconds=200),
    print_blob=False,
    backend="hypothesis",
)
settings.register_profile("default", default)
settings.load_profile("default")

assert settings.default is not None

CI = settings(
    derandomize=True,
    deadline=None,
    database=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile("ci", CI)


if is_in_ci():  # pragma: no cover # covered in ci, but not locally
    settings.load_profile("ci")

assert settings.default is not None


# Check that the kwonly args to settings.__init__ is the same as the set of
# defined settings - in case we've added or remove something from one but
# not the other.
assert set(all_settings) == {
    p.name
    for p in inspect.signature(settings.__init__).parameters.values()
    if p.kind == inspect.Parameter.KEYWORD_ONLY
}
