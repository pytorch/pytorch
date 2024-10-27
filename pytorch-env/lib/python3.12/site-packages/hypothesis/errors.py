# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from typing import Literal

from hypothesis.internal.compat import ExceptionGroup


class HypothesisException(Exception):
    """Generic parent class for exceptions thrown by Hypothesis."""


class _Trimmable(HypothesisException):
    """Hypothesis can trim these tracebacks even if they're raised internally."""


class UnsatisfiedAssumption(HypothesisException):
    """An internal error raised by assume.

    If you're seeing this error something has gone wrong.
    """

    def __init__(self, reason=None):
        self.reason = reason


class NoSuchExample(HypothesisException):
    """The condition we have been asked to satisfy appears to be always false.

    This does not guarantee that no example exists, only that we were
    unable to find one.
    """

    def __init__(self, condition_string, extra=""):
        super().__init__(f"No examples found of condition {condition_string}{extra}")


class Unsatisfiable(_Trimmable):
    """We ran out of time or examples before we could find enough examples
    which satisfy the assumptions of this hypothesis.

    This could be because the function is too slow. If so, try upping
    the timeout. It could also be because the function is using assume
    in a way that is too hard to satisfy. If so, try writing a custom
    strategy or using a better starting point (e.g if you are requiring
    a list has unique values you could instead filter out all duplicate
    values from the list)
    """


class Flaky(_Trimmable):
    """Base class for indeterministic failures. Usually one of the more
    specific subclasses (FlakyFailure or FlakyStrategyDefinition) is raised."""


class FlakyReplay(Flaky):
    """Internal error raised by the conjecture engine if flaky failures are
    detected during replay.

    Carries information allowing the runner to reconstruct the flakiness as
    a FlakyFailure exception group for final presentation.
    """

    def __init__(self, reason, interesting_origins=None):
        super().__init__(reason)
        self.reason = reason
        self._interesting_origins = interesting_origins


class FlakyStrategyDefinition(Flaky):
    """This function appears to cause inconsistent data generation.

    Common causes for this problem are:
        1. The strategy depends on external state. e.g. it uses an external
           random number generator. Try to make a version that passes all the
           relevant state in from Hypothesis.
    """


class _WrappedBaseException(Exception):
    """Used internally for wrapping BaseExceptions as components of FlakyFailure."""


class FlakyFailure(ExceptionGroup, Flaky):
    """This function appears to fail non-deterministically: We have seen it
    fail when passed this example at least once, but a subsequent invocation
    did not fail, or caused a distinct error.

    Common causes for this problem are:
        1. The function depends on external state. e.g. it uses an external
           random number generator. Try to make a version that passes all the
           relevant state in from Hypothesis.
        2. The function is suffering from too much recursion and its failure
           depends sensitively on where it's been called from.
        3. The function is timing sensitive and can fail or pass depending on
           how long it takes. Try breaking it up into smaller functions which
           don't do that and testing those instead.
    """

    def __new__(cls, msg, group):
        # The Exception mixin forces this an ExceptionGroup (only accepting
        # Exceptions, not BaseException). Usually BaseException is raised
        # directly and will hence not be part of a FlakyFailure, but I'm not
        # sure this assumption holds everywhere. So wrap any BaseExceptions.
        group = list(group)
        for i, exc in enumerate(group):
            if not isinstance(exc, Exception):
                err = _WrappedBaseException()
                err.__cause__ = err.__context__ = exc
                group[i] = err
        return ExceptionGroup.__new__(cls, msg, group)


class InvalidArgument(_Trimmable, TypeError):
    """Used to indicate that the arguments to a Hypothesis function were in
    some manner incorrect."""


class ResolutionFailed(InvalidArgument):
    """Hypothesis had to resolve a type to a strategy, but this failed.

    Type inference is best-effort, so this only happens when an
    annotation exists but could not be resolved for a required argument
    to the target of ``builds()``, or where the user passed ``...``.
    """


class InvalidState(HypothesisException):
    """The system is not in a state where you were allowed to do that."""


class InvalidDefinition(_Trimmable, TypeError):
    """Used to indicate that a class definition was not well put together and
    has something wrong with it."""


class HypothesisWarning(HypothesisException, Warning):
    """A generic warning issued by Hypothesis."""


class FailedHealthCheck(_Trimmable):
    """Raised when a test fails a healthcheck."""


class NonInteractiveExampleWarning(HypothesisWarning):
    """SearchStrategy.example() is designed for interactive use,
    but should never be used in the body of a test.
    """


class HypothesisDeprecationWarning(HypothesisWarning, FutureWarning):
    """A deprecation warning issued by Hypothesis.

    Actually inherits from FutureWarning, because DeprecationWarning is
    hidden by the default warnings filter.

    You can configure the Python :mod:`python:warnings` to handle these
    warnings differently to others, either turning them into errors or
    suppressing them entirely.  Obviously we would prefer the former!
    """


class HypothesisSideeffectWarning(HypothesisWarning):
    """A warning issued by Hypothesis when it sees actions that are
    discouraged at import or initialization time because they are
    slow or have user-visible side effects.
    """


class Frozen(HypothesisException):
    """Raised when a mutation method has been called on a ConjectureData object
    after freeze() has been called."""


def __getattr__(name):
    if name == "MultipleFailures":
        from hypothesis._settings import note_deprecation
        from hypothesis.internal.compat import BaseExceptionGroup

        note_deprecation(
            "MultipleFailures is deprecated; use the builtin `BaseExceptionGroup` type "
            "instead, or `exceptiongroup.BaseExceptionGroup` before Python 3.11",
            since="2022-08-02",
            has_codemod=False,  # This would be a great PR though!
            stacklevel=1,
        )
        return BaseExceptionGroup

    raise AttributeError(f"Module 'hypothesis.errors' has no attribute {name}")


class DeadlineExceeded(_Trimmable):
    """Raised when an individual test body has taken too long to run."""

    def __init__(self, runtime, deadline):
        super().__init__(
            "Test took %.2fms, which exceeds the deadline of %.2fms"
            % (runtime.total_seconds() * 1000, deadline.total_seconds() * 1000)
        )
        self.runtime = runtime
        self.deadline = deadline

    def __reduce__(self):
        return (type(self), (self.runtime, self.deadline))


class StopTest(BaseException):
    """Raised when a test should stop running and return control to
    the Hypothesis engine, which should then continue normally.
    """

    def __init__(self, testcounter: int) -> None:
        super().__init__(repr(testcounter))
        self.testcounter = testcounter


class DidNotReproduce(HypothesisException):
    pass


class Found(HypothesisException):
    """Signal that the example matches condition. Internal use only."""


class RewindRecursive(Exception):
    """Signal that the type inference should be rewound due to recursive types. Internal use only."""

    def __init__(self, target: object) -> None:
        self.target = target


class SmallSearchSpaceWarning(HypothesisWarning):
    """Indicates that an inferred strategy does not span the search space
    in a meaningful way, for example by only creating default instances."""


class BackendCannotProceed(HypothesisException):
    """UNSTABLE API

    Raised by alternative backends when the PrimitiveProvider cannot proceed.
    This is expected to occur inside one of the `.draw_*()` methods, or for
    symbolic execution perhaps in `.realize(...)`.

    The optional `scope` argument can enable smarter integration:

        verified:
            Do not request further test cases from this backend.  We _may_
            generate more test cases with other backends; if one fails then
            Hypothesis will report unsound verification in the backend too.

        exhausted:
            Do not request further test cases from this backend; finish testing
            with test cases generated with the default backend.  Common if e.g.
            native code blocks symbolic reasoning very early.

        discard_test_case:
            This particular test case could not be converted to concrete values;
            skip any further processing and continue with another test case from
            this backend.
    """

    def __init__(
        self,
        scope: Literal["verified", "exhausted", "discard_test_case", "other"] = "other",
        /,
    ) -> None:
        self.scope = scope
