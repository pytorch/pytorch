# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import importlib
import inspect
import math
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from random import Random
from typing import Literal, NoReturn, cast

from hypothesis import HealthCheck, Phase, Verbosity, settings as Settings
from hypothesis._settings import local_settings, note_deprecation
from hypothesis.database import ExampleDatabase, choices_from_bytes, choices_to_bytes
from hypothesis.errors import (
    BackendCannotProceed,
    FlakyBackendFailure,
    HypothesisException,
    InvalidArgument,
    StopTest,
)
from hypothesis.internal.cache import LRUReusedCache
from hypothesis.internal.compat import NotRequired, TypedDict, ceil, override
from hypothesis.internal.conjecture.choice import (
    ChoiceConstraintsT,
    ChoiceKeyT,
    ChoiceNode,
    ChoiceT,
    ChoiceTemplate,
    choices_key,
)
from hypothesis.internal.conjecture.data import (
    ConjectureData,
    ConjectureResult,
    DataObserver,
    Overrun,
    Status,
    _Overrun,
)
from hypothesis.internal.conjecture.datatree import (
    DataTree,
    PreviouslyUnseenBehaviour,
    TreeRecordingObserver,
)
from hypothesis.internal.conjecture.junkdrawer import (
    ensure_free_stackframes,
    startswith,
)
from hypothesis.internal.conjecture.pareto import NO_SCORE, ParetoFront, ParetoOptimiser
from hypothesis.internal.conjecture.providers import (
    AVAILABLE_PROVIDERS,
    HypothesisProvider,
    PrimitiveProvider,
)
from hypothesis.internal.conjecture.shrinker import Shrinker, ShrinkPredicateT, sort_key
from hypothesis.internal.escalation import InterestingOrigin
from hypothesis.internal.healthcheck import fail_health_check
from hypothesis.internal.observability import Observation, with_observability_callback
from hypothesis.reporting import base_report, report

# In most cases, the following constants are all Final. However, we do allow users
# to monkeypatch all of these variables, which means we cannot annotate them as
# Final or mypyc will inline them and render monkeypatching useless.

#: The maximum number of times the shrinker will reduce the complexity of a failing
#: input before giving up. This avoids falling down a trap of exponential (or worse)
#: complexity, where the shrinker appears to be making progress but will take a
#: substantially long time to finish completely.
MAX_SHRINKS: int = 500

# If the shrinking phase takes more than five minutes, abort it early and print
# a warning.   Many CI systems will kill a build after around ten minutes with
# no output, and appearing to hang isn't great for interactive use either -
# showing partially-shrunk examples is better than quitting with no examples!
# (but make it monkeypatchable, for the rare users who need to keep on shrinking)

#: The maximum total time in seconds that the shrinker will try to shrink a failure
#: for before giving up. This is across all shrinks for the same failure, so even
#: if the shrinker successfully reduces the complexity of a single failure several
#: times, it will stop when it hits |MAX_SHRINKING_SECONDS| of total time taken.
MAX_SHRINKING_SECONDS: int = 300

#: The maximum amount of entropy a single test case can use before giving up
#: while making random choices during input generation.
#:
#: The "unit" of one |BUFFER_SIZE| does not have any defined semantics, and you
#: should not rely on it, except that a linear increase |BUFFER_SIZE| will linearly
#: increase the amount of entropy a test case can use during generation.
BUFFER_SIZE: int = 8 * 1024
CACHE_SIZE: int = 10000
MIN_TEST_CALLS: int = 10

# we use this to isolate Hypothesis from interacting with the global random,
# to make it easier to reason about our global random warning logic easier (see
# deprecate_random_in_strategy).
_random = Random()


def shortlex(s):
    return (len(s), s)


@dataclass(slots=True, frozen=False)
class HealthCheckState:
    valid_examples: int = field(default=0)
    invalid_examples: int = field(default=0)
    overrun_examples: int = field(default=0)
    draw_times: defaultdict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @property
    def total_draw_time(self) -> float:
        return math.fsum(sum(self.draw_times.values(), start=[]))

    def timing_report(self) -> str:
        """Return a terminal report describing what was slow."""
        if not self.draw_times:
            return ""
        width = max(
            len(k.removeprefix("generate:").removesuffix(": ")) for k in self.draw_times
        )
        out = [f"\n  {'':^{width}}   count | fraction |    slowest draws (seconds)"]
        args_in_order = sorted(self.draw_times.items(), key=lambda kv: -sum(kv[1]))
        for i, (argname, times) in enumerate(args_in_order):  # pragma: no branch
            # If we have very many unique keys, which can happen due to interactive
            # draws with computed labels, we'll skip uninformative rows.
            if (
                5 <= i < (len(self.draw_times) - 2)
                and math.fsum(times) * 20 < self.total_draw_time
            ):
                out.append(f"  (skipped {len(self.draw_times) - i} rows of fast draws)")
                break
            # Compute the row to report, omitting times <1ms to focus on slow draws
            reprs = [f"{t:>6.3f}," for t in sorted(times)[-5:] if t > 5e-4]
            desc = " ".join((["    -- "] * 5 + reprs)[-5:]).rstrip(",")
            arg = argname.removeprefix("generate:").removesuffix(": ")
            out.append(
                f"  {arg:^{width}} | {len(times):>4}  | "
                f"{math.fsum(times)/self.total_draw_time:>7.0%}  |  {desc}"
            )
        return "\n".join(out)


class ExitReason(Enum):
    max_examples = "settings.max_examples={s.max_examples}"
    max_iterations = (
        "settings.max_examples={s.max_examples}, "
        "but < 10% of examples satisfied assumptions"
    )
    max_shrinks = f"shrunk example {MAX_SHRINKS} times"
    finished = "nothing left to do"
    flaky = "test was flaky"
    very_slow_shrinking = "shrinking was very slow"

    def describe(self, settings: Settings) -> str:
        return self.value.format(s=settings)


class RunIsComplete(Exception):
    pass


def _get_provider(backend: str) -> PrimitiveProvider | type[PrimitiveProvider]:
    provider_cls = AVAILABLE_PROVIDERS[backend]
    if isinstance(provider_cls, str):
        module_name, class_name = provider_cls.rsplit(".", 1)
        provider_cls = getattr(importlib.import_module(module_name), class_name)

    if provider_cls.lifetime == "test_function":
        return provider_cls(None)
    elif provider_cls.lifetime == "test_case":
        return provider_cls
    else:
        raise InvalidArgument(
            f"invalid lifetime {provider_cls.lifetime} for provider {provider_cls.__name__}. "
            "Expected one of 'test_function', 'test_case'."
        )


class CallStats(TypedDict):
    status: str
    runtime: float
    drawtime: float
    gctime: float
    events: list[str]


PhaseStatistics = TypedDict(
    "PhaseStatistics",
    {
        "duration-seconds": float,
        "test-cases": list[CallStats],
        "distinct-failures": int,
        "shrinks-successful": int,
    },
)
StatisticsDict = TypedDict(
    "StatisticsDict",
    {
        "generate-phase": NotRequired[PhaseStatistics],
        "reuse-phase": NotRequired[PhaseStatistics],
        "shrink-phase": NotRequired[PhaseStatistics],
        "stopped-because": NotRequired[str],
        "targets": NotRequired[dict[str, float]],
        "nodeid": NotRequired[str],
    },
)


def choice_count(choices: Sequence[ChoiceT | ChoiceTemplate]) -> int | None:
    count = 0
    for choice in choices:
        if isinstance(choice, ChoiceTemplate):
            if choice.count is None:
                return None
            count += choice.count
        else:
            count += 1
    return count


class DiscardObserver(DataObserver):
    @override
    def kill_branch(self) -> NoReturn:
        raise ContainsDiscard


def realize_choices(data: ConjectureData, *, for_failure: bool) -> None:
    # backwards-compatibility with backends without for_failure, can remove
    # in a few months
    kwargs = {}
    if for_failure:
        if "for_failure" in inspect.signature(data.provider.realize).parameters:
            kwargs["for_failure"] = True
        else:
            note_deprecation(
                f"{type(data.provider).__qualname__}.realize does not have the "
                "for_failure parameter. This will be an error in future versions "
                "of Hypothesis. (If you installed this backend from a separate "
                "package, upgrading that package may help).",
                has_codemod=False,
                since="2025-05-07",
            )

    for node in data.nodes:
        value = data.provider.realize(node.value, **kwargs)
        expected_type = {
            "string": str,
            "float": float,
            "integer": int,
            "boolean": bool,
            "bytes": bytes,
        }[node.type]
        if type(value) is not expected_type:
            raise HypothesisException(
                f"expected {expected_type} from "
                f"{data.provider.realize.__qualname__}, got {type(value)}"
            )

        constraints = cast(
            ChoiceConstraintsT,
            {
                k: data.provider.realize(v, **kwargs)
                for k, v in node.constraints.items()
            },
        )
        node.value = value
        node.constraints = constraints


class ConjectureRunner:
    def __init__(
        self,
        test_function: Callable[[ConjectureData], None],
        *,
        settings: Settings | None = None,
        random: Random | None = None,
        database_key: bytes | None = None,
        ignore_limits: bool = False,
        thread_overlap: dict[int, bool] | None = None,
    ) -> None:
        self._test_function: Callable[[ConjectureData], None] = test_function
        self.settings: Settings = settings or Settings()
        self.shrinks: int = 0
        self.finish_shrinking_deadline: float | None = None
        self.call_count: int = 0
        self.misaligned_count: int = 0
        self.valid_examples: int = 0
        self.invalid_examples: int = 0
        self.overrun_examples: int = 0
        self.random: Random = random or Random(_random.getrandbits(128))
        self.database_key: bytes | None = database_key
        self.ignore_limits: bool = ignore_limits
        self.thread_overlap = {} if thread_overlap is None else thread_overlap

        # Global dict of per-phase statistics, and a list of per-call stats
        # which transfer to the global dict at the end of each phase.
        self._current_phase: str = "(not a phase)"
        self.statistics: StatisticsDict = {}
        self.stats_per_test_case: list[CallStats] = []

        self.interesting_examples: dict[InterestingOrigin, ConjectureResult] = {}
        # We use call_count because there may be few possible valid_examples.
        self.first_bug_found_at: int | None = None
        self.last_bug_found_at: int | None = None
        self.first_bug_found_time: float = math.inf

        self.shrunk_examples: set[InterestingOrigin] = set()
        self.health_check_state: HealthCheckState | None = None
        self.tree: DataTree = DataTree()
        self.provider: PrimitiveProvider | type[PrimitiveProvider] = _get_provider(
            self.settings.backend
        )

        self.best_observed_targets: defaultdict[str, float] = defaultdict(
            lambda: NO_SCORE
        )
        self.best_examples_of_observed_targets: dict[str, ConjectureResult] = {}

        # We keep the pareto front in the example database if we have one. This
        # is only marginally useful at present, but speeds up local development
        # because it means that large targets will be quickly surfaced in your
        # testing.
        self.pareto_front: ParetoFront | None = None
        if self.database_key is not None and self.settings.database is not None:
            self.pareto_front = ParetoFront(self.random)
            self.pareto_front.on_evict(self.on_pareto_evict)

        # We want to be able to get the ConjectureData object that results
        # from running a choice sequence without recalculating, especially during
        # shrinking where we need to know about the structure of the
        # executed test case.
        self.__data_cache = LRUReusedCache[
            tuple[ChoiceKeyT, ...], ConjectureResult | _Overrun
        ](CACHE_SIZE)

        self.reused_previously_shrunk_test_case: bool = False

        self.__pending_call_explanation: str | None = None
        self._backend_found_failure: bool = False
        self._backend_exceeded_deadline: bool = False
        self._switch_to_hypothesis_provider: bool = False

        self.__failed_realize_count: int = 0
        # note unsound verification by alt backends
        self._verified_by: str | None = None

    @contextmanager
    def _with_switch_to_hypothesis_provider(
        self, value: bool
    ) -> Generator[None, None, None]:
        previous = self._switch_to_hypothesis_provider
        try:
            self._switch_to_hypothesis_provider = value
            yield
        finally:
            self._switch_to_hypothesis_provider = previous

    @property
    def using_hypothesis_backend(self) -> bool:
        return (
            self.settings.backend == "hypothesis" or self._switch_to_hypothesis_provider
        )

    def explain_next_call_as(self, explanation: str) -> None:
        self.__pending_call_explanation = explanation

    def clear_call_explanation(self) -> None:
        self.__pending_call_explanation = None

    @contextmanager
    def _log_phase_statistics(
        self, phase: Literal["reuse", "generate", "shrink"]
    ) -> Generator[None, None, None]:
        self.stats_per_test_case.clear()
        start_time = time.perf_counter()
        try:
            self._current_phase = phase
            yield
        finally:
            self.statistics[phase + "-phase"] = {  # type: ignore
                "duration-seconds": time.perf_counter() - start_time,
                "test-cases": list(self.stats_per_test_case),
                "distinct-failures": len(self.interesting_examples),
                "shrinks-successful": self.shrinks,
            }

    @property
    def should_optimise(self) -> bool:
        return Phase.target in self.settings.phases

    def __tree_is_exhausted(self) -> bool:
        return self.tree.is_exhausted and self.using_hypothesis_backend

    def __stoppable_test_function(self, data: ConjectureData) -> None:
        """Run ``self._test_function``, but convert a ``StopTest`` exception
        into a normal return and avoid raising anything flaky for RecursionErrors.
        """
        # We ensure that the test has this much stack space remaining, no
        # matter the size of the stack when called, to de-flake RecursionErrors
        # (#2494, #3671). Note, this covers the data generation part of the test;
        # the actual test execution is additionally protected at the call site
        # in hypothesis.core.execute_once.
        with ensure_free_stackframes():
            try:
                self._test_function(data)
            except StopTest as e:
                if e.testcounter == data.testcounter:
                    # This StopTest has successfully stopped its test, and can now
                    # be discarded.
                    pass
                else:
                    # This StopTest was raised by a different ConjectureData. We
                    # need to re-raise it so that it will eventually reach the
                    # correct engine.
                    raise

    def _cache_key(self, choices: Sequence[ChoiceT]) -> tuple[ChoiceKeyT, ...]:
        return choices_key(choices)

    def _cache(self, data: ConjectureData) -> None:
        result = data.as_result()
        key = self._cache_key(data.choices)
        self.__data_cache[key] = result

    def cached_test_function(
        self,
        choices: Sequence[ChoiceT | ChoiceTemplate],
        *,
        error_on_discard: bool = False,
        extend: int | Literal["full"] = 0,
    ) -> ConjectureResult | _Overrun:
        """
        If ``error_on_discard`` is set to True this will raise ``ContainsDiscard``
        in preference to running the actual test function. This is to allow us
        to skip test cases we expect to be redundant in some cases. Note that
        it may be the case that we don't raise ``ContainsDiscard`` even if the
        result has discards if we cannot determine from previous runs whether
        it will have a discard.
        """
        # node templates represent a not-yet-filled hole and therefore cannot
        # be cached or retrieved from the cache.
        if not any(isinstance(choice, ChoiceTemplate) for choice in choices):
            # this type cast is validated by the isinstance check above (ie, there
            # are no ChoiceTemplate elements).
            choices = cast(Sequence[ChoiceT], choices)
            key = self._cache_key(choices)
            try:
                cached = self.__data_cache[key]
                # if we have a cached overrun for this key, but we're allowing extensions
                # of the nodes, it could in fact run to a valid data if we try.
                if extend == 0 or cached.status is not Status.OVERRUN:
                    return cached
            except KeyError:
                pass

        if extend == "full":
            max_length = None
        elif (count := choice_count(choices)) is None:
            max_length = None
        else:
            max_length = count + extend

        # explicitly use a no-op DataObserver here instead of a TreeRecordingObserver.
        # The reason is we don't expect simulate_test_function to explore new choices
        # and write back to the tree, so we don't want the overhead of the
        # TreeRecordingObserver tracking those calls.
        trial_observer: DataObserver | None = DataObserver()
        if error_on_discard:
            trial_observer = DiscardObserver()

        try:
            trial_data = self.new_conjecture_data(
                choices, observer=trial_observer, max_choices=max_length
            )
            self.tree.simulate_test_function(trial_data)
        except PreviouslyUnseenBehaviour:
            pass
        else:
            trial_data.freeze()
            key = self._cache_key(trial_data.choices)
            if trial_data.status > Status.OVERRUN:
                try:
                    return self.__data_cache[key]
                except KeyError:
                    pass
            else:
                # if we simulated to an overrun, then we our result is certainly
                # an overrun; no need to consult the cache. (and we store this result
                # for simulation-less lookup later).
                self.__data_cache[key] = Overrun
                return Overrun
            try:
                return self.__data_cache[key]
            except KeyError:
                pass

        data = self.new_conjecture_data(choices, max_choices=max_length)
        # note that calling test_function caches `data` for us.
        self.test_function(data)
        return data.as_result()

    def test_function(self, data: ConjectureData) -> None:
        if self.__pending_call_explanation is not None:
            self.debug(self.__pending_call_explanation)
            self.__pending_call_explanation = None

        self.call_count += 1
        interrupted = False

        try:
            self.__stoppable_test_function(data)
        except KeyboardInterrupt:
            interrupted = True
            raise
        except BackendCannotProceed as exc:
            if exc.scope in ("verified", "exhausted"):
                self._switch_to_hypothesis_provider = True
                if exc.scope == "verified":
                    self._verified_by = self.settings.backend
            elif exc.scope == "discard_test_case":
                self.__failed_realize_count += 1
                if (
                    self.__failed_realize_count > 10
                    and (self.__failed_realize_count / self.call_count) > 0.2
                ):
                    self._switch_to_hypothesis_provider = True

            # treat all BackendCannotProceed exceptions as invalid. This isn't
            # great; "verified" should really be counted as self.valid_examples += 1.
            # But we check self.valid_examples == 0 to determine whether to raise
            # Unsatisfiable, and that would throw this check off.
            self.invalid_examples += 1

            # skip the post-test-case tracking; we're pretending this never happened
            interrupted = True
            data.cannot_proceed_scope = exc.scope
            data.freeze()
            return
        except BaseException:
            data.freeze()
            if self.settings.backend != "hypothesis":
                realize_choices(data, for_failure=True)
            self.save_choices(data.choices)
            raise
        finally:
            # No branch, because if we're interrupted we always raise
            # the KeyboardInterrupt, never continue to the code below.
            if not interrupted:  # pragma: no branch
                assert data.cannot_proceed_scope is None
                data.freeze()

                if self.settings.backend != "hypothesis":
                    realize_choices(data, for_failure=data.status is Status.INTERESTING)

                call_stats: CallStats = {
                    "status": data.status.name.lower(),
                    "runtime": data.finish_time - data.start_time,
                    "drawtime": math.fsum(data.draw_times.values()),
                    "gctime": data.gc_finish_time - data.gc_start_time,
                    "events": sorted(
                        k if v == "" else f"{k}: {v}" for k, v in data.events.items()
                    ),
                }
                self.stats_per_test_case.append(call_stats)

                self._cache(data)
                if data.misaligned_at is not None:  # pragma: no branch # coverage bug?
                    self.misaligned_count += 1

        self.debug_data(data)

        if (
            data.target_observations
            and self.pareto_front is not None
            and self.pareto_front.add(data.as_result())
        ):
            self.save_choices(data.choices, sub_key=b"pareto")

        if data.status >= Status.VALID:
            for k, v in data.target_observations.items():
                self.best_observed_targets[k] = max(self.best_observed_targets[k], v)

                if k not in self.best_examples_of_observed_targets:
                    data_as_result = data.as_result()
                    assert not isinstance(data_as_result, _Overrun)
                    self.best_examples_of_observed_targets[k] = data_as_result
                    continue

                existing_example = self.best_examples_of_observed_targets[k]
                existing_score = existing_example.target_observations[k]

                if v < existing_score:
                    continue

                if v > existing_score or sort_key(data.nodes) < sort_key(
                    existing_example.nodes
                ):
                    data_as_result = data.as_result()
                    assert not isinstance(data_as_result, _Overrun)
                    self.best_examples_of_observed_targets[k] = data_as_result

        if data.status is Status.VALID:
            self.valid_examples += 1
        if data.status is Status.INVALID:
            self.invalid_examples += 1
        if data.status is Status.OVERRUN:
            self.overrun_examples += 1

        if data.status == Status.INTERESTING:
            if not self.using_hypothesis_backend:
                # replay this failure on the hypothesis backend to ensure it still
                # finds a failure. otherwise, it is flaky.
                initial_exception = data.expected_exception
                data = ConjectureData.for_choices(data.choices)
                # we've already going to use the hypothesis provider for this
                # data, so the verb "switch" is a bit misleading here. We're really
                # setting this to inform our on_observation logic that the observation
                # generated here was from a hypothesis backend, and shouldn't be
                # sent to the on_observation of any alternative backend.
                with self._with_switch_to_hypothesis_provider(True):
                    self.__stoppable_test_function(data)
                data.freeze()
                # TODO: Should same-origin also be checked? (discussion in
                # https://github.com/HypothesisWorks/hypothesis/pull/4470#discussion_r2217055487)
                if data.status != Status.INTERESTING:
                    desc_new_status = {
                        data.status.VALID: "passed",
                        data.status.INVALID: "failed filters",
                        data.status.OVERRUN: "overran",
                    }[data.status]
                    raise FlakyBackendFailure(
                        f"Inconsistent results from replaying a failing test case! "
                        f"Raised {type(initial_exception).__name__} on "
                        f"backend={self.settings.backend!r}, but "
                        f"{desc_new_status} under backend='hypothesis'.",
                        [initial_exception],
                    )

                self._cache(data)

            assert data.interesting_origin is not None
            key = data.interesting_origin
            changed = False
            try:
                existing = self.interesting_examples[key]
            except KeyError:
                changed = True
                self.last_bug_found_at = self.call_count
                if self.first_bug_found_at is None:
                    self.first_bug_found_at = self.call_count
                    self.first_bug_found_time = time.monotonic()
            else:
                if sort_key(data.nodes) < sort_key(existing.nodes):
                    self.shrinks += 1
                    self.downgrade_choices(existing.choices)
                    self.__data_cache.unpin(self._cache_key(existing.choices))
                    changed = True

            if changed:
                self.save_choices(data.choices)
                self.interesting_examples[key] = data.as_result()  # type: ignore
                if not self.using_hypothesis_backend:
                    self._backend_found_failure = True
                self.__data_cache.pin(self._cache_key(data.choices), data.as_result())
                self.shrunk_examples.discard(key)

            if self.shrinks >= MAX_SHRINKS:
                self.exit_with(ExitReason.max_shrinks)

        if (
            not self.ignore_limits
            and self.finish_shrinking_deadline is not None
            and self.finish_shrinking_deadline < time.perf_counter()
        ):
            # See https://github.com/HypothesisWorks/hypothesis/issues/2340
            report(
                "WARNING: Hypothesis has spent more than five minutes working to shrink"
                " a failing example, and stopped because it is making very slow"
                " progress.  When you re-run your tests, shrinking will resume and may"
                " take this long before aborting again.\nPLEASE REPORT THIS if you can"
                " provide a reproducing example, so that we can improve shrinking"
                " performance for everyone."
            )
            self.exit_with(ExitReason.very_slow_shrinking)

        if not self.interesting_examples:
            # Note that this logic is reproduced to end the generation phase when
            # we have interesting examples.  Update that too if you change this!
            # (The doubled implementation is because here we exit the engine entirely,
            #  while in the other case below we just want to move on to shrinking.)
            if self.valid_examples >= self.settings.max_examples:
                self.exit_with(ExitReason.max_examples)
            if self.call_count >= max(
                self.settings.max_examples * 10,
                # We have a high-ish default max iterations, so that tests
                # don't become flaky when max_examples is too low.
                1000,
            ):
                self.exit_with(ExitReason.max_iterations)

        if self.__tree_is_exhausted():
            self.exit_with(ExitReason.finished)

        self.record_for_health_check(data)

    def on_pareto_evict(self, data: ConjectureResult) -> None:
        self.settings.database.delete(self.pareto_key, choices_to_bytes(data.choices))

    def generate_novel_prefix(self) -> tuple[ChoiceT, ...]:
        """Uses the tree to proactively generate a starting choice sequence
        that we haven't explored yet for this test.

        When this method is called, we assume that there must be at
        least one novel prefix left to find. If there were not, then the
        test run should have already stopped due to tree exhaustion.
        """
        return self.tree.generate_novel_prefix(self.random)

    def record_for_health_check(self, data: ConjectureData) -> None:
        # Once we've actually found a bug, there's no point in trying to run
        # health checks - they'll just mask the actually important information.
        if data.status == Status.INTERESTING:
            self.health_check_state = None

        state = self.health_check_state

        if state is None:
            return

        for k, v in data.draw_times.items():
            state.draw_times[k].append(v)

        if data.status == Status.VALID:
            state.valid_examples += 1
        elif data.status == Status.INVALID:
            state.invalid_examples += 1
        else:
            assert data.status == Status.OVERRUN
            state.overrun_examples += 1

        max_valid_draws = 10
        max_invalid_draws = 50
        max_overrun_draws = 20

        assert state.valid_examples <= max_valid_draws

        if state.valid_examples == max_valid_draws:
            self.health_check_state = None
            return

        if state.overrun_examples == max_overrun_draws:
            fail_health_check(
                self.settings,
                "Generated inputs routinely consumed more than the maximum "
                f"allowed entropy: {state.valid_examples} inputs were generated "
                f"successfully, while {state.overrun_examples} inputs exceeded the "
                f"maximum allowed entropy during generation."
                "\n\n"
                f"Testing with inputs this large tends to be slow, and to produce "
                "failures that are both difficult to shrink and difficult to understand. "
                "Try decreasing the amount of data generated, for example by "
                "decreasing the minimum size of collection strategies like "
                "st.lists()."
                "\n\n"
                "If you expect the average size of your input to be this large, "
                "you can disable this health check with "
                "@settings(suppress_health_check=[HealthCheck.data_too_large]). "
                "See "
                "https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck "
                "for details.",
                HealthCheck.data_too_large,
            )
        if state.invalid_examples == max_invalid_draws:
            fail_health_check(
                self.settings,
                "It looks like this test is filtering out a lot of inputs. "
                f"{state.valid_examples} inputs were generated successfully, "
                f"while {state.invalid_examples} inputs were filtered out. "
                "\n\n"
                "An input might be filtered out by calls to assume(), "
                "strategy.filter(...), or occasionally by Hypothesis internals."
                "\n\n"
                "Applying this much filtering makes input generation slow, since "
                "Hypothesis must discard inputs which are filtered out and try "
                "generating it again. It is also possible that applying this much "
                "filtering will distort the domain and/or distribution of the test, "
                "leaving your testing less rigorous than expected."
                "\n\n"
                "If you expect this many inputs to be filtered out during generation, "
                "you can disable this health check with "
                "@settings(suppress_health_check=[HealthCheck.filter_too_much]). See "
                "https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck "
                "for details.",
                HealthCheck.filter_too_much,
            )

        # Allow at least the greater of one second or 5x the deadline.  If deadline
        # is None, allow 30s - the user can disable the healthcheck too if desired.
        draw_time = state.total_draw_time
        draw_time_limit = 5 * (self.settings.deadline or timedelta(seconds=6))
        if (
            draw_time > max(1.0, draw_time_limit.total_seconds())
            # we disable HealthCheck.too_slow under concurrent threads, since
            # cpython may switch away from a thread for arbitrarily long.
            and not self.thread_overlap.get(threading.get_ident(), False)
        ):
            extra_str = []
            if state.invalid_examples:
                extra_str.append(f"{state.invalid_examples} invalid inputs")
            if state.overrun_examples:
                extra_str.append(
                    f"{state.overrun_examples} inputs which exceeded the "
                    "maximum allowed entropy"
                )
            extra_str = ", and ".join(extra_str)
            extra_str = f" ({extra_str})" if extra_str else ""

            fail_health_check(
                self.settings,
                "Input generation is slow: Hypothesis only generated "
                f"{state.valid_examples} valid inputs after {draw_time:.2f} "
                f"seconds{extra_str}."
                "\n" + state.timing_report() + "\n\n"
                "This could be for a few reasons:"
                "\n"
                "1. This strategy could be generating too much data per input. "
                "Try decreasing the amount of data generated, for example by "
                "decreasing the minimum size of collection strategies like "
                "st.lists()."
                "\n"
                "2. Some other expensive computation could be running during input "
                "generation. For example, "
                "if @st.composite or st.data() is interspersed with an expensive "
                "computation, HealthCheck.too_slow is likely to trigger. If this "
                "computation is unrelated to input generation, move it elsewhere. "
                "Otherwise, try making it more efficient, or disable this health "
                "check if that is not possible."
                "\n\n"
                "If you expect input generation to take this long, you can disable "
                "this health check with "
                "@settings(suppress_health_check=[HealthCheck.too_slow]). See "
                "https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck "
                "for details.",
                HealthCheck.too_slow,
            )

    def save_choices(
        self, choices: Sequence[ChoiceT], sub_key: bytes | None = None
    ) -> None:
        if self.settings.database is not None:
            key = self.sub_key(sub_key)
            if key is None:
                return
            self.settings.database.save(key, choices_to_bytes(choices))

    def downgrade_choices(self, choices: Sequence[ChoiceT]) -> None:
        buffer = choices_to_bytes(choices)
        if self.settings.database is not None and self.database_key is not None:
            self.settings.database.move(self.database_key, self.secondary_key, buffer)

    def sub_key(self, sub_key: bytes | None) -> bytes | None:
        if self.database_key is None:
            return None
        if sub_key is None:
            return self.database_key
        return b".".join((self.database_key, sub_key))

    @property
    def secondary_key(self) -> bytes | None:
        return self.sub_key(b"secondary")

    @property
    def pareto_key(self) -> bytes | None:
        return self.sub_key(b"pareto")

    def debug(self, message: str) -> None:
        if self.settings.verbosity >= Verbosity.debug:
            base_report(message)

    @property
    def report_debug_info(self) -> bool:
        return self.settings.verbosity >= Verbosity.debug

    def debug_data(self, data: ConjectureData | ConjectureResult) -> None:
        if not self.report_debug_info:
            return

        status = repr(data.status)
        if data.status == Status.INTERESTING:
            status = f"{status} ({data.interesting_origin!r})"

        self.debug(
            f"{len(data.choices)} choices {data.choices} -> {status}"
            f"{', ' + data.output if data.output else ''}"
        )

    def observe_for_provider(self) -> AbstractContextManager:
        def on_observation(observation: Observation) -> None:
            assert observation.type == "test_case"
            # because lifetime == "test_function"
            assert isinstance(self.provider, PrimitiveProvider)
            # only fire if we actually used that provider to generate this observation
            if not self._switch_to_hypothesis_provider:
                self.provider.on_observation(observation)

        if (
            self.settings.backend != "hypothesis"
            # only for lifetime = "test_function" providers (guaranteed
            # by this isinstance check)
            and isinstance(self.provider, PrimitiveProvider)
            # and the provider opted-in to observations
            and self.provider.add_observability_callback
        ):
            return with_observability_callback(on_observation)
        return nullcontext()

    def run(self) -> None:
        with local_settings(self.settings), self.observe_for_provider():
            try:
                self._run()
            except RunIsComplete:
                pass
            for v in self.interesting_examples.values():
                self.debug_data(v)
            self.debug(
                f"Run complete after {self.call_count} examples "
                f"({self.valid_examples} valid) and {self.shrinks} shrinks"
            )

    @property
    def database(self) -> ExampleDatabase | None:
        if self.database_key is None:
            return None
        return self.settings.database

    def has_existing_examples(self) -> bool:
        return self.database is not None and Phase.reuse in self.settings.phases

    def reuse_existing_examples(self) -> None:
        """If appropriate (we have a database and have been told to use it),
        try to reload existing examples from the database.

        If there are a lot we don't try all of them. We always try the
        smallest example in the database (which is guaranteed to be the
        last failure) and the largest (which is usually the seed example
        which the last failure came from but we don't enforce that). We
        then take a random sampling of the remainder and try those. Any
        examples that are no longer interesting are cleared out.
        """
        if self.has_existing_examples():
            self.debug("Reusing examples from database")
            # We have to do some careful juggling here. We have two database
            # corpora: The primary and secondary. The primary corpus is a
            # small set of minimized examples each of which has at one point
            # demonstrated a distinct bug. We want to retry all of these.

            # We also have a secondary corpus of examples that have at some
            # point demonstrated interestingness (currently only ones that
            # were previously non-minimal examples of a bug, but this will
            # likely expand in future). These are a good source of potentially
            # interesting examples, but there are a lot of them, so we down
            # sample the secondary corpus to a more manageable size.

            corpus = sorted(
                self.settings.database.fetch(self.database_key), key=shortlex
            )
            factor = 0.1 if (Phase.generate in self.settings.phases) else 1
            desired_size = max(2, ceil(factor * self.settings.max_examples))
            primary_corpus_size = len(corpus)

            if len(corpus) < desired_size:
                extra_corpus = list(self.settings.database.fetch(self.secondary_key))

                shortfall = desired_size - len(corpus)

                if len(extra_corpus) <= shortfall:
                    extra = extra_corpus
                else:
                    extra = self.random.sample(extra_corpus, shortfall)
                extra.sort(key=shortlex)
                corpus.extend(extra)

            # We want a fast path where every primary entry in the database was
            # interesting.
            found_interesting_in_primary = False
            all_interesting_in_primary_were_exact = True

            for i, existing in enumerate(corpus):
                if i >= primary_corpus_size and found_interesting_in_primary:
                    break
                choices = choices_from_bytes(existing)
                if choices is None:
                    # clear out any keys which fail deserialization
                    self.settings.database.delete(self.database_key, existing)
                    continue
                data = self.cached_test_function(choices, extend="full")
                if data.status != Status.INTERESTING:
                    self.settings.database.delete(self.database_key, existing)
                    self.settings.database.delete(self.secondary_key, existing)
                else:
                    if i < primary_corpus_size:
                        found_interesting_in_primary = True
                        assert not isinstance(data, _Overrun)
                        if choices_key(choices) != choices_key(data.choices):
                            all_interesting_in_primary_were_exact = False
                    if not self.settings.report_multiple_bugs:
                        break
            if found_interesting_in_primary:
                if all_interesting_in_primary_were_exact:
                    self.reused_previously_shrunk_test_case = True

            # Because self.database is not None (because self.has_existing_examples())
            # and self.database_key is not None (because we fetched using it above),
            # we can guarantee self.pareto_front is not None
            assert self.pareto_front is not None

            # If we've not found any interesting examples so far we try some of
            # the pareto front from the last run.
            if len(corpus) < desired_size and not self.interesting_examples:
                desired_extra = desired_size - len(corpus)
                pareto_corpus = list(self.settings.database.fetch(self.pareto_key))
                if len(pareto_corpus) > desired_extra:
                    pareto_corpus = self.random.sample(pareto_corpus, desired_extra)
                pareto_corpus.sort(key=shortlex)

                for existing in pareto_corpus:
                    choices = choices_from_bytes(existing)
                    if choices is None:
                        self.settings.database.delete(self.pareto_key, existing)
                        continue
                    data = self.cached_test_function(choices, extend="full")
                    if data not in self.pareto_front:
                        self.settings.database.delete(self.pareto_key, existing)
                    if data.status == Status.INTERESTING:
                        break

    def exit_with(self, reason: ExitReason) -> None:
        if self.ignore_limits:
            return
        self.statistics["stopped-because"] = reason.describe(self.settings)
        if self.best_observed_targets:
            self.statistics["targets"] = dict(self.best_observed_targets)
        self.debug(f"exit_with({reason.name})")
        self.exit_reason = reason
        raise RunIsComplete

    def should_generate_more(self) -> bool:
        # End the generation phase where we would have ended it if no bugs had
        # been found.  This reproduces the exit logic in `self.test_function`,
        # but with the important distinction that this clause will move on to
        # the shrinking phase having found one or more bugs, while the other
        # will exit having found zero bugs.
        if self.valid_examples >= self.settings.max_examples or self.call_count >= max(
            self.settings.max_examples * 10, 1000
        ):  # pragma: no cover
            return False

        # If we haven't found a bug, keep looking - if we hit any limits on
        # the number of tests to run that will raise an exception and stop
        # the run.
        if not self.interesting_examples:
            return True
        # Users who disable shrinking probably want to exit as fast as possible.
        # If we've found a bug and won't report more than one, stop looking.
        # If we first saw a bug more than 10 seconds ago, stop looking.
        elif (
            Phase.shrink not in self.settings.phases
            or not self.settings.report_multiple_bugs
            or time.monotonic() - self.first_bug_found_time > 10
        ):
            return False
        assert isinstance(self.first_bug_found_at, int)
        assert isinstance(self.last_bug_found_at, int)
        assert self.first_bug_found_at <= self.last_bug_found_at <= self.call_count
        # Otherwise, keep searching for between ten and 'a heuristic' calls.
        # We cap 'calls after first bug' so errors are reported reasonably
        # soon even for tests that are allowed to run for a very long time,
        # or sooner if the latest half of our test effort has been fruitless.
        return self.call_count < MIN_TEST_CALLS or self.call_count < min(
            self.first_bug_found_at + 1000, self.last_bug_found_at * 2
        )

    def generate_new_examples(self) -> None:
        if Phase.generate not in self.settings.phases:
            return
        if self.interesting_examples:
            # The example database has failing examples from a previous run,
            # so we'd rather report that they're still failing ASAP than take
            # the time to look for additional failures.
            return

        self.debug("Generating new examples")

        assert self.should_generate_more()
        self._switch_to_hypothesis_provider = True
        zero_data = self.cached_test_function((ChoiceTemplate("simplest", count=None),))
        if zero_data.status > Status.OVERRUN:
            assert isinstance(zero_data, ConjectureResult)
            # if the crosshair backend cannot proceed, it does not (and cannot)
            # realize the symbolic values, with the intent that Hypothesis will
            # throw away this test case. We usually do, but if it's the zero data
            # then we try to pin it here, which requires realizing the symbolics.
            #
            # We don't (yet) rely on the zero data being pinned, and so
            # it's simply a very slight performance loss to simply not pin it
            # if doing so would error.
            if zero_data.cannot_proceed_scope is None:  # pragma: no branch
                self.__data_cache.pin(
                    self._cache_key(zero_data.choices), zero_data.as_result()
                )  # Pin forever

        if zero_data.status == Status.OVERRUN or (
            zero_data.status == Status.VALID
            and isinstance(zero_data, ConjectureResult)
            and zero_data.length * 2 > BUFFER_SIZE
        ):
            fail_health_check(
                self.settings,
                "The smallest natural input for this test is very "
                "large. This makes it difficult for Hypothesis to generate "
                "good inputs, especially when trying to shrink failing inputs."
                "\n\n"
                "Consider reducing the amount of data generated by the strategy. "
                "Also consider introducing small alternative values for some "
                "strategies. For example, could you "
                "mark some arguments as optional by replacing `some_complex_strategy`"
                "with `st.none() | some_complex_strategy`?"
                "\n\n"
                "If you are confident that the size of the smallest natural input "
                "to your test cannot be reduced, you can suppress this health check "
                "with @settings(suppress_health_check=[HealthCheck.large_base_example]). "
                "See "
                "https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck "
                "for details.",
                HealthCheck.large_base_example,
            )

        self.health_check_state = HealthCheckState()

        # We attempt to use the size of the minimal generated test case starting
        # from a given novel prefix as a guideline to generate smaller test
        # cases for an initial period, by restriscting ourselves to test cases
        # that are not much larger than it.
        #
        # Calculating the actual minimal generated test case is hard, so we
        # take a best guess that zero extending a prefix produces the minimal
        # test case starting with that prefix (this is true for our built in
        # strategies). This is only a reasonable thing to do if the resulting
        # test case is valid. If we regularly run into situations where it is
        # not valid then this strategy is a waste of time, so we want to
        # abandon it early. In order to do this we track how many times in a
        # row it has failed to work, and abort small test case generation when
        # it has failed too many times in a row.
        consecutive_zero_extend_is_invalid = 0

        # We control growth during initial example generation, for two
        # reasons:
        #
        # * It gives us an opportunity to find small examples early, which
        #   gives us a fast path for easy to find bugs.
        # * It avoids low probability events where we might end up
        #   generating very large examples during health checks, which
        #   on slower machines can trigger HealthCheck.too_slow.
        #
        # The heuristic we use is that we attempt to estimate the smallest
        # extension of this prefix, and limit the size to no more than
        # an order of magnitude larger than that. If we fail to estimate
        # the size accurately, we skip over this prefix and try again.
        #
        # We need to tune the example size based on the initial prefix,
        # because any fixed size might be too small, and any size based
        # on the strategy in general can fall afoul of strategies that
        # have very different sizes for different prefixes.
        #
        # We previously set a minimum value of 10 on small_example_cap, with the
        # reasoning of avoiding flaky health checks. However, some users set a
        # low max_examples for performance. A hard lower bound in this case biases
        # the distribution towards small (and less powerful) examples. Flaky
        # and loud health checks are better than silent performance degradation.
        small_example_cap = min(self.settings.max_examples // 10, 50)
        optimise_at = max(self.settings.max_examples // 2, small_example_cap + 1, 10)
        ran_optimisations = False
        self._switch_to_hypothesis_provider = False

        while self.should_generate_more():
            # we don't yet integrate DataTree with backends. Instead of generating
            # a novel prefix, ask the backend for an input.
            if not self.using_hypothesis_backend:
                data = self.new_conjecture_data([])
                with suppress(BackendCannotProceed):
                    self.test_function(data)
                continue

            self._current_phase = "generate"
            prefix = self.generate_novel_prefix()
            if (
                self.valid_examples <= small_example_cap
                and self.call_count <= 5 * small_example_cap
                and not self.interesting_examples
                and consecutive_zero_extend_is_invalid < 5
            ):
                minimal_example = self.cached_test_function(
                    prefix + (ChoiceTemplate("simplest", count=None),)
                )

                if minimal_example.status < Status.VALID:
                    consecutive_zero_extend_is_invalid += 1
                    continue
                # Because the Status code is greater than Status.VALID, it cannot be
                # Status.OVERRUN, which guarantees that the minimal_example is a
                # ConjectureResult object.
                assert isinstance(minimal_example, ConjectureResult)
                consecutive_zero_extend_is_invalid = 0
                minimal_extension = len(minimal_example.choices) - len(prefix)
                max_length = len(prefix) + minimal_extension * 5

                # We could end up in a situation where even though the prefix was
                # novel when we generated it, because we've now tried zero extending
                # it not all possible continuations of it will be novel. In order to
                # avoid making redundant test calls, we rerun it in simulation mode
                # first. If this has a predictable result, then we don't bother
                # running the test function for real here. If however we encounter
                # some novel behaviour, we try again with the real test function,
                # starting from the new novel prefix that has discovered.
                trial_data = self.new_conjecture_data(prefix, max_choices=max_length)
                try:
                    self.tree.simulate_test_function(trial_data)
                    continue
                except PreviouslyUnseenBehaviour:
                    pass

                # If the simulation entered part of the tree that has been killed,
                # we don't want to run this.
                assert isinstance(trial_data.observer, TreeRecordingObserver)
                if trial_data.observer.killed:
                    continue

                # We might have hit the cap on number of examples we should
                # run when calculating the minimal example.
                if not self.should_generate_more():
                    break

                prefix = trial_data.choices
            else:
                max_length = None

            data = self.new_conjecture_data(prefix, max_choices=max_length)
            self.test_function(data)

            if (
                data.status is Status.OVERRUN
                and max_length is not None
                and "invalid because" not in data.events
            ):
                data.events["invalid because"] = (
                    "reduced max size for early examples (avoids flaky health checks)"
                )

            self.generate_mutations_from(data)

            # Although the optimisations are logically a distinct phase, we
            # actually normally run them as part of example generation. The
            # reason for this is that we cannot guarantee that optimisation
            # actually exhausts our budget: It might finish running and we
            # discover that actually we still could run a bunch more test cases
            # if we want.
            if (
                self.valid_examples >= max(small_example_cap, optimise_at)
                and not ran_optimisations
            ):
                ran_optimisations = True
                self._current_phase = "target"
                self.optimise_targets()

    def generate_mutations_from(self, data: ConjectureData | ConjectureResult) -> None:
        # A thing that is often useful but rarely happens by accident is
        # to generate the same value at multiple different points in the
        # test case.
        #
        # Rather than make this the responsibility of individual strategies
        # we implement a small mutator that just takes parts of the test
        # case with the same label and tries replacing one of them with a
        # copy of the other and tries running it. If we've made a good
        # guess about what to put where, this will run a similar generated
        # test case with more duplication.
        if (
            # An OVERRUN doesn't have enough information about the test
            # case to mutate, so we just skip those.
            data.status >= Status.INVALID
            # This has a tendency to trigger some weird edge cases during
            # generation so we don't let it run until we're done with the
            # health checks.
            and self.health_check_state is None
        ):
            initial_calls = self.call_count
            failed_mutations = 0

            while (
                self.should_generate_more()
                # We implement fairly conservative checks for how long we
                # we should run mutation for, as it's generally not obvious
                # how helpful it is for any given test case.
                and self.call_count <= initial_calls + 5
                and failed_mutations <= 5
            ):
                groups = data.spans.mutator_groups
                if not groups:
                    break

                group = self.random.choice(groups)
                (start1, end1), (start2, end2) = self.random.sample(sorted(group), 2)
                if start1 > start2:
                    (start1, end1), (start2, end2) = (start2, end2), (start1, end1)

                if (
                    start1 <= start2 <= end2 <= end1
                ):  # pragma: no cover  # flaky on conjecture-cover tests
                    # One span entirely contains the other. The strategy is very
                    # likely some kind of tree. e.g. we might have
                    #
                    #                   ┌─────┐
                    #             ┌─────┤  a  ├──────┐
                    #             │     └─────┘      │
                    #          ┌──┴──┐            ┌──┴──┐
                    #       ┌──┤  b  ├──┐      ┌──┤  c  ├──┐
                    #       │  └──┬──┘  │      │  └──┬──┘  │
                    #     ┌─┴─┐ ┌─┴─┐ ┌─┴─┐  ┌─┴─┐ ┌─┴─┐ ┌─┴─┐
                    #     │ d │ │ e │ │ f │  │ g │ │ h │ │ i │
                    #     └───┘ └───┘ └───┘  └───┘ └───┘ └───┘
                    #
                    # where each node is drawn from the same strategy and so
                    # has the same span label. We might have selected the spans
                    # corresponding to the a and c nodes, which is the entire
                    # tree and the subtree of (and including) c respectively.
                    #
                    # There are two possible mutations we could apply in this case:
                    # 1. replace a with c (replace child with parent)
                    # 2. replace c with a (replace parent with child)
                    #
                    # (1) results in multiple partial copies of the
                    # parent:
                    #                 ┌─────┐
                    #           ┌─────┤  a  ├────────────┐
                    #           │     └─────┘            │
                    #        ┌──┴──┐                   ┌─┴───┐
                    #     ┌──┤  b  ├──┐          ┌─────┤  a  ├──────┐
                    #     │  └──┬──┘  │          │     └─────┘      │
                    #   ┌─┴─┐ ┌─┴─┐ ┌─┴─┐     ┌──┴──┐            ┌──┴──┐
                    #   │ d │ │ e │ │ f │  ┌──┤  b  ├──┐      ┌──┤  c  ├──┐
                    #   └───┘ └───┘ └───┘  │  └──┬──┘  │      │  └──┬──┘  │
                    #                    ┌─┴─┐ ┌─┴─┐ ┌─┴─┐  ┌─┴─┐ ┌─┴─┐ ┌─┴─┐
                    #                    │ d │ │ e │ │ f │  │ g │ │ h │ │ i │
                    #                    └───┘ └───┘ └───┘  └───┘ └───┘ └───┘
                    #
                    # While (2) results in truncating part of the parent:
                    #
                    #                    ┌─────┐
                    #                 ┌──┤  c  ├──┐
                    #                 │  └──┬──┘  │
                    #               ┌─┴─┐ ┌─┴─┐ ┌─┴─┐
                    #               │ g │ │ h │ │ i │
                    #               └───┘ └───┘ └───┘
                    #
                    # (1) is the same as Example IV.4. in Nautilus (NDSS '19)
                    # (https://wcventure.github.io/FuzzingPaper/Paper/NDSS19_Nautilus.pdf),
                    # except we do not repeat the replacement additional times
                    # (the paper repeats it once for a total of two copies).
                    #
                    # We currently only apply mutation (1), and ignore mutation
                    # (2). The reason is that the attempt generated from (2) is
                    # always something that Hypothesis could easily have generated
                    # itself, by simply not making various choices. Whereas
                    # duplicating the exact value + structure of particular choices
                    # in (1) would have been hard for Hypothesis to generate by
                    # chance.
                    #
                    # TODO: an extension of this mutation might repeat (1) on
                    # a geometric distribution between 0 and ~10 times. We would
                    # need to find the corresponding span to recurse on in the new
                    # choices, probably just by using the choices index.

                    # case (1): duplicate the choices in start1:start2.
                    attempt = data.choices[:start2] + data.choices[start1:]
                else:
                    (start, end) = self.random.choice([(start1, end1), (start2, end2)])
                    replacement = data.choices[start:end]
                    # We attempt to replace both the examples with
                    # whichever choice we made. Note that this might end
                    # up messing up and getting the example boundaries
                    # wrong - labels matching are only a best guess as to
                    # whether the two are equivalent - but it doesn't
                    # really matter. It may not achieve the desired result,
                    # but it's still a perfectly acceptable choice sequence
                    # to try.
                    attempt = (
                        data.choices[:start1]
                        + replacement
                        + data.choices[end1:start2]
                        + replacement
                        + data.choices[end2:]
                    )

                try:
                    new_data = self.cached_test_function(
                        attempt,
                        # We set error_on_discard so that we don't end up
                        # entering parts of the tree we consider redundant
                        # and not worth exploring.
                        error_on_discard=True,
                    )
                except ContainsDiscard:
                    failed_mutations += 1
                    continue

                if new_data is Overrun:
                    failed_mutations += 1  # pragma: no cover # annoying case
                else:
                    assert isinstance(new_data, ConjectureResult)
                    if (
                        new_data.status >= data.status
                        and choices_key(data.choices) != choices_key(new_data.choices)
                        and all(
                            k in new_data.target_observations
                            and new_data.target_observations[k] >= v
                            for k, v in data.target_observations.items()
                        )
                    ):
                        data = new_data
                        failed_mutations = 0
                    else:
                        failed_mutations += 1

    def optimise_targets(self) -> None:
        """If any target observations have been made, attempt to optimise them
        all."""
        if not self.should_optimise:
            return
        from hypothesis.internal.conjecture.optimiser import Optimiser

        # We want to avoid running the optimiser for too long in case we hit
        # an unbounded target score. We start this off fairly conservatively
        # in case interesting examples are easy to find and then ramp it up
        # on an exponential schedule so we don't hamper the optimiser too much
        # if it needs a long time to find good enough improvements.
        max_improvements = 10
        while True:
            prev_calls = self.call_count

            any_improvements = False

            for target, data in list(self.best_examples_of_observed_targets.items()):
                optimiser = Optimiser(
                    self, data, target, max_improvements=max_improvements
                )
                optimiser.run()
                if optimiser.improvements > 0:
                    any_improvements = True

            if self.interesting_examples:
                break

            max_improvements *= 2

            if any_improvements:
                continue

            if self.best_observed_targets:
                self.pareto_optimise()

            if prev_calls == self.call_count:
                break

    def pareto_optimise(self) -> None:
        if self.pareto_front is not None:
            ParetoOptimiser(self).run()

    def _run(self) -> None:
        # have to use the primitive provider to interpret database bits...
        self._switch_to_hypothesis_provider = True
        with self._log_phase_statistics("reuse"):
            self.reuse_existing_examples()
        # Fast path for development: If the database gave us interesting
        # examples from the previously stored primary key, don't try
        # shrinking it again as it's unlikely to work.
        if self.reused_previously_shrunk_test_case:
            self.exit_with(ExitReason.finished)
        # ...but we should use the supplied provider when generating...
        self._switch_to_hypothesis_provider = False
        with self._log_phase_statistics("generate"):
            self.generate_new_examples()
            # We normally run the targeting phase mixed in with the generate phase,
            # but if we've been asked to run it but not generation then we have to
            # run it explicitly on its own here.
            if Phase.generate not in self.settings.phases:
                self._current_phase = "target"
                self.optimise_targets()
        # ...and back to the primitive provider when shrinking.
        self._switch_to_hypothesis_provider = True
        with self._log_phase_statistics("shrink"):
            self.shrink_interesting_examples()
        self.exit_with(ExitReason.finished)

    def new_conjecture_data(
        self,
        prefix: Sequence[ChoiceT | ChoiceTemplate],
        *,
        observer: DataObserver | None = None,
        max_choices: int | None = None,
    ) -> ConjectureData:
        provider = (
            HypothesisProvider if self._switch_to_hypothesis_provider else self.provider
        )
        observer = observer or self.tree.new_observer()
        if not self.using_hypothesis_backend:
            observer = DataObserver()

        return ConjectureData(
            prefix=prefix,
            observer=observer,
            provider=provider,
            max_choices=max_choices,
            random=self.random,
        )

    def shrink_interesting_examples(self) -> None:
        """If we've found interesting examples, try to replace each of them
        with a minimal interesting example with the same interesting_origin.

        We may find one or more examples with a new interesting_origin
        during the shrink process. If so we shrink these too.
        """
        if Phase.shrink not in self.settings.phases or not self.interesting_examples:
            return

        self.debug("Shrinking interesting examples")
        self.finish_shrinking_deadline = time.perf_counter() + MAX_SHRINKING_SECONDS

        for prev_data in sorted(
            self.interesting_examples.values(), key=lambda d: sort_key(d.nodes)
        ):
            assert prev_data.status == Status.INTERESTING
            data = self.new_conjecture_data(prev_data.choices)
            self.test_function(data)
            if data.status != Status.INTERESTING:
                self.exit_with(ExitReason.flaky)

        self.clear_secondary_key()

        while len(self.shrunk_examples) < len(self.interesting_examples):
            target, example = min(
                (
                    (k, v)
                    for k, v in self.interesting_examples.items()
                    if k not in self.shrunk_examples
                ),
                key=lambda kv: (sort_key(kv[1].nodes), shortlex(repr(kv[0]))),
            )
            self.debug(f"Shrinking {target!r}: {example.choices}")

            if not self.settings.report_multiple_bugs:
                # If multi-bug reporting is disabled, we shrink our currently-minimal
                # failure, allowing 'slips' to any bug with a smaller minimal example.
                self.shrink(example, lambda d: d.status == Status.INTERESTING)
                return

            def predicate(d: ConjectureResult | _Overrun) -> bool:
                if d.status < Status.INTERESTING:
                    return False
                d = cast(ConjectureResult, d)
                return d.interesting_origin == target

            self.shrink(example, predicate)

            self.shrunk_examples.add(target)

    def clear_secondary_key(self) -> None:
        if self.has_existing_examples():
            # If we have any smaller examples in the secondary corpus, now is
            # a good time to try them to see if they work as shrinks. They
            # probably won't, but it's worth a shot and gives us a good
            # opportunity to clear out the database.

            # It's not worth trying the primary corpus because we already
            # tried all of those in the initial phase.
            corpus = sorted(
                self.settings.database.fetch(self.secondary_key), key=shortlex
            )
            for c in corpus:
                choices = choices_from_bytes(c)
                if choices is None:
                    self.settings.database.delete(self.secondary_key, c)
                    continue
                primary = {
                    choices_to_bytes(v.choices)
                    for v in self.interesting_examples.values()
                }
                if shortlex(c) > max(map(shortlex, primary)):
                    break

                self.cached_test_function(choices)
                # We unconditionally remove c from the secondary key as it
                # is either now primary or worse than our primary example
                # of this reason for interestingness.
                self.settings.database.delete(self.secondary_key, c)

    def shrink(
        self,
        example: ConjectureData | ConjectureResult,
        predicate: ShrinkPredicateT | None = None,
        allow_transition: (
            Callable[[ConjectureData | ConjectureResult, ConjectureData], bool] | None
        ) = None,
    ) -> ConjectureData | ConjectureResult:
        s = self.new_shrinker(example, predicate, allow_transition)
        s.shrink()
        return s.shrink_target

    def new_shrinker(
        self,
        example: ConjectureData | ConjectureResult,
        predicate: ShrinkPredicateT | None = None,
        allow_transition: (
            Callable[[ConjectureData | ConjectureResult, ConjectureData], bool] | None
        ) = None,
    ) -> Shrinker:
        return Shrinker(
            self,
            example,
            predicate,
            allow_transition=allow_transition,
            explain=Phase.explain in self.settings.phases,
            in_target_phase=self._current_phase == "target",
        )

    def passing_choice_sequences(
        self, prefix: Sequence[ChoiceNode] = ()
    ) -> frozenset[tuple[ChoiceNode, ...]]:
        """Return a collection of choice sequence nodes which cause the test to pass.
        Optionally restrict this by a certain prefix, which is useful for explain mode.
        """
        return frozenset(
            cast(ConjectureResult, result).nodes
            for key in self.__data_cache
            if (result := self.__data_cache[key]).status is Status.VALID
            and startswith(cast(ConjectureResult, result).nodes, prefix)
        )


class ContainsDiscard(Exception):
    pass
