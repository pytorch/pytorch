# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Observability tools to spit out analysis-ready tables, one row per test case."""

import base64
import dataclasses
import json
import math
import os
import sys
import threading
import time
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from threading import Lock
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    TypeAlias,
    Union,
    cast,
)

from hypothesis.configuration import storage_directory
from hypothesis.errors import HypothesisWarning
from hypothesis.internal.conjecture.choice import (
    BooleanConstraints,
    BytesConstraints,
    ChoiceConstraintsT,
    ChoiceNode,
    ChoiceT,
    ChoiceTypeT,
    FloatConstraints,
    IntegerConstraints,
    StringConstraints,
)
from hypothesis.internal.escalation import InterestingOrigin
from hypothesis.internal.floats import float_to_int
from hypothesis.internal.intervalsets import IntervalSet

if TYPE_CHECKING:
    from hypothesis.internal.conjecture.data import ConjectureData, Spans, Status


Observation: TypeAlias = Union["InfoObservation", "TestCaseObservation"]
CallbackThreadT: TypeAlias = Callable[[Observation], None]
# for all_threads=True, we pass the thread id as well.
CallbackAllThreadsT: TypeAlias = Callable[[Observation, int], None]
CallbackT: TypeAlias = CallbackThreadT | CallbackAllThreadsT

# thread_id: list[callback]
_callbacks: dict[int | None, list[CallbackThreadT]] = {}
# callbacks where all_threads=True was set
_callbacks_all_threads: list[CallbackAllThreadsT] = []


@dataclass(slots=True, frozen=False)
class PredicateCounts:
    satisfied: int = 0
    unsatisfied: int = 0

    def update_count(self, *, condition: bool) -> None:
        if condition:
            self.satisfied += 1
        else:
            self.unsatisfied += 1


def _choice_to_json(choice: ChoiceT | None) -> Any:
    if choice is None:
        return None
    # see the note on the same check in to_jsonable for why we cast large
    # integers to floats.
    if (
        isinstance(choice, int)
        and not isinstance(choice, bool)
        and abs(choice) >= 2**63
    ):
        return ["integer", str(choice)]
    elif isinstance(choice, bytes):
        return ["bytes", base64.b64encode(choice).decode()]
    elif isinstance(choice, float) and math.isnan(choice):
        # handle nonstandard nan bit patterns. We don't need to do this for -0.0
        # vs 0.0 since json doesn't normalize -0.0 to 0.0.
        return ["float", float_to_int(choice)]
    return choice


def choices_to_json(choices: tuple[ChoiceT, ...]) -> list[Any]:
    return [_choice_to_json(choice) for choice in choices]


def _constraints_to_json(
    choice_type: ChoiceTypeT, constraints: ChoiceConstraintsT
) -> dict[str, Any]:
    constraints = constraints.copy()
    if choice_type == "integer":
        constraints = cast(IntegerConstraints, constraints)
        return {
            "min_value": _choice_to_json(constraints["min_value"]),
            "max_value": _choice_to_json(constraints["max_value"]),
            "weights": (
                None
                if constraints["weights"] is None
                # wrap up in a list, instead of a dict, because json dicts
                # require string keys
                else [
                    (_choice_to_json(k), v) for k, v in constraints["weights"].items()
                ]
            ),
            "shrink_towards": _choice_to_json(constraints["shrink_towards"]),
        }
    elif choice_type == "float":
        constraints = cast(FloatConstraints, constraints)
        return {
            "min_value": _choice_to_json(constraints["min_value"]),
            "max_value": _choice_to_json(constraints["max_value"]),
            "allow_nan": constraints["allow_nan"],
            "smallest_nonzero_magnitude": constraints["smallest_nonzero_magnitude"],
        }
    elif choice_type == "string":
        constraints = cast(StringConstraints, constraints)
        assert isinstance(constraints["intervals"], IntervalSet)
        return {
            "intervals": constraints["intervals"].intervals,
            "min_size": _choice_to_json(constraints["min_size"]),
            "max_size": _choice_to_json(constraints["max_size"]),
        }
    elif choice_type == "bytes":
        constraints = cast(BytesConstraints, constraints)
        return {
            "min_size": _choice_to_json(constraints["min_size"]),
            "max_size": _choice_to_json(constraints["max_size"]),
        }
    elif choice_type == "boolean":
        constraints = cast(BooleanConstraints, constraints)
        return {
            "p": constraints["p"],
        }
    else:
        raise NotImplementedError(f"unknown choice type {choice_type}")


def nodes_to_json(nodes: tuple[ChoiceNode, ...]) -> list[dict[str, Any]]:
    return [
        {
            "type": node.type,
            "value": _choice_to_json(node.value),
            "constraints": _constraints_to_json(node.type, node.constraints),
            "was_forced": node.was_forced,
        }
        for node in nodes
    ]


@dataclass(slots=True, frozen=True)
class ObservationMetadata:
    traceback: str | None
    reproduction_decorator: str | None
    predicates: dict[str, PredicateCounts]
    backend: dict[str, Any]
    sys_argv: list[str]
    os_getpid: int
    imported_at: float
    data_status: "Status"
    phase: str
    interesting_origin: InterestingOrigin | None
    choice_nodes: tuple[ChoiceNode, ...] | None
    choice_spans: Optional["Spans"]

    def to_json(self) -> dict[str, Any]:
        data = {
            "traceback": self.traceback,
            "reproduction_decorator": self.reproduction_decorator,
            "predicates": self.predicates,
            "backend": self.backend,
            "sys.argv": self.sys_argv,
            "os.getpid()": self.os_getpid,
            "imported_at": self.imported_at,
            "data_status": self.data_status,
            "phase": self.phase,
            "interesting_origin": self.interesting_origin,
            "choice_nodes": (
                None if self.choice_nodes is None else nodes_to_json(self.choice_nodes)
            ),
            "choice_spans": (
                None
                if self.choice_spans is None
                else [
                    (
                        # span.label is an int, but cast to string to avoid conversion
                        # to float (and loss of precision) for large label values.
                        #
                        # The value of this label is opaque to consumers anyway, so its
                        # type shouldn't matter as long as it's consistent.
                        str(span.label),
                        span.start,
                        span.end,
                        span.discarded,
                    )
                    for span in self.choice_spans
                ]
            ),
        }
        # check that we didn't forget one
        assert len(data) == len(dataclasses.fields(self))
        return data


@dataclass(slots=True, frozen=True)
class BaseObservation:
    type: Literal["test_case", "info", "alert", "error"]
    property: str
    run_start: float


InfoObservationType = Literal["info", "alert", "error"]
TestCaseStatus = Literal["gave_up", "passed", "failed"]


@dataclass(slots=True, frozen=True)
class InfoObservation(BaseObservation):
    type: InfoObservationType
    title: str
    content: str | dict


@dataclass(slots=True, frozen=True)
class TestCaseObservation(BaseObservation):
    __test__ = False  # no! bad pytest!

    type: Literal["test_case"]
    status: TestCaseStatus
    status_reason: str
    representation: str
    arguments: dict
    how_generated: str
    features: dict
    coverage: dict[str, list[int]] | None
    timing: dict[str, float]
    metadata: ObservationMetadata


def add_observability_callback(f: CallbackT, /, *, all_threads: bool = False) -> None:
    """
    Adds ``f`` as a callback for :ref:`observability <observability>`. ``f``
    should accept one argument, which is an observation. Whenever Hypothesis
    produces a new observation, it calls each callback with that observation.

    If Hypothesis tests are being run from multiple threads, callbacks are tracked
    per-thread. In other words, ``add_observability_callback(f)`` only adds ``f``
    as an observability callback for observations produced on that thread.

    If ``all_threads=True`` is passed, ``f`` will instead be registered as a
    callback for all threads. This means it will be called for observations
    generated by all threads, not just the thread which registered ``f`` as a
    callback. In this case, ``f`` will be passed two arguments: the first is the
    observation, and the second is the integer thread id from
    :func:`python:threading.get_ident` where that observation was generated.

    We recommend against registering ``f`` as a callback for both ``all_threads=True``
    and the default ``all_threads=False``, due to unclear semantics with
    |remove_observability_callback|.
    """
    if all_threads:
        _callbacks_all_threads.append(cast(CallbackAllThreadsT, f))
        return

    thread_id = threading.get_ident()
    if thread_id not in _callbacks:
        _callbacks[thread_id] = []

    _callbacks[thread_id].append(cast(CallbackThreadT, f))


def remove_observability_callback(f: CallbackT, /) -> None:
    """
    Removes ``f`` from the :ref:`observability <observability>` callbacks.

    If ``f`` is not in the list of observability callbacks, silently do nothing.

    If running under multiple threads, ``f`` will only be removed from the
    callbacks for this thread.
    """
    if f in _callbacks_all_threads:
        _callbacks_all_threads.remove(cast(CallbackAllThreadsT, f))

    thread_id = threading.get_ident()
    if thread_id not in _callbacks:
        return

    callbacks = _callbacks[thread_id]
    if f in callbacks:
        callbacks.remove(cast(CallbackThreadT, f))

    if not callbacks:
        del _callbacks[thread_id]


def observability_enabled() -> bool:
    """
    Returns whether or not Hypothesis considers :ref:`observability <observability>`
    to be enabled. Observability is enabled if there is at least one observability
    callback present.

    Callers might use this method to determine whether they should compute an
    expensive representation that is only used under observability, for instance
    by |alternative backends|.
    """
    return bool(_callbacks) or bool(_callbacks_all_threads)


@contextmanager
def with_observability_callback(
    f: Callable[[Observation], None], /, *, all_threads: bool = False
) -> Generator[None, None, None]:
    """
    A simple context manager which calls |add_observability_callback| on ``f``
    when it enters and |remove_observability_callback| on ``f`` when it exits.
    """
    add_observability_callback(f, all_threads=all_threads)
    try:
        yield
    finally:
        remove_observability_callback(f)


def deliver_observation(observation: Observation) -> None:
    thread_id = threading.get_ident()

    for callback in _callbacks.get(thread_id, []):
        callback(observation)

    for callback in _callbacks_all_threads:
        callback(observation, thread_id)


class _TestcaseCallbacks:
    def __bool__(self):
        self._note_deprecation()
        return bool(_callbacks)

    def _note_deprecation(self):
        from hypothesis._settings import note_deprecation

        note_deprecation(
            "hypothesis.internal.observability.TESTCASE_CALLBACKS is deprecated. "
            "Replace TESTCASE_CALLBACKS.append with add_observability_callback, "
            "TESTCASE_CALLBACKS.remove with remove_observability_callback, and "
            "bool(TESTCASE_CALLBACKS) with observability_enabled().",
            since="2025-08-01",
            has_codemod=False,
        )

    def append(self, f):
        self._note_deprecation()
        add_observability_callback(f)

    def remove(self, f):
        self._note_deprecation()
        remove_observability_callback(f)


#: .. warning::
#:
#:   Deprecated in favor of |add_observability_callback|,
#:   |remove_observability_callback|, and |observability_enabled|.
#:
#:   |TESTCASE_CALLBACKS| remains a thin compatibility
#:   shim which forwards ``.append``, ``.remove``, and ``bool()`` to those
#:   three methods. It is not an attempt to be fully compatible with the previous
#:   ``TESTCASE_CALLBACKS = []``, so iteration or other usages will not work
#:   anymore. Please update to using the new methods instead.
#:
#:   |TESTCASE_CALLBACKS| will eventually be removed.
TESTCASE_CALLBACKS = _TestcaseCallbacks()


def make_testcase(
    *,
    run_start: float,
    property: str,
    data: "ConjectureData",
    how_generated: str,
    representation: str = "<unknown>",
    timing: dict[str, float],
    arguments: dict | None = None,
    coverage: dict[str, list[int]] | None = None,
    phase: str | None = None,
    backend_metadata: dict[str, Any] | None = None,
    status: (
        Union[TestCaseStatus, "Status"] | None
    ) = None,  # overrides automatic calculation
    status_reason: str | None = None,  # overrides automatic calculation
    # added to calculated metadata. If keys overlap, the value from this `metadata`
    # is used
    metadata: dict[str, Any] | None = None,
) -> TestCaseObservation:
    from hypothesis.core import reproduction_decorator
    from hypothesis.internal.conjecture.data import Status

    # We should only be sending observability reports for datas that have finished
    # being modified.
    assert data.frozen

    if status_reason is not None:
        pass
    elif data.interesting_origin:
        status_reason = str(data.interesting_origin)
    elif phase == "shrink" and data.status == Status.OVERRUN:
        status_reason = "exceeded size of current best example"
    else:
        status_reason = str(data.events.pop("invalid because", ""))

    status_map: dict[Status, TestCaseStatus] = {
        Status.OVERRUN: "gave_up",
        Status.INVALID: "gave_up",
        Status.VALID: "passed",
        Status.INTERESTING: "failed",
    }

    if status is not None and isinstance(status, Status):
        status = status_map[status]
    if status is None:
        status = status_map[data.status]

    return TestCaseObservation(
        type="test_case",
        status=status,
        status_reason=status_reason,
        representation=representation,
        arguments={
            k.removeprefix("generate:"): v for k, v in (arguments or {}).items()
        },
        how_generated=how_generated,  # iid, mutation, etc.
        features={
            **{
                f"target:{k}".strip(":"): v for k, v in data.target_observations.items()
            },
            **data.events,
        },
        coverage=coverage,
        timing=timing,
        metadata=ObservationMetadata(
            **{
                "traceback": data.expected_traceback,
                "reproduction_decorator": (
                    reproduction_decorator(data.choices) if status == "failed" else None
                ),
                "predicates": dict(data._observability_predicates),
                "backend": backend_metadata or {},
                "data_status": data.status,
                "phase": phase,
                "interesting_origin": data.interesting_origin,
                "choice_nodes": data.nodes if OBSERVABILITY_CHOICES else None,
                "choice_spans": data.spans if OBSERVABILITY_CHOICES else None,
                **_system_metadata(),
                # unpack last so it takes precedence for duplicate keys
                **(metadata or {}),
            }
        ),
        run_start=run_start,
        property=property,
    )


_WROTE_TO = set()
_deliver_to_file_lock = Lock()


def _deliver_to_file(
    observation: Observation, thread_id: int
) -> None:  # pragma: no cover
    from hypothesis.strategies._internal.utils import to_jsonable

    kind = "testcases" if observation.type == "test_case" else "info"
    fname = storage_directory("observed", f"{date.today().isoformat()}_{kind}.jsonl")
    fname.parent.mkdir(exist_ok=True, parents=True)

    observation_bytes = (
        json.dumps(to_jsonable(observation, avoid_realization=False)) + "\n"
    )
    # only allow one conccurent file write to avoid write races. This is likely to make
    # HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY quite slow under threading. A queue
    # would be an improvement, but that requires a background thread, and I
    # would prefer to avoid a thread in the single-threaded case. We could
    # switch over to a queue if we detect multithreading, but it's tricky to get
    # right.
    with _deliver_to_file_lock:
        _WROTE_TO.add(fname)
        with fname.open(mode="a") as f:
            f.write(observation_bytes)


_imported_at = time.time()


@lru_cache
def _system_metadata() -> dict[str, Any]:
    return {
        "sys_argv": sys.argv,
        "os_getpid": os.getpid(),
        "imported_at": _imported_at,
    }


#: If ``False``, do not collect coverage information when observability is enabled.
#:
#: This is exposed both for performance (as coverage collection can be slow on
#: Python 3.11 and earlier) and size (if you do not use coverage information,
#: you may not want to store it in-memory).
OBSERVABILITY_COLLECT_COVERAGE = (
    "HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY_NOCOVER" not in os.environ
)
#: If ``True``, include the ``metadata.choice_nodes`` and ``metadata.spans`` keys
#: in test case observations.
#:
#: ``False`` by default. ``metadata.choice_nodes`` and ``metadata.spans`` can be
#: a substantial amount of data, and so must be opted-in to, even when
#: observability is enabled.
#:
#: .. warning::
#:
#:     EXPERIMENTAL AND UNSTABLE. We are actively working towards a better
#:     interface for this as of June 2025, and this attribute may disappear or
#:     be renamed without notice.
#:
OBSERVABILITY_CHOICES = "HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY_CHOICES" in os.environ

if OBSERVABILITY_COLLECT_COVERAGE is False and (
    sys.version_info[:2] >= (3, 12)
):  # pragma: no cover
    warnings.warn(
        "Coverage data collection should be quite fast in Python 3.12 or later "
        "so there should be no need to turn coverage reporting off.",
        HypothesisWarning,
        stacklevel=2,
    )

if (
    "HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY" in os.environ
    or OBSERVABILITY_COLLECT_COVERAGE is False
):  # pragma: no cover
    add_observability_callback(_deliver_to_file, all_threads=True)

    # Remove files more than a week old, to cap the size on disk
    max_age = (date.today() - timedelta(days=8)).isoformat()
    for f in storage_directory("observed", intent_to_write=False).glob("*.jsonl"):
        if f.stem < max_age:  # pragma: no branch
            f.unlink(missing_ok=True)
