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
import time
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property
from random import Random
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NoReturn,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from hypothesis.errors import (
    CannotProceedScopeT,
    ChoiceTooLarge,
    Frozen,
    InvalidArgument,
    StopTest,
)
from hypothesis.internal.cache import LRUCache
from hypothesis.internal.compat import add_note
from hypothesis.internal.conjecture.choice import (
    BooleanConstraints,
    BytesConstraints,
    ChoiceConstraintsT,
    ChoiceNode,
    ChoiceT,
    ChoiceTemplate,
    ChoiceTypeT,
    FloatConstraints,
    IntegerConstraints,
    StringConstraints,
    choice_constraints_key,
    choice_from_index,
    choice_permitted,
    choices_size,
)
from hypothesis.internal.conjecture.junkdrawer import IntList, gc_cumulative_time
from hypothesis.internal.conjecture.providers import (
    COLLECTION_DEFAULT_MAX_SIZE,
    HypothesisProvider,
    PrimitiveProvider,
)
from hypothesis.internal.conjecture.utils import calc_label_from_name
from hypothesis.internal.escalation import InterestingOrigin
from hypothesis.internal.floats import (
    SMALLEST_SUBNORMAL,
    float_to_int,
    int_to_float,
    sign_aware_lte,
)
from hypothesis.internal.intervalsets import IntervalSet
from hypothesis.internal.observability import PredicateCounts
from hypothesis.reporting import debug_report
from hypothesis.utils.conventions import not_set
from hypothesis.utils.threading import ThreadLocal

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy
    from hypothesis.strategies._internal.core import DataObject
    from hypothesis.strategies._internal.random import RandomState
    from hypothesis.strategies._internal.strategies import Ex


def __getattr__(name: str) -> Any:
    if name == "AVAILABLE_PROVIDERS":
        from hypothesis._settings import note_deprecation
        from hypothesis.internal.conjecture.providers import AVAILABLE_PROVIDERS

        note_deprecation(
            "hypothesis.internal.conjecture.data.AVAILABLE_PROVIDERS has been moved to "
            "hypothesis.internal.conjecture.providers.AVAILABLE_PROVIDERS.",
            since="2025-01-25",
            has_codemod=False,
            stacklevel=1,
        )
        return AVAILABLE_PROVIDERS

    raise AttributeError(
        f"Module 'hypothesis.internal.conjecture.data' has no attribute {name}"
    )


T = TypeVar("T")
TargetObservations = dict[str, int | float]
# index, choice_type, constraints, forced value
MisalignedAt: TypeAlias = tuple[int, ChoiceTypeT, ChoiceConstraintsT, ChoiceT | None]

TOP_LABEL = calc_label_from_name("top")
MAX_DEPTH = 100

threadlocal = ThreadLocal(global_test_counter=int)


class Status(IntEnum):
    OVERRUN = 0
    INVALID = 1
    VALID = 2
    INTERESTING = 3

    def __repr__(self) -> str:
        return f"Status.{self.name}"


@dataclass(slots=True, frozen=True)
class StructuralCoverageTag:
    label: int


STRUCTURAL_COVERAGE_CACHE: dict[int, StructuralCoverageTag] = {}


def structural_coverage(label: int) -> StructuralCoverageTag:
    try:
        return STRUCTURAL_COVERAGE_CACHE[label]
    except KeyError:
        return STRUCTURAL_COVERAGE_CACHE.setdefault(label, StructuralCoverageTag(label))


# This cache can be quite hot and so we prefer LRUCache over LRUReusedCache for
# performance. We lose scan resistance, but that's probably fine here.
POOLED_CONSTRAINTS_CACHE: LRUCache[tuple[Any, ...], ChoiceConstraintsT] = LRUCache(4096)


class Span:
    """A span tracks the hierarchical structure of choices within a single test run.

    Spans are created to mark regions of the choice sequence that that are
    logically related to each other. For instance, Hypothesis tracks:
    - A single top-level span for the entire choice sequence
    - A span for the choices made by each strategy
    - Some strategies define additional spans within their choices. For instance,
      st.lists() tracks the "should add another element" choice and the "add
      another element" choices as separate spans.

    Spans provide useful information to the shrinker, mutator, targeted PBT,
    and other subsystems of Hypothesis.

    Rather than store each ``Span`` as a rich object, it is actually
    just an index into the ``Spans`` class defined below. This has two
    purposes: Firstly, for most properties of spans we will never need
    to allocate storage at all, because most properties are not used on
    most spans. Secondly, by storing the spans as compact lists
    of integers, we save a considerable amount of space compared to
    Python's normal object size.

    This does have the downside that it increases the amount of allocation
    we do, and slows things down as a result, in some usage patterns because
    we repeatedly allocate the same Span or int objects, but it will
    often dramatically reduce our memory usage, so is worth it.
    """

    __slots__ = ("index", "owner")

    def __init__(self, owner: "Spans", index: int) -> None:
        self.owner = owner
        self.index = index

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Span):
            return NotImplemented
        return (self.owner is other.owner) and (self.index == other.index)

    def __ne__(self, other: object) -> bool:
        if self is other:
            return False
        if not isinstance(other, Span):
            return NotImplemented
        return (self.owner is not other.owner) or (self.index != other.index)

    def __repr__(self) -> str:
        return f"spans[{self.index}]"

    @property
    def label(self) -> int:
        """A label is an opaque value that associates each span with its
        approximate origin, such as a particular strategy class or a particular
        kind of draw."""
        return self.owner.labels[self.owner.label_indices[self.index]]

    @property
    def parent(self) -> int | None:
        """The index of the span that this one is nested directly within."""
        if self.index == 0:
            return None
        return self.owner.parentage[self.index]

    @property
    def start(self) -> int:
        return self.owner.starts[self.index]

    @property
    def end(self) -> int:
        return self.owner.ends[self.index]

    @property
    def depth(self) -> int:
        """
        Depth of this span in the span tree. The top-level span has a depth of 0.
        """
        return self.owner.depths[self.index]

    @property
    def discarded(self) -> bool:
        """True if this is span's ``stop_span`` call had ``discard`` set to
        ``True``. This means we believe that the shrinker should be able to delete
        this span completely, without affecting the value produced by its enclosing
        strategy. Typically set when a rejection sampler decides to reject a
        generated value and try again."""
        return self.index in self.owner.discarded

    @property
    def choice_count(self) -> int:
        """The number of choices in this span."""
        return self.end - self.start

    @property
    def children(self) -> "list[Span]":
        """The list of all spans with this as a parent, in increasing index
        order."""
        return [self.owner[i] for i in self.owner.children[self.index]]


class SpanProperty:
    """There are many properties of spans that we calculate by
    essentially rerunning the test case multiple times based on the
    calls which we record in SpanProperty.

    This class defines a visitor, subclasses of which can be used
    to calculate these properties.
    """

    def __init__(self, spans: "Spans"):
        self.span_stack: list[int] = []
        self.spans = spans
        self.span_count = 0
        self.choice_count = 0

    def run(self) -> Any:
        """Rerun the test case with this visitor and return the
        results of ``self.finish()``."""
        for record in self.spans.trail:
            if record == TrailType.STOP_SPAN_DISCARD:
                self.__pop(discarded=True)
            elif record == TrailType.STOP_SPAN_NO_DISCARD:
                self.__pop(discarded=False)
            elif record == TrailType.CHOICE:
                self.choice_count += 1
            else:
                # everything after TrailType.CHOICE is the label of a span start.
                self.__push(record - TrailType.CHOICE - 1)

        return self.finish()

    def __push(self, label_index: int) -> None:
        i = self.span_count
        assert i < len(self.spans)
        self.start_span(i, label_index=label_index)
        self.span_count += 1
        self.span_stack.append(i)

    def __pop(self, *, discarded: bool) -> None:
        i = self.span_stack.pop()
        self.stop_span(i, discarded=discarded)

    def start_span(self, i: int, label_index: int) -> None:
        """Called at the start of each span, with ``i`` the
        index of the span and ``label_index`` the index of
        its label in ``self.spans.labels``."""

    def stop_span(self, i: int, *, discarded: bool) -> None:
        """Called at the end of each span, with ``i`` the
        index of the span and ``discarded`` being ``True`` if ``stop_span``
        was called with ``discard=True``."""

    def finish(self) -> Any:
        raise NotImplementedError


class TrailType(IntEnum):
    STOP_SPAN_DISCARD = 1
    STOP_SPAN_NO_DISCARD = 2
    CHOICE = 3
    # every trail element larger than TrailType.CHOICE is the label of a span
    # start, offset by its index. So the first span label is stored as 4, the
    # second as 5, etc, regardless of its actual integer label.


class SpanRecord:
    """Records the series of ``start_span``, ``stop_span``, and
    ``draw_bits`` calls so that these may be stored in ``Spans`` and
    replayed when we need to know about the structure of individual
    ``Span`` objects.

    Note that there is significant similarity between this class and
    ``DataObserver``, and the plan is to eventually unify them, but
    they currently have slightly different functions and implementations.
    """

    def __init__(self) -> None:
        self.labels: list[int] = []
        self.__index_of_labels: dict[int, int] | None = {}
        self.trail = IntList()
        self.nodes: list[ChoiceNode] = []

    def freeze(self) -> None:
        self.__index_of_labels = None

    def record_choice(self) -> None:
        self.trail.append(TrailType.CHOICE)

    def start_span(self, label: int) -> None:
        assert self.__index_of_labels is not None
        try:
            i = self.__index_of_labels[label]
        except KeyError:
            i = self.__index_of_labels.setdefault(label, len(self.labels))
            self.labels.append(label)
        self.trail.append(TrailType.CHOICE + 1 + i)

    def stop_span(self, *, discard: bool) -> None:
        if discard:
            self.trail.append(TrailType.STOP_SPAN_DISCARD)
        else:
            self.trail.append(TrailType.STOP_SPAN_NO_DISCARD)


class _starts_and_ends(SpanProperty):
    def __init__(self, spans: "Spans") -> None:
        super().__init__(spans)
        self.starts = IntList.of_length(len(self.spans))
        self.ends = IntList.of_length(len(self.spans))

    def start_span(self, i: int, label_index: int) -> None:
        self.starts[i] = self.choice_count

    def stop_span(self, i: int, *, discarded: bool) -> None:
        self.ends[i] = self.choice_count

    def finish(self) -> tuple[IntList, IntList]:
        return (self.starts, self.ends)


class _discarded(SpanProperty):
    def __init__(self, spans: "Spans") -> None:
        super().__init__(spans)
        self.result: set[int] = set()

    def finish(self) -> frozenset[int]:
        return frozenset(self.result)

    def stop_span(self, i: int, *, discarded: bool) -> None:
        if discarded:
            self.result.add(i)


class _parentage(SpanProperty):
    def __init__(self, spans: "Spans") -> None:
        super().__init__(spans)
        self.result = IntList.of_length(len(self.spans))

    def stop_span(self, i: int, *, discarded: bool) -> None:
        if i > 0:
            self.result[i] = self.span_stack[-1]

    def finish(self) -> IntList:
        return self.result


class _depths(SpanProperty):
    def __init__(self, spans: "Spans") -> None:
        super().__init__(spans)
        self.result = IntList.of_length(len(self.spans))

    def start_span(self, i: int, label_index: int) -> None:
        self.result[i] = len(self.span_stack)

    def finish(self) -> IntList:
        return self.result


class _label_indices(SpanProperty):
    def __init__(self, spans: "Spans") -> None:
        super().__init__(spans)
        self.result = IntList.of_length(len(self.spans))

    def start_span(self, i: int, label_index: int) -> None:
        self.result[i] = label_index

    def finish(self) -> IntList:
        return self.result


class _mutator_groups(SpanProperty):
    def __init__(self, spans: "Spans") -> None:
        super().__init__(spans)
        self.groups: dict[int, set[tuple[int, int]]] = defaultdict(set)

    def start_span(self, i: int, label_index: int) -> None:
        # TODO should we discard start == end cases? occurs for eg st.data()
        # which is conditionally or never drawn from. arguably swapping
        # nodes with the empty list is a useful mutation enabled by start == end?
        key = (self.spans[i].start, self.spans[i].end)
        self.groups[label_index].add(key)

    def finish(self) -> Iterable[set[tuple[int, int]]]:
        # Discard groups with only one span, since the mutator can't
        # do anything useful with them.
        return [g for g in self.groups.values() if len(g) >= 2]


class Spans:
    """A lazy collection of ``Span`` objects, derived from
    the record of recorded behaviour in ``SpanRecord``.

    Behaves logically as if it were a list of ``Span`` objects,
    but actually mostly exists as a compact store of information
    for them to reference into. All properties on here are best
    understood as the backing storage for ``Span`` and are
    described there.
    """

    def __init__(self, record: SpanRecord) -> None:
        self.trail = record.trail
        self.labels = record.labels
        self.__length = self.trail.count(
            TrailType.STOP_SPAN_DISCARD
        ) + record.trail.count(TrailType.STOP_SPAN_NO_DISCARD)
        self.__children: list[Sequence[int]] | None = None

    @cached_property
    def starts_and_ends(self) -> tuple[IntList, IntList]:
        return _starts_and_ends(self).run()

    @property
    def starts(self) -> IntList:
        return self.starts_and_ends[0]

    @property
    def ends(self) -> IntList:
        return self.starts_and_ends[1]

    @cached_property
    def discarded(self) -> frozenset[int]:
        return _discarded(self).run()

    @cached_property
    def parentage(self) -> IntList:
        return _parentage(self).run()

    @cached_property
    def depths(self) -> IntList:
        return _depths(self).run()

    @cached_property
    def label_indices(self) -> IntList:
        return _label_indices(self).run()

    @cached_property
    def mutator_groups(self) -> list[set[tuple[int, int]]]:
        return _mutator_groups(self).run()

    @property
    def children(self) -> list[Sequence[int]]:
        if self.__children is None:
            children = [IntList() for _ in range(len(self))]
            for i, p in enumerate(self.parentage):
                if i > 0:
                    children[p].append(i)
            # Replace empty children lists with a tuple to reduce
            # memory usage.
            for i, c in enumerate(children):
                if not c:
                    children[i] = ()  # type: ignore
            self.__children = children  # type: ignore
        return self.__children  # type: ignore

    def __len__(self) -> int:
        return self.__length

    def __getitem__(self, i: int) -> Span:
        n = self.__length
        if i < -n or i >= n:
            raise IndexError(f"Index {i} out of range [-{n}, {n})")
        if i < 0:
            i += n
        return Span(self, i)

    # not strictly necessary as we have len/getitem, but required for mypy.
    # https://github.com/python/mypy/issues/9737
    def __iter__(self) -> Iterator[Span]:
        for i in range(len(self)):
            yield self[i]


class _Overrun:
    status: Status = Status.OVERRUN

    def __repr__(self) -> str:
        return "Overrun"


Overrun = _Overrun()


class DataObserver:
    """Observer class for recording the behaviour of a
    ConjectureData object, primarily used for tracking
    the behaviour in the tree cache."""

    def conclude_test(
        self,
        status: Status,
        interesting_origin: InterestingOrigin | None,
    ) -> None:
        """Called when ``conclude_test`` is called on the
        observed ``ConjectureData``, with the same arguments.

        Note that this is called after ``freeze`` has completed.
        """

    def kill_branch(self) -> None:
        """Mark this part of the tree as not worth re-exploring."""

    def draw_integer(
        self, value: int, *, constraints: IntegerConstraints, was_forced: bool
    ) -> None:
        pass

    def draw_float(
        self, value: float, *, constraints: FloatConstraints, was_forced: bool
    ) -> None:
        pass

    def draw_string(
        self, value: str, *, constraints: StringConstraints, was_forced: bool
    ) -> None:
        pass

    def draw_bytes(
        self, value: bytes, *, constraints: BytesConstraints, was_forced: bool
    ) -> None:
        pass

    def draw_boolean(
        self, value: bool, *, constraints: BooleanConstraints, was_forced: bool
    ) -> None:
        pass


@dataclass(slots=True, frozen=True)
class ConjectureResult:
    """Result class storing the parts of ConjectureData that we
    will care about after the original ConjectureData has outlived its
    usefulness."""

    status: Status
    interesting_origin: InterestingOrigin | None
    nodes: tuple[ChoiceNode, ...] = field(repr=False, compare=False)
    length: int
    output: str
    expected_exception: BaseException | None
    expected_traceback: str | None
    has_discards: bool
    target_observations: TargetObservations
    tags: frozenset[StructuralCoverageTag]
    spans: Spans = field(repr=False, compare=False)
    arg_slices: set[tuple[int, int]] = field(repr=False)
    slice_comments: dict[tuple[int, int], str] = field(repr=False)
    misaligned_at: MisalignedAt | None = field(repr=False)
    cannot_proceed_scope: CannotProceedScopeT | None = field(repr=False)

    def as_result(self) -> "ConjectureResult":
        return self

    @property
    def choices(self) -> tuple[ChoiceT, ...]:
        return tuple(node.value for node in self.nodes)


class ConjectureData:
    @classmethod
    def for_choices(
        cls,
        choices: Sequence[ChoiceTemplate | ChoiceT],
        *,
        observer: DataObserver | None = None,
        provider: PrimitiveProvider | type[PrimitiveProvider] = HypothesisProvider,
        random: Random | None = None,
    ) -> "ConjectureData":
        from hypothesis.internal.conjecture.engine import choice_count

        return cls(
            max_choices=choice_count(choices),
            random=random,
            prefix=choices,
            observer=observer,
            provider=provider,
        )

    def __init__(
        self,
        *,
        random: Random | None,
        observer: DataObserver | None = None,
        provider: PrimitiveProvider | type[PrimitiveProvider] = HypothesisProvider,
        prefix: Sequence[ChoiceTemplate | ChoiceT] | None = None,
        max_choices: int | None = None,
        provider_kw: dict[str, Any] | None = None,
    ) -> None:
        from hypothesis.internal.conjecture.engine import BUFFER_SIZE

        if observer is None:
            observer = DataObserver()
        if provider_kw is None:
            provider_kw = {}
        elif not isinstance(provider, type):
            raise InvalidArgument(
                f"Expected {provider=} to be a class since {provider_kw=} was "
                "passed, but got an instance instead."
            )

        assert isinstance(observer, DataObserver)
        self.observer = observer
        self.max_choices = max_choices
        self.max_length = BUFFER_SIZE
        self.overdraw = 0
        self._random = random

        self.length = 0
        self.index = 0
        self.output = ""
        self.status = Status.VALID
        self.frozen = False
        self.testcounter = threadlocal.global_test_counter
        threadlocal.global_test_counter += 1
        self.start_time = time.perf_counter()
        self.gc_start_time = gc_cumulative_time()
        self.events: dict[str, str | int | float] = {}
        self.interesting_origin: InterestingOrigin | None = None
        self.draw_times: dict[str, float] = {}
        self._stateful_run_times: dict[str, float] = defaultdict(float)
        self.max_depth = 0
        self.has_discards = False

        self.provider: PrimitiveProvider = (
            provider(self, **provider_kw) if isinstance(provider, type) else provider
        )
        assert isinstance(self.provider, PrimitiveProvider)

        self.__result: ConjectureResult | None = None

        # Observations used for targeted search.  They'll be aggregated in
        # ConjectureRunner.generate_new_examples and fed to TargetSelector.
        self.target_observations: TargetObservations = {}

        # Tags which indicate something about which part of the search space
        # this example is in. These are used to guide generation.
        self.tags: set[StructuralCoverageTag] = set()
        self.labels_for_structure_stack: list[set[int]] = []

        # Normally unpopulated but we need this in the niche case
        # that self.as_result() is Overrun but we still want the
        # examples for reporting purposes.
        self.__spans: Spans | None = None

        # We want the top level span to have depth 0, so we start
        # at -1.
        self.depth = -1
        self.__span_record = SpanRecord()

        # Slice indices for discrete reportable parts that which-parts-matter can
        # try varying, to report if the minimal example always fails anyway.
        self.arg_slices: set[tuple[int, int]] = set()
        self.slice_comments: dict[tuple[int, int], str] = {}
        self._observability_args: dict[str, Any] = {}
        self._observability_predicates: defaultdict[str, PredicateCounts] = defaultdict(
            PredicateCounts
        )

        self._sampled_from_all_strategies_elements_message: (
            tuple[str, object] | None
        ) = None
        self._shared_strategy_draws: dict[Hashable, tuple[Any, SearchStrategy]] = {}
        self._shared_data_strategy: DataObject | None = None
        self._stateful_repr_parts: list[Any] | None = None
        self.states_for_ids: dict[int, RandomState] | None = None
        self.seeds_to_states: dict[Any, RandomState] | None = None
        self.hypothesis_runner: Any = not_set

        self.expected_exception: BaseException | None = None
        self.expected_traceback: str | None = None

        self.prefix = prefix
        self.nodes: tuple[ChoiceNode, ...] = ()
        self.misaligned_at: MisalignedAt | None = None
        self.cannot_proceed_scope: CannotProceedScopeT | None = None
        self.start_span(TOP_LABEL)

    def __repr__(self) -> str:
        return "ConjectureData(%s, %d choices%s)" % (
            self.status.name,
            len(self.nodes),
            ", frozen" if self.frozen else "",
        )

    @property
    def choices(self) -> tuple[ChoiceT, ...]:
        return tuple(node.value for node in self.nodes)

    # draw_* functions might be called in one of two contexts: either "above" or
    # "below" the choice sequence. For instance, draw_string calls draw_boolean
    # from ``many`` when calculating the number of characters to return. We do
    # not want these choices to get written to the choice sequence, because they
    # are not true choices themselves.
    #
    # `observe` formalizes this. The choice will only be written to the choice
    # sequence if observe is True.

    @overload
    def _draw(
        self,
        choice_type: Literal["integer"],
        constraints: IntegerConstraints,
        *,
        observe: bool,
        forced: int | None,
    ) -> int: ...

    @overload
    def _draw(
        self,
        choice_type: Literal["float"],
        constraints: FloatConstraints,
        *,
        observe: bool,
        forced: float | None,
    ) -> float: ...

    @overload
    def _draw(
        self,
        choice_type: Literal["string"],
        constraints: StringConstraints,
        *,
        observe: bool,
        forced: str | None,
    ) -> str: ...

    @overload
    def _draw(
        self,
        choice_type: Literal["bytes"],
        constraints: BytesConstraints,
        *,
        observe: bool,
        forced: bytes | None,
    ) -> bytes: ...

    @overload
    def _draw(
        self,
        choice_type: Literal["boolean"],
        constraints: BooleanConstraints,
        *,
        observe: bool,
        forced: bool | None,
    ) -> bool: ...

    def _draw(
        self,
        choice_type: ChoiceTypeT,
        constraints: ChoiceConstraintsT,
        *,
        observe: bool,
        forced: ChoiceT | None,
    ) -> ChoiceT:
        # this is somewhat redundant with the length > max_length check at the
        # end of the function, but avoids trying to use a null self.random when
        # drawing past the node of a ConjectureData.for_choices data.
        if self.length == self.max_length:
            debug_report(f"overrun because hit {self.max_length=}")
            self.mark_overrun()
        if len(self.nodes) == self.max_choices:
            debug_report(f"overrun because hit {self.max_choices=}")
            self.mark_overrun()

        if observe and self.prefix is not None and self.index < len(self.prefix):
            value = self._pop_choice(choice_type, constraints, forced=forced)
        elif forced is None:
            value = getattr(self.provider, f"draw_{choice_type}")(**constraints)

        if forced is not None:
            value = forced

        # nan values generated via int_to_float break list membership:
        #
        #  >>> n = 18444492273895866368
        # >>> assert math.isnan(int_to_float(n))
        # >>> assert int_to_float(n) not in [int_to_float(n)]
        #
        # because int_to_float nans are not equal in the sense of either
        # `a == b` or `a is b`.
        #
        # This can lead to flaky errors when collections require unique
        # floats. What was happening is that in some places we provided math.nan
        # provide math.nan, and in others we provided
        # int_to_float(float_to_int(math.nan)), and which one gets used
        # was not deterministic across test iterations.
        #
        # To fix this, *never* provide a nan value which is equal (via `is`) to
        # another provided nan value. This sacrifices some test power; we should
        # bring that back (ABOVE the choice sequence layer) in the future.
        #
        # See https://github.com/HypothesisWorks/hypothesis/issues/3926.
        if choice_type == "float":
            assert isinstance(value, float)
            if math.isnan(value):
                value = int_to_float(float_to_int(value))

        if observe:
            was_forced = forced is not None
            getattr(self.observer, f"draw_{choice_type}")(
                value, constraints=constraints, was_forced=was_forced
            )
            size = 0 if self.provider.avoid_realization else choices_size([value])
            if self.length + size > self.max_length:
                debug_report(
                    f"overrun because {self.length=} + {size=} > {self.max_length=}"
                )
                self.mark_overrun()

            node = ChoiceNode(
                type=choice_type,
                value=value,
                constraints=constraints,
                was_forced=was_forced,
                index=len(self.nodes),
            )
            self.__span_record.record_choice()
            self.nodes += (node,)
            self.length += size

        return value

    def draw_integer(
        self,
        min_value: int | None = None,
        max_value: int | None = None,
        *,
        weights: dict[int, float] | None = None,
        shrink_towards: int = 0,
        forced: int | None = None,
        observe: bool = True,
    ) -> int:
        # Validate arguments
        if weights is not None:
            assert min_value is not None
            assert max_value is not None
            assert len(weights) <= 255  # arbitrary practical limit
            # We can and should eventually support total weights. But this
            # complicates shrinking as we can no longer assume we can force
            # a value to the unmapped probability mass if that mass might be 0.
            assert sum(weights.values()) < 1
            # similarly, things get simpler if we assume every value is possible.
            # we'll want to drop this restriction eventually.
            assert all(w != 0 for w in weights.values())

        if forced is not None and min_value is not None:
            assert min_value <= forced
        if forced is not None and max_value is not None:
            assert forced <= max_value

        constraints: IntegerConstraints = self._pooled_constraints(
            "integer",
            {
                "min_value": min_value,
                "max_value": max_value,
                "weights": weights,
                "shrink_towards": shrink_towards,
            },
        )
        return self._draw("integer", constraints, observe=observe, forced=forced)

    def draw_float(
        self,
        min_value: float = -math.inf,
        max_value: float = math.inf,
        *,
        allow_nan: bool = True,
        smallest_nonzero_magnitude: float = SMALLEST_SUBNORMAL,
        # TODO: consider supporting these float widths at the choice sequence
        # level in the future.
        # width: Literal[16, 32, 64] = 64,
        forced: float | None = None,
        observe: bool = True,
    ) -> float:
        assert smallest_nonzero_magnitude > 0
        assert not math.isnan(min_value)
        assert not math.isnan(max_value)

        if smallest_nonzero_magnitude == 0.0:  # pragma: no cover
            raise FloatingPointError(
                "Got allow_subnormal=True, but we can't represent subnormal floats "
                "right now, in violation of the IEEE-754 floating-point "
                "specification.  This is usually because something was compiled with "
                "-ffast-math or a similar option, which sets global processor state.  "
                "See https://simonbyrne.github.io/notes/fastmath/ for a more detailed "
                "writeup - and good luck!"
            )

        if forced is not None:
            assert allow_nan or not math.isnan(forced)
            assert math.isnan(forced) or (
                sign_aware_lte(min_value, forced) and sign_aware_lte(forced, max_value)
            )

        constraints: FloatConstraints = self._pooled_constraints(
            "float",
            {
                "min_value": min_value,
                "max_value": max_value,
                "allow_nan": allow_nan,
                "smallest_nonzero_magnitude": smallest_nonzero_magnitude,
            },
        )
        return self._draw("float", constraints, observe=observe, forced=forced)

    def draw_string(
        self,
        intervals: IntervalSet,
        *,
        min_size: int = 0,
        max_size: int = COLLECTION_DEFAULT_MAX_SIZE,
        forced: str | None = None,
        observe: bool = True,
    ) -> str:
        assert forced is None or min_size <= len(forced) <= max_size
        assert min_size >= 0
        if len(intervals) == 0:
            assert min_size == 0

        constraints: StringConstraints = self._pooled_constraints(
            "string",
            {
                "intervals": intervals,
                "min_size": min_size,
                "max_size": max_size,
            },
        )
        return self._draw("string", constraints, observe=observe, forced=forced)

    def draw_bytes(
        self,
        min_size: int = 0,
        max_size: int = COLLECTION_DEFAULT_MAX_SIZE,
        *,
        forced: bytes | None = None,
        observe: bool = True,
    ) -> bytes:
        assert forced is None or min_size <= len(forced) <= max_size
        assert min_size >= 0

        constraints: BytesConstraints = self._pooled_constraints(
            "bytes", {"min_size": min_size, "max_size": max_size}
        )
        return self._draw("bytes", constraints, observe=observe, forced=forced)

    def draw_boolean(
        self,
        p: float = 0.5,
        *,
        forced: bool | None = None,
        observe: bool = True,
    ) -> bool:
        assert (forced is not True) or p > 0
        assert (forced is not False) or p < 1

        constraints: BooleanConstraints = self._pooled_constraints("boolean", {"p": p})
        return self._draw("boolean", constraints, observe=observe, forced=forced)

    @overload
    def _pooled_constraints(
        self, choice_type: Literal["integer"], constraints: IntegerConstraints
    ) -> IntegerConstraints: ...

    @overload
    def _pooled_constraints(
        self, choice_type: Literal["float"], constraints: FloatConstraints
    ) -> FloatConstraints: ...

    @overload
    def _pooled_constraints(
        self, choice_type: Literal["string"], constraints: StringConstraints
    ) -> StringConstraints: ...

    @overload
    def _pooled_constraints(
        self, choice_type: Literal["bytes"], constraints: BytesConstraints
    ) -> BytesConstraints: ...

    @overload
    def _pooled_constraints(
        self, choice_type: Literal["boolean"], constraints: BooleanConstraints
    ) -> BooleanConstraints: ...

    def _pooled_constraints(
        self, choice_type: ChoiceTypeT, constraints: ChoiceConstraintsT
    ) -> ChoiceConstraintsT:
        """Memoize common dictionary objects to reduce memory pressure."""
        # caching runs afoul of nondeterminism checks
        if self.provider.avoid_realization:
            return constraints

        key = (choice_type, *choice_constraints_key(choice_type, constraints))
        try:
            return POOLED_CONSTRAINTS_CACHE[key]
        except KeyError:
            POOLED_CONSTRAINTS_CACHE[key] = constraints
            return constraints

    def _pop_choice(
        self,
        choice_type: ChoiceTypeT,
        constraints: ChoiceConstraintsT,
        *,
        forced: ChoiceT | None,
    ) -> ChoiceT:
        assert self.prefix is not None
        # checked in _draw
        assert self.index < len(self.prefix)

        value = self.prefix[self.index]
        if isinstance(value, ChoiceTemplate):
            node: ChoiceTemplate = value
            if node.count is not None:
                assert node.count >= 0
            # node templates have to be at the end for now, since it's not immediately
            # apparent how to handle overruning a node template while generating a single
            # node if the alternative is not "the entire data is an overrun".
            assert self.index == len(self.prefix) - 1
            if node.type == "simplest":
                if forced is not None:
                    choice = forced
                try:
                    choice = choice_from_index(0, choice_type, constraints)
                except ChoiceTooLarge:
                    self.mark_overrun()
            else:
                raise NotImplementedError

            if node.count is not None:
                node.count -= 1
                if node.count < 0:
                    self.mark_overrun()
            return choice

        choice = value
        node_choice_type = {
            str: "string",
            float: "float",
            int: "integer",
            bool: "boolean",
            bytes: "bytes",
        }[type(choice)]
        # If we're trying to:
        # * draw a different choice type at the same location
        # * draw the same choice type with a different constraints, which does not permit
        #   the current value
        #
        # then we call this a misalignment, because the choice sequence has
        # changed from what we expected at some point. An easy misalignment is
        #
        #   one_of(integers(0, 100), integers(101, 200))
        #
        # where the choice sequence [0, 100] has constraints {min_value: 0, max_value: 100}
        # at index 1, but [0, 101] has constraints {min_value: 101, max_value: 200} at
        # index 1 (which does not permit any of the values 0-100).
        #
        # When the choice sequence becomes misaligned, we generate a new value of the
        # type and constraints the strategy expects.
        if node_choice_type != choice_type or not choice_permitted(choice, constraints):
            # only track first misalignment for now.
            if self.misaligned_at is None:
                self.misaligned_at = (self.index, choice_type, constraints, forced)
            try:
                # Fill in any misalignments with index 0 choices. An alternative to
                # this is using the index of the misaligned choice instead
                # of index 0, which may be useful for maintaining
                # "similarly-complex choices" in the shrinker. This requires
                # attaching an index to every choice in ConjectureData.for_choices,
                # which we don't always have (e.g. when reading from db).
                #
                # If we really wanted this in the future we could make this complexity
                # optional, use it if present, and default to index 0 otherwise.
                # This complicates our internal api and so I'd like to avoid it
                # if possible.
                #
                # Additionally, I don't think slips which require
                # slipping to high-complexity values are common. Though arguably
                # we may want to expand a bit beyond *just* the simplest choice.
                # (we could for example consider sampling choices from index 0-10).
                choice = choice_from_index(0, choice_type, constraints)
            except ChoiceTooLarge:
                # should really never happen with a 0-index choice, but let's be safe.
                self.mark_overrun()

        self.index += 1
        return choice

    def as_result(self) -> ConjectureResult | _Overrun:
        """Convert the result of running this test into
        either an Overrun object or a ConjectureResult."""

        assert self.frozen
        if self.status == Status.OVERRUN:
            return Overrun
        if self.__result is None:
            self.__result = ConjectureResult(
                status=self.status,
                interesting_origin=self.interesting_origin,
                spans=self.spans,
                nodes=self.nodes,
                length=self.length,
                output=self.output,
                expected_traceback=self.expected_traceback,
                expected_exception=self.expected_exception,
                has_discards=self.has_discards,
                target_observations=self.target_observations,
                tags=frozenset(self.tags),
                arg_slices=self.arg_slices,
                slice_comments=self.slice_comments,
                misaligned_at=self.misaligned_at,
                cannot_proceed_scope=self.cannot_proceed_scope,
            )
            assert self.__result is not None
        return self.__result

    def __assert_not_frozen(self, name: str) -> None:
        if self.frozen:
            raise Frozen(f"Cannot call {name} on frozen ConjectureData")

    def note(self, value: Any) -> None:
        self.__assert_not_frozen("note")
        if not isinstance(value, str):
            value = repr(value)
        self.output += value

    def draw(
        self,
        strategy: "SearchStrategy[Ex]",
        label: int | None = None,
        observe_as: str | None = None,
    ) -> "Ex":
        from hypothesis.internal.observability import observability_enabled
        from hypothesis.strategies._internal.lazy import unwrap_strategies
        from hypothesis.strategies._internal.utils import to_jsonable

        at_top_level = self.depth == 0
        start_time = None
        if at_top_level:
            # We start this timer early, because accessing attributes on a LazyStrategy
            # can be almost arbitrarily slow.  In cases like characters() and text()
            # where we cache something expensive, this led to Flaky deadline errors!
            # See https://github.com/HypothesisWorks/hypothesis/issues/2108
            start_time = time.perf_counter()
            gc_start_time = gc_cumulative_time()

        strategy.validate()

        if strategy.is_empty:
            self.mark_invalid(f"empty strategy {self!r}")

        if self.depth >= MAX_DEPTH:
            self.mark_invalid("max depth exceeded")

        # Jump directly to the unwrapped strategy for the label and for do_draw.
        # This avoids adding an extra span to all lazy strategies.
        unwrapped = unwrap_strategies(strategy)
        if label is None:
            label = unwrapped.label
            assert isinstance(label, int)

        self.start_span(label=label)
        try:
            if not at_top_level:
                return unwrapped.do_draw(self)
            assert start_time is not None
            key = observe_as or f"generate:unlabeled_{len(self.draw_times)}"
            try:
                try:
                    v = unwrapped.do_draw(self)
                finally:
                    # Subtract the time spent in GC to avoid overcounting, as it is
                    # accounted for at the overall example level.
                    in_gctime = gc_cumulative_time() - gc_start_time
                    self.draw_times[key] = time.perf_counter() - start_time - in_gctime
            except Exception as err:
                add_note(
                    err,
                    f"while generating {key.removeprefix('generate:')!r} from {strategy!r}",
                )
                raise
            if observability_enabled():
                avoid = self.provider.avoid_realization
                self._observability_args[key] = to_jsonable(v, avoid_realization=avoid)
            return v
        finally:
            self.stop_span()

    def start_span(self, label: int) -> None:
        self.provider.span_start(label)
        self.__assert_not_frozen("start_span")
        self.depth += 1
        # Logically it would make sense for this to just be
        # ``self.depth = max(self.depth, self.max_depth)``, which is what it used to
        # be until we ran the code under tracemalloc and found a rather significant
        # chunk of allocation was happening here. This was presumably due to varargs
        # or the like, but we didn't investigate further given that it was easy
        # to fix with this check.
        if self.depth > self.max_depth:
            self.max_depth = self.depth
        self.__span_record.start_span(label)
        self.labels_for_structure_stack.append({label})

    def stop_span(self, *, discard: bool = False) -> None:
        self.provider.span_end(discard)
        if self.frozen:
            return
        if discard:
            self.has_discards = True
        self.depth -= 1
        assert self.depth >= -1
        self.__span_record.stop_span(discard=discard)

        labels_for_structure = self.labels_for_structure_stack.pop()

        if not discard:
            if self.labels_for_structure_stack:
                self.labels_for_structure_stack[-1].update(labels_for_structure)
            else:
                self.tags.update([structural_coverage(l) for l in labels_for_structure])

        if discard:
            # Once we've discarded a span, every test case starting with
            # this prefix contains discards. We prune the tree at that point so
            # as to avoid future test cases bothering with this region, on the
            # assumption that some span that you could have used instead
            # there would *not* trigger the discard. This greatly speeds up
            # test case generation in some cases, because it allows us to
            # ignore large swathes of the search space that are effectively
            # redundant.
            #
            # A scenario that can cause us problems but which we deliberately
            # have decided not to support is that if there are side effects
            # during data generation then you may end up with a scenario where
            # every good test case generates a discard because the discarded
            # section sets up important things for later. This is not terribly
            # likely and all that you see in this case is some degradation in
            # quality of testing, so we don't worry about it.
            #
            # Note that killing the branch does *not* mean we will never
            # explore below this point, and in particular we may do so during
            # shrinking. Any explicit request for a data object that starts
            # with the branch here will work just fine, but novel prefix
            # generation will avoid it, and we can use it to detect when we
            # have explored the entire tree (up to redundancy).

            self.observer.kill_branch()

    @property
    def spans(self) -> Spans:
        assert self.frozen
        if self.__spans is None:
            self.__spans = Spans(record=self.__span_record)
        return self.__spans

    def freeze(self) -> None:
        if self.frozen:
            return
        self.finish_time = time.perf_counter()
        self.gc_finish_time = gc_cumulative_time()

        # Always finish by closing all remaining spans so that we have a valid tree.
        while self.depth >= 0:
            self.stop_span()

        self.__span_record.freeze()
        self.frozen = True
        self.observer.conclude_test(self.status, self.interesting_origin)

    def choice(
        self,
        values: Sequence[T],
        *,
        forced: T | None = None,
        observe: bool = True,
    ) -> T:
        forced_i = None if forced is None else values.index(forced)
        i = self.draw_integer(
            0,
            len(values) - 1,
            forced=forced_i,
            observe=observe,
        )
        return values[i]

    def conclude_test(
        self,
        status: Status,
        interesting_origin: InterestingOrigin | None = None,
    ) -> NoReturn:
        assert (interesting_origin is None) or (status == Status.INTERESTING)
        self.__assert_not_frozen("conclude_test")
        self.interesting_origin = interesting_origin
        self.status = status
        self.freeze()
        raise StopTest(self.testcounter)

    def mark_interesting(self, interesting_origin: InterestingOrigin) -> NoReturn:
        self.conclude_test(Status.INTERESTING, interesting_origin)

    def mark_invalid(self, why: str | None = None) -> NoReturn:
        if why is not None:
            self.events["invalid because"] = why
        self.conclude_test(Status.INVALID)

    def mark_overrun(self) -> NoReturn:
        self.conclude_test(Status.OVERRUN)


def draw_choice(
    choice_type: ChoiceTypeT, constraints: ChoiceConstraintsT, *, random: Random
) -> ChoiceT:
    cd = ConjectureData(random=random)
    return cast(ChoiceT, getattr(cd.provider, f"draw_{choice_type}")(**constraints))
