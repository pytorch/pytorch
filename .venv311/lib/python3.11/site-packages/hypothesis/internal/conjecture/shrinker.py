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
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    cast,
)

from hypothesis.internal.conjecture.choice import (
    ChoiceNode,
    ChoiceT,
    choice_equal,
    choice_from_index,
    choice_key,
    choice_permitted,
    choice_to_index,
)
from hypothesis.internal.conjecture.data import (
    ConjectureData,
    ConjectureResult,
    Spans,
    Status,
    _Overrun,
    draw_choice,
)
from hypothesis.internal.conjecture.junkdrawer import (
    endswith,
    find_integer,
    replace_all,
    startswith,
)
from hypothesis.internal.conjecture.shrinking import (
    Bytes,
    Float,
    Integer,
    Ordering,
    String,
)
from hypothesis.internal.conjecture.shrinking.choicetree import (
    ChoiceTree,
    prefix_selection_order,
    random_selection_order,
)
from hypothesis.internal.floats import MAX_PRECISE_INTEGER

if TYPE_CHECKING:
    from random import Random

    from hypothesis.internal.conjecture.engine import ConjectureRunner

ShrinkPredicateT: TypeAlias = Callable[[ConjectureResult | _Overrun], bool]


def sort_key(nodes: Sequence[ChoiceNode]) -> tuple[int, tuple[int, ...]]:
    """Returns a sort key such that "simpler" choice sequences are smaller than
    "more complicated" ones.

    We define sort_key so that x is simpler than y if x is shorter than y or if
    they have the same length and map(choice_to_index, x) < map(choice_to_index, y).

    The reason for using this ordering is:

    1. If x is shorter than y then that means we had to make fewer decisions
       in constructing the test case when we ran x than we did when we ran y.
    2. If x is the same length as y then replacing a choice with a lower index
       choice corresponds to replacing it with a simpler/smaller choice.
    3. Because choices drawn early in generation potentially get used in more
       places they potentially have a more significant impact on the final
       result, so it makes sense to prioritise reducing earlier choices over
       later ones.
    """
    return (
        len(nodes),
        tuple(choice_to_index(node.value, node.constraints) for node in nodes),
    )


@dataclass(slots=True, frozen=False)
class ShrinkPass:
    function: Any
    name: str | None = None
    last_prefix: Any = ()

    # some execution statistics
    calls: int = 0
    misaligned: int = 0
    shrinks: int = 0
    deletions: int = 0

    def __post_init__(self):
        if self.name is None:
            self.name = self.function.__name__

    def __hash__(self):
        return hash(self.name)


class StopShrinking(Exception):
    pass


class Shrinker:
    """A shrinker is a child object of a ConjectureRunner which is designed to
    manage the associated state of a particular shrink problem. That is, we
    have some initial ConjectureData object and some property of interest
    that it satisfies, and we want to find a ConjectureData object with a
    shortlex (see sort_key above) smaller choice sequence that exhibits the same
    property.

    Currently the only property of interest we use is that the status is
    INTERESTING and the interesting_origin takes on some fixed value, but we
    may potentially be interested in other use cases later.
    However we assume that data with a status < VALID never satisfies the predicate.

    The shrinker keeps track of a value shrink_target which represents the
    current best known ConjectureData object satisfying the predicate.
    It refines this value by repeatedly running *shrink passes*, which are
    methods that perform a series of transformations to the current shrink_target
    and evaluate the underlying test function to find new ConjectureData
    objects. If any of these satisfy the predicate, the shrink_target
    is updated automatically. Shrinking runs until no shrink pass can
    improve the shrink_target, at which point it stops. It may also be
    terminated if the underlying engine throws RunIsComplete, but that
    is handled by the calling code rather than the Shrinker.

    =======================
    Designing Shrink Passes
    =======================

    Generally a shrink pass is just any function that calls
    cached_test_function and/or consider_new_nodes a number of times,
    but there are a couple of useful things to bear in mind.

    A shrink pass *makes progress* if running it changes self.shrink_target
    (i.e. it tries a shortlex smaller ConjectureData object satisfying
    the predicate). The desired end state of shrinking is to find a
    value such that no shrink pass can make progress, i.e. that we
    are at a local minimum for each shrink pass.

    In aid of this goal, the main invariant that a shrink pass much
    satisfy is that whether it makes progress must be deterministic.
    It is fine (encouraged even) for the specific progress it makes
    to be non-deterministic, but if you run a shrink pass, it makes
    no progress, and then you immediately run it again, it should
    never succeed on the second time. This allows us to stop as soon
    as we have run each shrink pass and seen no progress on any of
    them.

    This means that e.g. it's fine to try each of N deletions
    or replacements in a random order, but it's not OK to try N random
    deletions (unless you have already shrunk at least once, though we
    don't currently take advantage of this loophole).

    Shrink passes need to be written so as to be robust against
    change in the underlying shrink target. It is generally safe
    to assume that the shrink target does not change prior to the
    point of first modification - e.g. if you change no bytes at
    index ``i``, all spans whose start is ``<= i`` still exist,
    as do all blocks, and the data object is still of length
    ``>= i + 1``. This can only be violated by bad user code which
    relies on an external source of non-determinism.

    When the underlying shrink_target changes, shrink
    passes should not run substantially more test_function calls
    on success than they do on failure. Say, no more than a constant
    factor more. In particular shrink passes should not iterate to a
    fixed point.

    This means that shrink passes are often written with loops that
    are carefully designed to do the right thing in the case that no
    shrinks occurred and try to adapt to any changes to do a reasonable
    job. e.g. say we wanted to write a shrink pass that tried deleting
    each individual choice (this isn't an especially good pass,
    but it leads to a simple illustrative example), we might do it
    by iterating over the choice sequence like so:

    .. code-block:: python

        i = 0
        while i < len(self.shrink_target.nodes):
            if not self.consider_new_nodes(
                self.shrink_target.nodes[:i] + self.shrink_target.nodes[i + 1 :]
            ):
                i += 1

    The reason for writing the loop this way is that i is always a
    valid index into the current choice sequence, even if the current sequence
    changes as a result of our actions. When the choice sequence changes,
    we leave the index where it is rather than restarting from the
    beginning, and carry on. This means that the number of steps we
    run in this case is always bounded above by the number of steps
    we would run if nothing works.

    Another thing to bear in mind about shrink pass design is that
    they should prioritise *progress*. If you have N operations that
    you need to run, you should try to order them in such a way as
    to avoid stalling, where you have long periods of test function
    invocations where no shrinks happen. This is bad because whenever
    we shrink we reduce the amount of work the shrinker has to do
    in future, and often speed up the test function, so we ideally
    wanted those shrinks to happen much earlier in the process.

    Sometimes stalls are inevitable of course - e.g. if the pass
    makes no progress, then the entire thing is just one long stall,
    but it's helpful to design it so that stalls are less likely
    in typical behaviour.

    The two easiest ways to do this are:

    * Just run the N steps in random order. As long as a
      reasonably large proportion of the operations succeed, this
      guarantees the expected stall length is quite short. The
      book keeping for making sure this does the right thing when
      it succeeds can be quite annoying.
    * When you have any sort of nested loop, loop in such a way
      that both loop variables change each time. This prevents
      stalls which occur when one particular value for the outer
      loop is impossible to make progress on, rendering the entire
      inner loop into a stall.

    However, although progress is good, too much progress can be
    a bad sign! If you're *only* seeing successful reductions,
    that's probably a sign that you are making changes that are
    too timid. Two useful things to offset this:

    * It's worth writing shrink passes which are *adaptive*, in
      the sense that when operations seem to be working really
      well we try to bundle multiple of them together. This can
      often be used to turn what would be O(m) successful calls
      into O(log(m)).
    * It's often worth trying one or two special minimal values
      before trying anything more fine grained (e.g. replacing
      the whole thing with zero).

    """

    def derived_value(fn):
        """It's useful during shrinking to have access to derived values of
        the current shrink target.

        This decorator allows you to define these as cached properties. They
        are calculated once, then cached until the shrink target changes, then
        recalculated the next time they are used."""

        def accept(self):
            try:
                return self.__derived_values[fn.__name__]
            except KeyError:
                return self.__derived_values.setdefault(fn.__name__, fn(self))

        accept.__name__ = fn.__name__
        return property(accept)

    def __init__(
        self,
        engine: "ConjectureRunner",
        initial: ConjectureData | ConjectureResult,
        predicate: ShrinkPredicateT | None,
        *,
        allow_transition: (
            Callable[[ConjectureData | ConjectureResult, ConjectureData], bool] | None
        ),
        explain: bool,
        in_target_phase: bool = False,
    ):
        """Create a shrinker for a particular engine, with a given starting
        point and predicate. When shrink() is called it will attempt to find an
        example for which predicate is True and which is strictly smaller than
        initial.

        Note that initial is a ConjectureData object, and predicate
        takes ConjectureData objects.
        """
        assert predicate is not None or allow_transition is not None
        self.engine = engine
        self.__predicate = predicate or (lambda data: True)
        self.__allow_transition = allow_transition or (lambda source, destination: True)
        self.__derived_values: dict = {}

        self.initial_size = len(initial.choices)
        # We keep track of the current best example on the shrink_target
        # attribute.
        self.shrink_target = initial
        self.clear_change_tracking()
        self.shrinks = 0

        # We terminate shrinks that seem to have reached their logical
        # conclusion: If we've called the underlying test function at
        # least self.max_stall times since the last time we shrunk,
        # it's time to stop shrinking.
        self.max_stall = 200
        self.initial_calls = self.engine.call_count
        self.initial_misaligned = self.engine.misaligned_count
        self.calls_at_last_shrink = self.initial_calls

        self.shrink_passes: list[ShrinkPass] = [
            ShrinkPass(self.try_trivial_spans),
            self.node_program("X" * 5),
            self.node_program("X" * 4),
            self.node_program("X" * 3),
            self.node_program("X" * 2),
            self.node_program("X" * 1),
            ShrinkPass(self.pass_to_descendant),
            ShrinkPass(self.reorder_spans),
            ShrinkPass(self.minimize_duplicated_choices),
            ShrinkPass(self.minimize_individual_choices),
            ShrinkPass(self.redistribute_numeric_pairs),
            ShrinkPass(self.lower_integers_together),
            ShrinkPass(self.lower_duplicated_characters),
        ]

        # Because the shrinker is also used to `pareto_optimise` in the target phase,
        # we sometimes want to allow extending buffers instead of aborting at the end.
        self.__extend: Literal["full"] | int = "full" if in_target_phase else 0
        self.should_explain = explain

    @derived_value  # type: ignore
    def cached_calculations(self):
        return {}

    def cached(self, *keys):
        def accept(f):
            cache_key = (f.__name__, *keys)
            try:
                return self.cached_calculations[cache_key]
            except KeyError:
                return self.cached_calculations.setdefault(cache_key, f())

        return accept

    @property
    def calls(self) -> int:
        """Return the number of calls that have been made to the underlying
        test function."""
        return self.engine.call_count

    @property
    def misaligned(self) -> int:
        return self.engine.misaligned_count

    def check_calls(self) -> None:
        if self.calls - self.calls_at_last_shrink >= self.max_stall:
            raise StopShrinking

    def cached_test_function(
        self, nodes: Sequence[ChoiceNode]
    ) -> tuple[bool, ConjectureResult | _Overrun | None]:
        nodes = nodes[: len(self.nodes)]

        if startswith(nodes, self.nodes):
            return (True, None)

        if sort_key(self.nodes) < sort_key(nodes):
            return (False, None)

        # sometimes our shrinking passes try obviously invalid things. We handle
        # discarding them in one place here.
        if any(not choice_permitted(node.value, node.constraints) for node in nodes):
            return (False, None)

        result = self.engine.cached_test_function(
            [n.value for n in nodes], extend=self.__extend
        )
        previous = self.shrink_target
        self.incorporate_test_data(result)
        self.check_calls()
        return (previous is not self.shrink_target, result)

    def consider_new_nodes(self, nodes: Sequence[ChoiceNode]) -> bool:
        return self.cached_test_function(nodes)[0]

    def incorporate_test_data(self, data):
        """Takes a ConjectureData or Overrun object updates the current
        shrink_target if this data represents an improvement over it."""
        if data.status < Status.VALID or data is self.shrink_target:
            return
        if (
            self.__predicate(data)
            and sort_key(data.nodes) < sort_key(self.shrink_target.nodes)
            and self.__allow_transition(self.shrink_target, data)
        ):
            self.update_shrink_target(data)

    def debug(self, msg: str) -> None:
        self.engine.debug(msg)

    @property
    def random(self) -> "Random":
        return self.engine.random

    def shrink(self) -> None:
        """Run the full set of shrinks and update shrink_target.

        This method is "mostly idempotent" - calling it twice is unlikely to
        have any effect, though it has a non-zero probability of doing so.
        """

        try:
            self.initial_coarse_reduction()
            self.greedy_shrink()
        except StopShrinking:
            # If we stopped shrinking because we're making slow progress (instead of
            # reaching a local optimum), don't run the explain-phase logic.
            self.should_explain = False
        finally:
            if self.engine.report_debug_info:

                def s(n):
                    return "s" if n != 1 else ""

                total_deleted = self.initial_size - len(self.shrink_target.choices)
                calls = self.engine.call_count - self.initial_calls
                misaligned = self.engine.misaligned_count - self.initial_misaligned

                self.debug(
                    "---------------------\n"
                    "Shrink pass profiling\n"
                    "---------------------\n\n"
                    f"Shrinking made a total of {calls} call{s(calls)} of which "
                    f"{self.shrinks} shrank and {misaligned} were misaligned. This "
                    f"deleted {total_deleted} choices out of {self.initial_size}."
                )
                for useful in [True, False]:
                    self.debug("")
                    if useful:
                        self.debug("Useful passes:")
                    else:
                        self.debug("Useless passes:")
                    self.debug("")
                    for pass_ in sorted(
                        self.shrink_passes,
                        key=lambda t: (-t.calls, t.deletions, t.shrinks),
                    ):
                        if pass_.calls == 0:
                            continue
                        if (pass_.shrinks != 0) != useful:
                            continue

                        self.debug(
                            f"  * {pass_.name} made {pass_.calls} call{s(pass_.calls)} of which "
                            f"{pass_.shrinks} shrank and {pass_.misaligned} were misaligned, "
                            f"deleting {pass_.deletions} choice{s(pass_.deletions)}."
                        )
                self.debug("")
        self.explain()

    def explain(self) -> None:

        if not self.should_explain or not self.shrink_target.arg_slices:
            return

        self.max_stall = 2**100
        shrink_target = self.shrink_target
        nodes = self.nodes
        choices = self.choices
        chunks: dict[tuple[int, int], list[tuple[ChoiceT, ...]]] = defaultdict(list)

        # Before we start running experiments, let's check for known inputs which would
        # make them redundant.  The shrinking process means that we've already tried many
        # variations on the minimal example, so this can save a lot of time.
        seen_passing_seq = self.engine.passing_choice_sequences(
            prefix=self.nodes[: min(self.shrink_target.arg_slices)[0]]
        )

        # Now that we've shrunk to a minimal failing example, it's time to try
        # varying each part that we've noted will go in the final report.  Consider
        # slices in largest-first order
        for start, end in sorted(
            self.shrink_target.arg_slices, key=lambda x: (-(x[1] - x[0]), x)
        ):
            # Check for any previous examples that match the prefix and suffix,
            # so we can skip if we found a passing example while shrinking.
            if any(
                startswith(seen, nodes[:start]) and endswith(seen, nodes[end:])
                for seen in seen_passing_seq
            ):
                continue

            # Run our experiments
            n_same_failures = 0
            note = "or any other generated value"
            # TODO: is 100 same-failures out of 500 attempts a good heuristic?
            for n_attempt in range(500):  # pragma: no branch
                # no-branch here because we don't coverage-test the abort-at-500 logic.

                if n_attempt - 10 > n_same_failures * 5:
                    # stop early if we're seeing mostly invalid examples
                    break  # pragma: no cover

                # replace start:end with random values
                replacement = []
                for i in range(start, end):
                    node = nodes[i]
                    if not node.was_forced:
                        value = draw_choice(
                            node.type, node.constraints, random=self.random
                        )
                        node = node.copy(with_value=value)
                    replacement.append(node.value)

                attempt = choices[:start] + tuple(replacement) + choices[end:]
                result = self.engine.cached_test_function(attempt, extend="full")

                if result.status is Status.OVERRUN:
                    continue  # pragma: no cover  # flakily covered
                result = cast(ConjectureResult, result)
                if not (
                    len(attempt) == len(result.choices)
                    and endswith(result.nodes, nodes[end:])
                ):
                    # Turns out this was a variable-length part, so grab the infix...
                    for span1, span2 in zip(
                        shrink_target.spans, result.spans, strict=False
                    ):
                        assert span1.start == span2.start
                        assert span1.start <= start
                        assert span1.label == span2.label
                        if span1.start == start and span1.end == end:
                            result_end = span2.end
                            break
                    else:
                        raise NotImplementedError("Expected matching prefixes")

                    attempt = (
                        choices[:start]
                        + result.choices[start:result_end]
                        + choices[end:]
                    )
                    chunks[(start, end)].append(result.choices[start:result_end])
                    result = self.engine.cached_test_function(attempt)

                    if result.status is Status.OVERRUN:
                        continue  # pragma: no cover  # flakily covered
                    result = cast(ConjectureResult, result)
                else:
                    chunks[(start, end)].append(result.choices[start:end])

                if shrink_target is not self.shrink_target:  # pragma: no cover
                    # If we've shrunk further without meaning to, bail out.
                    self.shrink_target.slice_comments.clear()
                    return
                if result.status is Status.VALID:
                    # The test passed, indicating that this param can't vary freely.
                    # However, it's really hard to write a simple and reliable covering
                    # test, because of our `seen_passing_buffers` check above.
                    break  # pragma: no cover
                if self.__predicate(result):  # pragma: no branch
                    n_same_failures += 1
                    if n_same_failures >= 100:
                        self.shrink_target.slice_comments[(start, end)] = note
                        break

        # Finally, if we've found multiple independently-variable parts, check whether
        # they can all be varied together.
        if len(self.shrink_target.slice_comments) <= 1:
            return
        n_same_failures_together = 0
        chunks_by_start_index = sorted(chunks.items())
        for _ in range(500):  # pragma: no branch
            # no-branch here because we don't coverage-test the abort-at-500 logic.
            new_choices: list[ChoiceT] = []
            prev_end = 0
            for (start, end), ls in chunks_by_start_index:
                assert prev_end <= start < end, "these chunks must be nonoverlapping"
                new_choices.extend(choices[prev_end:start])
                new_choices.extend(self.random.choice(ls))
                prev_end = end

            result = self.engine.cached_test_function(new_choices)

            # This *can't* be a shrink because none of the components were.
            assert shrink_target is self.shrink_target
            if result.status == Status.VALID:
                self.shrink_target.slice_comments[(0, 0)] = (
                    "The test sometimes passed when commented parts were varied together."
                )
                break  # Test passed, this param can't vary freely.
            if self.__predicate(result):  # pragma: no branch
                n_same_failures_together += 1
                if n_same_failures_together >= 100:
                    self.shrink_target.slice_comments[(0, 0)] = (
                        "The test always failed when commented parts were varied together."
                    )
                    break

    def greedy_shrink(self) -> None:
        """Run a full set of greedy shrinks (that is, ones that will only ever
        move to a better target) and update shrink_target appropriately.

        This method iterates to a fixed point and so is idempontent - calling
        it twice will have exactly the same effect as calling it once.
        """
        self.fixate_shrink_passes(self.shrink_passes)

    def initial_coarse_reduction(self):
        """Performs some preliminary reductions that should not be
        repeated as part of the main shrink passes.

        The main reason why these can't be included as part of shrink
        passes is that they have much more ability to make the test
        case "worse". e.g. they might rerandomise part of it, significantly
        increasing the value of individual nodes, which works in direct
        opposition to the lexical shrinking and will frequently undo
        its work.
        """
        self.reduce_each_alternative()

    @derived_value  # type: ignore
    def spans_starting_at(self):
        result = [[] for _ in self.shrink_target.nodes]
        for i, ex in enumerate(self.spans):
            # We can have zero-length spans that start at the end
            if ex.start < len(result):
                result[ex.start].append(i)
        return tuple(map(tuple, result))

    def reduce_each_alternative(self):
        """This is a pass that is designed to rerandomise use of the
        one_of strategy or things that look like it, in order to try
        to move from later strategies to earlier ones in the branch
        order.

        It does this by trying to systematically lower each value it
        finds that looks like it might be the branch decision for
        one_of, and then attempts to repair any changes in shape that
        this causes.
        """
        i = 0
        while i < len(self.shrink_target.nodes):
            nodes = self.shrink_target.nodes
            node = nodes[i]
            if (
                node.type == "integer"
                and not node.was_forced
                and node.value <= 10
                and node.constraints["min_value"] == 0
            ):
                assert isinstance(node.value, int)

                # We've found a plausible candidate for a ``one_of`` choice.
                # We now want to see if the shape of the test case actually depends
                # on it. If it doesn't, then we don't need to do this (comparatively
                # costly) pass, and can let much simpler lexicographic reduction
                # handle it later.
                #
                # We test this by trying to set the value to zero and seeing if the
                # shape changes, as measured by either changing the number of subsequent
                # nodes, or changing the nodes in such a way as to cause one of the
                # previous values to no longer be valid in its position.
                zero_attempt = self.cached_test_function(
                    nodes[:i] + (nodes[i].copy(with_value=0),) + nodes[i + 1 :]
                )[1]
                if (
                    zero_attempt is not self.shrink_target
                    and zero_attempt is not None
                    and zero_attempt.status >= Status.VALID
                ):
                    changed_shape = len(zero_attempt.nodes) != len(nodes)

                    if not changed_shape:
                        for j in range(i + 1, len(nodes)):
                            zero_node = zero_attempt.nodes[j]
                            orig_node = nodes[j]
                            if (
                                zero_node.type != orig_node.type
                                or not choice_permitted(
                                    orig_node.value, zero_node.constraints
                                )
                            ):
                                changed_shape = True
                                break
                    if changed_shape:
                        for v in range(node.value):
                            if self.try_lower_node_as_alternative(i, v):
                                break
            i += 1

    def try_lower_node_as_alternative(self, i, v):
        """Attempt to lower `self.shrink_target.nodes[i]` to `v`,
        while rerandomising and attempting to repair any subsequent
        changes to the shape of the test case that this causes."""
        nodes = self.shrink_target.nodes
        if self.consider_new_nodes(
            nodes[:i] + (nodes[i].copy(with_value=v),) + nodes[i + 1 :]
        ):
            return True

        prefix = nodes[:i] + (nodes[i].copy(with_value=v),)
        initial = self.shrink_target
        spans = self.spans_starting_at[i]
        for _ in range(3):
            random_attempt = self.engine.cached_test_function(
                [n.value for n in prefix], extend=len(nodes)
            )
            if random_attempt.status < Status.VALID:
                continue
            self.incorporate_test_data(random_attempt)
            for j in spans:
                initial_span = initial.spans[j]
                attempt_span = random_attempt.spans[j]
                contents = random_attempt.nodes[attempt_span.start : attempt_span.end]
                self.consider_new_nodes(
                    nodes[:i] + contents + nodes[initial_span.end :]
                )
                if initial is not self.shrink_target:
                    return True
        return False

    @derived_value  # type: ignore
    def shrink_pass_choice_trees(self) -> dict[Any, ChoiceTree]:
        return defaultdict(ChoiceTree)

    def step(self, shrink_pass: ShrinkPass, *, random_order: bool = False) -> bool:
        tree = self.shrink_pass_choice_trees[shrink_pass]
        if tree.exhausted:
            return False

        initial_shrinks = self.shrinks
        initial_calls = self.calls
        initial_misaligned = self.misaligned
        size = len(self.shrink_target.choices)
        assert shrink_pass.name is not None
        self.engine.explain_next_call_as(shrink_pass.name)

        if random_order:
            selection_order = random_selection_order(self.random)
        else:
            selection_order = prefix_selection_order(shrink_pass.last_prefix)

        try:
            shrink_pass.last_prefix = tree.step(
                selection_order,
                lambda chooser: shrink_pass.function(chooser),
            )
        finally:
            shrink_pass.calls += self.calls - initial_calls
            shrink_pass.misaligned += self.misaligned - initial_misaligned
            shrink_pass.shrinks += self.shrinks - initial_shrinks
            shrink_pass.deletions += size - len(self.shrink_target.choices)
            self.engine.clear_call_explanation()
        return True

    def fixate_shrink_passes(self, passes: list[ShrinkPass]) -> None:
        """Run steps from each pass in ``passes`` until the current shrink target
        is a fixed point of all of them."""
        any_ran = True
        while any_ran:
            any_ran = False

            reordering = {}

            # We run remove_discarded after every pass to do cleanup
            # keeping track of whether that actually works. Either there is
            # no discarded data and it is basically free, or it reliably works
            # and deletes data, or it doesn't work. In that latter case we turn
            # it off for the rest of this loop through the passes, but will
            # try again once all of the passes have been run.
            can_discard = self.remove_discarded()

            calls_at_loop_start = self.calls

            # We keep track of how many calls can be made by a single step
            # without making progress and use this to test how much to pad
            # out self.max_stall by as we go along.
            max_calls_per_failing_step = 1

            for sp in passes:
                if can_discard:
                    can_discard = self.remove_discarded()

                before_sp = self.shrink_target

                # Run the shrink pass until it fails to make any progress
                # max_failures times in a row. This implicitly boosts shrink
                # passes that are more likely to work.
                failures = 0
                max_failures = 20
                while failures < max_failures:
                    # We don't allow more than max_stall consecutive failures
                    # to shrink, but this means that if we're unlucky and the
                    # shrink passes are in a bad order where only the ones at
                    # the end are useful, if we're not careful this heuristic
                    # might stop us before we've tried everything. In order to
                    # avoid that happening, we make sure that there's always
                    # plenty of breathing room to make it through a single
                    # iteration of the fixate_shrink_passes loop.
                    self.max_stall = max(
                        self.max_stall,
                        2 * max_calls_per_failing_step
                        + (self.calls - calls_at_loop_start),
                    )

                    prev = self.shrink_target
                    initial_calls = self.calls
                    # It's better for us to run shrink passes in a deterministic
                    # order, to avoid repeat work, but this can cause us to create
                    # long stalls when there are a lot of steps which fail to do
                    # anything useful. In order to avoid this, once we've noticed
                    # we're in a stall (i.e. half of max_failures calls have failed
                    # to do anything) we switch to randomly jumping around. If we
                    # find a success then we'll resume deterministic order from
                    # there which, with any luck, is in a new good region.
                    if not self.step(sp, random_order=failures >= max_failures // 2):
                        # step returns False when there is nothing to do because
                        # the entire choice tree is exhausted. If this happens
                        # we break because we literally can't run this pass any
                        # more than we already have until something else makes
                        # progress.
                        break
                    any_ran = True

                    # Don't count steps that didn't actually try to do
                    # anything as failures. Otherwise, this call is a failure
                    # if it failed to make any changes to the shrink target.
                    if initial_calls != self.calls:
                        if prev is not self.shrink_target:
                            failures = 0
                        else:
                            max_calls_per_failing_step = max(
                                max_calls_per_failing_step, self.calls - initial_calls
                            )
                            failures += 1

                # We reorder the shrink passes so that on our next run through
                # we try good ones first. The rule is that shrink passes that
                # did nothing useful are the worst, shrink passes that reduced
                # the length are the best.
                if self.shrink_target is before_sp:
                    reordering[sp] = 1
                elif len(self.choices) < len(before_sp.choices):
                    reordering[sp] = -1
                else:
                    reordering[sp] = 0

            passes.sort(key=reordering.__getitem__)

    @property
    def nodes(self) -> tuple[ChoiceNode, ...]:
        return self.shrink_target.nodes

    @property
    def choices(self) -> tuple[ChoiceT, ...]:
        return self.shrink_target.choices

    @property
    def spans(self) -> Spans:
        return self.shrink_target.spans

    @derived_value  # type: ignore
    def spans_by_label(self):
        """
        A mapping of labels to a list of spans with that label. Spans in the list
        are ordered by their normal index order.
        """

        spans_by_label = defaultdict(list)
        for ex in self.spans:
            spans_by_label[ex.label].append(ex)
        return dict(spans_by_label)

    @derived_value  # type: ignore
    def distinct_labels(self):
        return sorted(self.spans_by_label, key=str)

    def pass_to_descendant(self, chooser):
        """Attempt to replace each span with a descendant span.

        This is designed to deal with strategies that call themselves
        recursively. For example, suppose we had:

        binary_tree = st.deferred(
            lambda: st.one_of(
                st.integers(), st.tuples(binary_tree, binary_tree)))

        This pass guarantees that we can replace any binary tree with one of
        its subtrees - each of those will create an interval that the parent
        could validly be replaced with, and this pass will try doing that.

        This is pretty expensive - it takes O(len(intervals)^2) - so we run it
        late in the process when we've got the number of intervals as far down
        as possible.
        """

        label = chooser.choose(
            self.distinct_labels, lambda l: len(self.spans_by_label[l]) >= 2
        )

        spans = self.spans_by_label[label]
        i = chooser.choose(range(len(spans) - 1))
        ancestor = spans[i]

        if i + 1 == len(spans) or spans[i + 1].start >= ancestor.end:
            return

        @self.cached(label, i)
        def descendants():
            lo = i + 1
            hi = len(spans)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if spans[mid].start >= ancestor.end:
                    hi = mid
                else:
                    lo = mid
            return [
                span
                for span in spans[i + 1 : hi]
                if span.choice_count < ancestor.choice_count
            ]

        descendant = chooser.choose(descendants, lambda ex: ex.choice_count > 0)

        assert ancestor.start <= descendant.start
        assert ancestor.end >= descendant.end
        assert descendant.choice_count < ancestor.choice_count

        self.consider_new_nodes(
            self.nodes[: ancestor.start]
            + self.nodes[descendant.start : descendant.end]
            + self.nodes[ancestor.end :]
        )

    def lower_common_node_offset(self):
        """Sometimes we find ourselves in a situation where changes to one part
        of the choice sequence unlock changes to other parts. Sometimes this is
        good, but sometimes this can cause us to exhibit exponential slow
        downs!

        e.g. suppose we had the following:

        m = draw(integers(min_value=0))
        n = draw(integers(min_value=0))
        assert abs(m - n) > 1

        If this fails then we'll end up with a loop where on each iteration we
        reduce each of m and n by 2 - m can't go lower because of n, then n
        can't go lower because of m.

        This will take us O(m) iterations to complete, which is exponential in
        the data size, as we gradually zig zag our way towards zero.

        This can only happen if we're failing to reduce the size of the choice
        sequence: The number of iterations that reduce the length of the choice
        sequence is bounded by that length.

        So what we do is this: We keep track of which nodes are changing, and
        then if there's some non-zero common offset to them we try and minimize
        them all at once by lowering that offset.

        This may not work, and it definitely won't get us out of all possible
        exponential slow downs (an example of where it doesn't is where the
        shape of the nodes changes as a result of this bouncing behaviour),
        but it fails fast when it doesn't work and gets us out of a really
        nastily slow case when it does.
        """
        if len(self.__changed_nodes) <= 1:
            return

        changed = []
        for i in sorted(self.__changed_nodes):
            node = self.nodes[i]
            if node.trivial or node.type != "integer":
                continue
            changed.append(node)

        if not changed:
            return

        ints = [
            abs(node.value - node.constraints["shrink_towards"]) for node in changed
        ]
        offset = min(ints)
        assert offset > 0

        for i in range(len(ints)):
            ints[i] -= offset

        st = self.shrink_target

        def offset_node(node, n):
            return (
                node.index,
                node.index + 1,
                [node.copy(with_value=node.constraints["shrink_towards"] + n)],
            )

        def consider(n, sign):
            return self.consider_new_nodes(
                replace_all(
                    st.nodes,
                    [
                        offset_node(node, sign * (n + v))
                        for node, v in zip(changed, ints, strict=False)
                    ],
                )
            )

        # shrink from both sides
        Integer.shrink(offset, lambda n: consider(n, 1))
        Integer.shrink(offset, lambda n: consider(n, -1))
        self.clear_change_tracking()

    def clear_change_tracking(self):
        self.__last_checked_changed_at = self.shrink_target
        self.__all_changed_nodes = set()

    def mark_changed(self, i):
        self.__changed_nodes.add(i)

    @property
    def __changed_nodes(self) -> set[int]:
        if self.__last_checked_changed_at is self.shrink_target:
            return self.__all_changed_nodes

        prev_target = self.__last_checked_changed_at
        new_target = self.shrink_target
        assert prev_target is not new_target
        prev_nodes = prev_target.nodes
        new_nodes = new_target.nodes
        assert sort_key(new_target.nodes) < sort_key(prev_target.nodes)

        if len(prev_nodes) != len(new_nodes) or any(
            n1.type != n2.type for n1, n2 in zip(prev_nodes, new_nodes, strict=True)
        ):
            # should we check constraints are equal as well?
            self.__all_changed_nodes = set()
        else:
            assert len(prev_nodes) == len(new_nodes)
            for i, (n1, n2) in enumerate(zip(prev_nodes, new_nodes, strict=True)):
                assert n1.type == n2.type
                if not choice_equal(n1.value, n2.value):
                    self.__all_changed_nodes.add(i)

        return self.__all_changed_nodes

    def update_shrink_target(self, new_target):
        assert isinstance(new_target, ConjectureResult)
        self.shrinks += 1
        # If we are just taking a long time to shrink we don't want to
        # trigger this heuristic, so whenever we shrink successfully
        # we give ourselves a bit of breathing room to make sure we
        # would find a shrink that took that long to find the next time.
        # The case where we're taking a long time but making steady
        # progress is handled by `finish_shrinking_deadline` in engine.py
        self.max_stall = max(
            self.max_stall, (self.calls - self.calls_at_last_shrink) * 2
        )
        self.calls_at_last_shrink = self.calls
        self.shrink_target = new_target
        self.__derived_values = {}

    def try_shrinking_nodes(self, nodes, n):
        """Attempts to replace each node in the nodes list with n. Returns
        True if it succeeded (which may include some additional modifications
        to shrink_target).

        In current usage it is expected that each of the nodes currently have
        the same value and choice_type, although this is not essential. Note that
        n must be < the node at min(nodes) or this is not a valid shrink.

        This method will attempt to do some small amount of work to delete data
        that occurs after the end of the nodes. This is useful for cases where
        there is some size dependency on the value of a node.
        """
        # If the length of the shrink target has changed from under us such that
        # the indices are out of bounds, give up on the replacement.
        # TODO_BETTER_SHRINK: we probably want to narrow down the root cause here at some point.
        if any(node.index >= len(self.nodes) for node in nodes):
            return  # pragma: no cover

        initial_attempt = replace_all(
            self.nodes,
            [(node.index, node.index + 1, [node.copy(with_value=n)]) for node in nodes],
        )

        attempt = self.cached_test_function(initial_attempt)[1]

        if attempt is None:
            return False

        if attempt is self.shrink_target:
            # if the initial shrink was a success, try lowering offsets.
            self.lower_common_node_offset()
            return True

        # If this produced something completely invalid we ditch it
        # here rather than trying to persevere.
        if attempt.status is Status.OVERRUN:
            return False

        if attempt.status is Status.INVALID:
            return False

        if attempt.misaligned_at is not None:
            # we're invalid due to a misalignment in the tree. We'll try to fix
            # a very specific type of misalignment here: where we have a node of
            # {"size": n} and tried to draw the same node, but with {"size": m < n}.
            # This can occur with eg
            #
            #   n = data.draw_integer()
            #   s = data.draw_string(min_size=n)
            #
            # where we try lowering n, resulting in the test_function drawing a lower
            # min_size than our attempt had for the draw_string node.
            #
            # We'll now try realigning this tree by:
            # * replacing the constraints in our attempt with what test_function tried
            #   to draw in practice
            # * truncating the value of that node to match min_size
            #
            # This helps in the specific case of drawing a value and then drawing
            # a collection of that size...and not much else. In practice this
            # helps because this antipattern is fairly common.

            # TODO we'll probably want to apply the same trick as in the valid
            # case of this function of preserving from the right instead of
            # preserving from the left. see test_can_shrink_variable_string_draws.

            (index, attempt_choice_type, attempt_constraints, _attempt_forced) = (
                attempt.misaligned_at
            )
            node = self.nodes[index]
            if node.type != attempt_choice_type:
                return False  # pragma: no cover
            if node.was_forced:
                return False  # pragma: no cover

            if node.type in {"string", "bytes"}:
                # if the size *increased*, we would have to guess what to pad with
                # in order to try fixing up this attempt. Just give up.
                if node.constraints["min_size"] <= attempt_constraints["min_size"]:
                    # attempts which increase min_size tend to overrun rather than
                    # be misaligned, making a covering case difficult.
                    return False  # pragma: no cover
                # the size decreased in our attempt. Try again, but truncate the value
                # to that size by removing any elements past min_size.
                return self.consider_new_nodes(
                    initial_attempt[: node.index]
                    + [
                        initial_attempt[node.index].copy(
                            with_constraints=attempt_constraints,
                            with_value=initial_attempt[node.index].value[
                                : attempt_constraints["min_size"]
                            ],
                        )
                    ]
                    + initial_attempt[node.index :]
                )

        lost_nodes = len(self.nodes) - len(attempt.nodes)
        if lost_nodes <= 0:
            return False

        start = nodes[0].index
        end = nodes[-1].index + 1
        # We now look for contiguous regions to delete that might help fix up
        # this failed shrink. We only look for contiguous regions of the right
        # lengths because doing anything more than that starts to get very
        # expensive. See minimize_individual_choices for where we
        # try to be more aggressive.
        regions_to_delete = {(end, end + lost_nodes)}

        for ex in self.spans:
            if ex.start > start:
                continue
            if ex.end <= end:
                continue

            if ex.index >= len(attempt.spans):
                continue  # pragma: no cover

            replacement = attempt.spans[ex.index]
            in_original = [c for c in ex.children if c.start >= end]
            in_replaced = [c for c in replacement.children if c.start >= end]

            if len(in_replaced) >= len(in_original) or not in_replaced:
                continue

            # We've found a span where some of the children went missing
            # as a result of this change, and just replacing it with the data
            # it would have had and removing the spillover didn't work. This
            # means that some of its children towards the right must be
            # important, so we try to arrange it so that it retains its
            # rightmost children instead of its leftmost.
            regions_to_delete.add(
                (in_original[0].start, in_original[-len(in_replaced)].start)
            )

        for u, v in sorted(regions_to_delete, key=lambda x: x[1] - x[0], reverse=True):
            try_with_deleted = initial_attempt[:u] + initial_attempt[v:]
            if self.consider_new_nodes(try_with_deleted):
                return True

        return False

    def remove_discarded(self):
        """Try removing all bytes marked as discarded.

        This is primarily to deal with data that has been ignored while
        doing rejection sampling - e.g. as a result of an integer range, or a
        filtered strategy.

        Such data will also be handled by the adaptive_example_deletion pass,
        but that pass is necessarily more conservative and will try deleting
        each interval individually. The common case is that all data drawn and
        rejected can just be thrown away immediately in one block, so this pass
        will be much faster than trying each one individually when it works.

        returns False if there is discarded data and removing it does not work,
        otherwise returns True.
        """
        while self.shrink_target.has_discards:
            discarded = []

            for ex in self.shrink_target.spans:
                if (
                    ex.choice_count > 0
                    and ex.discarded
                    and (not discarded or ex.start >= discarded[-1][-1])
                ):
                    discarded.append((ex.start, ex.end))

            # This can happen if we have discards but they are all of
            # zero length. This shouldn't happen very often so it's
            # faster to check for it here than at the point of example
            # generation.
            if not discarded:
                break

            attempt = list(self.nodes)
            for u, v in reversed(discarded):
                del attempt[u:v]

            if not self.consider_new_nodes(tuple(attempt)):
                return False
        return True

    @derived_value  # type: ignore
    def duplicated_nodes(self):
        """Returns a list of nodes grouped (choice_type, value)."""
        duplicates = defaultdict(list)
        for node in self.nodes:
            duplicates[(node.type, choice_key(node.value))].append(node)
        return list(duplicates.values())

    def node_program(self, program: str) -> ShrinkPass:
        return ShrinkPass(
            lambda chooser: self._node_program(chooser, program),
            name=f"node_program_{program}",
        )

    def _node_program(self, chooser, program):
        n = len(program)
        # Adaptively attempt to run the node program at the current
        # index. If this successfully applies the node program ``k`` times
        # then this runs in ``O(log(k))`` test function calls.
        i = chooser.choose(range(len(self.nodes) - n + 1))

        # First, run the node program at the chosen index. If this fails,
        # don't do any extra work, so that failure is as cheap as possible.
        if not self.run_node_program(i, program, original=self.shrink_target):
            return

        # Because we run in a random order we will often find ourselves in the middle
        # of a region where we could run the node program. We thus start by moving
        # left to the beginning of that region if possible in order to to start from
        # the beginning of that region.
        def offset_left(k):
            return i - k * n

        i = offset_left(
            find_integer(
                lambda k: self.run_node_program(
                    offset_left(k), program, original=self.shrink_target
                )
            )
        )

        original = self.shrink_target
        # Now try to run the node program multiple times here.
        find_integer(
            lambda k: self.run_node_program(i, program, original=original, repeats=k)
        )

    def minimize_duplicated_choices(self, chooser):
        """Find choices that have been duplicated in multiple places and attempt
        to minimize all of the duplicates simultaneously.

        This lets us handle cases where two values can't be shrunk
        independently of each other but can easily be shrunk together.
        For example if we had something like:

        ls = data.draw(lists(integers()))
        y = data.draw(integers())
        assert y not in ls

        Suppose we drew y = 3 and after shrinking we have ls = [3]. If we were
        to replace both 3s with 0, this would be a valid shrink, but if we were
        to replace either 3 with 0 on its own the test would start passing.

        It is also useful for when that duplication is accidental and the value
        of the choices don't matter very much because it allows us to replace
        more values at once.
        """
        nodes = chooser.choose(self.duplicated_nodes)
        # we can't lower any nodes which are trivial. try proceeding with the
        # remaining nodes.
        nodes = [node for node in nodes if not node.trivial]
        if len(nodes) <= 1:
            return

        self.minimize_nodes(nodes)

    def redistribute_numeric_pairs(self, chooser):
        """If there is a sum of generated numbers that we need their sum
        to exceed some bound, lowering one of them requires raising the
        other. This pass enables that."""

        # look for a pair of nodes (node1, node2) which are both numeric
        # and aren't separated by too many other nodes. We'll decrease node1 and
        # increase node2 (note that the other way around doesn't make sense as
        # it's strictly worse in the ordering).
        def can_choose_node(node):
            # don't choose nan, inf, or floats above the threshold where f + 1 > f
            # (which is not necessarily true for floats above MAX_PRECISE_INTEGER).
            # The motivation for the last condition is to avoid trying weird
            # non-shrinks where we raise one node and think we lowered another
            # (but didn't).
            return node.type in {"integer", "float"} and not (
                node.type == "float"
                and (math.isnan(node.value) or abs(node.value) >= MAX_PRECISE_INTEGER)
            )

        node1 = chooser.choose(
            self.nodes,
            lambda node: can_choose_node(node) and not node.trivial,
        )
        node2 = chooser.choose(
            self.nodes,
            lambda node: can_choose_node(node)
            # Note that it's fine for node2 to be trivial, because we're going to
            # explicitly make it *not* trivial by adding to its value.
            and not node.was_forced
            # to avoid quadratic behavior, scan ahead only a small amount for
            # the related node.
            and node1.index < node.index <= node1.index + 4,
        )

        m: int | float = node1.value
        n: int | float = node2.value

        def boost(k: int) -> bool:
            # floats always shrink towards 0
            shrink_towards = (
                node1.constraints["shrink_towards"] if node1.type == "integer" else 0
            )
            if k > abs(m - shrink_towards):
                return False

            # We are trying to move node1 (m) closer to shrink_towards, and node2
            # (n) farther away from shrink_towards. If m is below shrink_towards,
            # we want to add to m and subtract from n, and vice versa if above
            # shrink_towards.
            if m < shrink_towards:
                k = -k

            try:
                v1 = m - k
                v2 = n + k
            except OverflowError:  # pragma: no cover
                # if n or m is a float and k is over sys.float_info.max, coercing
                # k to a float will overflow.
                return False

            # if we've increased node2 to the point that we're past max precision,
            # give up - things have become too unstable.
            if node1.type == "float" and abs(v2) >= MAX_PRECISE_INTEGER:
                return False

            return self.consider_new_nodes(
                self.nodes[: node1.index]
                + (node1.copy(with_value=v1),)
                + self.nodes[node1.index + 1 : node2.index]
                + (node2.copy(with_value=v2),)
                + self.nodes[node2.index + 1 :]
            )

        find_integer(boost)

    def lower_integers_together(self, chooser):
        node1 = chooser.choose(
            self.nodes, lambda n: n.type == "integer" and not n.trivial
        )
        # Search up to 3 nodes ahead, to avoid quadratic time.
        node2 = self.nodes[
            chooser.choose(
                range(node1.index + 1, min(len(self.nodes), node1.index + 3 + 1)),
                lambda i: self.nodes[i].type == "integer"
                and not self.nodes[i].was_forced,
            )
        ]

        # one might expect us to require node2 to be nontrivial, and to minimize
        # the node which is closer to its shrink_towards, rather than node1
        # unconditionally. In reality, it's acceptable for us to transition node2
        # from trivial to nontrivial, because the shrink ordering is dominated by
        # the complexity of the earlier node1. What matters is minimizing node1.
        shrink_towards = node1.constraints["shrink_towards"]

        def consider(n):
            return self.consider_new_nodes(
                self.nodes[: node1.index]
                + (node1.copy(with_value=node1.value - n),)
                + self.nodes[node1.index + 1 : node2.index]
                + (node2.copy(with_value=node2.value - n),)
                + self.nodes[node2.index + 1 :]
            )

        find_integer(lambda n: consider(shrink_towards - n))
        find_integer(lambda n: consider(n - shrink_towards))

    def lower_duplicated_characters(self, chooser):
        """
        Select two string choices no more than 4 choices apart and simultaneously
        lower characters which appear in both strings. This helps cases where the
        same character must appear in two strings, but the actual value of the
        character is not relevant.

        This shrinking pass currently only tries lowering *all* instances of the
        duplicated character in both strings. So for instance, given two choices:

            "bbac"
            "abbb"

        we would try lowering all five of the b characters simultaneously. This
        may fail to shrink some cases where only certain character indices are
        correlated, for instance if only the b at index 1 could be lowered
        simultaneously and the rest did in fact actually have to be a `b`.

        It would be nice to try shrinking that case as well, but we would need good
        safeguards because it could get very expensive to try all combinations.
        I expect lowering all duplicates to handle most cases in the meantime.
        """
        node1 = chooser.choose(
            self.nodes, lambda n: n.type == "string" and not n.trivial
        )

        # limit search to up to 4 choices ahead, to avoid quadratic behavior
        node2 = self.nodes[
            chooser.choose(
                range(node1.index + 1, min(len(self.nodes), node1.index + 1 + 4)),
                lambda i: self.nodes[i].type == "string" and not self.nodes[i].trivial
                # select nodes which have at least one of the same character present
                and set(node1.value) & set(self.nodes[i].value),
            )
        ]

        duplicated_characters = set(node1.value) & set(node2.value)
        # deterministic ordering
        char = chooser.choose(sorted(duplicated_characters))
        intervals = node1.constraints["intervals"]

        def copy_node(node, n):
            # replace all duplicate characters in each string. This might miss
            # some shrinks compared to only replacing some, but trying all possible
            # combinations of indices could get expensive if done without some
            # thought.
            return node.copy(
                with_value=node.value.replace(char, intervals.char_in_shrink_order(n))
            )

        Integer.shrink(
            intervals.index_from_char_in_shrink_order(char),
            lambda n: self.consider_new_nodes(
                self.nodes[: node1.index]
                + (copy_node(node1, n),)
                + self.nodes[node1.index + 1 : node2.index]
                + (copy_node(node2, n),)
                + self.nodes[node2.index + 1 :]
            ),
        )

    def minimize_nodes(self, nodes):
        choice_type = nodes[0].type
        value = nodes[0].value
        # unlike choice_type and value, constraints are *not* guaranteed to be equal among all
        # passed nodes. We arbitrarily use the constraints of the first node. I think
        # this is unsound (= leads to us trying shrinks that could not have been
        # generated), but those get discarded at test-time, and this enables useful
        # slips where constraints are not equal but are close enough that doing the
        # same operation on both basically just works.
        constraints = nodes[0].constraints
        assert all(
            node.type == choice_type and choice_equal(node.value, value)
            for node in nodes
        )

        if choice_type == "integer":
            shrink_towards = constraints["shrink_towards"]
            # try shrinking from both sides towards shrink_towards.
            # we're starting from n = abs(shrink_towards - value). Because the
            # shrinker will not check its starting value, we need to try
            # shrinking to n first.
            self.try_shrinking_nodes(nodes, abs(shrink_towards - value))
            Integer.shrink(
                abs(shrink_towards - value),
                lambda n: self.try_shrinking_nodes(nodes, shrink_towards + n),
            )
            Integer.shrink(
                abs(shrink_towards - value),
                lambda n: self.try_shrinking_nodes(nodes, shrink_towards - n),
            )
        elif choice_type == "float":
            self.try_shrinking_nodes(nodes, abs(value))
            Float.shrink(
                abs(value),
                lambda val: self.try_shrinking_nodes(nodes, val),
            )
            Float.shrink(
                abs(value),
                lambda val: self.try_shrinking_nodes(nodes, -val),
            )
        elif choice_type == "boolean":
            # must be True, otherwise would be trivial and not selected.
            assert value is True
            # only one thing to try: false!
            self.try_shrinking_nodes(nodes, False)
        elif choice_type == "bytes":
            Bytes.shrink(
                value,
                lambda val: self.try_shrinking_nodes(nodes, val),
                min_size=constraints["min_size"],
            )
        elif choice_type == "string":
            String.shrink(
                value,
                lambda val: self.try_shrinking_nodes(nodes, val),
                intervals=constraints["intervals"],
                min_size=constraints["min_size"],
            )
        else:
            raise NotImplementedError

    def try_trivial_spans(self, chooser):
        i = chooser.choose(range(len(self.spans)))

        prev = self.shrink_target
        nodes = self.shrink_target.nodes
        span = self.spans[i]
        prefix = nodes[: span.start]
        replacement = tuple(
            [
                (
                    node
                    if node.was_forced
                    else node.copy(
                        with_value=choice_from_index(0, node.type, node.constraints)
                    )
                )
                for node in nodes[span.start : span.end]
            ]
        )
        suffix = nodes[span.end :]
        attempt = self.cached_test_function(prefix + replacement + suffix)[1]

        if self.shrink_target is not prev:
            return

        if isinstance(attempt, ConjectureResult):
            new_span = attempt.spans[i]
            new_replacement = attempt.nodes[new_span.start : new_span.end]
            self.consider_new_nodes(prefix + new_replacement + suffix)

    def minimize_individual_choices(self, chooser):
        """Attempt to minimize each choice in sequence.

        This is the pass that ensures that e.g. each integer we draw is a
        minimum value. So it's the part that guarantees that if we e.g. do

        x = data.draw(integers())
        assert x < 10

        then in our shrunk example, x = 10 rather than say 97.

        If we are unsuccessful at minimizing a choice of interest we then
        check if that's because it's changing the size of the test case and,
        if so, we also make an attempt to delete parts of the test case to
        see if that fixes it.

        We handle most of the common cases in try_shrinking_nodes which is
        pretty good at clearing out large contiguous blocks of dead space,
        but it fails when there is data that has to stay in particular places
        in the list.
        """
        node = chooser.choose(self.nodes, lambda node: not node.trivial)
        initial_target = self.shrink_target

        self.minimize_nodes([node])
        if self.shrink_target is not initial_target:
            # the shrink target changed, so our shrink worked. Defer doing
            # anything more intelligent until this shrink fails.
            return

        # the shrink failed. One particularly common case where minimizing a
        # node can fail is the antipattern of drawing a size and then drawing a
        # collection of that size, or more generally when there is a size
        # dependency on some single node. We'll explicitly try and fix up this
        # common case here: if decreasing an integer node by one would reduce
        # the size of the generated input, we'll try deleting things after that
        # node and see if the resulting attempt works.

        if node.type != "integer":
            # Only try this fixup logic on integer draws. Almost all size
            # dependencies are on integer draws, and if it's not, it's doing
            # something convoluted enough that it is unlikely to shrink well anyway.
            # TODO: extent to floats? we probably currently fail on the following,
            # albeit convoluted example:
            # n = int(data.draw(st.floats()))
            # s = data.draw(st.lists(st.integers(), min_size=n, max_size=n))
            return

        lowered = (
            self.nodes[: node.index]
            + (node.copy(with_value=node.value - 1),)
            + self.nodes[node.index + 1 :]
        )
        attempt = self.cached_test_function(lowered)[1]
        if (
            attempt is None
            or attempt.status < Status.VALID
            or len(attempt.nodes) == len(self.nodes)
            or len(attempt.nodes) == node.index + 1
        ):
            # no point in trying our size-dependency-logic if our attempt at
            # lowering the node resulted in:
            # * an invalid conjecture data
            # * the same number of nodes as before
            # * no nodes beyond the lowered node (nothing to try to delete afterwards)
            return

        # If it were then the original shrink should have worked and we could
        # never have got here.
        assert attempt is not self.shrink_target

        @self.cached(node.index)
        def first_span_after_node():
            lo = 0
            hi = len(self.spans)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                span = self.spans[mid]
                if span.start >= node.index:
                    hi = mid
                else:
                    lo = mid
            return hi

        # we try deleting both entire spans, and single nodes.
        # If we wanted to get more aggressive, we could try deleting n
        # consecutive nodes (that don't cross a span boundary) for say
        # n <= 2 or n <= 3.
        if chooser.choose([True, False]):
            span = self.spans[
                chooser.choose(
                    range(first_span_after_node, len(self.spans)),
                    lambda i: self.spans[i].choice_count > 0,
                )
            ]
            self.consider_new_nodes(lowered[: span.start] + lowered[span.end :])
        else:
            node = self.nodes[chooser.choose(range(node.index + 1, len(self.nodes)))]
            self.consider_new_nodes(lowered[: node.index] + lowered[node.index + 1 :])

    def reorder_spans(self, chooser):
        """This pass allows us to reorder the children of each span.

        For example, consider the following:

        .. code-block:: python

            import hypothesis.strategies as st
            from hypothesis import given


            @given(st.text(), st.text())
            def test_not_equal(x, y):
                assert x != y

        Without the ability to reorder x and y this could fail either with
        ``x=""``, ``y="0"``, or the other way around. With reordering it will
        reliably fail with ``x=""``, ``y="0"``.
        """
        span = chooser.choose(self.spans)

        label = chooser.choose(span.children).label
        spans = [c for c in span.children if c.label == label]
        if len(spans) <= 1:
            return

        endpoints = [(span.start, span.end) for span in spans]
        st = self.shrink_target

        Ordering.shrink(
            range(len(spans)),
            lambda indices: self.consider_new_nodes(
                replace_all(
                    st.nodes,
                    [
                        (
                            u,
                            v,
                            st.nodes[spans[i].start : spans[i].end],
                        )
                        for (u, v), i in zip(endpoints, indices, strict=True)
                    ],
                )
            ),
            key=lambda i: sort_key(st.nodes[spans[i].start : spans[i].end]),
        )

    def run_node_program(self, i, program, original, repeats=1):
        """Node programs are a mini-DSL for node rewriting, defined as a sequence
        of commands that can be run at some index into the nodes

        Commands are:

            * "X", delete this node

        This method runs the node program in ``program`` at node index
        ``i`` on the ConjectureData ``original``. If ``repeats > 1`` then it
        will attempt to approximate the results of running it that many times.

        Returns True if this successfully changes the underlying shrink target,
        else False.
        """
        if i + len(program) > len(original.nodes) or i < 0:
            return False
        attempt = list(original.nodes)
        for _ in range(repeats):
            for k, command in reversed(list(enumerate(program))):
                j = i + k
                if j >= len(attempt):
                    return False

                if command == "X":
                    del attempt[j]
                else:
                    raise NotImplementedError(f"Unrecognised command {command!r}")

        return self.consider_new_nodes(attempt)
