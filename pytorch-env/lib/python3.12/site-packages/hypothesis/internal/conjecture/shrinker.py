# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union

import attr

from hypothesis.internal.compat import int_from_bytes, int_to_bytes
from hypothesis.internal.conjecture.choicetree import (
    ChoiceTree,
    prefix_selection_order,
    random_selection_order,
)
from hypothesis.internal.conjecture.data import (
    ConjectureData,
    ConjectureResult,
    Status,
    ir_value_equal,
    ir_value_key,
    ir_value_permitted,
)
from hypothesis.internal.conjecture.junkdrawer import find_integer, replace_all
from hypothesis.internal.conjecture.shrinking import (
    Bytes,
    Float,
    Integer,
    Ordering,
    String,
)

if TYPE_CHECKING:
    from random import Random

    from hypothesis.internal.conjecture.engine import ConjectureRunner

SortKeyT = TypeVar("SortKeyT", str, bytes)


def sort_key(buffer: SortKeyT) -> tuple[int, SortKeyT]:
    """Returns a sort key such that "simpler" buffers are smaller than
    "more complicated" ones.

    We define sort_key so that x is simpler than y if x is shorter than y or if
    they have the same length and x < y lexicographically. This is called the
    shortlex order.

    The reason for using the shortlex order is:

    1. If x is shorter than y then that means we had to make fewer decisions
       in constructing the test case when we ran x than we did when we ran y.
    2. If x is the same length as y then replacing a byte with a lower byte
       corresponds to reducing the value of an integer we drew with draw_bits
       towards zero.
    3. We want a total order, and given (2) the natural choices for things of
       the same size are either the lexicographic or colexicographic orders
       (the latter being the lexicographic order of the reverse of the string).
       Because values drawn early in generation potentially get used in more
       places they potentially have a more significant impact on the final
       result, so it makes sense to prioritise reducing earlier values over
       later ones. This makes the lexicographic order the more natural choice.
    """
    return (len(buffer), buffer)


SHRINK_PASS_DEFINITIONS: dict[str, "ShrinkPassDefinition"] = {}


@attr.s()
class ShrinkPassDefinition:
    """A shrink pass bundles together a large number of local changes to
    the current shrink target.

    Each shrink pass is defined by some function and some arguments to that
    function. The ``generate_arguments`` function returns all arguments that
    might be useful to run on the current shrink target.

    The guarantee made by methods defined this way is that after they are
    called then *either* the shrink target has changed *or* each of
    ``fn(*args)`` has been called for every ``args`` in ``generate_arguments(self)``.
    No guarantee is made that all of these will be called if the shrink target
    changes.
    """

    run_with_chooser = attr.ib()

    @property
    def name(self):
        return self.run_with_chooser.__name__

    def __attrs_post_init__(self):
        assert self.name not in SHRINK_PASS_DEFINITIONS, self.name
        SHRINK_PASS_DEFINITIONS[self.name] = self


def defines_shrink_pass():
    """A convenient decorator for defining shrink passes."""

    def accept(run_step):
        ShrinkPassDefinition(run_with_chooser=run_step)

        def run(self):
            raise NotImplementedError("Shrink passes should not be run directly")

        run.__name__ = run_step.__name__
        run.is_shrink_pass = True
        return run

    return accept


class Shrinker:
    """A shrinker is a child object of a ConjectureRunner which is designed to
    manage the associated state of a particular shrink problem. That is, we
    have some initial ConjectureData object and some property of interest
    that it satisfies, and we want to find a ConjectureData object with a
    shortlex (see sort_key above) smaller buffer that exhibits the same
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
    cached_test_function and/or incorporate_new_buffer a number of times,
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
    index ``i``, all examples whose start is ``<= i`` still exist,
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
    each individual byte (this isn't an especially good choice,
    but it leads to a simple illustrative example), we might do it
    by iterating over the buffer like so:

    .. code-block:: python

        i = 0
        while i < len(self.shrink_target.buffer):
            if not self.incorporate_new_buffer(
                self.shrink_target.buffer[:i] + self.shrink_target.buffer[i + 1 :]
            ):
                i += 1

    The reason for writing the loop this way is that i is always a
    valid index into the current buffer, even if the current buffer
    changes as a result of our actions. When the buffer changes,
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
        initial: Union[ConjectureData, ConjectureResult],
        predicate: Optional[Callable[[ConjectureData], bool]],
        *,
        allow_transition: Optional[
            Callable[[Union[ConjectureData, ConjectureResult], ConjectureData], bool]
        ],
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
        self.__pending_shrink_explanation = None

        self.initial_size = len(initial.buffer)

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

        self.passes_by_name: dict[str, ShrinkPass] = {}

        # Because the shrinker is also used to `pareto_optimise` in the target phase,
        # we sometimes want to allow extending buffers instead of aborting at the end.
        if in_target_phase:
            from hypothesis.internal.conjecture.engine import BUFFER_SIZE

            self.__extend = BUFFER_SIZE
        else:
            self.__extend = 0
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

    def add_new_pass(self, run):
        """Creates a shrink pass corresponding to calling ``run(self)``"""

        definition = SHRINK_PASS_DEFINITIONS[run]

        p = ShrinkPass(
            run_with_chooser=definition.run_with_chooser,
            shrinker=self,
            index=len(self.passes_by_name),
        )
        self.passes_by_name[p.name] = p
        return p

    def shrink_pass(self, name):
        """Return the ShrinkPass object for the pass with the given name."""
        if isinstance(name, ShrinkPass):
            return name
        if name not in self.passes_by_name:
            self.add_new_pass(name)
        return self.passes_by_name[name]

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

    def cached_test_function_ir(self, tree):
        # sometimes our shrinking passes try obviously invalid things. We handle
        # discarding them in one place here.
        for node in tree:
            if not ir_value_permitted(node.value, node.ir_type, node.kwargs):
                return None

        result = self.engine.cached_test_function_ir(tree)
        self.incorporate_test_data(result)
        self.check_calls()
        return result

    def consider_new_tree(self, tree):
        tree = tree[: len(self.nodes)]

        def startswith(t1, t2):
            return t1[: len(t2)] == t2

        if startswith(tree, self.nodes):
            return True

        if startswith(self.nodes, tree):
            return False

        previous = self.shrink_target
        self.cached_test_function_ir(tree)
        return previous is not self.shrink_target

    def consider_new_buffer(self, buffer):
        """Returns True if after running this buffer the result would be
        the current shrink_target."""
        buffer = bytes(buffer)
        return buffer.startswith(self.buffer) or self.incorporate_new_buffer(buffer)

    def incorporate_new_buffer(self, buffer):
        """Either runs the test function on this buffer and returns True if
        that changed the shrink_target, or determines that doing so would
        be useless and returns False without running it."""

        buffer = bytes(buffer[: self.shrink_target.index])
        # Sometimes an attempt at lexicographic minimization will do the wrong
        # thing because the buffer has changed under it (e.g. something has
        # turned into a write, the bit size has changed). The result would be
        # an invalid string, but it's better for us to just ignore it here as
        # it turns out to involve quite a lot of tricky book-keeping to get
        # this right and it's better to just handle it in one place.
        if sort_key(buffer) >= sort_key(self.shrink_target.buffer):
            return False

        if self.shrink_target.buffer.startswith(buffer):
            return False

        previous = self.shrink_target
        self.cached_test_function(buffer)
        return previous is not self.shrink_target

    def incorporate_test_data(self, data):
        """Takes a ConjectureData or Overrun object updates the current
        shrink_target if this data represents an improvement over it."""
        if data.status < Status.VALID or data is self.shrink_target:
            return
        if (
            self.__predicate(data)
            and sort_key(data.buffer) < sort_key(self.shrink_target.buffer)
            and self.__allow_transition(self.shrink_target, data)
        ):
            self.update_shrink_target(data)

    def cached_test_function(self, buffer):
        """Returns a cached version of the underlying test function, so
        that the result is either an Overrun object (if the buffer is
        too short to be a valid test case) or a ConjectureData object
        with status >= INVALID that would result from running this buffer."""
        buffer = bytes(buffer)
        result = self.engine.cached_test_function(buffer, extend=self.__extend)
        self.incorporate_test_data(result)
        self.check_calls()
        return result

    def debug(self, msg: str) -> None:
        self.engine.debug(msg)

    @property
    def random(self) -> "Random":
        return self.engine.random

    def shrink(self):
        """Run the full set of shrinks and update shrink_target.

        This method is "mostly idempotent" - calling it twice is unlikely to
        have any effect, though it has a non-zero probability of doing so.
        """
        # We assume that if an all-zero block of bytes is an interesting
        # example then we're not going to do better than that.
        # This might not technically be true: e.g. for integers() | booleans()
        # the simplest example is actually [1, 0]. Missing this case is fairly
        # harmless and this allows us to make various simplifying assumptions
        # about the structure of the data (principally that we're never
        # operating on a block of all zero bytes so can use non-zeroness as a
        # signpost of complexity).
        if not any(self.shrink_target.buffer) or self.incorporate_new_buffer(
            bytes(len(self.shrink_target.buffer))
        ):
            self.explain()
            return

        # There are multiple buffers that represent the same counterexample, eg
        # n=2 (from the 16 bit integer bucket) and n=2 (from the 32 bit integer
        # bucket). Before we start shrinking, we need to normalize to the minimal
        # such buffer, else a buffer-smaller but ir-larger value may be chosen
        # as the minimal counterexample.
        data = self.engine.new_conjecture_data_ir(self.nodes)
        self.engine.test_function(data)
        self.incorporate_test_data(data.as_result())

        try:
            self.greedy_shrink()
        except StopShrinking:
            # If we stopped shrinking because we're making slow progress (instead of
            # reaching a local optimum), don't run the explain-phase logic.
            self.should_explain = False
        finally:
            if self.engine.report_debug_info:

                def s(n):
                    return "s" if n != 1 else ""

                total_deleted = self.initial_size - len(self.shrink_target.buffer)
                calls = self.engine.call_count - self.initial_calls
                misaligned = self.engine.misaligned_count - self.initial_misaligned

                self.debug(
                    "---------------------\n"
                    "Shrink pass profiling\n"
                    "---------------------\n\n"
                    f"Shrinking made a total of {calls} call{s(calls)} of which "
                    f"{self.shrinks} shrank and {misaligned} were misaligned. This deleted {total_deleted} bytes out "
                    f"of {self.initial_size}."
                )
                for useful in [True, False]:
                    self.debug("")
                    if useful:
                        self.debug("Useful passes:")
                    else:
                        self.debug("Useless passes:")
                    self.debug("")
                    for p in sorted(
                        self.passes_by_name.values(),
                        key=lambda t: (-t.calls, t.deletions, t.shrinks),
                    ):
                        if p.calls == 0:
                            continue
                        if (p.shrinks != 0) != useful:
                            continue

                        self.debug(
                            f"  * {p.name} made {p.calls} call{s(p.calls)} of which "
                            f"{p.shrinks} shrank and {p.misaligned} were misaligned, "
                            f"deleting {p.deletions} byte{s(p.deletions)}."
                        )
                self.debug("")
        self.explain()

    def explain(self):
        if not self.should_explain or not self.shrink_target.arg_slices:
            return
        from hypothesis.internal.conjecture.engine import BUFFER_SIZE

        self.max_stall = 1e999
        shrink_target = self.shrink_target
        buffer = shrink_target.buffer
        chunks = defaultdict(list)

        # Before we start running experiments, let's check for known inputs which would
        # make them redundant.  The shrinking process means that we've already tried many
        # variations on the minimal example, so this can save a lot of time.
        seen_passing_buffers = self.engine.passing_buffers(
            prefix=buffer[: min(self.shrink_target.arg_slices)[0]]
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
                seen.startswith(buffer[:start]) and seen.endswith(buffer[end:])
                for seen in seen_passing_buffers
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

                buf_attempt_fixed = bytearray(buffer)
                buf_attempt_fixed[start:end] = [
                    self.random.randint(0, 255) for _ in range(end - start)
                ]
                result = self.engine.cached_test_function(
                    buf_attempt_fixed, extend=BUFFER_SIZE - len(buf_attempt_fixed)
                )

                # Turns out this was a variable-length part, so grab the infix...
                if result.status == Status.OVERRUN:
                    continue  # pragma: no cover  # flakily covered
                if not (
                    len(buf_attempt_fixed) == len(result.buffer)
                    and result.buffer.endswith(buffer[end:])
                ):
                    for ex, res in zip(shrink_target.examples, result.examples):
                        assert ex.start == res.start
                        assert ex.start <= start
                        assert ex.label == res.label
                        if start == ex.start and end == ex.end:
                            res_end = res.end
                            break
                    else:
                        raise NotImplementedError("Expected matching prefixes")

                    buf_attempt_fixed = (
                        buffer[:start] + result.buffer[start:res_end] + buffer[end:]
                    )
                    chunks[(start, end)].append(result.buffer[start:res_end])
                    result = self.engine.cached_test_function(buf_attempt_fixed)

                    if result.status == Status.OVERRUN:
                        continue  # pragma: no cover  # flakily covered
                else:
                    chunks[(start, end)].append(result.buffer[start:end])

                if shrink_target is not self.shrink_target:  # pragma: no cover
                    # If we've shrunk further without meaning to, bail out.
                    self.shrink_target.slice_comments.clear()
                    return
                if result.status == Status.VALID:
                    # The test passed, indicating that this param can't vary freely.
                    # However, it's really hard to write a simple and reliable covering
                    # test, because of our `seen_passing_buffers` check above.
                    break  # pragma: no cover
                elif self.__predicate(result):  # pragma: no branch
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
            new_buf = bytearray()
            prev_end = 0
            for (start, end), ls in chunks_by_start_index:
                assert prev_end <= start < end, "these chunks must be nonoverlapping"
                new_buf.extend(buffer[prev_end:start])
                new_buf.extend(self.random.choice(ls))
                prev_end = end

            result = self.engine.cached_test_function(new_buf)

            # This *can't* be a shrink because none of the components were.
            assert shrink_target is self.shrink_target
            if result.status == Status.VALID:
                self.shrink_target.slice_comments[(0, 0)] = (
                    "The test sometimes passed when commented parts were varied together."
                )
                break  # Test passed, this param can't vary freely.
            elif self.__predicate(result):  # pragma: no branch
                n_same_failures_together += 1
                if n_same_failures_together >= 100:
                    self.shrink_target.slice_comments[(0, 0)] = (
                        "The test always failed when commented parts were varied together."
                    )
                    break

    def greedy_shrink(self):
        """Run a full set of greedy shrinks (that is, ones that will only ever
        move to a better target) and update shrink_target appropriately.

        This method iterates to a fixed point and so is idempontent - calling
        it twice will have exactly the same effect as calling it once.
        """
        self.fixate_shrink_passes(
            [
                node_program("X" * 5),
                node_program("X" * 4),
                node_program("X" * 3),
                node_program("X" * 2),
                node_program("X" * 1),
                "pass_to_descendant",
                "reorder_examples",
                "minimize_duplicated_nodes",
                "minimize_individual_nodes",
                "redistribute_integer_pairs",
                "lower_blocks_together",
            ]
        )

    @derived_value  # type: ignore
    def shrink_pass_choice_trees(self):
        return defaultdict(ChoiceTree)

    def fixate_shrink_passes(self, passes):
        """Run steps from each pass in ``passes`` until the current shrink target
        is a fixed point of all of them."""
        passes = list(map(self.shrink_pass, passes))

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
                    if not sp.step(random_order=failures >= max_failures // 2):
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
                elif len(self.buffer) < len(before_sp.buffer):
                    reordering[sp] = -1
                else:
                    reordering[sp] = 0

            passes.sort(key=reordering.__getitem__)

    @property
    def buffer(self):
        return self.shrink_target.buffer

    @property
    def blocks(self):
        return self.shrink_target.blocks

    @property
    def nodes(self):
        return self.shrink_target.examples.ir_tree_nodes

    @property
    def examples(self):
        return self.shrink_target.examples

    @derived_value  # type: ignore
    def examples_by_label(self):
        """An index of all examples grouped by their label, with
        the examples stored in their normal index order."""

        examples_by_label = defaultdict(list)
        for ex in self.examples:
            examples_by_label[ex.label].append(ex)
        return dict(examples_by_label)

    @derived_value  # type: ignore
    def distinct_labels(self):
        return sorted(self.examples_by_label, key=str)

    @defines_shrink_pass()
    def pass_to_descendant(self, chooser):
        """Attempt to replace each example with a descendant example.

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
            self.distinct_labels, lambda l: len(self.examples_by_label[l]) >= 2
        )

        ls = self.examples_by_label[label]
        i = chooser.choose(range(len(ls) - 1))
        ancestor = ls[i]

        if i + 1 == len(ls) or ls[i + 1].ir_start >= ancestor.ir_end:
            return

        @self.cached(label, i)
        def descendants():
            lo = i + 1
            hi = len(ls)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if ls[mid].ir_start >= ancestor.ir_end:
                    hi = mid
                else:
                    lo = mid
            return [t for t in ls[i + 1 : hi] if t.ir_length < ancestor.ir_length]

        descendant = chooser.choose(descendants, lambda ex: ex.ir_length > 0)

        assert ancestor.ir_start <= descendant.ir_start
        assert ancestor.ir_end >= descendant.ir_end
        assert descendant.ir_length < ancestor.ir_length

        self.consider_new_tree(
            self.nodes[: ancestor.ir_start]
            + self.nodes[descendant.ir_start : descendant.ir_end]
            + self.nodes[ancestor.ir_end :]
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

        So what we do is this: We keep track of which blocks are changing, and
        then if there's some non-zero common offset to them we try and minimize
        them all at once by lowering that offset.

        This may not work, and it definitely won't get us out of all possible
        exponential slow downs (an example of where it doesn't is where the
        shape of the blocks changes as a result of this bouncing behaviour),
        but it fails fast when it doesn't work and gets us out of a really
        nastily slow case when it does.
        """
        if len(self.__changed_nodes) <= 1:
            return

        changed = []
        for i in sorted(self.__changed_nodes):
            node = self.nodes[i]
            if node.trivial or node.ir_type != "integer":
                continue
            changed.append(node)

        if not changed:
            return

        ints = [abs(node.value - node.kwargs["shrink_towards"]) for node in changed]
        offset = min(ints)
        assert offset > 0

        for i in range(len(ints)):
            ints[i] -= offset

        st = self.shrink_target

        def offset_node(node, n):
            return (
                node.index,
                node.index + 1,
                [node.copy(with_value=node.kwargs["shrink_towards"] + n)],
            )

        def consider(n, sign):
            return self.consider_new_tree(
                replace_all(
                    st.examples.ir_tree_nodes,
                    [
                        offset_node(node, sign * (n + v))
                        for node, v in zip(changed, ints)
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
    def __changed_nodes(self):
        if self.__last_checked_changed_at is self.shrink_target:
            return self.__all_changed_nodes

        prev_target = self.__last_checked_changed_at
        new_target = self.shrink_target
        assert prev_target is not new_target
        prev_nodes = prev_target.examples.ir_tree_nodes
        new_nodes = new_target.examples.ir_tree_nodes
        assert sort_key(new_target.buffer) < sort_key(prev_target.buffer)

        if len(prev_nodes) != len(new_nodes) or any(
            n1.ir_type != n2.ir_type for n1, n2 in zip(prev_nodes, new_nodes)
        ):
            # should we check kwargs are equal as well?
            self.__all_changed_nodes = set()
        else:
            assert len(prev_nodes) == len(new_nodes)
            for i, (n1, n2) in enumerate(zip(prev_nodes, new_nodes)):
                assert n1.ir_type == n2.ir_type
                if not ir_value_equal(n1.ir_type, n1.value, n2.value):
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
        the same value and ir type, although this is not essential. Note that
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

        attempt = self.cached_test_function_ir(initial_attempt)

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
            # * replacing the kwargs in our attempt with what test_function tried
            #   to draw in practice
            # * truncating the value of that node to match min_size
            #
            # This helps in the specific case of drawing a value and then drawing
            # a collection of that size...and not much else. In practice this
            # helps because this antipattern is fairly common.

            # TODO we'll probably want to apply the same trick as in the valid
            # case of this function of preserving from the right instead of
            # preserving from the left. see test_can_shrink_variable_string_draws.

            (index, attempt_ir_type, attempt_kwargs, _attempt_forced) = (
                attempt.misaligned_at
            )
            node = self.nodes[index]
            if node.ir_type != attempt_ir_type:
                return False  # pragma: no cover
            if node.was_forced:
                return False  # pragma: no cover

            if node.ir_type in {"string", "bytes"}:
                # if the size *increased*, we would have to guess what to pad with
                # in order to try fixing up this attempt. Just give up.
                if node.kwargs["min_size"] <= attempt_kwargs["min_size"]:
                    return False
                # the size decreased in our attempt. Try again, but replace with
                # the min_size that we would have gotten, and truncate the value
                # to that size by removing any elements past min_size.
                return self.consider_new_tree(
                    initial_attempt[: node.index]
                    + [
                        initial_attempt[node.index].copy(
                            with_kwargs=attempt_kwargs,
                            with_value=initial_attempt[node.index].value[
                                : attempt_kwargs["min_size"]
                            ],
                        )
                    ]
                    + initial_attempt[node.index :]
                )

        lost_nodes = len(self.nodes) - len(attempt.examples.ir_tree_nodes)
        if lost_nodes <= 0:
            return False

        start = nodes[0].index
        end = nodes[-1].index + 1
        # We now look for contiguous regions to delete that might help fix up
        # this failed shrink. We only look for contiguous regions of the right
        # lengths because doing anything more than that starts to get very
        # expensive. See minimize_individual_blocks for where we
        # try to be more aggressive.
        regions_to_delete = {(end, end + lost_nodes)}

        for ex in self.examples:
            if ex.ir_start > start:
                continue
            if ex.ir_end <= end:
                continue

            if ex.index >= len(attempt.examples):
                continue  # pragma: no cover

            replacement = attempt.examples[ex.index]
            in_original = [c for c in ex.children if c.ir_start >= end]
            in_replaced = [c for c in replacement.children if c.ir_start >= end]

            if len(in_replaced) >= len(in_original) or not in_replaced:
                continue

            # We've found an example where some of the children went missing
            # as a result of this change, and just replacing it with the data
            # it would have had and removing the spillover didn't work. This
            # means that some of its children towards the right must be
            # important, so we try to arrange it so that it retains its
            # rightmost children instead of its leftmost.
            regions_to_delete.add(
                (in_original[0].ir_start, in_original[-len(in_replaced)].ir_start)
            )

        for u, v in sorted(regions_to_delete, key=lambda x: x[1] - x[0], reverse=True):
            try_with_deleted = initial_attempt[:u] + initial_attempt[v:]
            if self.consider_new_tree(try_with_deleted):
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

            for ex in self.shrink_target.examples:
                if (
                    ex.length > 0
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

            attempt = bytearray(self.shrink_target.buffer)
            for u, v in reversed(discarded):
                del attempt[u:v]

            if not self.incorporate_new_buffer(attempt):
                return False
        return True

    @derived_value  # type: ignore
    def duplicated_nodes(self):
        """Returns a list of nodes grouped (ir_type, value)."""
        duplicates = defaultdict(list)
        for node in self.nodes:
            duplicates[(node.ir_type, ir_value_key(node.ir_type, node.value))].append(
                node
            )
        return list(duplicates.values())

    @defines_shrink_pass()
    def minimize_duplicated_nodes(self, chooser):
        """Find blocks that have been duplicated in multiple places and attempt
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
        of the blocks doesn't matter very much because it allows us to replace
        more values at once.
        """
        nodes = chooser.choose(self.duplicated_nodes)
        if len(nodes) <= 1:
            return

        # no point in lowering nodes together if one is already trivial.
        # TODO_BETTER_SHRINK: we could potentially just drop the trivial nodes
        # here and carry on with nontrivial ones?
        if any(node.trivial for node in nodes):
            return

        self.minimize_nodes(nodes)

    @defines_shrink_pass()
    def redistribute_integer_pairs(self, chooser):
        """If there is a sum of generated integers that we need their sum
        to exceed some bound, lowering one of them requires raising the
        other. This pass enables that."""
        # TODO_SHRINK let's extend this to floats as well.

        # look for a pair of nodes (node1, node2) which are both integers and
        # aren't separated by too many other nodes. We'll decrease node1 and
        # increase node2 (note that the other way around doesn't make sense as
        # it's strictly worse in the ordering).
        node1 = chooser.choose(
            self.nodes, lambda node: node.ir_type == "integer" and not node.trivial
        )
        node2 = chooser.choose(
            self.nodes,
            lambda node: node.ir_type == "integer"
            # Note that it's fine for node2 to be trivial, because we're going to
            # explicitly make it *not* trivial by adding to its value.
            and not node.was_forced
            # to avoid quadratic behavior, scan ahead only a small amount for
            # the related node.
            and node1.index < node.index <= node1.index + 4,
        )

        m = node1.value
        n = node2.value

        def boost(k):
            if k > m:
                return False

            node_value = m - k
            next_node_value = n + k

            return self.consider_new_tree(
                self.nodes[: node1.index]
                + [node1.copy(with_value=node_value)]
                + self.nodes[node1.index + 1 : node2.index]
                + [node2.copy(with_value=next_node_value)]
                + self.nodes[node2.index + 1 :]
            )

        find_integer(boost)

    @defines_shrink_pass()
    def lower_blocks_together(self, chooser):
        block = chooser.choose(self.blocks, lambda b: not b.trivial)

        # Choose the next block to be up to eight blocks onwards. We don't
        # want to go too far (to avoid quadratic time) but it's worth a
        # reasonable amount of lookahead, especially as we expect most
        # blocks are zero by this point anyway.
        next_block = self.blocks[
            chooser.choose(
                range(block.index + 1, min(len(self.blocks), block.index + 9)),
                lambda j: not self.blocks[j].trivial,
            )
        ]

        buffer = self.buffer

        m = int_from_bytes(buffer[block.start : block.end])
        n = int_from_bytes(buffer[next_block.start : next_block.end])

        def lower(k):
            if k > min(m, n):
                return False
            attempt = bytearray(buffer)
            attempt[block.start : block.end] = int_to_bytes(m - k, block.length)
            attempt[next_block.start : next_block.end] = int_to_bytes(
                n - k, next_block.length
            )
            assert len(attempt) == len(buffer)
            return self.consider_new_buffer(attempt)

        find_integer(lower)

    def minimize_nodes(self, nodes):
        ir_type = nodes[0].ir_type
        value = nodes[0].value
        # unlike ir_type and value, kwargs are *not* guaranteed to be equal among all
        # passed nodes. We arbitrarily use the kwargs of the first node. I think
        # this is unsound (= leads to us trying shrinks that could not have been
        # generated), but those get discarded at test-time, and this enables useful
        # slips where kwargs are not equal but are close enough that doing the
        # same operation on both basically just works.
        kwargs = nodes[0].kwargs
        assert all(
            node.ir_type == ir_type and ir_value_equal(ir_type, node.value, value)
            for node in nodes
        )

        if ir_type == "integer":
            shrink_towards = kwargs["shrink_towards"]
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
        elif ir_type == "float":
            self.try_shrinking_nodes(nodes, abs(value))
            Float.shrink(
                abs(value),
                lambda val: self.try_shrinking_nodes(nodes, val),
            )
            Float.shrink(
                abs(value),
                lambda val: self.try_shrinking_nodes(nodes, -val),
            )
        elif ir_type == "boolean":
            # must be True, otherwise would be trivial and not selected.
            assert value is True
            # only one thing to try: false!
            self.try_shrinking_nodes(nodes, False)
        elif ir_type == "bytes":
            Bytes.shrink(
                value,
                lambda val: self.try_shrinking_nodes(nodes, val),
            )
        elif ir_type == "string":
            String.shrink(
                value,
                lambda val: self.try_shrinking_nodes(nodes, val),
                intervals=kwargs["intervals"],
            )
        else:
            raise NotImplementedError

    @defines_shrink_pass()
    def minimize_individual_nodes(self, chooser):
        """Attempt to minimize each node in sequence.

        This is the pass that ensures that e.g. each integer we draw is a
        minimum value. So it's the part that guarantees that if we e.g. do

        x = data.draw(integers())
        assert x < 10

        then in our shrunk example, x = 10 rather than say 97.

        If we are unsuccessful at minimizing a node of interest we then
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

        if node.ir_type != "integer":
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
            + [node.copy(with_value=node.value - 1)]
            + self.nodes[node.index + 1 :]
        )
        attempt = self.cached_test_function_ir(lowered)
        if (
            attempt is None
            or attempt.status < Status.VALID
            or len(attempt.examples.ir_tree_nodes) == len(self.nodes)
            or len(attempt.examples.ir_tree_nodes) == node.index + 1
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
        def first_example_after_node():
            lo = 0
            hi = len(self.examples)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                ex = self.examples[mid]
                if ex.ir_start >= node.index:
                    hi = mid
                else:
                    lo = mid
            return hi

        # we try deleting both entire examples, and single nodes.
        # If we wanted to get more aggressive, we could try deleting n
        # consecutive nodes (that don't cross an example boundary) for say
        # n <= 2 or n <= 3.
        if chooser.choose([True, False]):
            ex = self.examples[
                chooser.choose(
                    range(first_example_after_node, len(self.examples)),
                    lambda i: self.examples[i].ir_length > 0,
                )
            ]
            self.consider_new_tree(lowered[: ex.ir_start] + lowered[ex.ir_end :])
        else:
            node = self.nodes[chooser.choose(range(node.index + 1, len(self.nodes)))]
            self.consider_new_tree(lowered[: node.index] + lowered[node.index + 1 :])

    @defines_shrink_pass()
    def reorder_examples(self, chooser):
        """This pass allows us to reorder the children of each example.

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
        ex = chooser.choose(self.examples)
        label = chooser.choose(ex.children).label

        examples = [c for c in ex.children if c.label == label]
        if len(examples) <= 1:
            return
        st = self.shrink_target
        endpoints = [(ex.ir_start, ex.ir_end) for ex in examples]

        Ordering.shrink(
            range(len(examples)),
            lambda indices: self.consider_new_tree(
                replace_all(
                    st.examples.ir_nodes,
                    [
                        (
                            u,
                            v,
                            st.examples.ir_nodes[
                                examples[i].ir_start : examples[i].ir_end
                            ],
                        )
                        for (u, v), i in zip(endpoints, indices)
                    ],
                )
            ),
            key=lambda i: st.buffer[examples[i].start : examples[i].end],
        )

    def run_node_program(self, i, description, original, repeats=1):
        """Node programs are a mini-DSL for node rewriting, defined as a sequence
        of commands that can be run at some index into the nodes

        Commands are:

            * "X", delete this node

        This method runs the node program in ``description`` at node index
        ``i`` on the ConjectureData ``original``. If ``repeats > 1`` then it
        will attempt to approximate the results of running it that many times.

        Returns True if this successfully changes the underlying shrink target,
        else False.
        """
        if i + len(description) > len(original.examples.ir_tree_nodes) or i < 0:
            return False
        attempt = list(original.examples.ir_tree_nodes)
        for _ in range(repeats):
            for k, command in reversed(list(enumerate(description))):
                j = i + k
                if j >= len(attempt):
                    return False

                if command == "X":
                    del attempt[j]
                else:
                    raise NotImplementedError(f"Unrecognised command {command!r}")

        return self.consider_new_tree(attempt)


def shrink_pass_family(f):
    def accept(*args):
        name = "{}({})".format(f.__name__, ", ".join(map(repr, args)))
        if name not in SHRINK_PASS_DEFINITIONS:

            def run(self, chooser):
                return f(self, chooser, *args)

            run.__name__ = name
            defines_shrink_pass()(run)
        assert name in SHRINK_PASS_DEFINITIONS
        return name

    return accept


@shrink_pass_family
def node_program(self, chooser, description):
    n = len(description)
    # Adaptively attempt to run the node program at the current
    # index. If this successfully applies the node program ``k`` times
    # then this runs in ``O(log(k))`` test function calls.
    i = chooser.choose(range(len(self.nodes) - n + 1))

    # First, run the node program at the chosen index. If this fails,
    # don't do any extra work, so that failure is as cheap as possible.
    if not self.run_node_program(i, description, original=self.shrink_target):
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
                offset_left(k), description, original=self.shrink_target
            )
        )
    )

    original = self.shrink_target
    # Now try to run the block program multiple times here.
    find_integer(
        lambda k: self.run_node_program(i, description, original=original, repeats=k)
    )


@attr.s(slots=True, eq=False)
class ShrinkPass:
    run_with_chooser = attr.ib()
    index = attr.ib()
    shrinker = attr.ib()

    last_prefix = attr.ib(default=())
    successes = attr.ib(default=0)
    calls = attr.ib(default=0)
    misaligned = attr.ib(default=0)
    shrinks = attr.ib(default=0)
    deletions = attr.ib(default=0)

    def step(self, *, random_order=False):
        tree = self.shrinker.shrink_pass_choice_trees[self]
        if tree.exhausted:
            return False

        initial_shrinks = self.shrinker.shrinks
        initial_calls = self.shrinker.calls
        initial_misaligned = self.shrinker.misaligned
        size = len(self.shrinker.shrink_target.buffer)
        self.shrinker.engine.explain_next_call_as(self.name)

        if random_order:
            selection_order = random_selection_order(self.shrinker.random)
        else:
            selection_order = prefix_selection_order(self.last_prefix)

        try:
            self.last_prefix = tree.step(
                selection_order,
                lambda chooser: self.run_with_chooser(self.shrinker, chooser),
            )
        finally:
            self.calls += self.shrinker.calls - initial_calls
            self.misaligned += self.shrinker.misaligned - initial_misaligned
            self.shrinks += self.shrinker.shrinks - initial_shrinks
            self.deletions += size - len(self.shrinker.shrink_target.buffer)
            self.shrinker.engine.clear_call_explanation()
        return True

    @property
    def name(self) -> str:
        return self.run_with_chooser.__name__


class StopShrinking(Exception):
    pass
