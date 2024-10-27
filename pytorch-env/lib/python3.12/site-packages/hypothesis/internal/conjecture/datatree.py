# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import itertools
import math
from typing import Optional, Union

import attr

from hypothesis.errors import (
    FlakyReplay,
    FlakyStrategyDefinition,
    HypothesisException,
    StopTest,
)
from hypothesis.internal import floats as flt
from hypothesis.internal.compat import int_to_bytes
from hypothesis.internal.conjecture.data import (
    BooleanKWargs,
    BytesKWargs,
    ConjectureData,
    DataObserver,
    FloatKWargs,
    IntegerKWargs,
    IRKWargsType,
    IRType,
    IRTypeName,
    Status,
    StringKWargs,
)
from hypothesis.internal.escalation import InterestingOrigin
from hypothesis.internal.floats import (
    count_between_floats,
    float_to_int,
    int_to_float,
    sign_aware_lte,
)


class PreviouslyUnseenBehaviour(HypothesisException):
    pass


def inconsistent_generation():
    raise FlakyStrategyDefinition(
        "Inconsistent data generation! Data generation behaved differently "
        "between different runs. Is your data generation depending on external "
        "state?"
    )


EMPTY: frozenset = frozenset()


@attr.s(slots=True)
class Killed:
    """Represents a transition to part of the tree which has been marked as
    "killed", meaning we want to treat it as not worth exploring, so it will
    be treated as if it were completely explored for the purposes of
    exhaustion."""

    next_node = attr.ib()

    def _repr_pretty_(self, p, cycle):
        assert cycle is False
        p.text("Killed")


def _node_pretty(ir_type, value, kwargs, *, forced):
    forced_marker = " [forced]" if forced else ""
    return f"{ir_type} {value}{forced_marker} {kwargs}"


@attr.s(slots=True)
class Branch:
    """Represents a transition where multiple choices can be made as to what
    to drawn."""

    kwargs = attr.ib()
    ir_type = attr.ib()
    children = attr.ib(repr=False)

    @property
    def max_children(self):
        max_children = compute_max_children(self.ir_type, self.kwargs)
        assert max_children > 0
        return max_children

    def _repr_pretty_(self, p, cycle):
        assert cycle is False
        for i, (value, child) in enumerate(self.children.items()):
            if i > 0:
                p.break_()
            p.text(_node_pretty(self.ir_type, value, self.kwargs, forced=False))
            with p.indent(2):
                p.break_()
                p.pretty(child)


@attr.s(slots=True, frozen=True)
class Conclusion:
    """Represents a transition to a finished state."""

    status: Status = attr.ib()
    interesting_origin: Optional[InterestingOrigin] = attr.ib()

    def _repr_pretty_(self, p, cycle):
        assert cycle is False
        o = self.interesting_origin
        # avoid str(o), which can include multiple lines of context
        origin = (
            "" if o is None else f", {o.exc_type.__name__} at {o.filename}:{o.lineno}"
        )
        p.text(f"Conclusion ({self.status!r}{origin})")


# The number of max children where, beyond this, it is practically impossible
# for hypothesis to saturate / explore all children nodes in a reasonable time
# frame. We use this to bail out of expensive max children computations early,
# where the numbers involved are so large that we know they will be larger than
# this number.
#
# Note that it's ok for us to underestimate the number of max children of a node
# by using this. We just may think the node is exhausted when in fact it has more
# possible children to be explored. This has the potential to finish generation
# early due to exhausting the entire tree, but that is quite unlikely: (1) the
# number of examples would have to be quite high, and (2) the tree would have to
# contain only one or two nodes, or generate_novel_prefix would simply switch to
# exploring another non-exhausted node.
#
# Also note that we may sometimes compute max children above this value. In other
# words, this is *not* a hard maximum on the computed max children. It's the point
# where further computation is not beneficial - but sometimes doing that computation
# unconditionally is cheaper than estimating against this value.
#
# The one case where this may be detrimental is fuzzing, where the throughput of
# examples is so high that it really may saturate important nodes. We'll cross
# that bridge when we come to it.
MAX_CHILDREN_EFFECTIVELY_INFINITE = 100_000


def _count_distinct_strings(*, alphabet_size, min_size, max_size):
    # We want to estimate if we're going to have more children than
    # MAX_CHILDREN_EFFECTIVELY_INFINITE, without computing a potentially
    # extremely expensive pow. We'll check if the number of strings in
    # the largest string size alone is enough to put us over this limit.
    # We'll also employ a trick of estimating against log, which is cheaper
    # than computing a pow.
    #
    # x = max_size
    # y = alphabet_size
    # n = MAX_CHILDREN_EFFECTIVELY_INFINITE
    #
    #     x**y > n
    # <=> log(x**y)  > log(n)
    # <=> y * log(x) > log(n)
    definitely_too_large = max_size * math.log(alphabet_size) > math.log(
        MAX_CHILDREN_EFFECTIVELY_INFINITE
    )
    if definitely_too_large:
        return MAX_CHILDREN_EFFECTIVELY_INFINITE

    return sum(alphabet_size**k for k in range(min_size, max_size + 1))


def compute_max_children(ir_type, kwargs):
    if ir_type == "integer":
        min_value = kwargs["min_value"]
        max_value = kwargs["max_value"]

        if min_value is None and max_value is None:
            # full 128 bit range.
            return 2**128 - 1
        if min_value is not None and max_value is not None:
            # count between min/max value.
            return max_value - min_value + 1

        # hard case: only one bound was specified. Here we probe either upwards
        # or downwards with our full 128 bit generation, but only half of these
        # (plus one for the case of generating zero) result in a probe in the
        # direction we want. ((2**128 - 1) // 2) + 1 == 2 ** 127
        assert (min_value is None) ^ (max_value is None)
        return 2**127
    elif ir_type == "boolean":
        p = kwargs["p"]
        # probabilities of 0 or 1 (or effectively 0 or 1) only have one choice.
        if p <= 2 ** (-64) or p >= (1 - 2 ** (-64)):
            return 1
        return 2
    elif ir_type == "bytes":
        return _count_distinct_strings(
            alphabet_size=2**8, min_size=kwargs["min_size"], max_size=kwargs["max_size"]
        )
    elif ir_type == "string":
        min_size = kwargs["min_size"]
        max_size = kwargs["max_size"]
        intervals = kwargs["intervals"]

        if len(intervals) == 0:
            # Special-case the empty alphabet to avoid an error in math.log(0).
            # Only possibility is the empty string.
            return 1

        # avoid math.log(1) == 0 and incorrectly failing our effectively_infinite
        # estimate, even when we definitely are too large.
        if len(intervals) == 1 and max_size > MAX_CHILDREN_EFFECTIVELY_INFINITE:
            return MAX_CHILDREN_EFFECTIVELY_INFINITE

        return _count_distinct_strings(
            alphabet_size=len(intervals), min_size=min_size, max_size=max_size
        )
    elif ir_type == "float":
        min_value = kwargs["min_value"]
        max_value = kwargs["max_value"]
        smallest_nonzero_magnitude = kwargs["smallest_nonzero_magnitude"]

        count = count_between_floats(min_value, max_value)

        # we have two intervals:
        # a. [min_value, max_value]
        # b. [-smallest_nonzero_magnitude, smallest_nonzero_magnitude]
        #
        # which could be subsets (in either order), overlapping, or disjoint. We
        # want the interval difference a - b.

        # next_down because endpoints are ok with smallest_nonzero_magnitude
        min_point = max(min_value, -flt.next_down(smallest_nonzero_magnitude))
        max_point = min(max_value, flt.next_down(smallest_nonzero_magnitude))

        if min_point > max_point:
            # case: disjoint intervals.
            return count

        count -= count_between_floats(min_point, max_point)
        if sign_aware_lte(min_value, -0.0) and sign_aware_lte(-0.0, max_value):
            # account for -0.0
            count += 1
        if sign_aware_lte(min_value, 0.0) and sign_aware_lte(0.0, max_value):
            # account for 0.0
            count += 1
        return count

    raise NotImplementedError(f"unhandled ir_type {ir_type}")


# In theory, this is a strict superset of the functionality of compute_max_children;
#
#   assert len(all_children(ir_type, kwargs)) == compute_max_children(ir_type, kwargs)
#
# In practice, we maintain two distinct implementations for efficiency and space
# reasons. If you just need the number of children, it is cheaper to use
# compute_max_children than to reify the list of children (only to immediately
# throw it away).
def all_children(ir_type, kwargs):
    if ir_type == "integer":
        min_value = kwargs["min_value"]
        max_value = kwargs["max_value"]

        if min_value is None and max_value is None:
            # full 128 bit range.
            yield from range(-(2**127) + 1, 2**127 - 1)

        elif min_value is not None and max_value is not None:
            yield from range(min_value, max_value + 1)
        else:
            assert (min_value is None) ^ (max_value is None)
            # hard case: only one bound was specified. Here we probe in 128 bits
            # around shrink_towards, and discard those above max_value or below
            # min_value respectively.
            shrink_towards = kwargs["shrink_towards"]
            if min_value is None:
                shrink_towards = min(max_value, shrink_towards)
                yield from range(shrink_towards - (2**127) + 1, max_value)
            else:
                assert max_value is None
                shrink_towards = max(min_value, shrink_towards)
                yield from range(min_value, shrink_towards + (2**127) - 1)

    if ir_type == "boolean":
        p = kwargs["p"]
        if p <= 2 ** (-64):
            yield False
        elif p >= (1 - 2 ** (-64)):
            yield True
        else:
            yield from [False, True]
    if ir_type == "bytes":
        for size in range(kwargs["min_size"], kwargs["max_size"] + 1):
            yield from (int_to_bytes(i, size) for i in range(2 ** (8 * size)))
    if ir_type == "string":
        min_size = kwargs["min_size"]
        max_size = kwargs["max_size"]
        intervals = kwargs["intervals"]

        # written unidiomatically in order to handle the case of max_size=inf.
        size = min_size
        while size <= max_size:
            for ords in itertools.product(intervals, repeat=size):
                yield "".join(chr(n) for n in ords)
            size += 1
    if ir_type == "float":

        def floats_between(a, b):
            for n in range(float_to_int(a), float_to_int(b) + 1):
                yield int_to_float(n)

        min_value = kwargs["min_value"]
        max_value = kwargs["max_value"]
        smallest_nonzero_magnitude = kwargs["smallest_nonzero_magnitude"]

        # handle zeroes separately so smallest_nonzero_magnitude can think of
        # itself as a complete interval (instead of a hole at ±0).
        if sign_aware_lte(min_value, -0.0) and sign_aware_lte(-0.0, max_value):
            yield -0.0
        if sign_aware_lte(min_value, 0.0) and sign_aware_lte(0.0, max_value):
            yield 0.0

        if flt.is_negative(min_value):
            if flt.is_negative(max_value):
                # case: both negative.
                max_point = min(max_value, -smallest_nonzero_magnitude)
                # float_to_int increases as negative magnitude increases, so
                # invert order.
                yield from floats_between(max_point, min_value)
            else:
                # case: straddles midpoint (which is between -0.0 and 0.0).
                yield from floats_between(-smallest_nonzero_magnitude, min_value)
                yield from floats_between(smallest_nonzero_magnitude, max_value)
        else:
            # case: both positive.
            min_point = max(min_value, smallest_nonzero_magnitude)
            yield from floats_between(min_point, max_value)


@attr.s(slots=True)
class TreeNode:
    """
    A node, or collection of directly descended nodes, in a DataTree.

    We store the DataTree as a radix tree (https://en.wikipedia.org/wiki/Radix_tree),
    which means that nodes that are the only child of their parent are collapsed
    into their parent to save space.

    Conceptually, you can unfold a single TreeNode storing n values in its lists
    into a sequence of n nodes, each a child of the last. In other words,
    (kwargs[i], values[i], ir_types[i]) corresponds to the single node at index
    i.

    Note that if a TreeNode represents a choice (i.e. the nodes cannot be compacted
    via the radix tree definition), then its lists will be empty and it will
    store a `Branch` representing that choce in its `transition`.

    Examples
    --------

    Consider sequentially drawing a boolean, then an integer.

            data.draw_boolean()
            data.draw_integer(1, 3)

    If we draw True and then 2, the tree may conceptually look like this.

                      ┌──────┐
                      │ root │
                      └──┬───┘
                      ┌──┴───┐
                      │ True │
                      └──┬───┘
                      ┌──┴───┐
                      │  2   │
                      └──────┘

    But since 2 is the only child of True, we will compact these nodes and store
    them as a single TreeNode.

                      ┌──────┐
                      │ root │
                      └──┬───┘
                    ┌────┴──────┐
                    │ [True, 2] │
                    └───────────┘

    If we then draw True and then 3, True will have multiple children and we
    can no longer store this compacted representation. We would call split_at(0)
    on the [True, 2] node to indicate that we need to add a choice at 0-index
    node (True).

                      ┌──────┐
                      │ root │
                      └──┬───┘
                      ┌──┴───┐
                    ┌─┤ True ├─┐
                    │ └──────┘ │
                  ┌─┴─┐      ┌─┴─┐
                  │ 2 │      │ 3 │
                  └───┘      └───┘
    """

    # The kwargs, value, and ir_types of the nodes stored here. These always
    # have the same length. The values at index i belong to node i.
    kwargs: list[IRKWargsType] = attr.ib(factory=list)
    values: list[IRType] = attr.ib(factory=list)
    ir_types: list[IRTypeName] = attr.ib(factory=list)

    # The indices of nodes which had forced values.
    #
    # Stored as None if no indices have been forced, purely for space saving
    # reasons (we force quite rarely).
    __forced: Optional[set] = attr.ib(default=None, init=False)

    # What happens next after drawing these nodes. (conceptually, "what is the
    # child/children of the last node stored here").
    #
    # One of:
    # - None (we don't know yet)
    # - Branch (we have seen multiple possible outcomes here)
    # - Conclusion (ConjectureData.conclude_test was called here)
    # - Killed (this branch is valid and may even have children, but should not
    #   be explored when generating novel prefixes)
    transition: Union[None, Branch, Conclusion, Killed] = attr.ib(default=None)

    # A tree node is exhausted if every possible sequence of draws below it has
    # been explored. We only update this when performing operations that could
    # change the answer.
    #
    # See also TreeNode.check_exhausted.
    is_exhausted: bool = attr.ib(default=False, init=False)

    @property
    def forced(self):
        if not self.__forced:
            return EMPTY
        return self.__forced

    def mark_forced(self, i):
        """
        Note that the draw at node i was forced.
        """
        assert 0 <= i < len(self.values)
        if self.__forced is None:
            self.__forced = set()
        self.__forced.add(i)

    def split_at(self, i):
        """
        Splits the tree so that it can incorporate a decision at the draw call
        corresponding to the node at position i.

        Raises FlakyStrategyDefinition if node i was forced.
        """

        if i in self.forced:
            inconsistent_generation()

        assert not self.is_exhausted

        key = self.values[i]

        child = TreeNode(
            ir_types=self.ir_types[i + 1 :],
            kwargs=self.kwargs[i + 1 :],
            values=self.values[i + 1 :],
            transition=self.transition,
        )
        self.transition = Branch(
            kwargs=self.kwargs[i], ir_type=self.ir_types[i], children={key: child}
        )
        if self.__forced is not None:
            child.__forced = {j - i - 1 for j in self.__forced if j > i}
            self.__forced = {j for j in self.__forced if j < i}
        child.check_exhausted()
        del self.ir_types[i:]
        del self.values[i:]
        del self.kwargs[i:]
        assert len(self.values) == len(self.kwargs) == len(self.ir_types) == i

    def check_exhausted(self):
        """
        Recalculates is_exhausted if necessary, and then returns it.

        A node is exhausted if:
        - Its transition is Conclusion or Killed
        - It has the maximum number of children (i.e. we have found all of its
          possible children), and all its children are exhausted

        Therefore, we only need to compute this for a node when:
        - We first create it in split_at
        - We set its transition to either Conclusion or Killed
          (TreeRecordingObserver.conclude_test or TreeRecordingObserver.kill_branch)
        - We exhaust any of its children
        """

        if (
            # a node cannot go from is_exhausted -> not is_exhausted.
            not self.is_exhausted
            # if we don't know what happens after this node, we don't have
            # enough information to tell if it's exhausted.
            and self.transition is not None
            # if there are still any nodes left which are the only child of their
            # parent (len(self.values) > 0), then this TreeNode must be not
            # exhausted, unless all of those nodes were forced.
            #
            # This is because we maintain an invariant of only adding nodes to
            # DataTree which have at least 2 possible values, so we know that if
            # they do not have any siblings that we still have more choices to
            # discover.
            #
            # (We actually *do* currently add single-valued nodes to the tree,
            # but immediately split them into a transition to avoid falsifying
            # this check. this is a bit of a hack.)
            and len(self.forced) == len(self.values)
        ):
            if isinstance(self.transition, (Conclusion, Killed)):
                self.is_exhausted = True
            elif len(self.transition.children) == self.transition.max_children:
                self.is_exhausted = all(
                    v.is_exhausted for v in self.transition.children.values()
                )
        return self.is_exhausted

    def _repr_pretty_(self, p, cycle):
        assert cycle is False
        indent = 0
        for i, (ir_type, kwargs, value) in enumerate(
            zip(self.ir_types, self.kwargs, self.values)
        ):
            with p.indent(indent):
                if i > 0:
                    p.break_()
                p.text(_node_pretty(ir_type, value, kwargs, forced=i in self.forced))
            indent += 2

        with p.indent(indent):
            if len(self.values) > 0:
                p.break_()
            if self.transition is not None:
                p.pretty(self.transition)
            else:
                p.text("unknown")


class DataTree:
    """
    A DataTree tracks the structured history of draws in some test function,
    across multiple ConjectureData objects.

    This information is used by ConjectureRunner to generate novel prefixes of
    this tree (see generate_novel_prefix). A novel prefix is a sequence of draws
    which the tree has not seen before, and therefore the ConjectureRunner has
    not generated as an input to the test function before.

    DataTree tracks the following:

    - Draws, at the ir level (with some ir_type, e.g. "integer")
      - ConjectureData.draw_integer()
      - ConjectureData.draw_float()
      - ConjectureData.draw_string()
      - ConjectureData.draw_boolean()
      - ConjectureData.draw_bytes()
    - Test conclusions (with some Status, e.g. Status.VALID)
      - ConjectureData.conclude_test()

    A DataTree is — surprise — a *tree*. A node in this tree is either a draw with
    some value, a test conclusion with some Status, or a special `Killed` value,
    which denotes that further draws may exist beyond this node but should not be
    considered worth exploring when generating novel prefixes. A node is a leaf
    iff it is a conclusion or Killed.

    A branch from node A to node B indicates that we have previously seen some
    sequence (a, b) of draws, where a and b are the values in nodes A and B.
    Similar intuition holds for conclusion and Killed nodes.

    Examples
    --------

    To see how a DataTree gets built through successive sets of draws, consider
    the following code that calls through to some ConjecutreData object `data`.
    The first call can be either True or False, and the second call can be any
    integer in the range [1, 3].

        data.draw_boolean()
        data.draw_integer(1, 3)

    To start, the corresponding DataTree object is completely empty.

                      ┌──────┐
                      │ root │
                      └──────┘

    We happen to draw True and then 2 in the above code. The tree tracks this.
    (2 also connects to a child Conclusion node with Status.VALID since it's the
    final draw in the code. I'll omit Conclusion nodes in diagrams for brevity.)

                      ┌──────┐
                      │ root │
                      └──┬───┘
                      ┌──┴───┐
                      │ True │
                      └──┬───┘
                      ┌──┴───┐
                      │  2   │
                      └──────┘

    This is a very boring tree so far! But now we happen to draw False and
    then 1. This causes a split in the tree. Remember, DataTree tracks history
    over all invocations of a function, not just one. The end goal is to know
    what invocations haven't been tried yet, after all.

                      ┌──────┐
                  ┌───┤ root ├───┐
                  │   └──────┘   │
               ┌──┴───┐        ┌─┴─────┐
               │ True │        │ False │
               └──┬───┘        └──┬────┘
                ┌─┴─┐           ┌─┴─┐
                │ 2 │           │ 1 │
                └───┘           └───┘

    If we were to ask DataTree for a novel prefix at this point, it might
    generate any of (True, 1), (True, 3), (False, 2), or (False, 3).

    Note that the novel prefix stops as soon as it generates a novel node. For
    instance, if we had generated a novel prefix back when the tree was only
    root -> True -> 2, we could have gotten any of (True, 1), (True, 3), or
    (False). But we could *not* have gotten (False, n), because both False and
    n were novel at that point, and we stop at the first novel node — False.

    I won't belabor this example. Here's what the tree looks like when fully
    explored:

                      ┌──────┐
               ┌──────┤ root ├──────┐
               │      └──────┘      │
            ┌──┴───┐              ┌─┴─────┐
         ┌──┤ True ├──┐       ┌───┤ False ├──┐
         │  └──┬───┘  │       │   └──┬────┘  │
       ┌─┴─┐ ┌─┴─┐  ┌─┴─┐   ┌─┴─┐  ┌─┴─┐   ┌─┴─┐
       │ 1 │ │ 2 │  │ 3 │   │ 1 │  │ 2 │   │ 3 │
       └───┘ └───┘  └───┘   └───┘  └───┘   └───┘

    You could imagine much more complicated trees than this arising in practice,
    and indeed they do. In particular, the tree need not be balanced or 'nice'
    like the tree above. For instance,

        b = data.draw_boolean()
        if b:
            data.draw_integer(1, 3)

    results in a tree with the entire right part lopped off, and False leading
    straight to a conclusion node with Status.VALID. As another example,

        n = data.draw_integers()
        assume(n >= 3)
        data.draw_string()

    results in a tree with the 0, 1, and 2 nodes leading straight to a
    conclusion node with Status.INVALID, and the rest branching off into all
    the possibilities of draw_string.

    Notes
    -----

    The above examples are slightly simplified and are intended to convey
    intuition. In practice, there are some implementation details to be aware
    of.

    - In draw nodes, we store the kwargs used in addition to the value drawn.
      E.g. the node corresponding to data.draw_float(min_value=1.0, max_value=1.5)
      would store {"min_value": 1.0, "max_value": 1.5, ...} (default values for
      other kwargs omitted).

      The kwargs parameters have the potential to change both the range of
      possible outputs of a node, and the probability distribution within that
      range, so we need to use these when drawing in DataTree as well. We draw
      values using these kwargs when (1) generating a novel value for a node
      and (2) choosing a random child when traversing the tree.

    - For space efficiency, rather than tracking the full tree structure, we
      store DataTree as a radix tree. This is conceptually equivalent (radix
      trees can always be "unfolded" to the full tree) but it means the internal
      representation may differ in practice.

      See TreeNode for more information.
    """

    def __init__(self):
        self.root = TreeNode()
        self._children_cache = {}

    @property
    def is_exhausted(self):
        """
        Returns True if every node is exhausted, and therefore the tree has
        been fully explored.
        """
        return self.root.is_exhausted

    def generate_novel_prefix(self, random):
        """Generate a short random string that (after rewriting) is not
        a prefix of any buffer previously added to the tree.

        The resulting prefix is essentially arbitrary - it would be nice
        for it to be uniform at random, but previous attempts to do that
        have proven too expensive.
        """

        assert not self.is_exhausted
        novel_prefix = bytearray()

        def append_buf(buf):
            novel_prefix.extend(buf)

        current_node = self.root
        while True:
            assert not current_node.is_exhausted
            for i, (ir_type, kwargs, value) in enumerate(
                zip(current_node.ir_types, current_node.kwargs, current_node.values)
            ):
                if i in current_node.forced:
                    if ir_type == "float":
                        value = int_to_float(value)
                    (_value, buf) = self._draw(
                        ir_type, kwargs, forced=value, random=random
                    )
                    append_buf(buf)
                else:
                    attempts = 0
                    while True:
                        if attempts <= 10:
                            try:
                                (v, buf) = self._draw(ir_type, kwargs, random=random)
                            except StopTest:  # pragma: no cover
                                # it is possible that drawing from a fresh data can
                                # overrun BUFFER_SIZE, due to eg unlucky rejection sampling
                                # of integer probes. Retry these cases.
                                attempts += 1
                                continue
                        else:
                            (v, buf) = self._draw_from_cache(
                                ir_type, kwargs, key=id(current_node), random=random
                            )

                        if v != value:
                            append_buf(buf)
                            break
                        attempts += 1
                        self._reject_child(
                            ir_type, kwargs, child=v, key=id(current_node)
                        )
                    # We've now found a value that is allowed to
                    # vary, so what follows is not fixed.
                    return bytes(novel_prefix)
            else:
                assert not isinstance(current_node.transition, (Conclusion, Killed))
                if current_node.transition is None:
                    return bytes(novel_prefix)
                branch = current_node.transition
                assert isinstance(branch, Branch)

                attempts = 0
                while True:
                    if attempts <= 10:
                        try:
                            (v, buf) = self._draw(
                                branch.ir_type, branch.kwargs, random=random
                            )
                        except StopTest:  # pragma: no cover
                            attempts += 1
                            continue
                    else:
                        (v, buf) = self._draw_from_cache(
                            branch.ir_type, branch.kwargs, key=id(branch), random=random
                        )
                    try:
                        child = branch.children[v]
                    except KeyError:
                        append_buf(buf)
                        return bytes(novel_prefix)
                    if not child.is_exhausted:
                        append_buf(buf)
                        current_node = child
                        break
                    attempts += 1
                    self._reject_child(
                        branch.ir_type, branch.kwargs, child=v, key=id(branch)
                    )

                    # We don't expect this assertion to ever fire, but coverage
                    # wants the loop inside to run if you have branch checking
                    # on, hence the pragma.
                    assert (  # pragma: no cover
                        attempts != 1000
                        or len(branch.children) < branch.max_children
                        or any(not v.is_exhausted for v in branch.children.values())
                    )

    def rewrite(self, buffer):
        """Use previously seen ConjectureData objects to return a tuple of
        the rewritten buffer and the status we would get from running that
        buffer with the test function. If the status cannot be predicted
        from the existing values it will be None."""
        buffer = bytes(buffer)

        data = ConjectureData.for_buffer(buffer)
        try:
            self.simulate_test_function(data)
            return (data.buffer, data.status)
        except PreviouslyUnseenBehaviour:
            return (buffer, None)

    def simulate_test_function(self, data):
        """Run a simulated version of the test function recorded by
        this tree. Note that this does not currently call ``stop_example``
        or ``start_example`` as these are not currently recorded in the
        tree. This will likely change in future."""
        node = self.root

        def draw(ir_type, kwargs, *, forced=None, convert_forced=True):
            if ir_type == "float" and forced is not None and convert_forced:
                forced = int_to_float(forced)

            draw_func = getattr(data, f"draw_{ir_type}")
            value = draw_func(**kwargs, forced=forced)

            if ir_type == "float":
                value = float_to_int(value)
            return value

        try:
            while True:
                for i, (ir_type, kwargs, previous) in enumerate(
                    zip(node.ir_types, node.kwargs, node.values)
                ):
                    v = draw(
                        ir_type, kwargs, forced=previous if i in node.forced else None
                    )
                    if v != previous:
                        raise PreviouslyUnseenBehaviour
                if isinstance(node.transition, Conclusion):
                    t = node.transition
                    data.conclude_test(t.status, t.interesting_origin)
                elif node.transition is None:
                    raise PreviouslyUnseenBehaviour
                elif isinstance(node.transition, Branch):
                    v = draw(node.transition.ir_type, node.transition.kwargs)
                    try:
                        node = node.transition.children[v]
                    except KeyError as err:
                        raise PreviouslyUnseenBehaviour from err
                else:
                    assert isinstance(node.transition, Killed)
                    data.observer.kill_branch()
                    node = node.transition.next_node
        except StopTest:
            pass

    def new_observer(self):
        return TreeRecordingObserver(self)

    def _draw(self, ir_type, kwargs, *, random, forced=None):
        from hypothesis.internal.conjecture.data import ir_to_buffer

        (value, buf) = ir_to_buffer(ir_type, kwargs, forced=forced, random=random)
        # using floats as keys into branch.children breaks things, because
        # e.g. hash(0.0) == hash(-0.0) would collide as keys when they are
        # in fact distinct child branches.
        # To distinguish floats here we'll use their bits representation. This
        # entails some bookkeeping such that we're careful about when the
        # float key is in its bits form (as a key into branch.children) and
        # when it is in its float form (as a value we want to write to the
        # buffer), and converting between the two forms as appropriate.
        if ir_type == "float":
            value = float_to_int(value)
        return (value, buf)

    def _get_children_cache(self, ir_type, kwargs, *, key):
        # cache the state of the children generator per node/branch (passed as
        # `key` here), such that we track which children we've already tried
        # for this branch across draws.
        # We take advantage of python generators here as one-way iterables,
        # so each time we iterate we implicitly store our position in the
        # children generator and don't re-draw children. `children` is the
        # concrete list of children draw from the generator that we will work
        # with. Whenever we need to top up this list, we will draw a new value
        # from the generator.
        if key not in self._children_cache:
            generator = all_children(ir_type, kwargs)
            children = []
            rejected = set()
            self._children_cache[key] = (generator, children, rejected)

        return self._children_cache[key]

    def _draw_from_cache(self, ir_type, kwargs, *, key, random):
        (generator, children, rejected) = self._get_children_cache(
            ir_type, kwargs, key=key
        )
        # Keep a stock of 100 potentially-valid children at all times.
        # This number is chosen to balance memory/speed vs randomness. Ideally
        # we would sample uniformly from all not-yet-rejected children, but
        # computing and storing said children is not free.
        # no-branch because coverage of the fall-through case here is a bit
        # annoying.
        if len(children) < 100:  # pragma: no branch
            for v in generator:
                if ir_type == "float":
                    v = float_to_int(v)
                if v in rejected:
                    continue
                children.append(v)
                if len(children) >= 100:
                    break

        forced = random.choice(children)
        if ir_type == "float":
            forced = int_to_float(forced)
        (value, buf) = self._draw(ir_type, kwargs, forced=forced, random=random)
        return (value, buf)

    def _reject_child(self, ir_type, kwargs, *, child, key):
        (_generator, children, rejected) = self._get_children_cache(
            ir_type, kwargs, key=key
        )
        rejected.add(child)
        # we remove a child from the list of possible children *only* when it is
        # rejected, and not when it is initially drawn in _draw_from_cache. The
        # reason is that a child being drawn does not guarantee that child will
        # be used in a way such that it is written back to the tree, so it needs
        # to be available for future draws until we are certain it has been
        # used.
        #
        # For instance, if we generated novel prefixes in a loop (but never used
        # those prefixes to generate new values!) then we don't want to remove
        # the drawn children from the available pool until they are actually
        # used.
        #
        # This does result in a small inefficiency: we may draw a child,
        # immediately use it (so we know it cannot be drawn again), but still
        # wait to draw and reject it here, because DataTree cannot guarantee
        # the drawn child has been used.
        if child in children:
            children.remove(child)

    def _repr_pretty_(self, p, cycle):
        assert cycle is False
        return p.pretty(self.root)


class TreeRecordingObserver(DataObserver):
    def __init__(self, tree):
        self.__current_node = tree.root
        self.__index_in_current_node = 0
        self.__trail = [self.__current_node]
        self.killed = False

    def draw_integer(
        self, value: int, *, was_forced: bool, kwargs: IntegerKWargs
    ) -> None:
        self.draw_value("integer", value, was_forced=was_forced, kwargs=kwargs)

    def draw_float(
        self, value: float, *, was_forced: bool, kwargs: FloatKWargs
    ) -> None:
        self.draw_value("float", value, was_forced=was_forced, kwargs=kwargs)

    def draw_string(
        self, value: str, *, was_forced: bool, kwargs: StringKWargs
    ) -> None:
        self.draw_value("string", value, was_forced=was_forced, kwargs=kwargs)

    def draw_bytes(
        self, value: bytes, *, was_forced: bool, kwargs: BytesKWargs
    ) -> None:
        self.draw_value("bytes", value, was_forced=was_forced, kwargs=kwargs)

    def draw_boolean(
        self, value: bool, *, was_forced: bool, kwargs: BooleanKWargs
    ) -> None:
        self.draw_value("boolean", value, was_forced=was_forced, kwargs=kwargs)

    def draw_value(
        self,
        ir_type: IRTypeName,
        value: IRType,
        *,
        was_forced: bool,
        kwargs: IRKWargsType,
    ) -> None:
        i = self.__index_in_current_node
        self.__index_in_current_node += 1
        node = self.__current_node

        if isinstance(value, float):
            value = float_to_int(value)

        assert len(node.kwargs) == len(node.values) == len(node.ir_types)
        if i < len(node.values):
            if ir_type != node.ir_types[i] or kwargs != node.kwargs[i]:
                inconsistent_generation()
            # Note that we don't check whether a previously
            # forced value is now free. That will be caught
            # if we ever split the node there, but otherwise
            # may pass silently. This is acceptable because it
            # means we skip a hash set lookup on every
            # draw and that's a pretty niche failure mode.
            if was_forced and i not in node.forced:
                inconsistent_generation()
            if value != node.values[i]:
                node.split_at(i)
                assert i == len(node.values)
                new_node = TreeNode()
                node.transition.children[value] = new_node
                self.__current_node = new_node
                self.__index_in_current_node = 0
        else:
            trans = node.transition
            if trans is None:
                node.ir_types.append(ir_type)
                node.kwargs.append(kwargs)
                node.values.append(value)
                if was_forced:
                    node.mark_forced(i)
                # generate_novel_prefix assumes the following invariant: any one
                # of the series of draws in a particular node can vary, i.e. the
                # max number of children is at least 2. However, some draws are
                # pseudo-choices and only have a single value, such as
                # integers(0, 0).
                #
                # Currently, we address this by forcefully splitting such
                # single-valued nodes into a transition when we see them. An
                # exception to this is if it was forced: forced pseudo-choices
                # do not cause the above issue because they inherently cannot
                # vary, and moreover they trip other invariants about never
                # splitting forced nodes.
                #
                # An alternative is not writing such choices to the tree at
                # all, and thus guaranteeing that each node has at least 2 max
                # children.
                if compute_max_children(ir_type, kwargs) == 1 and not was_forced:
                    node.split_at(i)
                    self.__current_node = node.transition.children[value]
                    self.__index_in_current_node = 0
            elif isinstance(trans, Conclusion):
                assert trans.status != Status.OVERRUN
                # We tried to draw where history says we should have
                # stopped
                inconsistent_generation()
            else:
                assert isinstance(trans, Branch), trans
                if ir_type != trans.ir_type or kwargs != trans.kwargs:
                    inconsistent_generation()
                try:
                    self.__current_node = trans.children[value]
                except KeyError:
                    self.__current_node = trans.children.setdefault(value, TreeNode())
                self.__index_in_current_node = 0
        if self.__trail[-1] is not self.__current_node:
            self.__trail.append(self.__current_node)

    def kill_branch(self):
        """Mark this part of the tree as not worth re-exploring."""
        if self.killed:
            return

        self.killed = True

        if self.__index_in_current_node < len(self.__current_node.values) or (
            self.__current_node.transition is not None
            and not isinstance(self.__current_node.transition, Killed)
        ):
            inconsistent_generation()

        if self.__current_node.transition is None:
            self.__current_node.transition = Killed(TreeNode())
            self.__update_exhausted()

        self.__current_node = self.__current_node.transition.next_node
        self.__index_in_current_node = 0
        self.__trail.append(self.__current_node)

    def conclude_test(self, status, interesting_origin):
        """Says that ``status`` occurred at node ``node``. This updates the
        node if necessary and checks for consistency."""
        if status == Status.OVERRUN:
            return
        i = self.__index_in_current_node
        node = self.__current_node

        if i < len(node.values) or isinstance(node.transition, Branch):
            inconsistent_generation()

        new_transition = Conclusion(status, interesting_origin)

        if node.transition is not None and node.transition != new_transition:
            # As an, I'm afraid, horrible bodge, we deliberately ignore flakiness
            # where tests go from interesting to valid, because it's much easier
            # to produce good error messages for these further up the stack.
            if isinstance(node.transition, Conclusion) and (
                node.transition.status != Status.INTERESTING
                or new_transition.status != Status.VALID
            ):
                old_origin = node.transition.interesting_origin
                new_origin = new_transition.interesting_origin
                raise FlakyReplay(
                    f"Inconsistent results from replaying a test case!\n"
                    f"  last: {node.transition.status.name} from {old_origin}\n"
                    f"  this: {new_transition.status.name} from {new_origin}",
                    (old_origin, new_origin),
                )
        else:
            node.transition = new_transition

        assert node is self.__trail[-1]
        node.check_exhausted()
        assert len(node.values) > 0 or node.check_exhausted()

        if not self.killed:
            self.__update_exhausted()

    def __update_exhausted(self):
        for t in reversed(self.__trail):
            # Any node we've traversed might have now become exhausted.
            # We check from the right. As soon as we hit a node that
            # isn't exhausted, this automatically implies that all of
            # its parents are not exhausted, so we stop.
            if not t.check_exhausted():
                break
