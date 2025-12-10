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
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import (
    Literal,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
)

from hypothesis.errors import ChoiceTooLarge
from hypothesis.internal.conjecture.floats import float_to_lex, lex_to_float
from hypothesis.internal.conjecture.utils import identity
from hypothesis.internal.floats import float_to_int, make_float_clamper, sign_aware_lte
from hypothesis.internal.intervalsets import IntervalSet

T = TypeVar("T")


class IntegerConstraints(TypedDict):
    min_value: int | None
    max_value: int | None
    weights: dict[int, float] | None
    shrink_towards: int


class FloatConstraints(TypedDict):
    min_value: float
    max_value: float
    allow_nan: bool
    smallest_nonzero_magnitude: float


class StringConstraints(TypedDict):
    intervals: IntervalSet
    min_size: int
    max_size: int


class BytesConstraints(TypedDict):
    min_size: int
    max_size: int


class BooleanConstraints(TypedDict):
    p: float


ChoiceT: TypeAlias = int | str | bool | float | bytes
ChoiceConstraintsT: TypeAlias = (
    IntegerConstraints
    | FloatConstraints
    | StringConstraints
    | BytesConstraints
    | BooleanConstraints
)
ChoiceTypeT: TypeAlias = Literal["integer", "string", "boolean", "float", "bytes"]
ChoiceKeyT: TypeAlias = (
    int | str | bytes | tuple[Literal["bool"], bool] | tuple[Literal["float"], int]
)


@dataclass(slots=True, frozen=False)
class ChoiceTemplate:
    type: Literal["simplest"]
    count: int | None

    def __post_init__(self) -> None:
        if self.count is not None:
            assert self.count > 0


@dataclass(slots=True, frozen=False)
class ChoiceNode:
    type: ChoiceTypeT
    value: ChoiceT
    constraints: ChoiceConstraintsT
    was_forced: bool
    index: int | None = None

    def copy(
        self,
        *,
        with_value: ChoiceT | None = None,
        with_constraints: ChoiceConstraintsT | None = None,
    ) -> "ChoiceNode":
        # we may want to allow this combination in the future, but for now it's
        # a footgun.
        if self.was_forced:
            assert with_value is None, "modifying a forced node doesn't make sense"
        # explicitly not copying index. node indices are only assigned via
        # ExampleRecord. This prevents footguns with relying on stale indices
        # after copying.
        return ChoiceNode(
            type=self.type,
            value=self.value if with_value is None else with_value,
            constraints=(
                self.constraints if with_constraints is None else with_constraints
            ),
            was_forced=self.was_forced,
        )

    @property
    def trivial(self) -> bool:
        """
        A node is trivial if it cannot be simplified any further. This does not
        mean that modifying a trivial node can't produce simpler test cases when
        viewing the tree as a whole. Just that when viewing this node in
        isolation, this is the simplest the node can get.
        """
        if self.was_forced:
            return True

        if self.type != "float":
            zero_value = choice_from_index(0, self.type, self.constraints)
            return choice_equal(self.value, zero_value)
        else:
            constraints = cast(FloatConstraints, self.constraints)
            min_value = constraints["min_value"]
            max_value = constraints["max_value"]
            shrink_towards = 0.0

            if min_value == -math.inf and max_value == math.inf:
                return choice_equal(self.value, shrink_towards)

            if (
                not math.isinf(min_value)
                and not math.isinf(max_value)
                and math.ceil(min_value) <= math.floor(max_value)
            ):
                # the interval contains an integer. the simplest integer is the
                # one closest to shrink_towards
                shrink_towards = max(math.ceil(min_value), shrink_towards)
                shrink_towards = min(math.floor(max_value), shrink_towards)
                return choice_equal(self.value, float(shrink_towards))

            # the real answer here is "the value in [min_value, max_value] with
            # the lowest denominator when represented as a fraction".
            # It would be good to compute this correctly in the future, but it's
            # also not incorrect to be conservative here.
            return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChoiceNode):
            return NotImplemented

        return (
            self.type == other.type
            and choice_equal(self.value, other.value)
            and choice_constraints_equal(self.type, self.constraints, other.constraints)
            and self.was_forced == other.was_forced
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.type,
                choice_key(self.value),
                choice_constraints_key(self.type, self.constraints),
                self.was_forced,
            )
        )

    def __repr__(self) -> str:
        forced_marker = " [forced]" if self.was_forced else ""
        return f"{self.type} {self.value!r}{forced_marker} {self.constraints!r}"


def _size_to_index(size: int, *, alphabet_size: int) -> int:
    # this is the closed form of this geometric series:
    # for i in range(size):
    #     index += alphabet_size**i
    if alphabet_size <= 0:
        assert size == 0
        return 0
    if alphabet_size == 1:
        return size
    v = (alphabet_size**size - 1) // (alphabet_size - 1)
    # mypy thinks (m: int) // (n: int) -> Any. assert it back to int.
    return cast(int, v)


def _index_to_size(index: int, alphabet_size: int) -> int:
    if alphabet_size == 0:
        return 0
    elif alphabet_size == 1:
        # there is only one string of each size, so the size is equal to its
        # ordering.
        return index

    # the closed-form inverse of _size_to_index is
    #   size = math.floor(math.log(index * (alphabet_size - 1) + 1, alphabet_size))
    # which is fast, but suffers from float precision errors. As performance is
    # relatively critical here, we'll use this formula by default, but fall back to
    # a much slower integer-only logarithm when the calculation is too close for
    # comfort.
    total = index * (alphabet_size - 1) + 1
    size = math.log(total, alphabet_size)

    # if this computation is close enough that it could have been affected by
    # floating point errors, use a much slower integer-only logarithm instead,
    # which is guaranteed to be precise.
    if 0 < math.ceil(size) - size < 1e-7:
        s = 0
        while total >= alphabet_size:
            total //= alphabet_size
            s += 1
        return s
    return math.floor(size)


def collection_index(
    choice: Sequence[T],
    *,
    min_size: int,
    alphabet_size: int,
    to_order: Callable[[T], int],
) -> int:
    # Collections are ordered by counting the number of values of each size,
    # starting with min_size. alphabet_size indicates how many options there
    # are for a single element. to_order orders an element by returning an n ≥ 0.

    # we start by adding the size to the index, relative to min_size.
    index = _size_to_index(len(choice), alphabet_size=alphabet_size) - _size_to_index(
        min_size, alphabet_size=alphabet_size
    )
    # We then add each element c to the index, starting from the end (so "ab" is
    # simpler than "ba"). Each loop takes c at position i in the sequence and
    # computes the number of sequences of size i which come before it in the ordering.

    # this running_exp computation is equivalent to doing
    #   index += (alphabet_size**i) * n
    # but reuses intermediate exponentiation steps for efficiency.
    running_exp = 1
    for c in reversed(choice):
        index += running_exp * to_order(c)
        running_exp *= alphabet_size
    return index


def collection_value(
    index: int,
    *,
    min_size: int,
    alphabet_size: int,
    from_order: Callable[[int], T],
) -> list[T]:
    from hypothesis.internal.conjecture.engine import BUFFER_SIZE

    # this function is probably easiest to make sense of as an inverse of
    # collection_index, tracking ~corresponding lines of code between the two.

    index += _size_to_index(min_size, alphabet_size=alphabet_size)
    size = _index_to_size(index, alphabet_size=alphabet_size)
    # index -> value computation can be arbitrarily expensive for arbitrarily
    # large min_size collections. short-circuit if the resulting size would be
    # obviously-too-large. callers will generally turn this into a .mark_overrun().
    if size >= BUFFER_SIZE:
        raise ChoiceTooLarge

    # subtract out the amount responsible for the size
    index -= _size_to_index(size, alphabet_size=alphabet_size)
    vals: list[T] = []
    for i in reversed(range(size)):
        # optimization for common case when we hit index 0. Exponentiation
        # on large integers is expensive!
        if index == 0:
            n = 0
        else:
            n = index // (alphabet_size**i)
            # subtract out the nearest multiple of alphabet_size**i
            index -= n * (alphabet_size**i)
        vals.append(from_order(n))
    return vals


def zigzag_index(value: int, *, shrink_towards: int) -> int:
    # value | 0  1 -1  2 -2  3 -3  4
    # index | 0  1  2  3  4  5  6  7
    index = 2 * abs(shrink_towards - value)
    if value > shrink_towards:
        index -= 1
    return index


def zigzag_value(index: int, *, shrink_towards: int) -> int:
    assert index >= 0
    # count how many "steps" away from shrink_towards we are.
    n = (index + 1) // 2
    # now check if we're stepping up or down from shrink_towards.
    if (index % 2) == 0:
        n *= -1
    return shrink_towards + n


def choice_to_index(choice: ChoiceT, constraints: ChoiceConstraintsT) -> int:
    # This function takes a choice in the choice sequence and returns the
    # complexity index of that choice from among its possible values, where 0
    # is the simplest.
    #
    # Note that the index of a choice depends on its constraints. The simplest value
    # (at index 0) for {"min_value": None, "max_value": None} is 0, while for
    # {"min_value": 1, "max_value": None} the simplest value is 1.
    #
    # choice_from_index inverts this function. An invariant on both functions is
    # that they must be injective. Unfortunately, floats do not currently respect
    # this. That's not *good*, but nothing has blown up - yet. And ordering
    # floats in a sane manner is quite hard, so I've left it for another day.

    if isinstance(choice, int) and not isinstance(choice, bool):
        # Let a = shrink_towards.
        # * Unbounded: Ordered by (|a - x|, sgn(a - x)). Think of a zigzag.
        #   [a, a + 1, a - 1, a + 2, a - 2, ...]
        # * Semi-bounded: Same as unbounded, except stop on one side when you hit
        #   {min, max}_value. so min_value=-1 a=0 has order
        #   [0, 1, -1, 2, 3, 4, ...]
        # * Bounded: Same as unbounded and semibounded, except stop on each side
        #   when you hit {min, max}_value.
        #
        # To simplify and gain intuition about this ordering, you can think about
        # the most common case where 0 is first (a = 0). We deviate from this only
        # rarely, e.g. for datetimes, where we generally want year 2000 to be
        # simpler than year 0.
        constraints = cast(IntegerConstraints, constraints)
        shrink_towards = constraints["shrink_towards"]
        min_value = constraints["min_value"]
        max_value = constraints["max_value"]

        if min_value is not None:
            shrink_towards = max(min_value, shrink_towards)
        if max_value is not None:
            shrink_towards = min(max_value, shrink_towards)

        if min_value is None and max_value is None:
            # case: unbounded
            return zigzag_index(choice, shrink_towards=shrink_towards)
        elif min_value is not None and max_value is None:
            # case: semibounded below

            # min_value = -2
            # index | 0  1  2  3  4  5  6  7
            #     v | 0  1 -1  2 -2  3  4  5
            if abs(choice - shrink_towards) <= (shrink_towards - min_value):
                return zigzag_index(choice, shrink_towards=shrink_towards)
            return choice - min_value
        elif max_value is not None and min_value is None:
            # case: semibounded above
            if abs(choice - shrink_towards) <= (max_value - shrink_towards):
                return zigzag_index(choice, shrink_towards=shrink_towards)
            return max_value - choice
        else:
            # case: bounded

            # range = [-2, 5]
            # shrink_towards = 2
            # index |  0  1  2  3  4  5  6  7
            #     v |  2  3  1  4  0  5 -1 -2
            #
            # ^ with zero weights at index = [0, 2, 6]
            # index |  0  1  2  3  4
            #     v |  3  4  0  5 -2

            assert min_value is not None
            assert max_value is not None
            assert constraints["weights"] is None or all(
                w > 0 for w in constraints["weights"].values()
            ), "technically possible but really annoying to support zero weights"

            # check which side gets exhausted first
            if (shrink_towards - min_value) < (max_value - shrink_towards):
                # Below shrink_towards gets exhausted first. Equivalent to
                # semibounded below
                if abs(choice - shrink_towards) <= (shrink_towards - min_value):
                    return zigzag_index(choice, shrink_towards=shrink_towards)
                return choice - min_value
            else:
                # Above shrink_towards gets exhausted first. Equivalent to semibounded
                # above
                if abs(choice - shrink_towards) <= (max_value - shrink_towards):
                    return zigzag_index(choice, shrink_towards=shrink_towards)
                return max_value - choice
    elif isinstance(choice, bool):
        constraints = cast(BooleanConstraints, constraints)
        # Ordered by [False, True].
        p = constraints["p"]
        if not (2 ** (-64) < p < (1 - 2 ** (-64))):
            # only one option is possible, so whatever it is is first.
            return 0
        return int(choice)
    elif isinstance(choice, bytes):
        constraints = cast(BytesConstraints, constraints)
        return collection_index(
            list(choice),
            min_size=constraints["min_size"],
            alphabet_size=2**8,
            to_order=identity,
        )
    elif isinstance(choice, str):
        constraints = cast(StringConstraints, constraints)
        intervals = constraints["intervals"]
        return collection_index(
            choice,
            min_size=constraints["min_size"],
            alphabet_size=len(intervals),
            to_order=intervals.index_from_char_in_shrink_order,
        )
    elif isinstance(choice, float):
        sign = int(math.copysign(1.0, choice) < 0)
        return (sign << 64) | float_to_lex(abs(choice))
    else:
        raise NotImplementedError


def choice_from_index(
    index: int, choice_type: ChoiceTypeT, constraints: ChoiceConstraintsT
) -> ChoiceT:
    assert index >= 0
    if choice_type == "integer":
        constraints = cast(IntegerConstraints, constraints)
        shrink_towards = constraints["shrink_towards"]
        min_value = constraints["min_value"]
        max_value = constraints["max_value"]

        if min_value is not None:
            shrink_towards = max(min_value, shrink_towards)
        if max_value is not None:
            shrink_towards = min(max_value, shrink_towards)

        if min_value is None and max_value is None:
            # case: unbounded
            return zigzag_value(index, shrink_towards=shrink_towards)
        elif min_value is not None and max_value is None:
            # case: semibounded below
            if index <= zigzag_index(min_value, shrink_towards=shrink_towards):
                return zigzag_value(index, shrink_towards=shrink_towards)
            return index + min_value
        elif max_value is not None and min_value is None:
            # case: semibounded above
            if index <= zigzag_index(max_value, shrink_towards=shrink_towards):
                return zigzag_value(index, shrink_towards=shrink_towards)
            return max_value - index
        else:
            # case: bounded
            assert min_value is not None
            assert max_value is not None
            assert constraints["weights"] is None or all(
                w > 0 for w in constraints["weights"].values()
            ), "possible but really annoying to support zero weights"

            if (shrink_towards - min_value) < (max_value - shrink_towards):
                # equivalent to semibounded below case
                if index <= zigzag_index(min_value, shrink_towards=shrink_towards):
                    return zigzag_value(index, shrink_towards=shrink_towards)
                return index + min_value
            else:
                # equivalent to semibounded above case
                if index <= zigzag_index(max_value, shrink_towards=shrink_towards):
                    return zigzag_value(index, shrink_towards=shrink_towards)
                return max_value - index
    elif choice_type == "boolean":
        constraints = cast(BooleanConstraints, constraints)
        # Ordered by [False, True].
        p = constraints["p"]
        only = None
        if p <= 2 ** (-64):
            only = False
        elif p >= (1 - 2 ** (-64)):
            only = True

        assert index in {0, 1}
        if only is not None:
            # only one choice
            assert index == 0
            return only
        return bool(index)
    elif choice_type == "bytes":
        constraints = cast(BytesConstraints, constraints)
        value_b = collection_value(
            index,
            min_size=constraints["min_size"],
            alphabet_size=2**8,
            from_order=identity,
        )
        return bytes(value_b)
    elif choice_type == "string":
        constraints = cast(StringConstraints, constraints)
        intervals = constraints["intervals"]
        # _s because mypy is unhappy with reusing different-typed names in branches,
        # even if the branches are disjoint.
        value_s = collection_value(
            index,
            min_size=constraints["min_size"],
            alphabet_size=len(intervals),
            from_order=intervals.char_in_shrink_order,
        )
        return "".join(value_s)
    elif choice_type == "float":
        constraints = cast(FloatConstraints, constraints)
        sign = -1 if index >> 64 else 1
        result = sign * lex_to_float(index & ((1 << 64) - 1))

        clamper = make_float_clamper(
            min_value=constraints["min_value"],
            max_value=constraints["max_value"],
            smallest_nonzero_magnitude=constraints["smallest_nonzero_magnitude"],
            allow_nan=constraints["allow_nan"],
        )
        return clamper(result)
    else:
        raise NotImplementedError


def choice_permitted(choice: ChoiceT, constraints: ChoiceConstraintsT) -> bool:
    if isinstance(choice, int) and not isinstance(choice, bool):
        constraints = cast(IntegerConstraints, constraints)
        min_value = constraints["min_value"]
        max_value = constraints["max_value"]
        if min_value is not None and choice < min_value:
            return False
        return not (max_value is not None and choice > max_value)
    elif isinstance(choice, float):
        constraints = cast(FloatConstraints, constraints)
        if math.isnan(choice):
            return constraints["allow_nan"]
        if 0 < abs(choice) < constraints["smallest_nonzero_magnitude"]:
            return False
        return sign_aware_lte(constraints["min_value"], choice) and sign_aware_lte(
            choice, constraints["max_value"]
        )
    elif isinstance(choice, str):
        constraints = cast(StringConstraints, constraints)
        if len(choice) < constraints["min_size"]:
            return False
        if (
            constraints["max_size"] is not None
            and len(choice) > constraints["max_size"]
        ):
            return False
        return all(ord(c) in constraints["intervals"] for c in choice)
    elif isinstance(choice, bytes):
        constraints = cast(BytesConstraints, constraints)
        if len(choice) < constraints["min_size"]:
            return False
        return constraints["max_size"] is None or len(choice) <= constraints["max_size"]
    elif isinstance(choice, bool):
        constraints = cast(BooleanConstraints, constraints)
        if constraints["p"] <= 0:
            return choice is False
        if constraints["p"] >= 1:
            return choice is True
        return True
    else:
        raise NotImplementedError(f"unhandled type {type(choice)} with value {choice}")


def choices_key(choices: Sequence[ChoiceT]) -> tuple[ChoiceKeyT, ...]:
    return tuple(choice_key(choice) for choice in choices)


def choice_key(choice: ChoiceT) -> ChoiceKeyT:
    if isinstance(choice, float):
        # float_to_int to distinguish -0.0/0.0, signaling/nonsignaling nans, etc,
        # and then add a "float" key to avoid colliding with actual integers.
        return ("float", float_to_int(choice))
    if isinstance(choice, bool):
        # avoid choice_key(0) == choice_key(False)
        return ("bool", choice)
    return choice


def choice_equal(choice1: ChoiceT, choice2: ChoiceT) -> bool:
    assert type(choice1) is type(choice2), (choice1, choice2)
    return choice_key(choice1) == choice_key(choice2)


def choice_constraints_equal(
    choice_type: ChoiceTypeT,
    constraints1: ChoiceConstraintsT,
    constraints2: ChoiceConstraintsT,
) -> bool:
    return choice_constraints_key(choice_type, constraints1) == choice_constraints_key(
        choice_type, constraints2
    )


def choice_constraints_key(
    choice_type: ChoiceTypeT, constraints: ChoiceConstraintsT
) -> tuple[Hashable, ...]:
    if choice_type == "float":
        constraints = cast(FloatConstraints, constraints)
        return (
            float_to_int(constraints["min_value"]),
            float_to_int(constraints["max_value"]),
            constraints["allow_nan"],
            constraints["smallest_nonzero_magnitude"],
        )
    if choice_type == "integer":
        constraints = cast(IntegerConstraints, constraints)
        return (
            constraints["min_value"],
            constraints["max_value"],
            None if constraints["weights"] is None else tuple(constraints["weights"]),
            constraints["shrink_towards"],
        )
    return tuple(constraints[key] for key in sorted(constraints))  # type: ignore


def choices_size(choices: Iterable[ChoiceT]) -> int:
    from hypothesis.database import choices_to_bytes

    return len(choices_to_bytes(choices))
