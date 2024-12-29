# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import sys
import warnings
from collections import abc, defaultdict
from collections.abc import Sequence
from functools import lru_cache
from random import shuffle
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    TypeVar,
    Union,
    cast,
    overload,
)

from hypothesis._settings import HealthCheck, Phase, Verbosity, settings
from hypothesis.control import _current_build_context, current_build_context
from hypothesis.errors import (
    HypothesisException,
    HypothesisWarning,
    InvalidArgument,
    NonInteractiveExampleWarning,
    UnsatisfiedAssumption,
)
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.conjecture.utils import (
    calc_label_from_cls,
    calc_label_from_name,
    combine_labels,
)
from hypothesis.internal.coverage import check_function
from hypothesis.internal.reflection import (
    get_pretty_function_description,
    is_identity_function,
)
from hypothesis.strategies._internal.utils import defines_strategy
from hypothesis.utils.conventions import UniqueIdentifier

# TODO: Use `(3, 13)` once Python 3.13 is released.
if sys.version_info >= (3, 13, 0, "final"):
    Ex = TypeVar("Ex", covariant=True, default=Any)
elif TYPE_CHECKING:
    from typing_extensions import TypeVar  # type: ignore[assignment]

    Ex = TypeVar("Ex", covariant=True, default=Any)  # type: ignore[call-arg,misc]
else:
    Ex = TypeVar("Ex", covariant=True)

Ex_Inv = TypeVar("Ex_Inv")
T = TypeVar("T")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

calculating = UniqueIdentifier("calculating")

MAPPED_SEARCH_STRATEGY_DO_DRAW_LABEL = calc_label_from_name(
    "another attempted draw in MappedStrategy"
)

FILTERED_SEARCH_STRATEGY_DO_DRAW_LABEL = calc_label_from_name(
    "single loop iteration in FilteredStrategy"
)


def recursive_property(name, default):
    """Handle properties which may be mutually recursive among a set of
    strategies.

    These are essentially lazily cached properties, with the ability to set
    an override: If the property has not been explicitly set, we calculate
    it on first access and memoize the result for later.

    The problem is that for properties that depend on each other, a naive
    calculation strategy may hit infinite recursion. Consider for example
    the property is_empty. A strategy defined as x = st.deferred(lambda: x)
    is certainly empty (in order to draw a value from x we would have to
    draw a value from x, for which we would have to draw a value from x,
    ...), but in order to calculate it the naive approach would end up
    calling x.is_empty in order to calculate x.is_empty in order to etc.

    The solution is one of fixed point calculation. We start with a default
    value that is the value of the property in the absence of evidence to
    the contrary, and then update the values of the property for all
    dependent strategies until we reach a fixed point.

    The approach taken roughly follows that in section 4.2 of Adams,
    Michael D., Celeste Hollenbeck, and Matthew Might. "On the complexity
    and performance of parsing with derivatives." ACM SIGPLAN Notices 51.6
    (2016): 224-236.
    """
    cache_key = "cached_" + name
    calculation = "calc_" + name
    force_key = "force_" + name

    def forced_value(target):
        try:
            return getattr(target, force_key)
        except AttributeError:
            return getattr(target, cache_key)

    def accept(self):
        try:
            return forced_value(self)
        except AttributeError:
            pass

        mapping = {}
        sentinel = object()
        hit_recursion = [False]

        # For a first pass we do a direct recursive calculation of the
        # property, but we block recursively visiting a value in the
        # computation of its property: When that happens, we simply
        # note that it happened and return the default value.
        def recur(strat):
            try:
                return forced_value(strat)
            except AttributeError:
                pass
            result = mapping.get(strat, sentinel)
            if result is calculating:
                hit_recursion[0] = True
                return default
            elif result is sentinel:
                mapping[strat] = calculating
                mapping[strat] = getattr(strat, calculation)(recur)
                return mapping[strat]
            return result

        recur(self)

        # If we hit self-recursion in the computation of any strategy
        # value, our mapping at the end is imprecise - it may or may
        # not have the right values in it. We now need to proceed with
        # a more careful fixed point calculation to get the exact
        # values. Hopefully our mapping is still pretty good and it
        # won't take a large number of updates to reach a fixed point.
        if hit_recursion[0]:
            needs_update = set(mapping)

            # We track which strategies use which in the course of
            # calculating their property value. If A ever uses B in
            # the course of calculating its value, then whenever the
            # value of B changes we might need to update the value of
            # A.
            listeners = defaultdict(set)
        else:
            needs_update = None

        def recur2(strat):
            def recur_inner(other):
                try:
                    return forced_value(other)
                except AttributeError:
                    pass
                listeners[other].add(strat)
                result = mapping.get(other, sentinel)
                if result is sentinel:
                    needs_update.add(other)
                    mapping[other] = default
                    return default
                return result

            return recur_inner

        count = 0
        seen = set()
        while needs_update:
            count += 1
            # If we seem to be taking a really long time to stabilize we
            # start tracking seen values to attempt to detect an infinite
            # loop. This should be impossible, and most code will never
            # hit the count, but having an assertion for it means that
            # testing is easier to debug and we don't just have a hung
            # test.
            # Note: This is actually covered, by test_very_deep_deferral
            # in tests/cover/test_deferred_strategies.py. Unfortunately it
            # runs into a coverage bug. See
            # https://github.com/nedbat/coveragepy/issues/605
            # for details.
            if count > 50:  # pragma: no cover
                key = frozenset(mapping.items())
                assert key not in seen, (key, name)
                seen.add(key)
            to_update = needs_update
            needs_update = set()
            for strat in to_update:
                new_value = getattr(strat, calculation)(recur2(strat))
                if new_value != mapping[strat]:
                    needs_update.update(listeners[strat])
                    mapping[strat] = new_value

        # We now have a complete and accurate calculation of the
        # property values for everything we have seen in the course of
        # running this calculation. We simultaneously update all of
        # them (not just the strategy we started out with).
        for k, v in mapping.items():
            setattr(k, cache_key, v)
        return getattr(self, cache_key)

    accept.__name__ = name
    return property(accept)


class SearchStrategy(Generic[Ex]):
    """A SearchStrategy is an object that knows how to explore data of a given
    type.

    Except where noted otherwise, methods on this class are not part of
    the public API and their behaviour may change significantly between
    minor version releases. They will generally be stable between patch
    releases.
    """

    supports_find = True
    validate_called = False
    __label = None
    __module__ = "hypothesis.strategies"

    def available(self, data):
        """Returns whether this strategy can *currently* draw any
        values. This typically useful for stateful testing where ``Bundle``
        grows over time a list of value to choose from.

        Unlike ``empty`` property, this method's return value may change
        over time.
        Note: ``data`` parameter will only be used for introspection and no
        value drawn from it.
        """
        return not self.is_empty

    # Returns True if this strategy can never draw a value and will always
    # result in the data being marked invalid.
    # The fact that this returns False does not guarantee that a valid value
    # can be drawn - this is not intended to be perfect, and is primarily
    # intended to be an optimisation for some cases.
    is_empty = recursive_property("is_empty", True)

    # Returns True if values from this strategy can safely be reused without
    # this causing unexpected behaviour.

    # True if values from this strategy can be implicitly reused (e.g. as
    # background values in a numpy array) without causing surprising
    # user-visible behaviour. Should be false for built-in strategies that
    # produce mutable values, and for strategies that have been mapped/filtered
    # by arbitrary user-provided functions.
    has_reusable_values = recursive_property("has_reusable_values", True)

    # Whether this strategy is suitable for holding onto in a cache.
    is_cacheable = recursive_property("is_cacheable", True)

    def calc_is_cacheable(self, recur):
        return True

    def calc_is_empty(self, recur):
        # Note: It is correct and significant that the default return value
        # from calc_is_empty is False despite the default value for is_empty
        # being true. The reason for this is that strategies should be treated
        # as empty absent evidence to the contrary, but most basic strategies
        # are trivially non-empty and it would be annoying to have to override
        # this method to show that.
        return False

    def calc_has_reusable_values(self, recur):
        return False

    def example(self) -> Ex:
        """Provide an example of the sort of value that this strategy
        generates. This is biased to be slightly simpler than is typical for
        values from this strategy, for clarity purposes.

        This method shouldn't be taken too seriously. It's here for interactive
        exploration of the API, not for any sort of real testing.

        This method is part of the public API.
        """
        if getattr(sys, "ps1", None) is None:  # pragma: no branch
            # The other branch *is* covered in cover/test_examples.py; but as that
            # uses `pexpect` for an interactive session `coverage` doesn't see it.
            warnings.warn(
                "The `.example()` method is good for exploring strategies, but should "
                "only be used interactively.  We recommend using `@given` for tests - "
                "it performs better, saves and replays failures to avoid flakiness, "
                "and reports minimal examples. (strategy: %r)" % (self,),
                NonInteractiveExampleWarning,
                stacklevel=2,
            )

        context = _current_build_context.value
        if context is not None:
            if context.data is not None and context.data.depth > 0:
                raise HypothesisException(
                    "Using example() inside a strategy definition is a bad "
                    "idea. Instead consider using hypothesis.strategies.builds() "
                    "or @hypothesis.strategies.composite to define your strategy."
                    " See https://hypothesis.readthedocs.io/en/latest/data.html"
                    "#hypothesis.strategies.builds or "
                    "https://hypothesis.readthedocs.io/en/latest/data.html"
                    "#composite-strategies for more details."
                )
            else:
                raise HypothesisException(
                    "Using example() inside a test function is a bad "
                    "idea. Instead consider using hypothesis.strategies.data() "
                    "to draw more examples during testing. See "
                    "https://hypothesis.readthedocs.io/en/latest/data.html"
                    "#drawing-interactively-in-tests for more details."
                )

        try:
            return self.__examples.pop()
        except (AttributeError, IndexError):
            self.__examples: list[Ex] = []

        from hypothesis.core import given

        # Note: this function has a weird name because it might appear in
        # tracebacks, and we want users to know that they can ignore it.
        @given(self)
        @settings(
            database=None,
            max_examples=100,
            deadline=None,
            verbosity=Verbosity.quiet,
            phases=(Phase.generate,),
            suppress_health_check=list(HealthCheck),
        )
        def example_generating_inner_function(ex):
            self.__examples.append(ex)

        example_generating_inner_function()
        shuffle(self.__examples)
        return self.__examples.pop()

    def map(self, pack: Callable[[Ex], T]) -> "SearchStrategy[T]":
        """Returns a new strategy that generates values by generating a value
        from this strategy and then calling pack() on the result, giving that.

        This method is part of the public API.
        """
        if is_identity_function(pack):
            return self  # type: ignore  # Mypy has no way to know that `Ex == T`
        return MappedStrategy(self, pack=pack)

    def flatmap(
        self, expand: Callable[[Ex], "SearchStrategy[T]"]
    ) -> "SearchStrategy[T]":
        """Returns a new strategy that generates values by generating a value
        from this strategy, say x, then generating a value from
        strategy(expand(x))

        This method is part of the public API.
        """
        from hypothesis.strategies._internal.flatmapped import FlatMapStrategy

        return FlatMapStrategy(expand=expand, strategy=self)

    def filter(self, condition: Callable[[Ex], Any]) -> "SearchStrategy[Ex]":
        """Returns a new strategy that generates values from this strategy
        which satisfy the provided condition. Note that if the condition is too
        hard to satisfy this might result in your tests failing with
        Unsatisfiable.

        This method is part of the public API.
        """
        return FilteredStrategy(conditions=(condition,), strategy=self)

    def _filter_for_filtered_draw(self, condition):
        # Hook for parent strategies that want to perform fallible filtering
        # on one of their internal strategies (e.g. UniqueListStrategy).
        # The returned object must have a `.do_filtered_draw(data)` method
        # that behaves like `do_draw`, but returns the sentinel object
        # `filter_not_satisfied` if the condition could not be satisfied.

        # This is separate from the main `filter` method so that strategies
        # can override `filter` without having to also guarantee a
        # `do_filtered_draw` method.
        return FilteredStrategy(conditions=(condition,), strategy=self)

    @property
    def branches(self) -> list["SearchStrategy[Ex]"]:
        return [self]

    def __or__(self, other: "SearchStrategy[T]") -> "SearchStrategy[Union[Ex, T]]":
        """Return a strategy which produces values by randomly drawing from one
        of this strategy or the other strategy.

        This method is part of the public API.
        """
        if not isinstance(other, SearchStrategy):
            raise ValueError(f"Cannot | a SearchStrategy with {other!r}")
        return OneOfStrategy((self, other))

    def __bool__(self) -> bool:
        warnings.warn(
            f"bool({self!r}) is always True, did you mean to draw a value?",
            HypothesisWarning,
            stacklevel=2,
        )
        return True

    def validate(self) -> None:
        """Throw an exception if the strategy is not valid.

        This can happen due to lazy construction
        """
        if self.validate_called:
            return
        try:
            self.validate_called = True
            self.do_validate()
            self.is_empty
            self.has_reusable_values
        except Exception:
            self.validate_called = False
            raise

    LABELS: ClassVar[dict[type, int]] = {}

    @property
    def class_label(self):
        cls = self.__class__
        try:
            return cls.LABELS[cls]
        except KeyError:
            pass
        result = calc_label_from_cls(cls)
        cls.LABELS[cls] = result
        return result

    @property
    def label(self) -> int:
        if self.__label is calculating:
            return 0
        if self.__label is None:
            self.__label = calculating
            self.__label = self.calc_label()
        return cast(int, self.__label)

    def calc_label(self):
        return self.class_label

    def do_validate(self):
        pass

    def do_draw(self, data: ConjectureData) -> Ex:
        raise NotImplementedError(f"{type(self).__name__}.do_draw")

    def __init__(self):
        pass


def is_simple_data(value):
    try:
        hash(value)
        return True
    except TypeError:
        return False


class SampledFromStrategy(SearchStrategy):
    """A strategy which samples from a set of elements. This is essentially
    equivalent to using a OneOfStrategy over Just strategies but may be more
    efficient and convenient.
    """

    _MAX_FILTER_CALLS = 10_000

    def __init__(self, elements, repr_=None, transformations=()):
        super().__init__()
        self.elements = cu.check_sample(elements, "sampled_from")
        assert self.elements
        self.repr_ = repr_
        self._transformations = transformations

    def map(self, pack):
        return type(self)(
            self.elements,
            repr_=self.repr_,
            transformations=(*self._transformations, ("map", pack)),
        )

    def filter(self, condition):
        return type(self)(
            self.elements,
            repr_=self.repr_,
            transformations=(*self._transformations, ("filter", condition)),
        )

    def __repr__(self):
        return (
            self.repr_
            or "sampled_from(["
            + ", ".join(map(get_pretty_function_description, self.elements))
            + "])"
        ) + "".join(
            f".{name}({get_pretty_function_description(f)})"
            for name, f in self._transformations
        )

    def calc_has_reusable_values(self, recur):
        # Because our custom .map/.filter implementations skip the normal
        # wrapper strategies (which would automatically return False for us),
        # we need to manually return False here if any transformations have
        # been applied.
        return not self._transformations

    def calc_is_cacheable(self, recur):
        return is_simple_data(self.elements)

    def _transform(self, element):
        # Used in UniqueSampledListStrategy
        for name, f in self._transformations:
            if name == "map":
                result = f(element)
                if build_context := _current_build_context.value:
                    build_context.record_call(result, f, [element], {})
                element = result
            else:
                assert name == "filter"
                if not f(element):
                    return filter_not_satisfied
        return element

    def do_draw(self, data):
        result = self.do_filtered_draw(data)
        if isinstance(result, SearchStrategy) and all(
            isinstance(x, SearchStrategy) for x in self.elements
        ):
            data._sampled_from_all_strategies_elements_message = (
                "sample_from was given a collection of strategies: "
                "{!r}. Was one_of intended?",
                self.elements,
            )
        if result is filter_not_satisfied:
            data.mark_invalid(f"Aborted test because unable to satisfy {self!r}")
        return result

    def get_element(self, i):
        return self._transform(self.elements[i])

    def do_filtered_draw(self, data):
        # Set of indices that have been tried so far, so that we never test
        # the same element twice during a draw.
        known_bad_indices = set()

        # Start with ordinary rejection sampling. It's fast if it works, and
        # if it doesn't work then it was only a small amount of overhead.
        for _ in range(3):
            i = data.draw_integer(0, len(self.elements) - 1)
            if i not in known_bad_indices:
                element = self.get_element(i)
                if element is not filter_not_satisfied:
                    return element
                if not known_bad_indices:
                    data.events[f"Retried draw from {self!r} to satisfy filter"] = ""
                known_bad_indices.add(i)

        # If we've tried all the possible elements, give up now.
        max_good_indices = len(self.elements) - len(known_bad_indices)
        if not max_good_indices:
            return filter_not_satisfied

        # Impose an arbitrary cutoff to prevent us from wasting too much time
        # on very large element lists.
        max_good_indices = min(max_good_indices, self._MAX_FILTER_CALLS - 3)

        # Before building the list of allowed indices, speculatively choose
        # one of them. We don't yet know how many allowed indices there will be,
        # so this choice might be out-of-bounds, but that's OK.
        speculative_index = data.draw_integer(0, max_good_indices - 1)

        # Calculate the indices of allowed values, so that we can choose one
        # of them at random. But if we encounter the speculatively-chosen one,
        # just use that and return immediately.  Note that we also track the
        # allowed elements, in case of .map(some_stateful_function)
        allowed = []
        for i in range(min(len(self.elements), self._MAX_FILTER_CALLS - 3)):
            if i not in known_bad_indices:
                element = self.get_element(i)
                if element is not filter_not_satisfied:
                    allowed.append((i, element))
                    if len(allowed) > speculative_index:
                        # Early-exit case: We reached the speculative index, so
                        # we just return the corresponding element.
                        data.draw_integer(0, len(self.elements) - 1, forced=i)
                        return element

        # The speculative index didn't work out, but at this point we've built
        # and can choose from the complete list of allowed indices and elements.
        if allowed:
            i, element = data.choice(allowed)
            data.draw_integer(0, len(self.elements) - 1, forced=i)
            return element
        # If there are no allowed indices, the filter couldn't be satisfied.
        return filter_not_satisfied


class OneOfStrategy(SearchStrategy[Ex]):
    """Implements a union of strategies. Given a number of strategies this
    generates values which could have come from any of them.

    The conditional distribution draws uniformly at random from some
    non-empty subset of these strategies and then draws from the
    conditional distribution of that strategy.
    """

    def __init__(self, strategies):
        super().__init__()
        strategies = tuple(strategies)
        self.original_strategies = list(strategies)
        self.__element_strategies = None
        self.__in_branches = False

    def calc_is_empty(self, recur):
        return all(recur(e) for e in self.original_strategies)

    def calc_has_reusable_values(self, recur):
        return all(recur(e) for e in self.original_strategies)

    def calc_is_cacheable(self, recur):
        return all(recur(e) for e in self.original_strategies)

    @property
    def element_strategies(self):
        if self.__element_strategies is None:
            # While strategies are hashable, they use object.__hash__ and are
            # therefore distinguished only by identity.
            #
            # In principle we could "just" define a __hash__ method
            # (and __eq__, but that's easy in terms of type() and hash())
            # to make this more powerful, but this is harder than it sounds:
            #
            # 1. Strategies are often distinguished by non-hashable attributes,
            #    or by attributes that have the same hash value ("^.+" / b"^.+").
            # 2. LazyStrategy: can't reify the wrapped strategy without breaking
            #    laziness, so there's a hash each for the lazy and the nonlazy.
            #
            # Having made several attempts, the minor benefits of making strategies
            # hashable are simply not worth the engineering effort it would take.
            # See also issues #2291 and #2327.
            seen = {self}
            strategies = []
            for arg in self.original_strategies:
                check_strategy(arg)
                if not arg.is_empty:
                    for s in arg.branches:
                        if s not in seen and not s.is_empty:
                            seen.add(s)
                            strategies.append(s)
            self.__element_strategies = strategies
        return self.__element_strategies

    def calc_label(self):
        return combine_labels(
            self.class_label, *(p.label for p in self.original_strategies)
        )

    def do_draw(self, data: ConjectureData) -> Ex:
        strategy = data.draw(
            SampledFromStrategy(self.element_strategies).filter(
                lambda s: s.available(data)
            )
        )
        return data.draw(strategy)

    def __repr__(self):
        return "one_of(%s)" % ", ".join(map(repr, self.original_strategies))

    def do_validate(self):
        for e in self.element_strategies:
            e.validate()

    @property
    def branches(self):
        if not self.__in_branches:
            try:
                self.__in_branches = True
                return self.element_strategies
            finally:
                self.__in_branches = False
        else:
            return [self]

    def filter(self, condition):
        return FilteredStrategy(
            OneOfStrategy([s.filter(condition) for s in self.original_strategies]),
            conditions=(),
        )


@overload
def one_of(
    __args: Sequence[SearchStrategy[Any]],
) -> SearchStrategy[Any]:  # pragma: no cover
    ...


@overload
def one_of(__a1: SearchStrategy[Ex]) -> SearchStrategy[Ex]:  # pragma: no cover
    ...


@overload
def one_of(
    __a1: SearchStrategy[Ex], __a2: SearchStrategy[T]
) -> SearchStrategy[Union[Ex, T]]:  # pragma: no cover
    ...


@overload
def one_of(
    __a1: SearchStrategy[Ex], __a2: SearchStrategy[T], __a3: SearchStrategy[T3]
) -> SearchStrategy[Union[Ex, T, T3]]:  # pragma: no cover
    ...


@overload
def one_of(
    __a1: SearchStrategy[Ex],
    __a2: SearchStrategy[T],
    __a3: SearchStrategy[T3],
    __a4: SearchStrategy[T4],
) -> SearchStrategy[Union[Ex, T, T3, T4]]:  # pragma: no cover
    ...


@overload
def one_of(
    __a1: SearchStrategy[Ex],
    __a2: SearchStrategy[T],
    __a3: SearchStrategy[T3],
    __a4: SearchStrategy[T4],
    __a5: SearchStrategy[T5],
) -> SearchStrategy[Union[Ex, T, T3, T4, T5]]:  # pragma: no cover
    ...


@overload
def one_of(*args: SearchStrategy[Any]) -> SearchStrategy[Any]:  # pragma: no cover
    ...


@defines_strategy(never_lazy=True)
def one_of(
    *args: Union[Sequence[SearchStrategy[Any]], SearchStrategy[Any]]
) -> SearchStrategy[Any]:
    # Mypy workaround alert:  Any is too loose above; the return parameter
    # should be the union of the input parameters.  Unfortunately, Mypy <=0.600
    # raises errors due to incompatible inputs instead.  See #1270 for links.
    # v0.610 doesn't error; it gets inference wrong for 2+ arguments instead.
    """Return a strategy which generates values from any of the argument
    strategies.

    This may be called with one iterable argument instead of multiple
    strategy arguments, in which case ``one_of(x)`` and ``one_of(*x)`` are
    equivalent.

    Examples from this strategy will generally shrink to ones that come from
    strategies earlier in the list, then shrink according to behaviour of the
    strategy that produced them. In order to get good shrinking behaviour,
    try to put simpler strategies first. e.g. ``one_of(none(), text())`` is
    better than ``one_of(text(), none())``.

    This is especially important when using recursive strategies. e.g.
    ``x = st.deferred(lambda: st.none() | st.tuples(x, x))`` will shrink well,
    but ``x = st.deferred(lambda: st.tuples(x, x) | st.none())`` will shrink
    very badly indeed.
    """
    if len(args) == 1 and not isinstance(args[0], SearchStrategy):
        try:
            args = tuple(args[0])
        except TypeError:
            pass
    if len(args) == 1 and isinstance(args[0], SearchStrategy):
        # This special-case means that we can one_of over lists of any size
        # without incurring any performance overhead when there is only one
        # strategy, and keeps our reprs simple.
        return args[0]
    if args and not any(isinstance(a, SearchStrategy) for a in args):
        # And this special case is to give a more-specific error message if it
        # seems that the user has confused `one_of()` for  `sampled_from()`;
        # the remaining validation is left to OneOfStrategy.  See PR #2627.
        raise InvalidArgument(
            f"Did you mean st.sampled_from({list(args)!r})?  st.one_of() is used "
            "to combine strategies, but all of the arguments were of other types."
        )
    return OneOfStrategy(args)


class MappedStrategy(SearchStrategy[Ex]):
    """A strategy which is defined purely by conversion to and from another
    strategy.

    Its parameter and distribution come from that other strategy.
    """

    def __init__(self, strategy, pack):
        super().__init__()
        self.mapped_strategy = strategy
        self.pack = pack

    def calc_is_empty(self, recur):
        return recur(self.mapped_strategy)

    def calc_is_cacheable(self, recur):
        return recur(self.mapped_strategy)

    def __repr__(self):
        if not hasattr(self, "_cached_repr"):
            self._cached_repr = f"{self.mapped_strategy!r}.map({get_pretty_function_description(self.pack)})"
        return self._cached_repr

    def do_validate(self):
        self.mapped_strategy.validate()

    def do_draw(self, data: ConjectureData) -> Any:
        with warnings.catch_warnings():
            if isinstance(self.pack, type) and issubclass(
                self.pack, (abc.Mapping, abc.Set)
            ):
                warnings.simplefilter("ignore", BytesWarning)
            for _ in range(3):
                try:
                    data.start_example(MAPPED_SEARCH_STRATEGY_DO_DRAW_LABEL)
                    x = data.draw(self.mapped_strategy)
                    result = self.pack(x)  # type: ignore
                    data.stop_example()
                    current_build_context().record_call(result, self.pack, [x], {})
                    return result
                except UnsatisfiedAssumption:
                    data.stop_example(discard=True)
        raise UnsatisfiedAssumption

    @property
    def branches(self) -> list[SearchStrategy[Ex]]:
        return [
            MappedStrategy(strategy, pack=self.pack)
            for strategy in self.mapped_strategy.branches
        ]

    def filter(self, condition: Callable[[Ex], Any]) -> "SearchStrategy[Ex]":
        # Includes a special case so that we can rewrite filters on collection
        # lengths, when most collections are `st.lists(...).map(the_type)`.
        ListStrategy = _list_strategy_type()
        if not isinstance(self.mapped_strategy, ListStrategy) or not (
            (isinstance(self.pack, type) and issubclass(self.pack, abc.Collection))
            or self.pack in _collection_ish_functions()
        ):
            return super().filter(condition)

        # Check whether our inner list strategy can rewrite this filter condition.
        # If not, discard the result and _only_ apply a new outer filter.
        new = ListStrategy.filter(self.mapped_strategy, condition)
        if getattr(new, "filtered_strategy", None) is self.mapped_strategy:
            return super().filter(condition)  # didn't rewrite

        # Apply a new outer filter even though we rewrote the inner strategy,
        # because some collections can change the list length (dict, set, etc).
        return FilteredStrategy(type(self)(new, self.pack), conditions=(condition,))


@lru_cache
def _list_strategy_type():
    from hypothesis.strategies._internal.collections import ListStrategy

    return ListStrategy


def _collection_ish_functions():
    funcs = [sorted]
    if np := sys.modules.get("numpy"):
        # c.f. https://numpy.org/doc/stable/reference/routines.array-creation.html
        # Probably only `np.array` and `np.asarray` will be used in practice,
        # but why should that stop us when we've already gone this far?
        funcs += [
            np.empty_like,
            np.eye,
            np.identity,
            np.ones_like,
            np.zeros_like,
            np.array,
            np.asarray,
            np.asanyarray,
            np.ascontiguousarray,
            np.asmatrix,
            np.copy,
            np.rec.array,
            np.rec.fromarrays,
            np.rec.fromrecords,
            np.diag,
            # bonus undocumented functions from tab-completion:
            np.asarray_chkfinite,
            np.asfortranarray,
        ]

    return funcs


filter_not_satisfied = UniqueIdentifier("filter not satisfied")


class FilteredStrategy(SearchStrategy[Ex]):
    def __init__(self, strategy, conditions):
        super().__init__()
        if isinstance(strategy, FilteredStrategy):
            # Flatten chained filters into a single filter with multiple conditions.
            self.flat_conditions = strategy.flat_conditions + conditions
            self.filtered_strategy = strategy.filtered_strategy
        else:
            self.flat_conditions = conditions
            self.filtered_strategy = strategy

        assert isinstance(self.flat_conditions, tuple)
        assert not isinstance(self.filtered_strategy, FilteredStrategy)

        self.__condition = None

    def calc_is_empty(self, recur):
        return recur(self.filtered_strategy)

    def calc_is_cacheable(self, recur):
        return recur(self.filtered_strategy)

    def __repr__(self):
        if not hasattr(self, "_cached_repr"):
            self._cached_repr = "{!r}{}".format(
                self.filtered_strategy,
                "".join(
                    f".filter({get_pretty_function_description(cond)})"
                    for cond in self.flat_conditions
                ),
            )
        return self._cached_repr

    def do_validate(self):
        # Start by validating our inner filtered_strategy.  If this was a LazyStrategy,
        # validation also reifies it so that subsequent calls to e.g. `.filter()` will
        # be passed through.
        self.filtered_strategy.validate()
        # So now we have a reified inner strategy, we'll replay all our saved
        # predicates in case some or all of them can be rewritten.  Note that this
        # replaces the `fresh` strategy too!
        fresh = self.filtered_strategy
        for cond in self.flat_conditions:
            fresh = fresh.filter(cond)
        if isinstance(fresh, FilteredStrategy):
            # In this case we have at least some non-rewritten filter predicates,
            # so we just re-initialize the strategy.
            FilteredStrategy.__init__(
                self, fresh.filtered_strategy, fresh.flat_conditions
            )
        else:
            # But if *all* the predicates were rewritten... well, do_validate() is
            # an in-place method so we still just re-initialize the strategy!
            FilteredStrategy.__init__(self, fresh, ())

    def filter(self, condition):
        # If we can, it's more efficient to rewrite our strategy to satisfy the
        # condition.  We therefore exploit the fact that the order of predicates
        # doesn't matter (`f(x) and g(x) == g(x) and f(x)`) by attempting to apply
        # condition directly to our filtered strategy as the inner-most filter.
        out = self.filtered_strategy.filter(condition)
        # If it couldn't be rewritten, we'll get a new FilteredStrategy - and then
        # combine the conditions of each in our expected newest=last order.
        if isinstance(out, FilteredStrategy):
            return FilteredStrategy(
                out.filtered_strategy, self.flat_conditions + out.flat_conditions
            )
        # But if it *could* be rewritten, we can return the more efficient form!
        return FilteredStrategy(out, self.flat_conditions)

    @property
    def condition(self):
        if self.__condition is None:
            if len(self.flat_conditions) == 1:
                # Avoid an extra indirection in the common case of only one condition.
                self.__condition = self.flat_conditions[0]
            elif len(self.flat_conditions) == 0:
                # Possible, if unlikely, due to filter predicate rewriting
                self.__condition = lambda _: True
            else:
                self.__condition = lambda x: all(
                    cond(x) for cond in self.flat_conditions
                )
        return self.__condition

    def do_draw(self, data: ConjectureData) -> Ex:
        result = self.do_filtered_draw(data)
        if result is not filter_not_satisfied:
            return result

        data.mark_invalid(f"Aborted test because unable to satisfy {self!r}")
        raise NotImplementedError("Unreachable, for Mypy")

    def do_filtered_draw(self, data):
        for i in range(3):
            data.start_example(FILTERED_SEARCH_STRATEGY_DO_DRAW_LABEL)
            value = data.draw(self.filtered_strategy)
            if self.condition(value):
                data.stop_example()
                return value
            else:
                data.stop_example(discard=True)
                if i == 0:
                    data.events[f"Retried draw from {self!r} to satisfy filter"] = ""

        return filter_not_satisfied

    @property
    def branches(self) -> list[SearchStrategy[Ex]]:
        return [
            FilteredStrategy(strategy=strategy, conditions=self.flat_conditions)
            for strategy in self.filtered_strategy.branches
        ]


@check_function
def check_strategy(arg, name=""):
    assert isinstance(name, str)
    if not isinstance(arg, SearchStrategy):
        hint = ""
        if isinstance(arg, (list, tuple)):
            hint = ", such as st.sampled_from({}),".format(name or "...")
        if name:
            name += "="
        raise InvalidArgument(
            "Expected a SearchStrategy%s but got %s%r (type=%s)"
            % (hint, name, arg, type(arg).__name__)
        )
