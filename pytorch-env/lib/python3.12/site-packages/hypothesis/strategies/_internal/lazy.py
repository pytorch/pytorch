# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from collections.abc import MutableMapping
from inspect import signature
from weakref import WeakKeyDictionary

from hypothesis.configuration import check_sideeffect_during_initialization
from hypothesis.internal.reflection import (
    convert_keyword_arguments,
    convert_positional_arguments,
    get_pretty_function_description,
    repr_call,
)
from hypothesis.strategies._internal.strategies import SearchStrategy

unwrap_cache: MutableMapping[SearchStrategy, SearchStrategy] = WeakKeyDictionary()
unwrap_depth = 0


def unwrap_strategies(s):
    global unwrap_depth

    if not isinstance(s, SearchStrategy):
        return s
    try:
        return unwrap_cache[s]
    except KeyError:
        pass

    unwrap_cache[s] = s

    try:
        unwrap_depth += 1
        try:
            result = unwrap_strategies(s.wrapped_strategy)
            unwrap_cache[s] = result
            try:
                assert result.force_has_reusable_values == s.force_has_reusable_values
            except AttributeError:
                pass

            try:
                result.force_has_reusable_values = s.force_has_reusable_values
            except AttributeError:
                pass
            return result
        except AttributeError:
            return s
    finally:
        unwrap_depth -= 1
        if unwrap_depth <= 0:
            unwrap_cache.clear()
        assert unwrap_depth >= 0


class LazyStrategy(SearchStrategy):
    """A strategy which is defined purely by conversion to and from another
    strategy.

    Its parameter and distribution come from that other strategy.
    """

    def __init__(self, function, args, kwargs, *, transforms=(), force_repr=None):
        super().__init__()
        self.__wrapped_strategy = None
        self.__representation = force_repr
        self.function = function
        self.__args = args
        self.__kwargs = kwargs
        self._transformations = transforms

    @property
    def supports_find(self):
        return self.wrapped_strategy.supports_find

    def calc_is_empty(self, recur):
        return recur(self.wrapped_strategy)

    def calc_has_reusable_values(self, recur):
        return recur(self.wrapped_strategy)

    def calc_is_cacheable(self, recur):
        for source in (self.__args, self.__kwargs.values()):
            for v in source:
                if isinstance(v, SearchStrategy) and not v.is_cacheable:
                    return False
        return True

    @property
    def wrapped_strategy(self):
        if self.__wrapped_strategy is None:
            check_sideeffect_during_initialization("lazy evaluation of {!r}", self)

            unwrapped_args = tuple(unwrap_strategies(s) for s in self.__args)
            unwrapped_kwargs = {
                k: unwrap_strategies(v) for k, v in self.__kwargs.items()
            }

            base = self.function(*self.__args, **self.__kwargs)
            if unwrapped_args == self.__args and unwrapped_kwargs == self.__kwargs:
                self.__wrapped_strategy = base
            else:
                self.__wrapped_strategy = self.function(
                    *unwrapped_args, **unwrapped_kwargs
                )
            for method, fn in self._transformations:
                self.__wrapped_strategy = getattr(self.__wrapped_strategy, method)(fn)
        return self.__wrapped_strategy

    def __with_transform(self, method, fn):
        repr_ = self.__representation
        if repr_:
            repr_ = f"{repr_}.{method}({get_pretty_function_description(fn)})"
        return type(self)(
            self.function,
            self.__args,
            self.__kwargs,
            transforms=(*self._transformations, (method, fn)),
            force_repr=repr_,
        )

    def map(self, pack):
        return self.__with_transform("map", pack)

    def filter(self, condition):
        return self.__with_transform("filter", condition)

    def do_validate(self):
        w = self.wrapped_strategy
        assert isinstance(w, SearchStrategy), f"{self!r} returned non-strategy {w!r}"
        w.validate()

    def __repr__(self):
        if self.__representation is None:
            sig = signature(self.function)
            pos = [p for p in sig.parameters.values() if "POSITIONAL" in p.kind.name]
            if len(pos) > 1 or any(p.default is not sig.empty for p in pos):
                _args, _kwargs = convert_positional_arguments(
                    self.function, self.__args, self.__kwargs
                )
            else:
                _args, _kwargs = convert_keyword_arguments(
                    self.function, self.__args, self.__kwargs
                )
            kwargs_for_repr = {
                k: v
                for k, v in _kwargs.items()
                if k not in sig.parameters or v is not sig.parameters[k].default
            }
            self.__representation = repr_call(
                self.function, _args, kwargs_for_repr, reorder=False
            ) + "".join(
                f".{method}({get_pretty_function_description(fn)})"
                for method, fn in self._transformations
            )
        return self.__representation

    def do_draw(self, data):
        return data.draw(self.wrapped_strategy)

    @property
    def label(self):
        return self.wrapped_strategy.label
