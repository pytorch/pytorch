# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import abc
import inspect
import math
from dataclasses import dataclass, field
from random import Random
from typing import Any

from hypothesis.control import should_note
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.reflection import define_function_signature
from hypothesis.reporting import report
from hypothesis.strategies._internal.core import lists, permutations, sampled_from
from hypothesis.strategies._internal.numbers import floats, integers
from hypothesis.strategies._internal.strategies import SearchStrategy


class HypothesisRandom(Random, abc.ABC):
    """A subclass of Random designed to expose the seed it was initially
    provided with."""

    def __init__(self, *, note_method_calls: bool) -> None:
        self._note_method_calls = note_method_calls

    def __deepcopy__(self, table):
        return self.__copy__()

    @abc.abstractmethod
    def seed(self, seed):
        raise NotImplementedError

    @abc.abstractmethod
    def getstate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def setstate(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def _hypothesis_do_random(self, method, kwargs):
        raise NotImplementedError

    def _hypothesis_log_random(self, method, kwargs, result):
        if not (self._note_method_calls and should_note()):
            return

        args, kwargs = convert_kwargs(method, kwargs)
        argstr = ", ".join(
            list(map(repr, args)) + [f"{k}={v!r}" for k, v in kwargs.items()]
        )
        report(f"{self!r}.{method}({argstr}) -> {result!r}")


RANDOM_METHODS = [
    name
    for name in [
        "_randbelow",
        "betavariate",
        "binomialvariate",
        "choice",
        "choices",
        "expovariate",
        "gammavariate",
        "gauss",
        "getrandbits",
        "lognormvariate",
        "normalvariate",
        "paretovariate",
        "randint",
        "random",
        "randrange",
        "sample",
        "shuffle",
        "triangular",
        "uniform",
        "vonmisesvariate",
        "weibullvariate",
        "randbytes",
    ]
    if hasattr(Random, name)
]


# Fake shims to get a good signature
def getrandbits(self, n: int) -> int:  # type: ignore
    raise NotImplementedError


def random(self) -> float:  # type: ignore
    raise NotImplementedError


def _randbelow(self, n: int) -> int:  # type: ignore
    raise NotImplementedError


STUBS = {f.__name__: f for f in [getrandbits, random, _randbelow]}


SIGNATURES: dict[str, inspect.Signature] = {}


def sig_of(name):
    try:
        return SIGNATURES[name]
    except KeyError:
        pass

    target = getattr(Random, name)
    result = inspect.signature(STUBS.get(name, target))
    SIGNATURES[name] = result
    return result


def define_copy_method(name):
    target = getattr(Random, name)

    def implementation(self, **kwargs):
        result = self._hypothesis_do_random(name, kwargs)
        self._hypothesis_log_random(name, kwargs, result)
        return result

    sig = inspect.signature(STUBS.get(name, target))

    result = define_function_signature(target.__name__, target.__doc__, sig)(
        implementation
    )

    result.__module__ = __name__
    result.__qualname__ = "HypothesisRandom." + result.__name__

    setattr(HypothesisRandom, name, result)


for r in RANDOM_METHODS:
    define_copy_method(r)


@dataclass(slots=True, frozen=False)
class RandomState:
    next_states: dict = field(default_factory=dict)
    state_id: Any = None


def state_for_seed(data, seed):
    if data.seeds_to_states is None:
        data.seeds_to_states = {}

    seeds_to_states = data.seeds_to_states
    try:
        state = seeds_to_states[seed]
    except KeyError:
        state = RandomState()
        seeds_to_states[seed] = state

    return state


def normalize_zero(f: float) -> float:
    if f == 0.0:
        return 0.0
    else:
        return f


class ArtificialRandom(HypothesisRandom):
    VERSION = 10**6

    def __init__(self, *, note_method_calls: bool, data: ConjectureData) -> None:
        super().__init__(note_method_calls=note_method_calls)
        self.__data = data
        self.__state = RandomState()

    def __repr__(self) -> str:
        return "HypothesisRandom(generated data)"

    def __copy__(self) -> "ArtificialRandom":
        result = ArtificialRandom(
            note_method_calls=self._note_method_calls,
            data=self.__data,
        )
        result.setstate(self.getstate())
        return result

    def __convert_result(self, method, kwargs, result):
        if method == "choice":
            return kwargs.get("seq")[result]
        if method in ("choices", "sample"):
            seq = kwargs["population"]
            return [seq[i] for i in result]
        if method == "shuffle":
            seq = kwargs["x"]
            original = list(seq)
            for i, i2 in enumerate(result):
                seq[i] = original[i2]
            return None
        return result

    def _hypothesis_do_random(self, method, kwargs):
        if method == "choices":
            key = (method, len(kwargs["population"]), kwargs.get("k"))
        elif method == "choice":
            key = (method, len(kwargs["seq"]))
        elif method == "shuffle":
            key = (method, len(kwargs["x"]))
        else:
            key = (method, *sorted(kwargs))

        try:
            result, self.__state = self.__state.next_states[key]
        except KeyError:
            pass
        else:
            return self.__convert_result(method, kwargs, result)

        if method == "_randbelow":
            result = self.__data.draw_integer(0, kwargs["n"] - 1)
        elif method == "random":
            # See https://github.com/HypothesisWorks/hypothesis/issues/4297
            # for numerics/bounds of "random" and "betavariate"
            result = self.__data.draw(floats(0, 1, exclude_max=True))
        elif method == "betavariate":
            result = self.__data.draw(floats(0, 1))
        elif method == "uniform":
            a = normalize_zero(kwargs["a"])
            b = normalize_zero(kwargs["b"])
            result = self.__data.draw(floats(a, b))
        elif method in ("weibullvariate", "gammavariate"):
            result = self.__data.draw(floats(min_value=0.0, allow_infinity=False))
        elif method in ("gauss", "normalvariate"):
            mu = kwargs["mu"]
            result = mu + self.__data.draw(
                floats(allow_nan=False, allow_infinity=False)
            )
        elif method == "vonmisesvariate":
            result = self.__data.draw(floats(0, 2 * math.pi))
        elif method == "randrange":
            if kwargs["stop"] is None:
                stop = kwargs["start"]
                start = 0
            else:
                start = kwargs["start"]
                stop = kwargs["stop"]

            step = kwargs["step"]
            if start == stop:
                raise ValueError(f"empty range for randrange({start}, {stop}, {step})")

            if step != 1:
                endpoint = (stop - start) // step
                if (start - stop) % step == 0:
                    endpoint -= 1

                i = self.__data.draw_integer(0, endpoint)
                result = start + i * step
            else:
                result = self.__data.draw_integer(start, stop - 1)
        elif method == "randint":
            result = self.__data.draw_integer(kwargs["a"], kwargs["b"])
        # New in Python 3.12, so not taken by our coverage job
        elif method == "binomialvariate":  # pragma: no cover
            result = self.__data.draw_integer(0, kwargs["n"])
        elif method == "choice":
            seq = kwargs["seq"]
            result = self.__data.draw_integer(0, len(seq) - 1)
        elif method == "choices":
            k = kwargs["k"]
            result = self.__data.draw(
                lists(
                    integers(0, len(kwargs["population"]) - 1),
                    min_size=k,
                    max_size=k,
                )
            )
        elif method == "sample":
            k = kwargs["k"]
            seq = kwargs["population"]

            if k > len(seq) or k < 0:
                raise ValueError(
                    f"Sample size {k} not in expected range 0 <= k <= {len(seq)}"
                )

            if k == 0:
                result = []
            else:
                result = self.__data.draw(
                    lists(
                        sampled_from(range(len(seq))),
                        min_size=k,
                        max_size=k,
                        unique=True,
                    )
                )

        elif method == "getrandbits":
            result = self.__data.draw_integer(0, 2 ** kwargs["n"] - 1)
        elif method == "triangular":
            low = normalize_zero(kwargs["low"])
            high = normalize_zero(kwargs["high"])
            mode = normalize_zero(kwargs["mode"])
            if mode is None:
                result = self.__data.draw(floats(low, high))
            elif self.__data.draw_boolean(0.5):
                result = self.__data.draw(floats(mode, high))
            else:
                result = self.__data.draw(floats(low, mode))
        elif method in ("paretovariate", "expovariate", "lognormvariate"):
            result = self.__data.draw(floats(min_value=0.0))
        elif method == "shuffle":
            result = self.__data.draw(permutations(range(len(kwargs["x"]))))
        elif method == "randbytes":
            n = int(kwargs["n"])
            result = self.__data.draw_bytes(min_size=n, max_size=n)
        else:
            raise NotImplementedError(method)

        new_state = RandomState()
        self.__state.next_states[key] = (result, new_state)
        self.__state = new_state

        return self.__convert_result(method, kwargs, result)

    def seed(self, seed):
        self.__state = state_for_seed(self.__data, seed)

    def getstate(self):
        if self.__state.state_id is not None:
            return self.__state.state_id

        if self.__data.states_for_ids is None:
            self.__data.states_for_ids = {}
        states_for_ids = self.__data.states_for_ids
        self.__state.state_id = len(states_for_ids)
        states_for_ids[self.__state.state_id] = self.__state

        return self.__state.state_id

    def setstate(self, state):
        self.__state = self.__data.states_for_ids[state]


DUMMY_RANDOM = Random(0)


def convert_kwargs(name, kwargs):
    kwargs = dict(kwargs)

    signature = sig_of(name)
    params = signature.parameters

    bound = signature.bind(DUMMY_RANDOM, **kwargs)
    bound.apply_defaults()

    for k in list(kwargs):
        if (
            kwargs[k] is params[k].default
            or params[k].kind != inspect.Parameter.KEYWORD_ONLY
        ):
            kwargs.pop(k)

    arg_names = list(params)[1:]

    args = []

    for a in arg_names:
        if params[a].kind == inspect.Parameter.KEYWORD_ONLY:
            break
        args.append(bound.arguments[a])
        kwargs.pop(a, None)

    while args:
        name = arg_names[len(args) - 1]
        if args[-1] is params[name].default:
            args.pop()
        else:
            break

    return (args, kwargs)


class TrueRandom(HypothesisRandom):
    def __init__(self, seed, note_method_calls):
        super().__init__(note_method_calls=note_method_calls)
        self.__seed = seed
        self.__random = Random(seed)

    def _hypothesis_do_random(self, method, kwargs):
        fn = getattr(self.__random, method)
        try:
            return fn(**kwargs)
        except TypeError:
            pass
        args, kwargs = convert_kwargs(method, kwargs)
        return fn(*args, **kwargs)

    def __copy__(self) -> "TrueRandom":
        result = TrueRandom(
            seed=self.__seed,
            note_method_calls=self._note_method_calls,
        )
        result.setstate(self.getstate())
        return result

    def __repr__(self) -> str:
        return f"Random({self.__seed!r})"

    def seed(self, seed):
        self.__random.seed(seed)
        self.__seed = seed

    def getstate(self):
        return self.__random.getstate()

    def setstate(self, state):
        self.__random.setstate(state)


class RandomStrategy(SearchStrategy[HypothesisRandom]):
    def __init__(self, *, note_method_calls: bool, use_true_random: bool) -> None:
        super().__init__()
        self.__note_method_calls = note_method_calls
        self.__use_true_random = use_true_random

    def do_draw(self, data: ConjectureData) -> HypothesisRandom:
        if self.__use_true_random:
            seed = data.draw_integer(0, 2**64 - 1)
            return TrueRandom(seed=seed, note_method_calls=self.__note_method_calls)
        else:
            return ArtificialRandom(
                note_method_calls=self.__note_method_calls, data=data
            )
