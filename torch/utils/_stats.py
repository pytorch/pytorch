# NOTE! PLEASE KEEP THIS FILE *FREE* OF TORCH DEPS! IT SHOULD BE IMPORTABLE ANYWHERE.
# IF YOU FEEL AN OVERWHELMING URGE TO ADD A TORCH DEP, MAKE A TRAMPOLINE FILE A LA torch._dynamo.utils
# AND SCRUB AWAY TORCH NOTIONS THERE.
import collections
import functools
from collections import OrderedDict
from typing import Callable, TypeVar
from typing_extensions import ParamSpec


simple_call_counter: OrderedDict[str, int] = collections.OrderedDict()

_P = ParamSpec("_P")
_R = TypeVar("_R")


def count_label(label: str) -> None:
    prev = simple_call_counter.setdefault(label, 0)
    simple_call_counter[label] = prev + 1


def count(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if fn.__qualname__ not in simple_call_counter:
            simple_call_counter[fn.__qualname__] = 0
        simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
        return fn(*args, **kwargs)

    return wrapper
