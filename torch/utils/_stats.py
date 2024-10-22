# mypy: allow-untyped-defs
# NOTE! PLEASE KEEP THIS FILE *FREE* OF TORCH DEPS! IT SHOULD BE IMPORTABLE ANYWHERE.
# IF YOU FEEL AN OVERWHELMING URGE TO ADD A TORCH DEP, MAKE A TRAMPOLINE FILE A LA torch._dynamo.utils
# AND SCRUB AWAY TORCH NOTIONS THERE.
import collections
import functools
from typing import OrderedDict

simple_call_counter: OrderedDict[str, int] = collections.OrderedDict()

def count_label(label):
    prev = simple_call_counter.setdefault(label, 0)
    simple_call_counter[label] = prev + 1

def count(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if fn.__qualname__ not in simple_call_counter:
            simple_call_counter[fn.__qualname__] = 0
        simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
        return fn(*args, **kwargs)
    return wrapper
