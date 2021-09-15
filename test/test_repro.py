import torch
import torch.nn as nn
import asyncio
import functools
import re
import sys
import textwrap
import traceback
import types
import unittest.mock
import warnings
from asyncio.events import AbstractEventLoop
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Set, Tuple, TypeVar

import testslide

"""
Repro instructions:
1. `pip install testslide`
2. `cd test` and `testslide test_repro.py`
"""

# NB: the internal code comment on async_test might be suspicious.
# (see https://fburl.com/diffusion/ebwww689).
# Maybe the bug happens as a result of a user error?
def async_test(func):
    if not asyncio.iscoroutinefunction(func):
        raise TypeError("Only 'async def' is supported, please fix your call site")

    @functools.wraps(func)
    def wrapper(*args: Any, **kws: Any) -> None:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(func(*args, **kws))

    return wrapper

class TestBlah(testslide.TestCase):
    # async_test is required for this repro
    @async_test
    async def test_one_frame(self):
        # net doesn't even get used! But somehow it is required for the repro
        net = nn.Linear(2, 1)

        def foo():
            # Create a nn.Parameter, then del it. This makes it so that
            # we flip the ownership bit on the Tensor.
            x = torch.nn.Parameter(torch.randn(3))
            y = x * x
            return y

        y = foo()
