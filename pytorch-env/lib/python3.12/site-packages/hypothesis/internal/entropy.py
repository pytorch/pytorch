# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import contextlib
import gc
import random
import sys
import warnings
from collections.abc import Hashable
from itertools import count
from typing import TYPE_CHECKING, Any, Callable
from weakref import WeakValueDictionary

import hypothesis.core
from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.internal.compat import FREE_THREADED_CPYTHON, GRAALPY, PYPY

if TYPE_CHECKING:
    from typing import Protocol

    # we can't use this at runtime until from_type supports
    # protocols -- breaks ghostwriter tests
    class RandomLike(Protocol):
        seed: Callable[..., Any]
        getstate: Callable[[], Any]
        setstate: Callable[..., Any]

else:  # pragma: no cover
    RandomLike = random.Random

# This is effectively a WeakSet, which allows us to associate the saved states
# with their respective Random instances even as new ones are registered and old
# ones go out of scope and get garbage collected.  Keys are ascending integers.
_RKEY = count()
RANDOMS_TO_MANAGE: WeakValueDictionary = WeakValueDictionary({next(_RKEY): random})


class NumpyRandomWrapper:
    def __init__(self):
        assert "numpy" in sys.modules
        # This class provides a shim that matches the numpy to stdlib random,
        # and lets us avoid importing Numpy until it's already in use.
        import numpy.random

        self.seed = numpy.random.seed
        self.getstate = numpy.random.get_state
        self.setstate = numpy.random.set_state


NP_RANDOM = None


if not (PYPY or GRAALPY):

    def _get_platform_base_refcount(r: Any) -> int:
        return sys.getrefcount(r)

    # Determine the number of refcounts created by function scope for
    # the given platform / version of Python.
    _PLATFORM_REF_COUNT = _get_platform_base_refcount(object())
else:  # pragma: no cover
    # PYPY and GRAALPY don't have `sys.getrefcount`
    _PLATFORM_REF_COUNT = -1


def register_random(r: RandomLike) -> None:
    """Register (a weakref to) the given Random-like instance for management by
    Hypothesis.

    You can pass instances of structural subtypes of ``random.Random``
    (i.e., objects with seed, getstate, and setstate methods) to
    ``register_random(r)`` to have their states seeded and restored in the same
    way as the global PRNGs from the ``random`` and ``numpy.random`` modules.

    All global PRNGs, from e.g. simulation or scheduling frameworks, should
    be registered to prevent flaky tests. Hypothesis will ensure that the
    PRNG state is consistent for all test runs, always seeding them to zero and
    restoring the previous state after the test, or, reproducibly varied if you
    choose to use the :func:`~hypothesis.strategies.random_module` strategy.

    ``register_random`` only makes `weakrefs
    <https://docs.python.org/3/library/weakref.html#module-weakref>`_ to ``r``,
    thus ``r`` will only be managed by Hypothesis as long as it has active
    references elsewhere at runtime. The pattern ``register_random(MyRandom())``
    will raise a ``ReferenceError`` to help protect users from this issue.
    This check does not occur for the PyPy interpreter. See the following example for
    an illustration of this issue

    .. code-block:: python


       def my_BROKEN_hook():
           r = MyRandomLike()

           # `r` will be garbage collected after the hook resolved
           # and Hypothesis will 'forget' that it was registered
           register_random(r)  # Hypothesis will emit a warning


       rng = MyRandomLike()


       def my_WORKING_hook():
           register_random(rng)
    """
    if not (hasattr(r, "seed") and hasattr(r, "getstate") and hasattr(r, "setstate")):
        raise InvalidArgument(f"{r=} does not have all the required methods")

    if r in RANDOMS_TO_MANAGE.values():
        return

    if not (PYPY or GRAALPY):  # pragma: no branch
        # PYPY and GRAALPY do not have `sys.getrefcount`.
        gc.collect()
        if not gc.get_referrers(r):
            if sys.getrefcount(r) <= _PLATFORM_REF_COUNT:
                raise ReferenceError(
                    f"`register_random` was passed `r={r}` which will be "
                    "garbage collected immediately after `register_random` creates a "
                    "weakref to it. This will prevent Hypothesis from managing this "
                    "PRNG. See the docs for `register_random` for more "
                    "details."
                )
            elif not FREE_THREADED_CPYTHON:  # pragma: no branch
                # On CPython, check for the free-threaded build because
                # gc.get_referrers() ignores objects with immortal refcounts
                # and objects are immortalized in the Python 3.13
                # free-threading implementation at runtime.

                warnings.warn(
                    "It looks like `register_random` was passed an object that could "
                    "be garbage collected immediately after `register_random` creates "
                    "a weakref to it. This will prevent Hypothesis from managing this "
                    "PRNG. See the docs for `register_random` for more details.",
                    HypothesisWarning,
                    stacklevel=2,
                )

    RANDOMS_TO_MANAGE[next(_RKEY)] = r


def get_seeder_and_restorer(
    seed: Hashable = 0,
) -> tuple[Callable[[], None], Callable[[], None]]:
    """Return a pair of functions which respectively seed all and restore
    the state of all registered PRNGs.

    This is used by the core engine via `deterministic_PRNG`, and by users
    via `register_random`.  We support registration of additional random.Random
    instances (or other objects with seed, getstate, and setstate methods)
    to force determinism on simulation or scheduling frameworks which avoid
    using the global random state.  See e.g. #1709.
    """
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32
    states: dict = {}

    if "numpy" in sys.modules:
        global NP_RANDOM
        if NP_RANDOM is None:
            # Protect this from garbage-collection by adding it to global scope
            NP_RANDOM = RANDOMS_TO_MANAGE[next(_RKEY)] = NumpyRandomWrapper()

    def seed_all():
        assert not states
        for k, r in RANDOMS_TO_MANAGE.items():
            states[k] = r.getstate()
            r.seed(seed)

    def restore_all():
        for k, state in states.items():
            r = RANDOMS_TO_MANAGE.get(k)
            if r is not None:  # i.e., hasn't been garbage-collected
                r.setstate(state)
        states.clear()

    return seed_all, restore_all


@contextlib.contextmanager
def deterministic_PRNG(seed=0):
    """Context manager that handles random.seed without polluting global state.

    See issue #1255 and PR #1295 for details and motivation - in short,
    leaving the global pseudo-random number generator (PRNG) seeded is a very
    bad idea in principle, and breaks all kinds of independence assumptions
    in practice.
    """
    if hypothesis.core._hypothesis_global_random is None:  # pragma: no cover
        hypothesis.core._hypothesis_global_random = random.Random()
        register_random(hypothesis.core._hypothesis_global_random)

    seed_all, restore_all = get_seeder_and_restorer(seed)
    seed_all()
    try:
        yield
    finally:
        restore_all()
