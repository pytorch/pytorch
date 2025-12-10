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
from collections.abc import Callable, Generator, Hashable
from itertools import count
from random import Random
from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary

import hypothesis.core
from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.internal.compat import FREE_THREADED_CPYTHON, GRAALPY, PYPY

if TYPE_CHECKING:
    from typing import Protocol

    # we can't use this at runtime until from_type supports
    # protocols -- breaks ghostwriter tests
    class RandomLike(Protocol):
        def seed(self, *args: Any, **kwargs: Any) -> Any: ...
        def getstate(self, *args: Any, **kwargs: Any) -> Any: ...
        def setstate(self, *args: Any, **kwargs: Any) -> Any: ...

else:  # pragma: no cover
    RandomLike = random.Random

_RKEY = count()
_global_random_rkey = next(_RKEY)
# This is effectively a WeakSet, which allows us to associate the saved states
# with their respective Random instances even as new ones are registered and old
# ones go out of scope and get garbage collected.  Keys are ascending integers.
RANDOMS_TO_MANAGE: WeakValueDictionary[int, RandomLike] = WeakValueDictionary(
    {_global_random_rkey: random}
)


class NumpyRandomWrapper:
    def __init__(self) -> None:
        assert "numpy" in sys.modules
        # This class provides a shim that matches the numpy to stdlib random,
        # and lets us avoid importing Numpy until it's already in use.
        import numpy.random

        self.seed = numpy.random.seed
        self.getstate = numpy.random.get_state
        self.setstate = numpy.random.set_state


NP_RANDOM: RandomLike | None = None


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

    if r in [
        random
        for ref in RANDOMS_TO_MANAGE.data.copy().values()  # type: ignore
        if (random := ref()) is not None
    ]:
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


# Used to make the warning issued by `deprecate_random_in_strategy` thread-safe,
# as well as to avoid warning on uses of st.randoms().
# Store just the hash to reduce memory consumption. This is an underapproximation
# of membership (distinct items might have the same hash), which is fine for the
# warning, as it results in missed alarms, not false alarms.
_known_random_state_hashes: set[Any] = set()


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
    states: dict[int, object] = {}

    if "numpy" in sys.modules:
        global NP_RANDOM
        if NP_RANDOM is None:
            # Protect this from garbage-collection by adding it to global scope
            NP_RANDOM = RANDOMS_TO_MANAGE[next(_RKEY)] = NumpyRandomWrapper()

    def seed_all() -> None:
        assert not states
        # access .data.copy().items() instead of .items() to avoid a "dictionary
        # changed size during iteration" error under multithreading.
        #
        # I initially expected this to be fixed by
        # https://github.com/python/cpython/commit/96d37dbcd23e65a7a57819aeced9034296ef747e,
        # but I believe that is addressing the size change from weakrefs expiring
        # during gc, not from the user adding new elements to the dict.
        #
        # Since we're accessing .data, we have to manually handle checking for
        # expired ref instances during iteration. Normally WeakValueDictionary
        # handles this for us.
        #
        # This command reproduces at time of writing:
        #   pytest hypothesis-python/tests/ -k test_intervals_are_equivalent_to_their_lists
        #   --parallel-threads 2
        for k, ref in RANDOMS_TO_MANAGE.data.copy().items():  # type: ignore
            r = ref()
            if r is None:
                # ie the random instance has been gc'd
                continue  # pragma: no cover
            states[k] = r.getstate()
            if k == _global_random_rkey:
                # r.seed sets the random's state. We want to add that state to
                # _known_random_states before calling r.seed, in case a thread
                # switch occurs between the two. To figure out the seed -> state
                # mapping, set the seed on a dummy random and add that state to
                # _known_random_state.
                #
                # we could use a global dummy random here, but then we'd have to
                # put a lock around it, and it's not clear to me if that's more
                # efficient than constructing a new instance each time.
                dummy_random = Random()
                dummy_random.seed(seed)
                _known_random_state_hashes.add(hash(dummy_random.getstate()))
                # we expect `assert r.getstate() == dummy_random.getstate()` to
                # hold here, but thread switches means it might not.

            r.seed(seed)

    def restore_all() -> None:
        for k, state in states.items():
            r = RANDOMS_TO_MANAGE.get(k)
            if r is None:  # i.e., has been garbage-collected
                continue

            if k == _global_random_rkey:
                _known_random_state_hashes.add(hash(state))
            r.setstate(state)

        states.clear()

    return seed_all, restore_all


@contextlib.contextmanager
def deterministic_PRNG(seed: int = 0) -> Generator[None, None, None]:
    """Context manager that handles random.seed without polluting global state.

    See issue #1255 and PR #1295 for details and motivation - in short,
    leaving the global pseudo-random number generator (PRNG) seeded is a very
    bad idea in principle, and breaks all kinds of independence assumptions
    in practice.
    """
    if (
        hypothesis.core.threadlocal._hypothesis_global_random is None
    ):  # pragma: no cover
        hypothesis.core.threadlocal._hypothesis_global_random = Random()
        register_random(hypothesis.core.threadlocal._hypothesis_global_random)

    seed_all, restore_all = get_seeder_and_restorer(seed)
    seed_all()
    try:
        yield
    finally:
        restore_all()
        # TODO it would be nice to clean up _known_random_state_hashes when no
        # active deterministic_PRNG contexts remain, to free memory (see similar
        # logic in StackframeLimiter). But it's a bit annoying to get right, and
        # likely not a big deal.
