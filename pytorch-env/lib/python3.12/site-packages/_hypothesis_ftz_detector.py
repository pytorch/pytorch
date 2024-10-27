# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
This is a toolkit for determining which module set the "flush to zero" flag.

For details, see the docstring and comments in `identify_ftz_culprit()`.  This module
is defined outside the main Hypothesis namespace so that we can avoid triggering
import of Hypothesis itself from each subprocess which must import the worker function.
"""

import importlib
import sys
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from multiprocessing import Queue
    from typing import TypeAlias

FTZCulprits: "TypeAlias" = tuple[Optional[bool], set[str]]


KNOWN_EVER_CULPRITS = (
    # https://moyix.blogspot.com/2022/09/someones-been-messing-with-my-subnormals.html
    # fmt: off
    "archive-pdf-tools", "bgfx-python", "bicleaner-ai-glove", "BTrees", "cadbiom",
    "ctranslate2", "dyNET", "dyNET38", "gevent", "glove-python-binary", "higra",
    "hybridq", "ikomia", "ioh", "jij-cimod", "lavavu", "lavavu-osmesa", "MulticoreTSNE",
    "neural-compressor", "nwhy", "openjij", "openturns", "perfmetrics", "pHashPy",
    "pyace-lite", "pyapr", "pycompadre", "pycompadre-serial", "PyKEP", "pykep",
    "pylimer-tools", "pyqubo", "pyscf", "PyTAT", "python-prtree", "qiskit-aer",
    "qiskit-aer-gpu", "RelStorage", "sail-ml", "segmentation", "sente", "sinr",
    "snapml", "superman", "symengine", "systran-align", "texture-tool", "tsne-mp",
    "xcsf",
    # fmt: on
)


def flush_to_zero() -> bool:
    # If this subnormal number compares equal to zero we have a problem
    return 2.0**-1073 == 0


def run_in_process(fn: Callable[..., FTZCulprits], *args: object) -> FTZCulprits:
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    q: "Queue[FTZCulprits]" = mp.Queue()
    p = mp.Process(target=target, args=(q, fn, *args))
    p.start()
    retval = q.get()
    p.join()
    return retval


def target(
    q: "Queue[FTZCulprits]", fn: Callable[..., FTZCulprits], *args: object
) -> None:
    q.put(fn(*args))


def always_imported_modules() -> FTZCulprits:
    return flush_to_zero(), set(sys.modules)


def modules_imported_by(mod: str) -> FTZCulprits:
    """Return the set of modules imported transitively by mod."""
    before = set(sys.modules)
    try:
        importlib.import_module(mod)
    except Exception:
        return None, set()
    imports = set(sys.modules) - before
    return flush_to_zero(), imports


# We don't want to redo all the expensive process-spawning checks when we've already
# done them, so we cache known-good packages and a known-FTZ result if we have one.
KNOWN_FTZ = None
CHECKED_CACHE = set()


def identify_ftz_culprits() -> str:
    """Find the modules in sys.modules which cause "mod" to be imported."""
    # If we've run this function before, return the same result.
    global KNOWN_FTZ
    if KNOWN_FTZ:
        return KNOWN_FTZ
    # Start by determining our baseline: the FTZ and sys.modules state in a fresh
    # process which has only imported this module and nothing else.
    always_enables_ftz, always_imports = run_in_process(always_imported_modules)
    if always_enables_ftz:
        raise RuntimeError("Python is always in FTZ mode, even without imports!")
    CHECKED_CACHE.update(always_imports)

    # Next, we'll search through sys.modules looking for a package (or packages) such
    # that importing them in a new process sets the FTZ state.  As a heuristic, we'll
    # start with packages known to have ever enabled FTZ, then top-level packages as
    # a way to eliminate large fractions of the search space relatively quickly.
    def key(name: str) -> tuple[bool, int, str]:
        """Prefer known-FTZ modules, then top-level packages, then alphabetical."""
        return (name not in KNOWN_EVER_CULPRITS, name.count("."), name)

    # We'll track the set of modules to be checked, and those which do trigger FTZ.
    candidates = set(sys.modules) - CHECKED_CACHE
    triggering_modules = {}
    while candidates:
        mod = min(candidates, key=key)
        candidates.discard(mod)
        enables_ftz, imports = run_in_process(modules_imported_by, mod)
        imports -= CHECKED_CACHE
        if enables_ftz:
            triggering_modules[mod] = imports
            candidates &= imports
        else:
            candidates -= imports
            CHECKED_CACHE.update(imports)

    # We only want to report the 'top level' packages which enable FTZ - for example,
    # if the enabling code is in `a.b`, and `a` in turn imports `a.b`, we prefer to
    # report `a`.  On the other hand, if `a` does _not_ import `a.b`, as is the case
    # for `hypothesis.extra.*` modules, then `a` will not be in `triggering_modules`
    # and we'll report `a.b` here instead.
    prefixes = tuple(n + "." for n in triggering_modules)
    result = {k for k in triggering_modules if not k.startswith(prefixes)}

    # Suppose that `bar` enables FTZ, and `foo` imports `bar`.  At this point we're
    # tracking both, but only want to report the latter.
    for a in sorted(result):
        for b in sorted(result):
            if a in triggering_modules[b] and b not in triggering_modules[a]:
                result.discard(b)

    # There may be a cyclic dependency which that didn't handle, or simply two
    # separate modules which both enable FTZ.  We already gave up comprehensive
    # reporting for speed above (`candidates &= imports`), so we'll also buy
    # simpler reporting by arbitrarily selecting the alphabetically first package.
    KNOWN_FTZ = min(result)  # Cache the result - it's likely this will trigger again!
    return KNOWN_FTZ


if __name__ == "__main__":
    # This would be really really annoying to write automated tests for, so I've
    # done some manual exploratory testing: `pip install grequests gevent==21.12.0`,
    # and call print() as desired to observe behavior.
    import grequests  # noqa

    # To test without skipping to a known answer, uncomment the following line and
    # change the last element of key from `name` to `-len(name)` so that we check
    # grequests before gevent.
    # KNOWN_EVER_CULPRITS = [c for c in KNOWN_EVER_CULPRITS if c != "gevent"]
    print(identify_ftz_culprits())
