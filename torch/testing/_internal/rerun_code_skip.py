# mypy: allow-untyped-defs
"""Known-failure skips that rerun-disabled-tests runs instead of skipping.

Only tests that opt in with ``common_utils.known_failure_skip_if`` are marked;
other skips, including ``skipIfRocm``, stay skipped.

In rerun-disabled-tests mode a marked test runs even though it is not in the
disabled-tests JSON. test/conftest.py reports its skip reason as the
``code_skip`` JUnit property, and tools/stats/check_disabled_tests.py uploads
those results to their own collection.
"""

import functools
import os


RUN_CODE_SKIPS = os.getenv("PYTORCH_TEST_RERUN_DISABLED_TESTS", "0") == "1"

# (reason, condition) pairs, one per stacked decorator. functools.wraps copies
# them onto outer wrappers such as instantiate_test.
_MARK_ATTR = "__pt_rerun_code_skips__"


def _forwarding_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def mark_rerun_code_skip(fn, reason, condition):
    """Mark ``fn`` so rerun-disabled-tests runs it while ``condition()`` is true.

    ``condition`` is checked when the mark is read, not at import, since some
    conditions (e.g. the ROCm arch) initialize the GPU.
    """
    # Decorators are often applied to a function shared by several tests
    # (OpInfo variants, device classes), so mark a fresh wrapper instead.
    marked = fn if isinstance(fn, type) else _forwarding_wrapper(fn)
    setattr(marked, _MARK_ATTR, (*getattr(fn, _MARK_ATTR, ()), (reason, condition)))
    return marked


def rerun_code_skip_reason(fn, cls):
    """The reason rerun-disabled-tests runs this test, or None if it does not."""
    for obj in (fn, cls):
        for reason, condition in getattr(obj, _MARK_ATTR, ()):
            if condition():
                return reason
    return None
