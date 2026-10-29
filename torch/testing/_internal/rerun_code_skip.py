# mypy: allow-untyped-defs
"""Known-failure skips that rerun-disabled-tests runs instead of skipping.

Only tests that opt in with ``rerun_code_skip`` or
``common_utils.known_failure_skip_if`` are marked; other skips, including
``skipIfRocm``, stay skipped.

In rerun-disabled-tests mode a marked test runs even though it is not in the
disabled-tests JSON. test/conftest.py reports its skip reason as the
``code_skip`` JUnit property, and tools/stats/check_disabled_tests.py uploads
those results to their own collection.
"""

import functools
import os
import unittest


RUN_CODE_SKIPS = os.getenv("PYTORCH_TEST_RERUN_DISABLED_TESTS", "0") == "1"

# Maps a device type, or None for every device, to (reason, condition).
# functools.wraps copies it onto outer wrappers such as instantiate_test.
_REASONS_ATTR = "__pt_rerun_code_skip_reasons__"


def _forwarding_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def mark_rerun_code_skip(fn, reason, device_type=None, condition=None):
    """Mark ``fn`` so rerun-disabled-tests runs it.

    ``device_type`` limits the mark to that device of device-generic tests.
    ``condition`` is checked when the mark is read, not at import, since some
    conditions (e.g. the ROCm arch) initialize the GPU.
    """
    # Decorators are often applied to a function shared by several tests
    # (OpInfo variants, device classes), so mark a fresh wrapper instead.
    marked = fn if isinstance(fn, type) else _forwarding_wrapper(fn)
    reasons = {**getattr(fn, _REASONS_ATTR, {}), device_type: (reason, condition)}
    setattr(marked, _REASONS_ATTR, reasons)
    return marked


def rerun_code_skip(reason):
    """unittest.skip for a known failure, which rerun-disabled-tests runs instead."""
    if RUN_CODE_SKIPS:
        return functools.partial(mark_rerun_code_skip, reason=reason)
    return unittest.skip(reason)


def rerun_code_skip_reason(fn, cls):
    """The reason rerun-disabled-tests runs this test, or None if it does not."""
    device_type = getattr(cls, "device_type", None)
    for obj in (fn, cls):
        reasons = getattr(obj, _REASONS_ATTR, {})
        entry = reasons.get(device_type, reasons.get(None))
        if entry is None:
            continue
        reason, condition = entry
        if condition is None or condition():
            return reason
    return None
