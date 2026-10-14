# mypy: allow-untyped-defs
"""Stamps for tests that rerun-disabled-tests should execute.

Importing this module must not import torch. The rerun test file loads it
even when the extension is not built.
"""

import unittest


# Set on @unittest.skip by test/conftest.py when the caller is not skipIf
# or skipUnless. Duplicated here so check_if_enable can see it without
# importing the pytest plugin.
UNCONDITIONAL_SKIP_ATTR = "__pt_unconditional_skip__"
RERUN_CODE_SKIP_ATTR = "__pt_rerun_code_skip__"
# Device-instantiation skips (skipMPS and friends). Collection runs the
# body only when the instantiated class device matches this value.
RERUN_SKIP_DEVICE_ATTR = "__pt_rerun_skip_device__"
# functools.wraps copies the stamps above onto every outer wrapper.
# The value is the id of the function that was actually stamped, so a
# copy on onlyCUDA or instantiated_test does not look owned.
RERUN_STAMP_OWNER_ATTR = "__pt_rerun_stamp_owner__"

# Messages that mean "this ROCm/CUDA API is not there", not "this test is
# a known failure". Checked before the known-bug phrases.
_UNSUPPORTED_API_MARKERS = (
    "ptx",
    "cuda-specific",
    "cuda specific",
    "hipblas",
    "nvidia-only",
    "nvidia only",
)


def _own_rerun_stamp(fn):
    try:
        setattr(fn, RERUN_STAMP_OWNER_ATTR, id(fn))
    except (AttributeError, TypeError):
        return fn
    return fn


def rerun_code_skip(reason):
    """unittest.skip that rerun-disabled-tests will execute."""

    def decorator(fn):
        skipped = unittest.skip(reason)(fn)
        try:
            setattr(skipped, RERUN_CODE_SKIP_ATTR, True)
        except (AttributeError, TypeError):
            return skipped
        return _own_rerun_stamp(skipped)

    return decorator


def stamp_rerun_device_skip(fn, device_type: str):
    """Mark a device-instantiation wrapper. Collection unwraps only that device."""
    try:
        setattr(fn, RERUN_CODE_SKIP_ATTR, True)
        setattr(fn, RERUN_SKIP_DEVICE_ATTR, device_type)
    except (AttributeError, TypeError):
        return fn
    return _own_rerun_stamp(fn)


def has_rerun_skip_stamp(func, cls=None) -> bool:
    for obj in (func, cls):
        if obj is None:
            continue
        if getattr(obj, UNCONDITIONAL_SKIP_ATTR, False) or getattr(
            obj, RERUN_CODE_SKIP_ATTR, False
        ):
            return True
    return False


def rocm_message_is_known_bug(msg: str) -> bool:
    """True when a ROCm skip message is a known failure, not a missing API.

    Known failure: a GitHub issue URL, "doesn't currently work", or a
    numerical-mismatch reason. Missing API: PTX, CUDA-specific codegen,
    hipBLAS, or NVIDIA-only.
    """
    text = msg.lower()
    if any(marker in text for marker in _UNSUPPORTED_API_MARKERS):
        return False
    if "github.com/" in text and "/issues/" in text:
        return True
    if "doesn't currently work" in text or "does not currently work" in text:
        return True
    if "numerical" in text:
        return True
    return False
