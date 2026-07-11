"""Runtime-only utilities extracted from upstream FA4 test helpers."""

from torch._guards import active_fake_mode


def is_fake_mode() -> bool:
    return active_fake_mode() is not None
