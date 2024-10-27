from __future__ import annotations

from typing import final


@final
class UsageError(Exception):
    """Error in pytest usage or invocation."""


class PrintHelp(Exception):
    """Raised when pytest should print its help to skip the rest of the
    argument parsing and validation."""
