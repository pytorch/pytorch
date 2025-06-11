# mypy: ignore-errors

from . import graph_break_hints
from .exc import unimplemented_v2


def dummy_method():
    unimplemented_v2(
        gb_type="Testing gb_type",
        context="Testing context",
        explanation="testing",
        hints=[*graph_break_hints.USER_ERROR],
    )
