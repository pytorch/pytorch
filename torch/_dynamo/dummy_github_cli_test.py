# mypy: ignore-errors
from .exc import unimplemented_v2


def dummy_method():
    unimplemented_v2(
        gb_type="Testing gb_type",
        context="Testing context",
        explanation="",
        hints=["Testing purposes"],
    )
