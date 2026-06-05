from typing import Any

from typing_extensions import assert_type

from torch._dynamo.utils import istype


def check_istype_tuple_narrowing(x: object) -> None:
    if istype(x, (str, int)):
        assert_type(x, str | int)

    if istype(x, (list, tuple)):
        assert_type(x, list[Any] | tuple[Any, ...])


def check_istype_preserves_container_parameters(
    x: list[int] | tuple[int, ...] | str,
) -> None:
    if istype(x, (list, tuple)):
        assert_type(x, list[int] | tuple[int, ...])
