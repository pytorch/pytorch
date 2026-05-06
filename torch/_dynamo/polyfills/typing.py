import sys
import typing
from typing import Any

from ..decorators import substitute_in_graph


__all__: list[str] = []
if sys.version_info >= (3, 13):
    __all__.append("typevar_typing_prepare_subst")


if sys.version_info >= (3, 13):
    # ref: https://github.com/python/cpython/blob/v3.13.13/Objects/typevarobject.c#L514-L563
    @substitute_in_graph(
        typing.TypeVar.__typing_prepare_subst__  # pyrefly: ignore [missing-attribute]
    )
    def typevar_typing_prepare_subst(
        self: typing.TypeVar, alias: Any, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        params = alias.__parameters__  # pyrefly: ignore [missing-attribute]
        i = params.index(self)
        if i < len(args):
            return args
        if (
            i == len(args)
            and self.__default__  # pyrefly: ignore [missing-attribute]
            is not typing.NoDefault  # pyrefly: ignore [missing-attribute]
        ):
            return args + (self.__default__,)  # pyrefly: ignore [missing-attribute]
        raise TypeError(
            f"Too few arguments for {alias}; actual {len(args)}, expected at least {i + 1}"
        )
