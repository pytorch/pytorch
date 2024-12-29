from functools import lru_cache
from typing import Callable, Iterable, Iterator, TypeVar, Union, overload

import jaraco.text as text
from packaging.requirements import Requirement

_T = TypeVar("_T")
_StrOrIter = Union[str, Iterable[str]]


parse_req: Callable[[str], Requirement] = lru_cache()(Requirement)
# Setuptools parses the same requirement many times
# (e.g. first for validation than for normalisation),
# so it might be worth to cache.


def parse_strings(strs: _StrOrIter) -> Iterator[str]:
    """
    Yield requirement strings for each specification in `strs`.

    `strs` must be a string, or a (possibly-nested) iterable thereof.
    """
    return text.join_continuation(map(text.drop_comment, text.yield_lines(strs)))


@overload
def parse(strs: _StrOrIter) -> Iterator[Requirement]: ...


@overload
def parse(strs: _StrOrIter, parser: Callable[[str], _T]) -> Iterator[_T]: ...


def parse(strs, parser=parse_req):
    """
    Replacement for ``pkg_resources.parse_requirements`` that uses ``packaging``.
    """
    return map(parser, parse_strings(strs))
