""" brain-dead simple parser for ini-style files.
(C) Ronny Pfannschmidt, Holger Krekel -- MIT licensed
"""
from __future__ import annotations
from typing import (
    Callable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    TYPE_CHECKING,
    NoReturn,
    NamedTuple,
    overload,
    cast,
)

import os

if TYPE_CHECKING:
    from typing_extensions import Final

__all__ = ["IniConfig", "ParseError", "COMMENTCHARS", "iscommentline"]

from .exceptions import ParseError
from . import _parse
from ._parse import COMMENTCHARS, iscommentline

_D = TypeVar("_D")
_T = TypeVar("_T")


class SectionWrapper:
    config: Final[IniConfig]
    name: Final[str]

    def __init__(self, config: IniConfig, name: str) -> None:
        self.config = config
        self.name = name

    def lineof(self, name: str) -> int | None:
        return self.config.lineof(self.name, name)

    @overload
    def get(self, key: str) -> str | None:
        ...

    @overload
    def get(
        self,
        key: str,
        convert: Callable[[str], _T],
    ) -> _T | None:
        ...

    @overload
    def get(
        self,
        key: str,
        default: None,
        convert: Callable[[str], _T],
    ) -> _T | None:
        ...

    @overload
    def get(self, key: str, default: _D, convert: None = None) -> str | _D:
        ...

    @overload
    def get(
        self,
        key: str,
        default: _D,
        convert: Callable[[str], _T],
    ) -> _T | _D:
        ...

    # TODO: investigate possible mypy bug wrt matching the passed over data
    def get(  # type: ignore [misc]
        self,
        key: str,
        default: _D | None = None,
        convert: Callable[[str], _T] | None = None,
    ) -> _D | _T | str | None:
        return self.config.get(self.name, key, convert=convert, default=default)

    def __getitem__(self, key: str) -> str:
        return self.config.sections[self.name][key]

    def __iter__(self) -> Iterator[str]:
        section: Mapping[str, str] = self.config.sections.get(self.name, {})

        def lineof(key: str) -> int:
            return self.config.lineof(self.name, key)  # type: ignore[return-value]

        yield from sorted(section, key=lineof)

    def items(self) -> Iterator[tuple[str, str]]:
        for name in self:
            yield name, self[name]


class IniConfig:
    path: Final[str]
    sections: Final[Mapping[str, Mapping[str, str]]]

    def __init__(
        self,
        path: str | os.PathLike[str],
        data: str | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self.path = os.fspath(path)
        if data is None:
            with open(self.path, encoding=encoding) as fp:
                data = fp.read()

        tokens = _parse.parse_lines(self.path, data.splitlines(True))

        self._sources = {}
        sections_data: dict[str, dict[str, str]]
        self.sections = sections_data = {}

        for lineno, section, name, value in tokens:
            if section is None:
                raise ParseError(self.path, lineno, "no section header defined")
            self._sources[section, name] = lineno
            if name is None:
                if section in self.sections:
                    raise ParseError(
                        self.path, lineno, f"duplicate section {section!r}"
                    )
                sections_data[section] = {}
            else:
                if name in self.sections[section]:
                    raise ParseError(self.path, lineno, f"duplicate name {name!r}")
                assert value is not None
                sections_data[section][name] = value

    def lineof(self, section: str, name: str | None = None) -> int | None:
        lineno = self._sources.get((section, name))
        return None if lineno is None else lineno + 1

    @overload
    def get(
        self,
        section: str,
        name: str,
    ) -> str | None:
        ...

    @overload
    def get(
        self,
        section: str,
        name: str,
        convert: Callable[[str], _T],
    ) -> _T | None:
        ...

    @overload
    def get(
        self,
        section: str,
        name: str,
        default: None,
        convert: Callable[[str], _T],
    ) -> _T | None:
        ...

    @overload
    def get(
        self, section: str, name: str, default: _D, convert: None = None
    ) -> str | _D:
        ...

    @overload
    def get(
        self,
        section: str,
        name: str,
        default: _D,
        convert: Callable[[str], _T],
    ) -> _T | _D:
        ...

    def get(  # type: ignore
        self,
        section: str,
        name: str,
        default: _D | None = None,
        convert: Callable[[str], _T] | None = None,
    ) -> _D | _T | str | None:
        try:
            value: str = self.sections[section][name]
        except KeyError:
            return default
        else:
            if convert is not None:
                return convert(value)
            else:
                return value

    def __getitem__(self, name: str) -> SectionWrapper:
        if name not in self.sections:
            raise KeyError(name)
        return SectionWrapper(self, name)

    def __iter__(self) -> Iterator[SectionWrapper]:
        for name in sorted(self.sections, key=self.lineof):  # type: ignore
            yield SectionWrapper(self, name)

    def __contains__(self, arg: str) -> bool:
        return arg in self.sections
