from __future__ import annotations

import contextlib
import dataclasses
import math
import sys
import textwrap
from io import StringIO
from typing import Any, NamedTuple, TYPE_CHECKING
from typing_extensions import Self


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


class LineContext(NamedTuple):
    context: Any


@dataclasses.dataclass
class ValueWithLineMap:
    value: str
    line_map: list[tuple[int, LineContext]]


class DeferredLineBase:
    """A line that can be 'unwritten' at a later time"""

    def __init__(self, line: str):
        if not line.strip():
            line = ""
        self.line = line

    def __call__(self) -> str | None:
        """Returns either self.line or None to indicate the line has been 'unwritten'"""
        raise NotImplementedError

    def _new_line(self, line: str) -> Self:
        """Returns a new deferred line with the same condition"""
        raise NotImplementedError

    def with_prefix(self, prefix: str) -> Self:
        return self._new_line(f"{prefix}{self.line}")

    def lstrip(self) -> Self:
        return self._new_line(self.line.lstrip())

    def __getitem__(self, index: int | slice) -> Self:
        return self._new_line(self.line[index])

    def __bool__(self) -> bool:
        return bool(self.line)

    def __len__(self) -> int:
        return len(self.line)


class IndentedBuffer:
    """Indent-aware buffer for code generation that can preserve line context."""

    tabwidth = 4

    def __init__(self, initial_indent: int = 0) -> None:
        self._lines: list[DeferredLineBase | LineContext | str] = []
        self._indent = initial_indent

    @contextlib.contextmanager
    def set_tabwidth(self, tabwidth: int) -> Iterator[None]:
        prev = self.tabwidth
        try:
            self.tabwidth = tabwidth
            yield
        finally:
            self.tabwidth = prev

    def getvaluewithlinemap(self) -> ValueWithLineMap:
        buf = StringIO()
        p = 1
        linemap: list[tuple[int, LineContext]] = []
        for li in self._lines:
            if isinstance(li, DeferredLineBase):
                line = li()
                if line is None:
                    continue
            elif isinstance(li, LineContext):
                linemap.append((p, li.context))
                continue
            else:
                line = li
            assert isinstance(line, str)
            buf.write(line)
            buf.write("\n")
            p += 1 + line.count("\n")
        return ValueWithLineMap(buf.getvalue(), linemap)

    def getvalue(self) -> str:
        return self.getvaluewithlinemap().value

    def getrawvalue(self) -> str:
        buf = StringIO()
        for li in self._lines:
            if isinstance(li, DeferredLineBase):
                line = li()
                if line is None:
                    continue
            elif isinstance(li, LineContext):
                continue
            else:
                line = li
            assert isinstance(line, str)
            # backslash implies line continuation
            if line.endswith("\\"):
                buf.write(line[:-1])
            else:
                buf.write(line)
                buf.write("\n")
        return buf.getvalue()

    def get_lines_ref(self):
        return self._lines

    def clear(self) -> None:
        self._lines.clear()

    def __bool__(self) -> bool:
        return bool(self._lines)

    def prefix(self) -> str:
        return " " * (self._indent * self.tabwidth)

    def newline(self) -> None:
        self.writeline("\n")

    def writeline(self, line: LineContext | DeferredLineBase | str) -> None:
        if isinstance(line, LineContext):
            self._lines.append(line)
        elif isinstance(line, DeferredLineBase):
            self._lines.append(line.with_prefix(self.prefix()))
        elif line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def writelines(self, lines: Sequence[LineContext | DeferredLineBase | str]) -> None:
        for line in lines:
            self.writeline(line)

    def indent(self, offset: int = 1) -> contextlib.AbstractContextManager[None]:
        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()

    def do_indent(self, offset: int = 1) -> None:
        self._indent += offset

    def do_unindent(self, offset: int = 1) -> None:
        self._indent -= offset

    def splice(self, other_code: IndentedBuffer | str, strip: bool = False) -> None:
        if isinstance(other_code, IndentedBuffer):
            dedent = float("inf")

            for line in other_code._lines:
                if not isinstance(line, LineContext) and line:
                    dedent = min(dedent, len(line) - len(line.lstrip()))
            if math.isinf(dedent):
                dedent = 0
            for line in other_code._lines:
                if isinstance(line, LineContext):
                    self._lines.append(line)
                else:
                    IndentedBuffer.writeline(self, line[int(dedent) :])
        else:
            other_code = textwrap.dedent(other_code)
            if strip:
                other_code = other_code.lstrip()
            if not other_code:
                return
            other_code = other_code.rstrip()
            for s in other_code.split("\n"):
                self.writeline(s)

    def map(self, func: Callable[[Any], Any]) -> IndentedBuffer:
        res = IndentedBuffer(initial_indent=self._indent)
        res._lines = [func(line) for line in self._lines]
        return res

    def __repr__(self) -> str:
        return f"{type(self)}({self.getvalue()})"

    def __add__(self, other: Self) -> IndentedBuffer:
        assert self._indent == other._indent
        res = IndentedBuffer(initial_indent=self._indent)
        # TODO(rec): or should this be self.__class__(initial_indent=self._indent)?
        res.writelines(self._lines)
        res.writelines(other._lines)
        return res

    def contains(self, new_line: DeferredLineBase | LineContext | str) -> bool:
        return new_line in self._lines


class FakeIndentedBuffer(IndentedBuffer):
    def __init__(self) -> None:
        super().__init__()

    def __getattribute__(self, name: str) -> Any:
        if name == "__class__":  # Allow access to the class attribute
            return object.__getattribute__(self, name)
        raise RuntimeError(
            f"Tried to call self.{name} on FakeIndentedBuffer. This buffer"
            "is currently used on TritonTemplateKernel to prevent actual"
            "writes to the body without explicitly specifying the body with"
            "`TritonTemplateKernel.set_subgraph_body(name)`"
        )


@contextlib.contextmanager
def restore_stdout_stderr() -> Iterator[None]:
    initial_stdout, initial_stderr = sys.stdout, sys.stderr
    try:
        yield
    finally:
        sys.stdout, sys.stderr = initial_stdout, initial_stderr


class DelayReplaceLine(DeferredLineBase):
    """At end of codegen call `line.replace(key, value_fn())`"""

    def __init__(self, key: str, value_fn: Callable[[], str], line: str):
        super().__init__(line)
        self.key = key
        self.value_fn = value_fn

    def __call__(self) -> str:
        return self.line.replace(self.key, self.value_fn())

    def _new_line(self, line: str) -> DelayReplaceLine:
        return DelayReplaceLine(self.key, self.value_fn, line)
