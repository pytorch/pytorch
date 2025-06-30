from __future__ import annotations

import json
import sys
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import Never

from . import ParseError
from .argument_parser import ArgumentParser
from .messages import LintResult
from .python_file import PythonFile


if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Iterator, Sequence


class ErrorLines:
    """How many lines to display before and after an error"""

    WINDOW = 5
    BEFORE = 2
    AFTER = WINDOW - BEFORE - 1


class FileLinter:
    """The base class that all token-based linters inherit from"""

    description: str
    linter_name: str

    epilog: str | None = None
    is_fixer: bool = True
    report_column_numbers: bool = False

    @abstractmethod
    def _lint(self, python_file: PythonFile) -> Iterator[LintResult]:
        raise NotImplementedError

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        self.argv = argv
        self.parser = ArgumentParser(
            is_fixer=self.is_fixer,
            description=self.description,
            epilog=self.epilog,
        )
        self.result_shown = False

    @classmethod
    def run(cls) -> Never:
        sys.exit(not cls().lint_all())

    def lint_all(self) -> bool:
        if self.args.fix and self.args.lintrunner:
            raise ValueError("--fix and --lintrunner are incompatible")

        success = True
        for p in self.paths:
            success = self._lint_file(p) and success
        return self.args.lintrunner or success

    @classmethod
    def make_file(cls, pc: Path | str | None = None) -> PythonFile:
        return PythonFile.make(cls.linter_name, pc)

    @cached_property
    def args(self) -> Namespace:
        args = self.parser.parse_args(self.argv)

        return args

    @cached_property
    def code(self) -> str:
        return self.linter_name.upper()

    @cached_property
    def paths(self) -> list[Path]:
        files = []
        file_parts = (f for fp in self.args.files for f in fp.split(":"))
        for f in file_parts:
            if f.startswith("@"):
                files.extend(Path(f[1:]).read_text().splitlines())
            elif f != "--":
                files.append(f)
        return sorted(Path(f) for f in files)

    def _lint_file(self, p: Path) -> bool:
        if self.args.verbose:
            print(p, "Reading", file=sys.stderr)

        pf = self.make_file(p)
        replacement, results = self._replace(pf)

        if display := list(self._display(pf, results)):
            print(*display, sep="\n")
        if results and self.args.fix and pf.path and pf.contents != replacement:
            pf.path.write_text(replacement)

        return not results or self.args.fix and all(r.is_edit for r in results)

    def _error(self, pf: PythonFile, result: LintResult) -> None:
        """Called on files that are unparsable"""

    def _replace(self, pf: PythonFile) -> tuple[str, list[LintResult]]:
        # Because of recursive replacements, we need to repeat replacing and reparsing
        # from the inside out until all possible replacements are complete
        previous_result_count = float("inf")
        first_results = None
        original = replacement = pf.contents

        while True:
            try:
                results = sorted(self._lint(pf), key=LintResult.sort_key)
            except IndentationError as e:
                error, (_name, lineno, column, _line) = e.args

                results = [LintResult(error, lineno, column)]
                self._error(pf, *results)

            except ParseError as e:
                results = [LintResult(str(e), *e.token.start)]
                self._error(pf, *results)

            for i, ri in enumerate(results):
                if not ri.is_recursive:
                    for rj in results[i + 1 :]:
                        if ri.contains(rj):
                            rj.is_recursive = True
                        else:
                            break

            first_results = first_results or results
            if not results or len(results) >= previous_result_count:
                break
            previous_result_count = len(results)

            lines = pf.lines[:]
            for r in reversed(results):
                if r.is_edit and not r.is_recursive:
                    r.apply(lines)
            replacement = "".join(lines)

            if not any(r.is_recursive for r in results):
                break
            pf = pf.with_contents(replacement)

        if first_results and self.args.lintrunner:
            name = f"Suggested fixes for {self.linter_name}"
            msg = LintResult(name=name, original=original, replacement=replacement)
            first_results.append(msg)

        return replacement, first_results

    def _display(self, pf: PythonFile, results: list[LintResult]) -> Iterator[str]:
        """Emit a series of human-readable strings representing the results"""
        for r in results:
            if self.args.lintrunner:
                msg = r.as_message(code=self.code, path=str(pf.path))
                yield json.dumps(msg.asdict(), sort_keys=True)
            else:
                if self.result_shown:
                    yield ""
                else:
                    self.result_shown = True
                if r.line is None:
                    yield f"{pf.path}: {r.name}"
                else:
                    yield from (i.rstrip() for i in self._display_window(pf, r))

    def _display_window(self, pf: PythonFile, r: LintResult) -> Iterator[str]:
        """Display a window onto the code with an error"""
        if r.char is None or not self.report_column_numbers:
            yield f"{pf.path}:{r.line}: {r.name}"
        else:
            yield f"{pf.path}:{r.line}:{r.char + 1}: {r.name}"

        begin = max((r.line or 0) - ErrorLines.BEFORE, 1)
        end = min(begin + ErrorLines.WINDOW, 1 + len(pf.lines))

        for lineno in range(begin, end):
            source_line = pf.lines[lineno - 1].rstrip()
            yield f"{lineno:5} | {source_line}"
            if lineno == r.line:
                spaces = 8 + (r.char or 0)
                carets = len(source_line) if r.char is None else (r.length or 1)
                yield spaces * " " + carets * "^"
