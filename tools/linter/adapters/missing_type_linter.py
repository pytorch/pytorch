from __future__ import annotations

import dataclasses as dc
import json
import re
import shlex
import sys
from functools import cached_property, partial
from pathlib import Path
from typing import Any, TYPE_CHECKING


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _PARENT not in _PATH:
    from ._linter import FileLinter, LintResult, PythonFile
else:
    from _linter import FileLinter, LintResult, PythonFile

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


GRANDFATHER = Path(__file__).parent / "missing_type_linter_grandfather.txt"

DESCRIPTION = """`missing_type_linter` is a lintrunner linter which uses pyrefly to detect
new symbols that have been created but have not been given a type.
"""

EPILOG = """
"""

TYPE_CHECK_COMMAND = "pyrefly report"
PARAM_RE = re.compile('Type of parameter "(.*)" is unknown')

PUBLIC_NAMES = "__init__", "__main__"
SUFFIXES = ".py", ".pyi"

# TODO: This file causes the block generator to hang!
BAD_FILE = Path("torch/nn/functional.pyi")


_log = partial(print, file=sys.stderr)


@dc.dataclass(frozen=True)
class MissingAnnotation:
    python_file: PythonFile
    name: str
    location: dict[str, Any]
    param_name: str | None = None

    @cached_property
    def block_name(self) -> str:
        # Triggers a read and tokenize of the whole file the first time called
        return self.python_file.block_name(self.line + 1)

    @cached_property
    def column(self) -> int:
        return self.location["start"]["column"]

    @cached_property
    def grandfather(self) -> str:
        parts = self.python_parts
        if self.block_name:
            parts = *parts, self.block_name
        else:
            parts = *parts, self.name
        return ".".join(parts) + self.suffix

    @cached_property
    def is_public(self) -> bool:
        return is_public(*self.python_parts)

    @cached_property
    def length(self) -> int | None:
        end = self.location["end"]
        return 1 + end["column"] - self.column if self.line == end["line"] else None

    @cached_property
    def line(self) -> int:
        return self.location["start"]["line"]

    @cached_property
    def lint_result(self) -> LintResult:
        category = "parameter" if self.param_name else "return"
        name = f"Missing {category} type for {self.grandfather}"
        return LintResult(
            char=self.column, length=self.length, line=self.line, name=name
        )

    @cached_property
    def python_parts(self) -> tuple[str, ...]:
        return self.python_file.python_parts

    @cached_property
    def suffix(self) -> str:
        return f"({self.param_name}=)" if self.param_name else ""


class MissingTypeLinter(FileLinter):
    linter_name = "missing_type_linter"
    description = DESCRIPTION
    epilog = EPILOG
    report_column_numbers = True

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        super().__init__(argv)
        add = self.parser.add_argument

        help = "The paths to check"
        add("--path", "-p", nargs="*", help=help)
        # add("path", nargs="*", help=help)

        help = f"Set the grandfather list (default={GRANDFATHER})"
        add("--grandfather", "-g", default=GRANDFATHER, type=Path, help=help)

        help = f"The command line for type checking (default='{TYPE_CHECK_COMMAND}')"
        add("--type-check", "-t", default=TYPE_CHECK_COMMAND, help=help)

        help = "A JSON file with the type checking results (so --type-check is ignored)"
        add("--type-result", "-r", default=None, type=Path, help=help)

        help = "Rewrite the grandfather list"
        add("--write-grandfather", "-w", action="store_true", help=help)

    def lint_all(self) -> bool:
        if self.write_grandfather:
            with self.args.grandfather.open("w") as fp:
                fp.writelines(g + "\n" for g in self.grandfather)
            return True
        else:
            for k in self.missing_annotations_added:
                self._lint_file(Path(k))
            return not self.missing_annotations_added

    def _lint(self, pf: PythonFile) -> Iterator[LintResult]:
        yield from (m.lint_result for m in self.missing_annotations_added[str(pf.path)])

    @cached_property
    def missing_annotations_added(self) -> dict[str, list[MissingAnnotation]]:
        with self.args.grandfather.open() as fp:
            grandfather = {i.strip() for i in fp}

        def grandfathered(lm: list[MissingAnnotation]) -> list[MissingAnnotation]:
            return [m for m in lm if m.grandfather not in grandfather]

        items = self.missing_annotations.items()
        return {k: g for k, v in items if (g := grandfathered(v))}

    @cached_property
    def grandfather(self) -> list[str]:
        annotations = self.missing_annotations.values()
        return sorted(m.grandfather for v in annotations for m in v)

    @cached_property
    def missing_annotations(self) -> dict[str, list[MissingAnnotation]]:
        def missing(i: int, pf: PythonFile) -> Iterator[MissingAnnotation]:
            if pf.path == BAD_FILE:
                return

            functions = self.type_results[str(pf.path.absolute())]["functions"]
            functions = [f for f in functions if is_public(f["name"])]

            if self.args.verbose:
                m = f"{i + 1:0{digits}d}:{len(functions):03d}:{pf.filename}"
                _log(m)

            for func in functions:
                make = partial(MissingAnnotation, python_file=pf, name=func["name"])
                if not func["return_annotation"]:
                    yield make(location=func["location"])

                for p in func["parameters"]:
                    if (
                        not p["annotation"]
                        and is_public(n := p["name"])
                        and n != "self"
                    ):
                        yield make(location=p["location"], param_name=n)

        def public(ma: Iterator[MissingAnnotation]) -> list[MissingAnnotation]:
            return [m for m in ma if m.is_public]

        python_files = [self.make_file(Path(f)) for f in self.type_results]
        python_files = [pf for pf in python_files if is_public(*pf.python_parts)]
        digits = len(str(len(python_files) + 1))
        if self.args.verbose:
            _log(len(python_files), "files")

        ipf = enumerate(python_files)
        return {pf.filename: v for i, pf in ipf if (v := public(missing(i, pf)))}

    @cached_property
    def type_results(self) -> dict[str, Any]:
        if self.args.type_result:
            text = self.args.type_result.read_text()
        else:
            path = self.args.path or ["torch"]
            cmd = *shlex.split(self.args.type_check), *path
            text = self.call(cmd)

        items = sorted(json.loads(text).items())
        return {k: v for k, v in items if k.endswith(SUFFIXES)}

    @cached_property
    def write_grandfather(self) -> bool:
        return self.args.write_grandfather or not self.args.grandfather.exists()


def is_public(*p: str) -> bool:
    return not any(i.startswith("_") and i not in PUBLIC_NAMES for i in p)


if __name__ == "__main__":
    MissingTypeLinter.run()
