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
    from collections.abc import Callable, Iterator, Sequence


DESCRIPTION = """`missing_type_linter` is a lintrunner linter which uses pyrefly to detect
new symbols that have been created but have not been given a type.
"""
EPILOG = """
"""

_log = partial(print, file=sys.stderr)

GRANDFATHER = Path(__file__).parent / "missing_type_linter_grandfather.txt"

TYPE_CHECK_COMMAND = "pyrefly report --config=pyrefly.toml"
PARAM_RE = re.compile('Type of parameter "(.*)" is unknown')
SUFFIXES = ".py", ".pyi"
PUBLIC_NAMES = "__init__", "__main__"


def _add_arguments(add: Callable[..., Any]) -> None:
    # Also inherits arguments from ._linter.file_linter.FileLinter

    help = "Run, but do not print lint checks or write grandfather file"
    add("--dry-run", "-d", action="store_true", help=help)

    help = f"Set the grandfather file name (default={GRANDFATHER})"
    add("--grandfather", "-g", default=GRANDFATHER, type=Path, help=help)

    help = "The paths to check"
    add("--path", "-p", nargs="*", help=help)

    help = f"The command line for type checking (default='{TYPE_CHECK_COMMAND}')"
    add("--type-check", "-t", default=TYPE_CHECK_COMMAND, help=help)

    help = "The name of a JSON file with type checking results (ignore --type-check)"
    add("--type-result", "-r", default=None, type=Path, help=help)

    help = "Rewrite the grandfather list"
    add("--write-grandfather", "-w", action="store_true", help=help)


def _is_public(*p: str) -> bool:
    # TODO: this rule is simple and correct as far as it goes, but incomplete.
    #
    # What is missing is checking `__all__`: see
    # https://github.com/pytorch/pytorch/wiki/Public-API-definition-and-documentation

    it = (j for i in p for j in i.split("."))
    return not any(i.startswith("_") and i not in PUBLIC_NAMES for i in it)


class MissingTypeLinter(FileLinter):
    linter_name = "missing_type_linter"
    description = DESCRIPTION
    epilog = EPILOG
    report_column_numbers = True
    summary: dict[str, float | int]

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        super().__init__(argv)
        self.summary = {}

        _add_arguments(self.parser.add_argument)

    def lint_all(self) -> bool:
        if self.must_write_grandfather:
            self._write_grandfather()
        else:
            for path in self.missing_annotations_delta:
                self._lint_file(path)

        self._summarize()
        return self.must_write_grandfather or not self.missing_annotations_delta

    def _lint(self, pf: PythonFile) -> Iterator[LintResult]:
        assert pf.path is not None
        lr = [m.lint_result() for m in self.missing_annotations_delta[pf.path]]
        if not self.args.dry_run:
            yield from lr

    def _summarize(self) -> None:
        # TODO: add commit ID, commit message, timestamp, more? to the summary report
        s = dict(self.summary)

        def percent(name: str, base: str = "public_functions") -> None:
            s[name + "_percent"] = round(100 * s[name] / s[base], 4)

        percent("full_annotations")
        percent("partial_annotations")
        percent("unannotated")

        percent("return_annotations")
        percent("self_parameters")

        percent("parameter_annotations", "public_functions_public_parameters")

        s = dict(sorted(s.items()))
        print(json.dumps(s, indent=4), file=sys.stderr)

    @cached_property
    def missing_annotations_delta(self) -> dict[Path, list[MissingAnnotation]]:
        """Missing annotations which are not grandfathered"""
        with self.args.grandfather.open() as fp:
            grandfather = {i.strip() for i in fp}
        self.summary["grandfather"] = len(grandfather)

        def grandfathered(lm: list[MissingAnnotation]) -> list[MissingAnnotation]:
            return [m for m in lm if m.name not in grandfather]

        items = self.missing_annotations.items()
        return {k: g for k, v in items if (g := grandfathered(v))}

    @cached_property
    def missing_annotations(self) -> dict[Path, list[MissingAnnotation]]:
        """All unannotated functions and parameters, indexed by Path"""

        def count(key: str) -> None:
            self.summary[key] = 1 + self.summary.get(key, 0)

        def function(f: dict[str, Any]) -> Iterator[MissingAnnotation]:
            count("all_functions")
            if not _is_public(f["name"]):
                return
            count("public_functions")

            if annotated := bool(f["return_annotation"]):
                count("return_annotations")
            else:
                yield MissingAnnotation(f["name"], f["location"])
            has_annotations = [annotated]  # pyrefly: ignore[unbound-name]

            for p in f["parameters"]:
                count("public_functions_all_parameters")
                if p["name"] == "self":
                    count("self_parameters")
                    continue
                if not _is_public(p["name"]):
                    continue
                count("public_functions_public_parameters")

                if annotated := bool(p["annotation"]):
                    count("parameter_annotations")
                else:
                    yield MissingAnnotation(f["name"], p["location"], p["name"])
                has_annotations.append(annotated)  # pyrefly: ignore[unbound-name]

            if all(has_annotations):
                count("full_annotations")
            elif any(has_annotations):
                count("partial_annotations")
            else:
                count("unannotated")

        def missing(i: int, pf: PythonFile) -> Iterator[MissingAnnotation]:
            count("all_files")

            assert pf.path is not None
            functions = self.type_results[pf.path.absolute()]["functions"]

            if self.args.verbose:
                msg = f"{i + 1:03d}:{len(functions):03d}:{pf.path}"
                _log(msg)

            for f in functions:
                yield from function(f)

        all_files = [self.make_file(Path(f)) for f in self.type_results]
        missed = ((pf, missing(i, pf)) for i, pf in enumerate(all_files))
        return {pf.path: v for pf, m in missed if (v := [i for i in m if i.is_public])}

    def _write_grandfather(self) -> None:
        """Names of symbols that are grandfathered into not having type annotations"""
        annotations = self.missing_annotations.values()
        grandfather = sorted({m.name for v in annotations for m in v})
        self.summary["grandfather"] = len(grandfather)
        if not self.args.dry_run:
            with self.args.grandfather.open("w") as fp:
                fp.writelines(g + "\n" for g in grandfather)

    @cached_property
    def _raw_type_results(self) -> dict[Path, Any]:
        """`Results from calling `pyrefly` and un-JSONing it"""
        if self.args.type_result:
            text = self.args.type_result.read_text()
        else:
            path = self.args.path or ["torch"]
            cmd = *shlex.split(self.args.type_check), *path
            text = self.call(cmd)
        return json.loads(text)

    @cached_property
    def type_results(self) -> dict[Path, Any]:
        """Map sorted absolute file paths to lists of `pyrefly` function reports"""
        it = ((Path(k), v) for k, v in sorted(self._raw_type_results.items()))
        return {p: v for p, v in it if p.suffix in SUFFIXES}

    @cached_property
    def must_write_grandfather(self) -> bool:
        return self.args.write_grandfather or not self.args.grandfather.exists()


@dc.dataclass(frozen=True)
class MissingAnnotation:
    name: str
    location: dict[str, Any]
    param_name: str | None = None

    is_fixer = False

    @cached_property
    def column(self) -> int:
        column = self.location["start"]["column"]
        assert isinstance(column, int)
        return column

    @cached_property
    def is_public(self) -> bool:
        return _is_public(self.name)

    @cached_property
    def length(self) -> int | None:
        end = self.location["end"]
        return 1 + end["column"] - self.column if self.line == end["line"] else 0

    @cached_property
    def line(self) -> int:
        line = self.location["start"]["line"]
        assert isinstance(line, int)
        return line

    def lint_result(self) -> LintResult:
        category = "parameter" if self.param_name else "return"
        name = f"Missing {category} type for {self.name}"
        return LintResult(
            char=self.column, length=self.length, line=self.line, name=name
        )

    @cached_property
    def suffix(self) -> str:
        return f"({self.param_name}=)" if self.param_name else ""


if __name__ == "__main__":
    MissingTypeLinter.run()
