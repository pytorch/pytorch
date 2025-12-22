from __future__ import annotations

import dataclasses as dc
import json
import re
import shlex
import sys
from functools import cached_property, partial
from pathlib import Path
from typing import Any, TYPE_CHECKING, TypeAlias
from typing_extensions import override


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _PARENT not in _PATH:
    from ._linter import FileLinter, is_public, LintResult, PythonFile
else:
    from _linter import FileLinter, is_public, LintResult, PythonFile

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Sequence


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

    help = "Use block_name instead of pyrefly_name (slower but more compatible)"
    add("--use-block-name", "-u", action="store_true", help=help)

    help = "Rewrite the grandfather list"
    add("--write-grandfather", "-w", action="store_true", help=help)


class MissingTypeLinter(FileLinter):
    linter_name = "missing_type_linter"
    description = DESCRIPTION
    epilog = EPILOG
    report_column_numbers = True

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        super().__init__(argv)
        _add_arguments(self.parser.add_argument)

    @override
    def lint_all(self) -> bool:
        if self.must_write_grandfather:
            self._write_grandfather()
        else:
            for path in self.failing_annotations:
                self._lint_file(path)

        self._report()
        return self.must_write_grandfather or not self.failing_annotations

    @override
    def _lint(self, pf: PythonFile) -> Iterator[LintResult]:
        lr = [m.lint_result() for m in self.failing_annotations[pf.path]]
        if not self.args.dry_run:
            yield from lr

    @cached_property
    def all_functions(self) -> list[Function]:
        """All Functions, including empty and non-public"""

        def functions(filename: str, d: dict[str, Any]) -> Iterator[Function]:
            path = Path(filename)
            if path.suffix not in SUFFIXES:
                return
            pf = PythonFile(MissingTypeLinter.linter_name, path=path)
            use = self.args.use_block_name
            for f in d["functions"]:
                yield Function(
                    Annotation.make(f, pf, f["name"], use),
                    [Annotation.make(p, pf, f["name"], use) for p in f["parameters"]],
                )

        items = sorted(self.type_results.items())
        return [i for k, v in items for i in functions(k, v)]

    @cached_property
    def functions(self) -> list[Function]:
        """All public annotations, grouped by function"""
        return [f.public for f in self.all_functions if f.public]

    @cached_property
    def failing_annotations(self) -> dict[Path, list[Annotation]]:
        """Public, missing annotations that are not grandfathered"""
        d: dict[Path, list[Annotation]] = {}
        it = (a for f in self.functions for a in f)
        for a in it:
            if a.missing_annotation and a.grandfather_name not in self.grandfather:
                d.setdefault(a.python_file.path, []).append(a)
        return d

    @cached_property
    def grandfather(self) -> set[str]:
        """
        Read lines from grandfather file, ignore comments and blank lines, return as set
        """
        with self.args.grandfather.open() as fp:
            return {v for i in fp if (v := i.split("#")[0].strip())}

    @cached_property
    def must_write_grandfather(self) -> bool:
        return self.args.write_grandfather or not self.args.grandfather.exists()

    @cached_property
    def type_results(self) -> dict[str, Any]:
        """Results from calling `pyrefly` and un-JSONing it, then making names unique"""
        if self.args.type_result:
            text = self.args.type_result.read_text()
        else:
            path = self.args.path or ["torch"]
            cmd = *shlex.split(self.args.type_check), *path
            text = self.call(cmd)

        type_results = dict(sorted(json.loads(text).items()))
        name_count: dict[str, int] = {}

        # Uniquify pyrefly names
        for contents in type_results.values():
            for f in contents["functions"]:
                name = f["name"]
                if count := name_count.get(name, 0):
                    f["name"] = f"{name}[{count + 1}]"
                name_count[name] = count + 1  # pyrefly: ignore[unbound-name]

        return type_results

    def _report(self) -> None:
        def count(name: str, value: Number | Collection[Any]) -> None:
            report[name] = value if isinstance(value, Number) else len(value)

        def percent(name: str, base: str) -> None:
            report[name + "_percent"] = round(100 * report[name] / report[base], 4)

        Number: TypeAlias = int | float
        report: dict[str, Number] = {}

        full = "fully_annotated_functions"  # Our metric
        part = "partially_annotated_functions"
        un = "unannotated_functions"
        report.update({full: 0, part: 0, un: 0})

        for f in self.functions:
            assert f.public is not None
            a = [i.missing_annotation for i in f.public]
            category = un if all(a) else part if any(a) else full
            report[category] += 1

        count("functions", self.functions)
        for category in (full, part, un):
            percent(category, "functions")

        all_annotations = [a for f in self.all_functions for a in f]
        annotations = [a for f in self.functions for a in f]
        params = [a for a in annotations if a.param_name]
        returns = [a for a in annotations if not a.param_name]

        count("all_files", {a.python_file.path for a in all_annotations})
        count("public_files", {f.path for f in self.functions})

        count("all_functions", sum(not a.param_name for a in all_annotations))
        count("public_functions", sum(not a.param_name for a in annotations))

        count("all_annotations", all_annotations)
        count("public_annotations", annotations)

        count("all_params", sum(bool(a.param_name) for a in all_annotations))
        count("public_params", params)

        count("grandfather", self.grandfather)

        def missing_percent(name: str, annotations: Collection[Annotation]) -> None:
            report[name + "_needed"] = sum(a.needs_annotation for a in annotations)
            report[name + "_missing"] = sum(a.missing_annotation for a in annotations)
            percent(name + "_missing", name + "_needed")

        missing_percent("annotations", annotations)
        missing_percent("return_annotations", returns)
        missing_percent("param_annotations", params)

        print(json.dumps(report, indent=4), file=sys.stderr)

    def _write_grandfather(self) -> None:
        it = (i for f in self.functions for i in f.missing())
        grandfather = sorted(a.grandfather_name for a in it)
        if not self.args.dry_run:
            with self.args.grandfather.open("w") as fp:
                fp.writelines(g + "\n" for g in grandfather)


@dc.dataclass(frozen=True)
class Annotation:
    """
    Represents one piece of information from `pyrefly` about a possible annotation,
    which might be None.
    """

    annotation: str | None
    location: dict[str, Any]
    param_name: str | None
    pyrefly_name: str
    python_file: PythonFile
    use_block_name: bool

    @staticmethod
    def make(
        d: dict[str, Any], pf: PythonFile, pyrefly_name: str, use_block_name: bool
    ) -> Annotation:
        if (annotation := d.get("annotation", d)) is d:
            # It's a return annotation
            annotation = d["return_annotation"]
            param_name = None
        else:
            # It's a parameter annotation
            param_name = d["name"]

        return Annotation(
            annotation=annotation,  # pyrefly: ignore[unbound-name]
            location=d["location"],
            param_name=param_name,
            pyrefly_name=pyrefly_name,
            python_file=pf,
            use_block_name=use_block_name,
        )

    @cached_property
    def block_name(self) -> str:
        # This triggers a somewhat expensive tokenization of the file.
        return self.python_file.block_name(self.line + 1)

    @cached_property
    def column(self) -> int:
        column = self.location["start"]["column"]
        assert isinstance(column, int)
        return column

    @cached_property
    def function_name(self) -> str:
        name = self.block_name if self.use_block_name else self.pyrefly_name
        return name.split(".")[-1]

    @cached_property
    def grandfather_name(self) -> str:
        if self.use_block_name:
            name = ".".join((*self.python_file.python_parts, self.block_name))
        else:
            name = self.pyrefly_name
        assert not name.endswith("."), (name, self)
        return f"{name}({self.param_name}=)" if self.param_name else name

    @cached_property
    def is_public(self) -> bool:
        return (
            (self.param_name is None or is_public(self.param_name))
            and is_public(self.pyrefly_name)
            and (
                not self.use_block_name
                or (self.python_file.is_public and is_public(self.block_name))
            )
        )

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
        name = f"Missing {category} type for {self.pyrefly_name}"
        return LintResult(
            char=self.column, length=self.length, line=self.line, name=name
        )

    @cached_property
    def missing_annotation(self) -> bool:
        """Does this need an annotation but doesn't have one?"""
        return self.annotation is None and self.needs_annotation

    @cached_property
    def needs_annotation(self) -> bool:
        """Does this actually require annotation?

        Parameters named `self` and `cls` do not need annotations, nor do the returns
        of double underscore functions.
        """
        return (
            self.is_public
            and self.param_name not in ("self", "cls")
            and (bool(self.param_name) or not self.function_name.startswith("__"))
        )


@dc.dataclass
class Function:
    return_: Annotation
    params: Sequence[Annotation]

    @cached_property
    def public(self) -> Function | None:
        if self.return_.is_public:
            params = [a for a in self.params if a.is_public]
            return Function(self.return_, params)
        return None

    def missing(self) -> Iterator[Annotation]:
        if self.return_.missing_annotation:
            yield self.return_
        yield from (p for p in self.params if p.missing_annotation)

    def path(self) -> Path:
        return self.return_.python_file.path

    def __iter__(self) -> Iterator[Annotation]:
        yield self.return_
        yield from self.params


if __name__ == "__main__":
    MissingTypeLinter.run()
