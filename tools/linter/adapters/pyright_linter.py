from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _PARENT not in _PATH:
    from . import _linter
else:
    import _linter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


GRANDFATHER = Path(__file__).parent / "pyright_linter-grandfather.txt"

DESCRIPTION = """`pyright_linter` is a lintrunner linter which uses pyright to detect
new symbols that have been created but have not been given a type.
"""

EPILOG = """
"""

PYRIGHT = "pyright --ignoreexternal --outputjson --verifytypes torch"
PARAM_RE = re.compile('Type of parameter "(.*)" is unknown')


class PyrightLinter(_linter.FileLinter):
    linter_name = "pyright_linter"
    description = DESCRIPTION
    epilog = EPILOG
    report_column_numbers = True

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        super().__init__(argv)
        add = self.parser.add_argument

        help = "Set the grandfather list"
        add("--grandfather", "-g", default=GRANDFATHER, type=Path, help=help)

        help = "Rewrite the grandfather list"
        add("--write-grandfather", "-w", action="store_true", help=help)

    def lint_all(self) -> bool:
        unknown = {}
        res = self.call(PYRIGHT, check=False)  # Always fails!

        for symbol in json.loads(res)["typeCompleteness"]["symbols"]:
            name = symbol["name"]
            for d in symbol.get("diagnostics", ()):
                if d["message"] == "Return type is unknown":
                    unknown[name] = d
                elif m := PARAM_RE.match(d["message"]):
                    unknown[f"{name}({m.group(1)}=)"] = d

        if not self.args.grandfather.exists() or self.args.write_grandfather:
            with self.args.grandfather.open("w") as fp:
                print(*sorted(unknown), file=fp, sep="\n")
            return True

        grandfather = set(self.args.grandfather.read_text().splitlines())
        if not (missing := [v for k, v in unknown.items() if k not in grandfather]):
            return True

        self._path_to_diagnostics = dict[Path, list[dict[str, Any]]]()
        for d in missing:
            path = Path(d["file"]).relative_to(Path(".").absolute())
            self._path_to_diagnostics.setdefault(path, []).append(d)

        for filename in sorted(self._path_to_diagnostics):
            self._lint_file(filename)
        return False

    def _lint(self, pf: _linter.PythonFile) -> Iterator[_linter.LintResult]:
        assert pf.path is not None
        for d in self._path_to_diagnostics[pf.path]:
            try:
                start = d["range"]["start"]
                end = d["range"]["end"]
            except KeyError:
                continue

            if (line := start["line"]) != end["line"]:
                # Not sure how to handle this!
                continue
            char = start["character"]
            length = end["character"] - char
            name = d["message"]
            yield _linter.LintResult(name=name, line=line, char=char, length=length)


if __name__ == "__main__":
    PyrightLinter.run()
