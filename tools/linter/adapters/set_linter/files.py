from __future__ import annotations

from pathlib import Path
from typing import Sequence


def resolve_python_files(include: list[str], exclude: list[str]) -> list[Path]:
    include = [j for i in include for j in i.split(":")]
    exclude = [j for i in exclude or () for j in i.split(":")]

    iglobs = python_glob(include, check_errors=True)
    eglobs = python_glob(exclude, check_errors=False)

    return sorted(iglobs - eglobs)


def python_glob(strings: Sequence[str], *, check_errors: bool) -> set[Path]:
    result: set[Path] = set()

    nonexistent: list[str] = []
    not_python: list[str] = []

    for s in strings:
        p = Path(s).expanduser()
        if p.is_dir():
            result.update(p.glob("**/*.py"))
        elif p.suffix != ".py":
            not_python.append(str(p))
        elif p.exists():
            result.add(p)
        else:
            nonexistent.append(str(p))

    if check_errors and (nonexistent or not_python):
        raise ValueError(
            "\n".join(
                [
                    (nonexistent and f'Nonexistent: {" ".join(nonexistent)}') or "",
                    (not_python and f'Not Python: {" ".join(not_python)}') or "",
                ]
            )
        )

    return result
