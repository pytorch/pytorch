from collections.abc import Sequence
from pathlib import Path
from re import match as _match
from typing import Optional, Union


def read_file(fname: Union[Path, str]) -> list[str]:
    with open(fname, encoding="utf-8") as f:
        return f.readlines()


def _embed_headers(
    content: list[str], include_dirs: list[Path], processed_files: set[str]
) -> str:
    for line_idx, cur_line in enumerate(content):
        # Eliminate warning: `#pragma once in main file`
        if cur_line.startswith("#pragma once"):
            content[line_idx] = ""
            continue
        m = _match('^\\s*#include\\s*[<"]([^>"]+)[>"]', cur_line)
        if m is None:
            continue
        for include_dir in include_dirs:
            path = include_dir / m[1]
            if not path.exists():
                continue
            if str(path) in processed_files:
                content[line_idx] = ""
                continue
            processed_files.add(str(path))
            content[line_idx] = _embed_headers(
                read_file(path), include_dirs, processed_files
            )
            break
    return "".join(content)


def embed_headers(
    fname: str, include_dirs: Optional[Union[Sequence[str], Sequence[Path], str]] = None
) -> str:
    if include_dirs is None:
        base_dir = Path(__file__).parent.parent.parent
        include_dirs = [base_dir, base_dir / "aten" / "src"]
    elif isinstance(include_dirs, str):
        include_dirs = [Path(include_dirs)]
    else:
        include_dirs = [Path(x) for x in include_dirs]

    return _embed_headers(read_file(fname), include_dirs, {fname})


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:\n {sys.argv[0]} filename")
        sys.exit(1)
    print(embed_headers(sys.argv[1]))
