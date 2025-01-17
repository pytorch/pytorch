from typing import Optional, List, Sequence, Set, Union
from pathlib import Path
from re import match as _match


def read_file(fname: str) -> List[str]:
    with open(fname, encoding='utf-8') as f:
        return f.readlines()


def embed_headers(fname: str, include_dirs:Optional[Union[Sequence[str], Sequence[Path], str]] = None) -> str:
    def _worker(fname: str, include_dirs: List[Path], processed_files: Set[str]) -> str:
        if fname in processed_files:
            return ""
        processed_files.add(fname)
        content = read_file(fname)
        for line_idx, cur_line in enumerate(content):
            m = _match('^\\s*#include\\s*[<"]([^>"]+)[>"]', cur_line)
            if m is None:
                continue
            fname = m[1]
            for include_dir in include_dirs:
                if (include_dir / fname).exists():
                    content[line_idx] = _worker(str(include_dir / fname), include_dirs, processed_files)
                    break
        return "".join(content)

    if include_dirs is None:
        include_dirs = [Path(__file__).parent.parent.parent]
    elif isinstance(include_dirs, str):
        include_dirs = [Path(include_dirs)]
    else:
        include_dirs = [Path(x) for x in include_dirs]

    return _worker(fname, include_dirs, set())



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:\n {sys.argv[0]} filename")
        sys.exit(1)
    print(embed_headers(sys.argv[1]))
