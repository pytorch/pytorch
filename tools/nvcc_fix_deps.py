"""Tool to fix the nvcc's dependecy file output

Usage: python nvcc_fix_deps.py nvcc [nvcc args]...

This wraps nvcc to ensure that the dependency file created by nvcc with the
-MD flag always uses absolute paths. nvcc sometimes outputs relative paths,
which ninja interprets as an unresolved dependency, so it triggers a rebuild
of that file every time.

The easiest way to use this is to define:

CMAKE_CUDA_COMPILER_LAUNCHER="python;tools/nvcc_fix_deps.py;ccache"

"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, TextIO


def resolve_include(path: Path, include_dirs: List[Path]) -> Path:
    for include_path in include_dirs:
        abs_path = include_path / path
        if abs_path.exists():
            return abs_path

    paths = "\n    ".join(str(d / path) for d in include_dirs)
    raise RuntimeError(
        f"""
ERROR: Failed to resolve dependency:
    {path}
Tried the following paths, but none existed:
    {paths}
"""
    )


def repair_depfile(depfile: TextIO, include_dirs: List[Path]) -> None:
    changes_made = False
    out = ""
    for line in depfile:
        if ":" in line:
            colon_pos = line.rfind(":")
            out += line[: colon_pos + 1]
            line = line[colon_pos + 1 :]

        line = line.strip()

        if line.endswith("\\"):
            end = " \\"
            line = line[:-1].strip()
        else:
            end = ""

        path = Path(line)
        if not path.is_absolute():
            changes_made = True
            path = resolve_include(path, include_dirs)
        out += f"    {path}{end}\n"

    # If any paths were changed, rewrite the entire file
    if changes_made:
        depfile.seek(0)
        depfile.write(out)
        depfile.truncate()


PRE_INCLUDE_ARGS = ["-include", "--pre-include"]
POST_INCLUDE_ARGS = ["-I", "--include-path", "-isystem", "--system-include"]


def extract_include_arg(include_dirs: List[Path], i: int, args: List[str]) -> None:
    def extract_one(name: str, i: int, args: List[str]) -> Optional[str]:
        arg = args[i]
        if arg == name:
            return args[i + 1]
        if arg.startswith(name):
            arg = arg[len(name) :]
            return arg[1:] if arg[0] == "=" else arg
        return None

    for name in PRE_INCLUDE_ARGS:
        path = extract_one(name, i, args)
        if path is not None:
            include_dirs.insert(0, Path(path).resolve())
            return

    for name in POST_INCLUDE_ARGS:
        path = extract_one(name, i, args)
        if path is not None:
            include_dirs.append(Path(path).resolve())
            return


if __name__ == "__main__":
    ret = subprocess.run(
        sys.argv[1:], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr
    )

    depfile_path = None
    include_dirs = []

    # Parse only the nvcc arguments we care about
    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg == "-MF":
            depfile_path = Path(args[i + 1])
        elif arg == "-c":
            # Include the base path of the cuda file
            include_dirs.append(Path(args[i + 1]).resolve().parent)
        else:
            extract_include_arg(include_dirs, i, args)

    if depfile_path is not None and depfile_path.exists():
        with depfile_path.open("r+") as f:
            repair_depfile(f, include_dirs)

    sys.exit(ret.returncode)
