"""Merge compile_commands.json from CMake build subdirectories.

Collects compile_commands.json files produced by Ninja and CMake sub-builds,
fixes gcc -> g++ for cquery compatibility, and writes a merged file to the
project root.

Called by cmake/PostBuildSteps.cmake as a build-time custom command.
"""

import argparse
import itertools
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=pathlib.Path, help="CMake binary directory")
    parser.add_argument(
        "source_dir", type=pathlib.Path, help="Project source directory"
    )
    args = parser.parse_args()

    build = args.build_dir
    ninja = list(build.glob("*compile_commands.json"))
    cmake_sub_dir = build / "torch" / "lib" / "build"
    cmake_sub = (
        list(cmake_sub_dir.glob("*/compile_commands.json"))
        if cmake_sub_dir.exists()
        else []
    )

    cmds = [
        entry
        for f in itertools.chain(ninja, cmake_sub)
        for entry in json.loads(f.read_text(encoding="utf-8"))
    ]

    # cquery does not like C++ compiles that start with gcc — it forgets to
    # include the C++ header directories.  Replace with g++ as a workaround.
    for c in cmds:
        if c.get("command", "").startswith("gcc "):
            c["command"] = "g++ " + c["command"][4:]

    out = args.source_dir / "compile_commands.json"
    new = json.dumps(cmds, indent=2)
    if not out.exists() or out.read_text(encoding="utf-8") != new:
        out.write_text(new, encoding="utf-8")


if __name__ == "__main__":
    main()
