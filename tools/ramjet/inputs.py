"""The repository files a build reads, as ninja sees them: every edge's inputs, the
headers each compile discovered, and the files CMake reruns for.

ramjet runs this on the builder after each build, so your machine can tell whether
a change needs the builder. It runs standalone there:

    python inputs.py REPO_ROOT OUTPUT
"""

import os
import subprocess
import sys


def build_inputs(root: str) -> list[str]:
    build = os.path.join(root, "build")

    def ninja_tool(tool: str) -> list[str]:
        # -n: ninja may rewrite its logs as it opens them, except in a dry run
        command = ["ninja", "-n", "-C", build, "-t", tool]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout.splitlines()

    paths = set(ninja_tool("inputs"))
    paths.update(line.strip() for line in ninja_tool("deps") if line.startswith(" "))
    with open(os.path.join(build, "build.ninja")) as manifest:
        for line in manifest:
            # CMake 4 adds cmake_install.cmake to this edge's outputs
            if line.startswith("build build.ninja") and ": RERUN_CMAKE" in line:
                paths.update(line.partition(" | ")[2].split())
    relative = {os.path.relpath(os.path.join(build, path), root) for path in paths}
    return sorted(p for p in relative if p.split(os.sep)[0] not in ("..", "build"))


if __name__ == "__main__":
    root, output = sys.argv[1:]
    with open(output, "w") as f:
        f.writelines(path + "\n" for path in build_inputs(root))
