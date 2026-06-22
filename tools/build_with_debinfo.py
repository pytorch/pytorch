#!/usr/bin/env python3
# Tool to quickly rebuild one or two files with debug info.
#
# It recompiles each named source with -g (in place of -O2/-O3), reusing the
# exact compile command CMake recorded for it, then relinks libtorch_python
# and symlinks the result into torch/lib so an editable `import torch` picks
# it up. Use --dry-run to print the plan without building.
#
# Why not `ninja -n torch_python | sed 's/-O[23]/-g/' | sh` (the old approach):
# the build uses file(GLOB ... CONFIGURE_DEPENDS), which wires a glob-check
# into build.ninja's own regeneration. In dry-run (-n) mode ninja cannot run
# that check or reload the regenerated graph, so `ninja -n <target>` only ever
# reports the regeneration step (VerifyGlobs + regenerate-during-build) and
# never the real compile/link commands. We therefore source the per-file
# compile command from build/compile_commands.json and the link command from
# `ninja -t commands` (a graph walk, not a dry run), neither of which is
# affected by the glob-check.

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PYTORCH_ROOTDIR = Path(__file__).resolve().parent.parent
TORCH_DIR = PYTORCH_ROOTDIR / "torch"
TORCH_LIB_DIR = TORCH_DIR / "lib"
BUILD_DIR = PYTORCH_ROOTDIR / "build"
BUILD_LIB_DIR = BUILD_DIR / "lib"
COMPILE_COMMANDS = BUILD_DIR / "compile_commands.json"


def check_output(args: list[str], cwd: str | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd).decode("utf-8")


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Incremental build PyTorch with debinfo")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rebuild plan (compile + link commands) without building.",
    )
    parser.add_argument("files", nargs="*")
    return parser.parse_args()


def get_lib_extension() -> str:
    if sys.platform == "linux":
        return "so"
    if sys.platform == "darwin":
        return "dylib"
    raise RuntimeError(f"Unsupported platform {sys.platform}")


def create_symlinks() -> None:
    """Creates symlinks from build/lib to torch/lib"""
    if not TORCH_LIB_DIR.exists():
        raise RuntimeError(f"Can't create symlinks as {TORCH_LIB_DIR} does not exist")
    if not BUILD_LIB_DIR.exists():
        raise RuntimeError(f"Can't create symlinks as {BUILD_LIB_DIR} does not exist")
    for torch_lib in TORCH_LIB_DIR.glob(f"*.{get_lib_extension()}"):
        if torch_lib.is_symlink():
            continue
        build_lib = BUILD_LIB_DIR / torch_lib.name
        if not build_lib.exists():
            raise RuntimeError(f"Can't find {build_lib} corresponding to {torch_lib}")
        torch_lib.unlink()
        torch_lib.symlink_to(build_lib)


def has_build_ninja() -> bool:
    return (BUILD_DIR / "build.ninja").exists()


def is_devel_setup() -> bool:
    output = check_output([sys.executable, "-c", "import torch;print(torch.__file__)"])
    return output.strip() == str(TORCH_DIR / "__init__.py")


def debugify(cmd: str) -> str:
    """Swap optimization flags for debug info, leaving everything else intact."""
    cmd = cmd.replace("-O2", "-g").replace("-O3", "-g")
    # Build Metal shaders with debug information.
    if "xcrun metal " in cmd and "-frecord-sources" not in cmd:
        cmd += " -frecord-sources -gline-tables-only"
    return cmd


def index_compile_commands(
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map each absolute source path to its compile_commands.json entry."""
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        src = (Path(entry["directory"]) / entry["file"]).resolve()
        result[str(src)] = entry
    return result


def load_compile_commands() -> dict[str, dict[str, Any]]:
    return index_compile_commands(json.loads(COMPILE_COMMANDS.read_text()))


def entry_command(entry: dict[str, Any]) -> str:
    cmd = entry.get("command")
    if cmd is None:
        cmd = " ".join(shlex.quote(arg) for arg in entry["arguments"])
    return cmd


def extract_link_command(ninja_commands: str, lib: str) -> str:
    """Pick the link command producing `lib` from `ninja -t commands` output.

    The link is the last command mentioning `lib`; ninja wraps linker rules
    as `: && <cmd> && :`.
    """
    link = None
    for line in ninja_commands.split("\n"):
        if lib not in line:
            continue
        line = line.strip()
        if line.startswith(": &&") and line.endswith("&& :"):
            line = line[4:-4].strip()
        link = line
    if link is None:
        raise RuntimeError(f"Could not find the {lib} link command")
    return link


def torch_python_link_command() -> str:
    """Return the libtorch_python link command via a ninja graph walk.

    `ninja -t commands` expands a target's commands without the dry-run
    staleness logic that CONFIGURE_DEPENDS defeats.
    """
    output = check_output(
        ["ninja", "-t", "commands", "torch_python"], cwd=str(BUILD_DIR)
    )
    return extract_link_command(output, f"libtorch_python.{get_lib_extension()}")


def main() -> None:
    if sys.platform == "win32":
        print("Not supported on Windows yet")
        sys.exit(-95)
    args = parse_args()
    if not has_build_ninja():
        print("Only ninja build system is supported at the moment")
        sys.exit(-1)
    if not COMPILE_COMMANDS.exists():
        print(
            f"{COMPILE_COMMANDS} not found; configure with "
            "CMAKE_EXPORT_COMPILE_COMMANDS=ON (PyTorch's build sets this by default)"
        )
        sys.exit(-1)
    # The symlink step rewrites torch/lib, so a real run must target the in-tree
    # (editable) torch. --dry-run only reads the build tree, so it works against
    # any build -- e.g. a CI wheel-build job where torch isn't installed -e.
    if not args.dry_run and not is_devel_setup():
        print(
            "Not a devel setup of PyTorch, "
            "please run `python -m pip install --no-build-isolation -v -e .` first"
        )
        sys.exit(-1)

    files = [f for f in args.files if f]
    if not files:
        return print("Nothing to do")

    compile_commands = load_compile_commands()
    plan: list[tuple[str, str, str]] = []
    for file in files:
        src = str(Path(file).resolve())
        entry = compile_commands.get(src)
        if entry is None:
            print(
                f"No compile command for {file}; is it a source compiled into "
                "the build? (try a path relative to the repo root)"
            )
            sys.exit(-1)
        plan.append((src, debugify(entry_command(entry)), entry["directory"]))

    link = torch_python_link_command()

    if args.dry_run:
        for name, cmd, _ in plan:
            print(f"# debug rebuild: {name}")
            print(cmd)
        print("# relink libtorch_python")
        print(link)
        return

    for idx, (name, cmd, cwd) in enumerate(plan):
        print(f"[{idx + 1} / {len(plan)}] Building {Path(name).name} with debug info")
        if args.verbose:
            print(cmd)
        subprocess.check_call(["sh", "-c", cmd], cwd=cwd)

    print("Relinking libtorch_python")
    if args.verbose:
        print(link)
    subprocess.check_call(["sh", "-c", link], cwd=str(BUILD_DIR))

    create_symlinks()


if __name__ == "__main__":
    main()
