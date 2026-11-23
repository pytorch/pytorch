#!/usr/bin/env python3
# Tool to quickly rebuild one or two files with debug info.
#
# It recompiles each named source with -g (in place of -O2/-O3), reusing the
# exact command ninja recorded for it, then follows ninja's command graph to
# relink its dependents. Use --dry-run to print the plan without building.
#
# Why not `ninja -n torch_python | sed 's/-O[23]/-g/' | sh` (the old approach):
# the build uses file(GLOB ... CONFIGURE_DEPENDS), which wires a glob-check
# into build.ninja's own regeneration. In dry-run (-n) mode ninja cannot run
# that check or reload the regenerated graph, so `ninja -n <target>` only ever
# reports the regeneration step (VerifyGlobs + regenerate-during-build) and
# never the real compile/link commands. We therefore source the per-file
# commands from `ninja -t commands` (a graph walk, not a dry run), which is not
# affected by the glob-check.

from __future__ import annotations

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


def command_tokens(cmd: str, build_dir: Path) -> list[tuple[str, Path]]:
    tokens = shlex.split(cmd)
    cwd = build_dir
    if len(tokens) > 2 and tokens[0] == "cd" and tokens[2] == "&&":
        cwd = Path(tokens[1])
        tokens = tokens[3:]
    return [(token, (cwd / token).resolve()) for token in tokens]


def command_outputs(cmd: str, build_dir: Path) -> set[Path]:
    tokens = command_tokens(cmd, build_dir)
    return {
        tokens[idx + 1][1]
        for idx, (token, _) in enumerate(tokens[:-1])
        if token == "-o"
    }


def create_build_plan(
    files: list[str], ninja_commands: str, build_dir: Path
) -> list[tuple[str, str]]:
    commands = [line.strip() for line in ninja_commands.splitlines() if line.strip()]
    sources = {Path(file).resolve() for file in files}
    parsed = []
    for cmd in commands:
        paths = {path for _, path in command_tokens(cmd, build_dir)}
        parsed.append(
            (cmd, paths, command_outputs(cmd, build_dir), bool(sources & paths))
        )
    selected: set[int] = set()
    outputs: set[Path] = set()

    for idx, (_, _, cmd_outputs, is_source) in enumerate(parsed):
        if not is_source:
            continue
        selected.add(idx)
        outputs.update(cmd_outputs)

    if not selected:
        raise RuntimeError("Could not find build commands for the requested files")

    for idx, (cmd, paths, cmd_outputs, _) in enumerate(parsed):
        if idx in selected:
            continue
        has_embedded_output = any(str(output) in cmd for output in outputs)
        if outputs.isdisjoint(paths) and not has_embedded_output:
            continue
        selected.add(idx)
        outputs.update(cmd_outputs)

    return [
        (
            "debug rebuild" if is_source else "rebuild dependent",
            debugify(cmd) if is_source else cmd,
        )
        for idx, (cmd, _, _, is_source) in enumerate(parsed)
        if idx in selected
    ]


def main() -> None:
    if sys.platform == "win32":
        print("Not supported on Windows yet")
        sys.exit(-95)
    args = parse_args()
    if not has_build_ninja():
        print("Only ninja build system is supported at the moment")
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

    ninja_commands = check_output(
        ["ninja", "-t", "commands", "torch_python"], cwd=str(BUILD_DIR)
    )
    try:
        plan = create_build_plan(files, ninja_commands, BUILD_DIR)
    except RuntimeError as error:
        print(error)
        sys.exit(-1)

    if args.dry_run:
        for kind, cmd in plan:
            print(f"# {kind}")
            print(cmd)
        return

    for idx, (kind, cmd) in enumerate(plan):
        print(f"[{idx + 1} / {len(plan)}] {kind}")
        if args.verbose:
            print(cmd)
        subprocess.check_call(["sh", "-c", cmd], cwd=str(BUILD_DIR))

    create_symlinks()


if __name__ == "__main__":
    main()
