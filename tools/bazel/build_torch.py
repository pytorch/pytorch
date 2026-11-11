"""Builds PyTorch's native code for Bazel by running the regular pip build.

Bazel runs this as one unsandboxed action. The declared sources are copied into
a directory that persists between runs, so the build sees only its declared
inputs and cannot write to the checkout, while CMake still rebuilds
incrementally.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cmake


def mirror(src: Path, files: dict[str, str]) -> None:
    """Makes src hold a copy of every file, and none that an earlier run copied."""
    listing = src.parent / "inputs.txt"
    previous = listing.read_text().splitlines() if listing.exists() else []
    for path in set(previous).difference(files):
        (src / path).unlink(missing_ok=True)
    for path, source in files.items():
        # Copies, not links to the checkout: the build rewrites some of the
        # files it installs, and CMake installs a link as a link.
        if stamp(source) != stamp(src / path):
            (src / path).parent.mkdir(parents=True, exist_ok=True)
            (src / path).unlink(missing_ok=True)
            shutil.copy2(source, src / path, follow_symlinks=False)
    listing.write_text("\n".join(files))


def stamp(path: str | Path) -> tuple[int, int, int] | None:
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        return None
    return status.st_mode, status.st_size, status.st_mtime_ns


def tree(root: Path) -> dict[str, str]:
    """Every file and link under root, keyed by its path from root's parent."""
    files = {}
    for directory, dirs, names in os.walk(root):
        links = [name for name in dirs if os.path.islink(f"{directory}/{name}")]
        for path in (f"{directory}/{name}" for name in names + links):
            files[os.path.relpath(path, root.parent)] = path
    return files


def build(src: Path, installed: Path, log: Path) -> None:
    env = dict(os.environ)
    # pip installs a wheel's scripts and data files into sys.prefix, which is
    # how the build finds ninja on PATH and CMake finds the MKL libraries.
    # Bazel keeps each wheel in its own directory instead, so point at those.
    wheels = [Path(p).parent for p in sys.path if Path(p).name == "site-packages"]
    scripts = [str(wheel / "bin") for wheel in wheels if (wheel / "bin").is_dir()]
    data = [str(wheel / "data") for wheel in wheels if (wheel / "data").is_dir()]
    # A venv also has its interpreter on PATH. NCCL's makefile relies on that: it
    # starts its code generators with `#!/usr/bin/env python3`.
    python = os.path.dirname(sys.executable)
    # The type checker takes cmake for this repository's cmake/ directory.
    # pyrefly: ignore [missing-attribute]
    env["PATH"] = os.pathsep.join([python, cmake.CMAKE_BIN_DIR, *scripts, env["PATH"]])
    env["CMAKE_PREFIX_PATH"] = os.pathsep.join(data)
    # CMake remembers where it found tools and libraries, so make it look again
    # when the places to find them in change, as they do when a wheel is updated.
    places = [env["PATH"], env["CMAKE_PREFIX_PATH"], env.get("CUDA_PATH", "")]
    record = src.parent / "places.txt"
    if not record.exists() or record.read_text() != "\n".join(places):
        (src / "build/CMakeCache.txt").unlink(missing_ok=True)
        record.write_text("\n".join(places))
    # Bazel's Python launcher exports PYTHONSAFEPATH, but CMake runs
    # `python -m torchgen.gen` from the source root and needs it on sys.path.
    env.pop("PYTHONSAFEPATH", None)

    # An editable install, like a developer's: its install tree keeps the same
    # path from one build to the next, which the CMake cache depends on, and
    # nothing is zipped into a wheel. pip adds the distribution's metadata and
    # its own redirect files to that tree.
    command = [sys.executable, "-m", "pip", "install", "--verbose", "--no-deps"]
    command += ["--no-build-isolation", "--no-cache-dir", "--disable-pip-version-check"]
    command += ["--config-settings", f"editable.rebuild-dir={installed}"]
    command += ["--upgrade", "--target", str(installed), "--editable", str(src)]
    # Run from the copy, so that no relative path leads back to the checkout.
    with log.open("w") as output:
        result = subprocess.run(command, cwd=src, env=env, stdout=output, stderr=output)
    if result.returncode != 0:
        tail = log.read_text(errors="replace").splitlines()[-200:]
        sys.exit("\n".join(tail) + f"\n\nPyTorch build failed; full log: {log}")


def collect(installed: Path, outputs: dict[str, str], files: dict[str, str]) -> None:
    # Some installed names carry a tag that a BUILD file cannot know:
    # _C.cpython-314-x86_64-linux-gnu.so, torch-2.16.0a0.dist-info.
    tagged = {".so": ".*.so", ".dist-info": "-*.dist-info"}
    for path, output in outputs.items():
        file = installed / path
        if pattern := tagged.get(file.suffix):
            file = next(installed.glob(path.removesuffix(file.suffix) + pattern), file)
        if not file.exists():
            sys.exit(f"the build did not install {path}")
        if os.path.isdir(output):
            os.rmdir(output)
        shutil.move(file, output)

    # Whatever is left must be a copy of a mirrored source file. Anything else
    # is a build product that no Bazel target would see.
    packages = [installed / name for name in ("torch", "torchgen", "functorch")]
    left = [f for package in packages for f in package.rglob("*") if f.is_file()]
    undeclared = sorted({str(f.relative_to(installed)) for f in left}.difference(files))
    if undeclared:
        names = "\n  ".join(undeclared[:50])
        sys.exit(f"installed files that are not declared as outputs:\n  {names}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcs", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--output", action="append", default=[])
    args = parser.parse_args()

    # The action runs in <output base>/execroot/<workspace>; keeping the state
    # in the output base makes `bazel clean --expunge` remove it.
    state = Path.cwd().parents[1] / "torch_native"
    src, installed = state / "src", state / "install"
    src.mkdir(parents=True, exist_ok=True)

    status_lines = args.status.read_text().splitlines()
    status = dict(line.split(" ", 1) for line in status_lines)
    # Bazel lists its inputs through links into the checkout.
    files = {f: os.path.realpath(f) for f in args.srcs.read_text().splitlines()}
    files |= tree(Path(status["STABLE_THIRD_PARTY_DIR"]))
    mirror(src, files)
    build(src, installed, state / "build.log")
    collect(installed, dict(item.split("=", 1) for item in args.output), files)


if __name__ == "__main__":
    main()
