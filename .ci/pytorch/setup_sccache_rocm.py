#!/usr/bin/env python3

import argparse
import os
import shutil
import stat
from pathlib import Path


BACKUP_DIR = "backup_for_sccache"


def find_llvm_bin(rocm_path: Path) -> Path:
    for candidate in (
        rocm_path / "lib" / "llvm" / "bin",
        rocm_path / "llvm" / "bin",
    ):
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"Could not find the ROCm LLVM bin directory under {rocm_path}")


def wrap_compiler(compiler: Path, sccache: Path) -> None:
    backup_dir = compiler.parent / BACKUP_DIR
    backup_dir.mkdir(exist_ok=True)
    metadata = backup_dir / f"{compiler.name}.path"
    backup_binary = backup_dir / compiler.name
    if metadata.exists():
        raise RuntimeError(f"{compiler} is already wrapped")

    real_compiler = compiler.resolve(strict=True)
    if compiler.is_symlink():
        metadata.write_text(f"symlink:{os.readlink(compiler)}")
        compiler.unlink()
        wrapped_compiler = real_compiler
    else:
        metadata.write_text("binary")
        shutil.move(compiler, backup_binary)
        wrapped_compiler = backup_binary

    compiler.write_text(
        f'#!/bin/sh\nexec "{sccache}" "{wrapped_compiler}" "$@"\n'
    )
    compiler.chmod(
        stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )


def restore_compiler(compiler: Path) -> None:
    backup_dir = compiler.parent / BACKUP_DIR
    metadata = backup_dir / f"{compiler.name}.path"
    if not metadata.exists():
        return

    path_info = metadata.read_text().strip()
    compiler.unlink(missing_ok=True)
    if path_info.startswith("symlink:"):
        compiler.symlink_to(path_info.removeprefix("symlink:"))
    elif path_info == "binary":
        shutil.move(backup_dir / compiler.name, compiler)
    else:
        raise RuntimeError(f"Invalid backup metadata for {compiler}: {path_info}")

    metadata.unlink()
    try:
        backup_dir.rmdir()
    except OSError:
        pass


def setup(rocm_path: Path, sccache: Path) -> None:
    llvm_bin = find_llvm_bin(rocm_path)
    wrapped: list[Path] = []
    try:
        for name in ("clang", "clang++"):
            compiler = llvm_bin / name
            wrap_compiler(compiler, sccache)
            wrapped.append(compiler)
    except Exception:
        for compiler in reversed(wrapped):
            restore_compiler(compiler)
        raise


def restore(rocm_path: Path) -> None:
    llvm_bin = find_llvm_bin(rocm_path)
    for name in ("clang", "clang++"):
        restore_compiler(llvm_bin / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocm-path", type=Path, required=True)
    parser.add_argument("--sccache-path", type=Path)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if args.restore:
        restore(args.rocm_path)
    else:
        if args.sccache_path is None:
            raise RuntimeError("--sccache-path is required when setting up wrappers")
        setup(args.rocm_path, args.sccache_path.resolve(strict=True))


if __name__ == "__main__":
    main()
