#!/usr/bin/env python3
"""Extract libtorch package from a PyTorch wheel.

Creates a libtorch zip from a pre-built wheel by copying the C++ libraries,
headers, and CMake files. On Linux, optionally splits debug symbols from
libtorch_cpu.so into a separate debug zip.

Usage:
    python extract_libtorch_from_wheel.py \
        --wheel-dir DIR --output-dir DIR --platform linux|macos|windows
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def find_wheel(wheel_dir: str) -> Path:
    wheels = glob.glob(os.path.join(wheel_dir, "*.whl"))
    if not wheels:
        raise FileNotFoundError(f"No .whl files found in {wheel_dir}")
    if len(wheels) > 1:
        raise RuntimeError(f"Multiple .whl files found in {wheel_dir}: {wheels}")
    return Path(wheels[0])


def parse_version_from_wheel(wheel_path: Path) -> str:
    # Wheel filename format: {name}-{version}(-{build})?-{python}-{abi}-{platform}.whl
    name = wheel_path.stem
    parts = name.split("-")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse version from wheel filename: {wheel_path.name}")
    return parts[1]


def extract_wheel(wheel_path: Path, extract_dir: Path) -> Path:
    with zipfile.ZipFile(wheel_path, "r") as zf:
        zf.extractall(extract_dir)
    # Find the torch directory
    torch_dir = extract_dir / "torch"
    if not torch_dir.is_dir():
        raise FileNotFoundError(
            f"No 'torch' directory found in extracted wheel at {extract_dir}"
        )
    return torch_dir


def should_exclude_lib(filename: str) -> bool:
    """Return True for files that should not go into the libtorch package."""
    if filename.startswith("libtorch_python"):
        return True
    if re.match(r"_C\.cpython.*", filename):
        return True
    if filename.endswith((".py", ".pyc")):
        return True
    if filename == "__init__.py":
        return True
    return False


def _is_lib_file(name: str, platform: str) -> bool:
    """Return True if the file looks like a library or header to include."""
    if platform == "linux":
        return ".so" in name or name.endswith(".a")
    elif platform == "macos":
        return name.endswith((".dylib", ".a"))
    elif platform == "windows":
        return name.endswith((".dll", ".lib", ".pdb"))
    return False


def copy_libraries(torch_dir: Path, libtorch_lib: Path, platform: str) -> None:
    """Copy libraries from torch/lib/ to libtorch/lib/."""
    torch_lib = torch_dir / "lib"
    if not torch_lib.is_dir():
        raise FileNotFoundError(f"torch/lib/ not found at {torch_lib}")

    for item in torch_lib.iterdir():
        if item.is_dir():
            # Copy subdirectories (e.g. libshm/) as-is
            shutil.copytree(item, libtorch_lib / item.name, dirs_exist_ok=True)
            continue
        if should_exclude_lib(item.name):
            continue
        if _is_lib_file(item.name, platform):
            shutil.copy2(item, libtorch_lib / item.name)

    # On macOS, also copy delocated dylibs from torch/.dylibs/ if present
    if platform == "macos":
        dylibs_dir = torch_dir / ".dylibs"
        if dylibs_dir.is_dir():
            for item in dylibs_dir.iterdir():
                if item.suffix == ".dylib" and not should_exclude_lib(item.name):
                    shutil.copy2(item, libtorch_lib / item.name)


def copy_includes(torch_dir: Path, libtorch_include: Path) -> None:
    torch_include = torch_dir / "include"
    if not torch_include.is_dir():
        # Some older wheels might have include under torch/lib/include
        torch_include = torch_dir / "lib" / "include"
    if not torch_include.is_dir():
        raise FileNotFoundError("include/ not found in torch directory")
    shutil.copytree(torch_include, libtorch_include, dirs_exist_ok=True)


def copy_cmake(torch_dir: Path, libtorch_share: Path) -> None:
    torch_cmake = torch_dir / "share" / "cmake"
    if not torch_cmake.is_dir():
        print(f"Warning: share/cmake/ not found at {torch_cmake}", file=sys.stderr)
        return
    cmake_dest = libtorch_share / "cmake"
    shutil.copytree(torch_cmake, cmake_dest, dirs_exist_ok=True)


def copy_bin(torch_dir: Path, libtorch_bin: Path, platform: str) -> None:
    """Copy binary executables (mainly relevant for Windows)."""
    if platform == "windows":
        torch_lib = torch_dir / "lib"
        if torch_lib.is_dir():
            for item in torch_lib.iterdir():
                if item.suffix == ".dll" and not should_exclude_lib(item.name):
                    shutil.copy2(item, libtorch_bin / item.name)


def write_metadata(libtorch_dir: Path, version: str, git_hash: str) -> None:
    (libtorch_dir / "build-version").write_text(version + "\n")
    (libtorch_dir / "build-hash").write_text(git_hash + "\n")


def get_git_hash(torch_dir: Path) -> str:
    """Read git_version from the wheel's torch/version.py."""
    version_file = torch_dir / "version.py"
    if not version_file.exists():
        return "unknown"
    from ast import literal_eval

    for line in version_file.read_text().splitlines():
        if line.strip().startswith("git_version"):
            try:
                return literal_eval(line.partition("=")[2].strip())
            except Exception:
                pass
    return "unknown"


def split_debug_symbols(
    libtorch_dir: Path, output_dir: Path, zip_prefix: str, version: str
) -> None:
    """Split debug symbols from libtorch_cpu.so (Linux only)."""
    libtorch_cpu = libtorch_dir / "lib" / "libtorch_cpu.so"
    if not libtorch_cpu.exists():
        print(
            "Warning: libtorch_cpu.so not found, skipping debug symbol split",
            file=sys.stderr,
        )
        return

    debug_dir = libtorch_dir.parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    dbg_file = debug_dir / "libtorch_cpu.so.dbg"

    # Copy to create debug file
    shutil.copy2(libtorch_cpu, dbg_file)

    # Keep only debug symbols
    subprocess.run(
        ["strip", "--only-keep-debug", str(dbg_file)],
        check=True,
    )

    # Strip debug info from release lib
    subprocess.run(
        ["strip", "--strip-debug", str(libtorch_cpu)],
        check=True,
    )

    # Add debug link
    subprocess.run(
        ["objcopy", str(libtorch_cpu), f"--add-gnu-debuglink={dbg_file}"],
        check=True,
        cwd=str(libtorch_dir / "lib"),
    )

    # Extract CRC32 from the debug link section
    try:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"objcopy --dump-section .gnu_debuglink=>(tail -c4 | od -t x4 -An | xargs echo) {libtorch_cpu}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        crc32 = result.stdout.strip()
    except subprocess.CalledProcessError:
        crc32 = "unknown"

    # Create debug zip
    debug_zip = output_dir / f"debug-{zip_prefix}-{version}-{crc32}.zip"
    with zipfile.ZipFile(debug_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(dbg_file, "debug/libtorch_cpu.so.dbg")

    print(f"Debug symbols zip: {debug_zip}")


def create_libtorch_zip(
    libtorch_dir: Path,
    output_dir: Path,
    zip_prefix: str,
    version: str,
) -> Path:
    zip_path = output_dir / f"{zip_prefix}-{version}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(libtorch_dir):
            for f in files:
                filepath = Path(root) / f
                arcname = filepath.relative_to(libtorch_dir.parent)
                zf.write(filepath, arcname)
    # Create latest symlink
    latest_zip = output_dir / f"{zip_prefix}-latest.zip"
    latest_zip.symlink_to(zip_path.name)

    print(f"Libtorch zip: {zip_path}")
    print(f"Libtorch latest zip: {latest_zip}")
    return zip_path


def compute_zip_prefix(platform: str, desired_cuda: str, libtorch_variant: str) -> str:
    """Compute the zip filename prefix matching existing naming conventions.

    Linux:  libtorch-shared-with-deps
    macOS:  libtorch-macos-arm64
    Windows: libtorch-win-shared-with-deps (or libtorch-win-arm64-shared-with-deps)
    """
    if platform == "macos":
        return "libtorch-macos-arm64"
    elif platform == "windows":
        return f"libtorch-win-{libtorch_variant}"
    else:
        return f"libtorch-{libtorch_variant}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract libtorch from PyTorch wheel")
    parser.add_argument(
        "--wheel-dir", required=True, help="Directory containing the .whl file"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for output zip files"
    )
    parser.add_argument(
        "--platform",
        required=True,
        choices=["linux", "macos", "windows"],
        help="Target platform",
    )
    parser.add_argument(
        "--desired-cuda",
        default="cpu",
        help="CUDA variant (cpu, cu126, cu128, rocm7.1, etc.)",
    )
    parser.add_argument(
        "--libtorch-variant",
        default="shared-with-deps",
        help="Libtorch variant (shared-with-deps, etc.)",
    )
    parser.add_argument(
        "--git-hash",
        default="",
        help="Git hash to use for build-hash (auto-detected if not set)",
    )
    args = parser.parse_args()

    wheel_dir = Path(args.wheel_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find and extract wheel
    wheel_path = find_wheel(str(wheel_dir))
    version = parse_version_from_wheel(wheel_path)
    print(f"Found wheel: {wheel_path}")
    print(f"Version: {version}")

    extract_dir = wheel_dir / "_extract_tmp"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()

    try:
        torch_dir = extract_wheel(wheel_path, extract_dir)

        # Create libtorch directory structure
        libtorch_dir = extract_dir / "libtorch"
        libtorch_dir.mkdir()
        for subdir in ["lib", "bin", "include", "share"]:
            (libtorch_dir / subdir).mkdir()

        # Copy components
        copy_libraries(torch_dir, libtorch_dir / "lib", args.platform)
        copy_includes(torch_dir, libtorch_dir / "include")
        copy_cmake(torch_dir, libtorch_dir / "share")
        copy_bin(torch_dir, libtorch_dir / "bin", args.platform)

        # Write metadata
        git_hash = args.git_hash or get_git_hash(torch_dir)
        write_metadata(libtorch_dir, version, git_hash)

        # Compute zip prefix
        zip_prefix = compute_zip_prefix(
            args.platform, args.desired_cuda, args.libtorch_variant
        )

        # Split debug symbols on Linux
        if args.platform == "linux":
            split_debug_symbols(libtorch_dir, output_dir, zip_prefix, version)

        # Create the zip
        create_libtorch_zip(libtorch_dir, output_dir, zip_prefix, version)

    finally:
        shutil.rmtree(extract_dir)


if __name__ == "__main__":
    main()
