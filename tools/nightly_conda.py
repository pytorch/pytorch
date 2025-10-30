#!/usr/bin/env python3
"""Install nightly PyTorch binaries and type stubs for type checking in non-git repos.

This is a simplified version of tools/nightly.py for use with conda environments
where you don't have a git repository or don't want to checkout specific commits.

Usage:
    python tools/nightly_conda.py --prefix ./conda_env
    python tools/nightly_conda.py --prefix ./conda_env --cuda 12.6
    python tools/nightly_conda.py --prefix ./conda_env --rocm 6.4
"""

from __future__ import annotations

import argparse
import itertools
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from platform import system as platform_system
from typing import Any, NamedTuple


REPO_ROOT = Path(__file__).absolute().parent.parent
PACKAGES_TO_INSTALL = (
    "torch",
    "numpy",
    "cmake",
    "ninja",
    "packaging",
    "ruff",
    "mypy",
    "pytest",
    "hypothesis",
    "ipython",
    "rich",
    "clang-format",
    "clang-tidy",
    "sphinx",
)

PLATFORM = platform_system().replace("Darwin", "macOS")
LINUX = PLATFORM == "Linux"
MACOS = PLATFORM == "macOS"
WINDOWS = PLATFORM == "Windows"


class PipSource(NamedTuple):
    name: str
    index_url: str
    supported_platforms: set[str]
    accelerator: str


PYTORCH_NIGHTLY_PIP_INDEX_URL = "https://download.pytorch.org/whl/nightly"
PIP_SOURCES = {
    "cpu": PipSource(
        name="cpu",
        index_url=f"{PYTORCH_NIGHTLY_PIP_INDEX_URL}/cpu",
        supported_platforms={"Linux", "macOS", "Windows"},
        accelerator="cpu",
    ),
    "cuda-12.6": PipSource(
        name="cuda-12.6",
        index_url=f"{PYTORCH_NIGHTLY_PIP_INDEX_URL}/cu126",
        supported_platforms={"Linux", "Windows"},
        accelerator="cuda",
    ),
    "cuda-12.8": PipSource(
        name="cuda-12.8",
        index_url=f"{PYTORCH_NIGHTLY_PIP_INDEX_URL}/cu128",
        supported_platforms={"Linux", "Windows"},
        accelerator="cuda",
    ),
    "cuda-13.0": PipSource(
        name="cuda-13.0",
        index_url=f"{PYTORCH_NIGHTLY_PIP_INDEX_URL}/cu130",
        supported_platforms={"Linux", "Windows"},
        accelerator="cuda",
    ),
    "rocm-6.4": PipSource(
        name="rocm-6.4",
        index_url=f"{PYTORCH_NIGHTLY_PIP_INDEX_URL}/rocm6.4",
        supported_platforms={"Linux"},
        accelerator="rocm",
    ),
    "rocm-7.0": PipSource(
        name="rocm-7.0",
        index_url=f"{PYTORCH_NIGHTLY_PIP_INDEX_URL}/rocm7.0",
        supported_platforms={"Linux"},
        accelerator="rocm",
    ),
}


def run_command(
    *args: str | Path, check: bool = True, capture_output: bool = False, **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    """Run a command with consistent settings."""
    cmd = [str(arg) for arg in args]
    print(f"Running: {shlex.join(cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        encoding="utf-8",
        capture_output=capture_output,
        **kwargs,
    )


def get_conda_python(prefix: Path) -> Path:
    """Get the Python executable in a conda environment."""
    if WINDOWS:
        python = prefix / "python.exe"
    else:
        python = prefix / "bin" / "python"

    if not python.exists():
        raise RuntimeError(f"Python executable not found at {python}")

    return python


def pip_download_wheel(
    python: Path,
    pip_source: PipSource,
    dest_dir: Path,
) -> Path:
    """Download the torch nightly wheel."""
    print(f"\nDownloading torch wheel from {pip_source.index_url}...")

    env = {
        **os.environ,
        "PIP_EXTRA_INDEX_URL": pip_source.index_url,
    }

    run_command(
        python,
        "-m",
        "pip",
        "download",
        "--pre",
        "--no-deps",
        f"--dest={dest_dir}",
        "torch",
        env=env,
    )

    wheels = list(dest_dir.glob("torch-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one torch wheel, got {wheels}")

    print(f"Downloaded: {wheels[0].name}")
    return wheels[0]


def unpack_wheel(python: Path, wheel: Path, dest: Path) -> Path:
    """Unpack a wheel file into a directory."""
    print(f"\nUnpacking wheel to {dest}...")

    # Install wheel package if needed
    run_command(python, "-m", "pip", "install", "--quiet", "wheel")

    run_command(python, "-m", "wheel", "unpack", f"--dest={dest}", str(wheel))

    subdirs = [p for p in dest.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(f"Expected exactly one directory in {dest}, got {subdirs}")

    return subdirs[0]


def get_listing_linux(source_dir: Path) -> list[Path]:
    """Get files to copy on Linux."""
    return list(
        itertools.chain(
            source_dir.glob("*.so"),
            (source_dir / "lib").glob("*.so"),
            (source_dir / "lib").glob("*.so.*"),
        )
    )


def get_listing_macos(source_dir: Path) -> list[Path]:
    """Get files to copy on macOS."""
    return list(
        itertools.chain(
            source_dir.glob("*.so"),
            (source_dir / "lib").glob("*.dylib"),
        )
    )


def get_listing_windows(source_dir: Path) -> list[Path]:
    """Get files to copy on Windows."""
    return list(
        itertools.chain(
            source_dir.glob("*.pyd"),
            (source_dir / "lib").glob("*.lib"),
            (source_dir / "lib").glob("*.dll"),
        )
    )


def glob_pyis(d: Path) -> set[str]:
    """Find all .pyi stub files in a directory."""
    return {p.relative_to(d).as_posix() for p in d.rglob("*.pyi")}


def find_missing_pyi(source_dir: Path, target_dir: Path) -> list[Path]:
    """Find .pyi files in source that are missing in target."""
    source_pyis = glob_pyis(source_dir)
    target_pyis = glob_pyis(target_dir)
    missing_pyis = sorted(source_dir / p for p in (source_pyis - target_pyis))
    return missing_pyis


def get_listing(source_dir: Path, target_dir: Path) -> list[Path]:
    """Get list of files to copy from wheel to repo."""
    if LINUX:
        listing = get_listing_linux(source_dir)
    elif MACOS:
        listing = get_listing_macos(source_dir)
    elif WINDOWS:
        listing = get_listing_windows(source_dir)
    else:
        raise RuntimeError(f"Platform {PLATFORM!r} not recognized")

    # Add .pyi stub files for type checking
    listing.extend(find_missing_pyi(source_dir, target_dir))

    # Add other important files
    listing.append(source_dir / "version.py")

    # Add directories if they exist
    for dir_name in ["bin", "include"]:
        dir_path = source_dir / dir_name
        if dir_path.exists():
            listing.append(dir_path)

    testing_generated = source_dir / "testing" / "_internal" / "generated"
    if testing_generated.exists():
        listing.append(testing_generated)

    return listing


def remove_existing(path: Path) -> None:
    """Remove a file or directory if it exists."""
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def copy_single(src: Path, source_dir: Path, target_dir: Path) -> None:
    """Copy a single file or directory."""
    relpath = src.relative_to(source_dir)
    trg = target_dir / relpath
    remove_existing(trg)

    if src.is_dir():
        trg.mkdir(parents=True, exist_ok=True)
        for root, dirs, files in os.walk(src):
            relroot = Path(root).relative_to(src)
            for name in files:
                relname = relroot / name
                s = src / relname
                t = trg / relname
                t.parent.mkdir(parents=True, exist_ok=True)
                print(f"  Copying {relname}")
                shutil.copy2(s, t)
            for name in dirs:
                (trg / relroot / name).mkdir(parents=True, exist_ok=True)
    else:
        print(f"  Copying {relpath}")
        trg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, trg)


def copy_files(listing: list[Path], source_dir: Path, target_dir: Path) -> None:
    """Copy files from wheel to repo."""
    print(f"\nCopying files to {target_dir}...")
    for src in listing:
        if src.exists():
            copy_single(src, source_dir, target_dir)


def parse_dependencies(python: Path, wheel_site_dir: Path) -> list[str]:
    """Parse dependencies from the torch wheel's metadata."""
    dist_info_dirs = list(wheel_site_dir.glob("*.dist-info"))
    if len(dist_info_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one .dist-info directory in {wheel_site_dir}, "
            f"got {dist_info_dirs}"
        )
    dist_info_dir = dist_info_dirs[0]
    if not (dist_info_dir / "METADATA").is_file():
        raise RuntimeError(
            f"Expected METADATA file in {dist_info_dir}, but it does not exist."
        )

    dependencies = (
        run_command(
            python,
            "-c",
            textwrap.dedent(
                """
                from packaging.metadata import Metadata

                with open("METADATA", encoding="utf-8") as f:
                    metadata = Metadata.from_email(f.read())
                for req in metadata.requires_dist:
                    if req.marker is None or req.marker.evaluate():
                        print(req)
                """
            ).strip(),
            cwd=dist_info_dir,
            capture_output=True,
        )
        .stdout.strip()
        .splitlines()
    )
    return [dep.strip() for dep in dependencies]


def install_dependencies(
    python: Path, dependencies: list[str], packages: list[str]
) -> None:
    """Install dependencies and additional packages."""
    all_packages = [p for p in packages if p != "torch"] + dependencies
    all_packages = list(dict.fromkeys(all_packages))  # Remove duplicates

    if all_packages:
        print(f"\nInstalling {len(all_packages)} packages...")
        run_command(python, "-m", "pip", "install", *all_packages)


def write_pth_file(python: Path) -> None:
    """Write .pth file to make local torch importable."""
    # Get site-packages directory
    output = run_command(
        python,
        "-c",
        "import site; print(site.getsitepackages()[0])",
        capture_output=True,
    ).stdout.strip()

    site_packages = Path(output)
    pth_file = site_packages / "pytorch-nightly.pth"

    print(f"\nWriting .pth file to {pth_file}...")
    pth_file.write_text(
        "# This file was autogenerated by PyTorch's tools/nightly_conda.py\n"
        "# Please delete this file if you no longer need the following development\n"
        "# version of PyTorch to be importable\n"
        f"{REPO_ROOT}\n",
        encoding="utf-8",
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Install PyTorch nightly binaries and type stubs for conda environments"
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=Path,
        help="Path to conda environment directory",
        required=True,
    )
    parser.add_argument(
        "--cuda",
        help="CUDA version to install (e.g., 12.6)",
        default=None,
    )
    parser.add_argument(
        "--rocm",
        help="ROCm version to install (e.g., 6.4)",
        default=None,
    )

    args = parser.parse_args()

    if args.cuda and args.rocm:
        parser.error("Cannot specify both --cuda and --rocm")

    prefix = args.prefix.absolute()
    if not prefix.exists():
        print(f"Error: Conda environment not found at {prefix}")
        print("Please create it first with: conda create -p <prefix>")
        sys.exit(1)

    # Determine pip source
    pip_source = PIP_SOURCES["cpu"]  # default

    if args.cuda:
        source_name = f"cuda-{args.cuda}"
        if source_name in PIP_SOURCES:
            pip_source = PIP_SOURCES[source_name]
        else:
            print(f"Error: CUDA version {args.cuda} not available")
            print(
                f"Available versions: {[k.replace('cuda-', '') for k in PIP_SOURCES if k.startswith('cuda-')]}"
            )
            sys.exit(1)
    elif args.rocm:
        source_name = f"rocm-{args.rocm}"
        if source_name in PIP_SOURCES:
            pip_source = PIP_SOURCES[source_name]
        else:
            print(f"Error: ROCm version {args.rocm} not available")
            print(
                f"Available versions: {[k.replace('rocm-', '') for k in PIP_SOURCES if k.startswith('rocm-')]}"
            )
            sys.exit(1)

    if PLATFORM not in pip_source.supported_platforms:
        print(f"Error: {pip_source.name} not supported on {PLATFORM}")
        sys.exit(1)

    print(f"Using pip source: {pip_source.name} ({pip_source.index_url})")

    # Get Python executable
    python = get_conda_python(prefix)
    print(f"Using Python: {python}")

    # Create temp directory for wheel download and extraction
    with tempfile.TemporaryDirectory(prefix="torch-wheel-") as tmpdir:
        tmppath = Path(tmpdir)

        # Download wheel
        wheel = pip_download_wheel(python, pip_source, tmppath)

        # Unpack wheel
        unpacked = unpack_wheel(python, wheel, tmppath / "unpacked")
        wheel_site_dir = unpacked

        # Parse dependencies
        print("\nParsing dependencies...")
        dependencies = parse_dependencies(python, wheel_site_dir)
        print(f"Found {len(dependencies)} dependencies")

        # Install dependencies
        install_dependencies(python, dependencies, list(PACKAGES_TO_INSTALL))

        # Copy files to repo
        source_dir = wheel_site_dir / "torch"
        target_dir = REPO_ROOT / "torch"
        listing = get_listing(source_dir, target_dir)
        copy_files(listing, source_dir, target_dir)

        # Write .pth file
        write_pth_file(python)

    print("\n" + "=" * 70)
    print("PyTorch nightly binaries installed successfully!")
    print("=" * 70)
    print("\nActivate your environment with:")
    print(f"  conda activate {prefix}")


if __name__ == "__main__":
    main()
