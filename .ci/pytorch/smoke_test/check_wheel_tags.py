"""Validate wheel platform tags and macOS dylib minos.
Supports two modes:
1. Pre-install: reads .whl files from PYTORCH_FINAL_PACKAGE_DIR
2. Post-install: reads metadata from installed torch package (soft warnings)
- (macOS only) dylib minos matches the wheel platform tag
"""

import os
import platform
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


EXPECTED_PLATFORM_TAGS: dict[str, str] = {
    "linux": r"_x86_64$",
    "linux-aarch64": r"_aarch64$",
    "windows": r"^win_amd64$",
    "win32": r"^win_amd64$",
    "macos-arm64": r"^macosx_\d+_\d+_arm64$",
    "darwin": r"^macosx_\d+_\d+_(arm64|x86_64)$",
}


def _extract_wheel_tags(whl_path: Path) -> list[str]:
    """Extract Tag values from the WHEEL metadata file inside a .whl archive."""
    tags = []
    with zipfile.ZipFile(whl_path, "r") as zf:
        wheel_files = [n for n in zf.namelist() if n.endswith("/WHEEL")]
        if not wheel_files:
            return tags
        content = zf.read(wheel_files[0]).decode("utf-8")
        for line in content.splitlines():
            if line.startswith("Tag:"):
                tags.append(line.split(":", 1)[1].strip())
    return tags


def _extract_installed_wheel_tags(package: str = "torch") -> list[str]:
    """Extract Tag values from an installed package's WHEEL metadata."""
    from importlib.metadata import distribution

    dist = distribution(package)
    wheel_text = dist.read_text("WHEEL")
    if not wheel_text:
        return []
    tags = []
    for line in wheel_text.splitlines():
        if line.startswith("Tag:"):
            tags.append(line.split(":", 1)[1].strip())
    return tags


def check_wheel_platform_tag() -> None:
    """Validate that wheel Tags in WHEEL metadata match the expected platform.

    Mode 1: PYTORCH_FINAL_PACKAGE_DIR set → read .whl file (strict, raises on mismatch)
    Mode 2: No wheel dir → read from installed torch package (soft, prints warnings)
    """
    wheel_dir = os.getenv("PYTORCH_FINAL_PACKAGE_DIR", "")

    target_os = os.getenv("TARGET_OS", sys.platform)
    if target_os == "linux" and platform.machine() == "aarch64":
        target_os = "linux-aarch64"
    expected_python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    import sysconfig

    abiflags = getattr(sys, "abiflags", "")
    if not abiflags and (
        os.getenv("MATRIX_PYTHON_VERSION", "").endswith("t")
        or bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        or not getattr(sys, "_is_gil_enabled", lambda: True)()
    ):
        abiflags = "t"
    expected_abi = f"cp{sys.version_info.major}{sys.version_info.minor}{abiflags}"
    print(f"Expected ABI tag: {expected_abi}")

    platform_pattern = EXPECTED_PLATFORM_TAGS.get(target_os)
    if not platform_pattern:
        print(
            f"No expected platform pattern for TARGET_OS={target_os}, "
            "skipping wheel tag check"
        )
        return

    # Mode 1: Read from .whl file
    if wheel_dir and os.path.isdir(wheel_dir):
        whls = list(Path(wheel_dir).glob("torch-*.whl"))
        if not whls:
            print(f"No torch wheel found in {wheel_dir}, skipping wheel tag check")
            return
        if len(whls) > 1:
            raise RuntimeError(
                f"Expected exactly one torch wheel in {wheel_dir}, "
                f"found {len(whls)}: {[w.name for w in whls]}"
            )
        whl = whls[0]
        print(f"Checking wheel platform tag for: {whl.name}")
        tags = _extract_wheel_tags(whl)
        source = whl.name
    else:
        # Mode 2: Read from installed package (soft)
        print("PYTORCH_FINAL_PACKAGE_DIR not set, reading from installed torch package")
        try:
            tags = _extract_installed_wheel_tags("torch")
            source = "installed torch"
        except Exception as e:
            print(f"Could not read installed torch metadata: {e}, skipping")
            return

    if not tags:
        raise RuntimeError(f"No Tag found in WHEEL metadata of {source}")

    for tag_str in tags:
        parts = tag_str.split("-")
        if len(parts) != 3:
            msg = (
                f"Malformed wheel tag '{tag_str}' in {source}, "
                f"expected format: <python>-<abi>-<platform>"
            )
            raise RuntimeError(msg)

        python_tag, abi_tag, platform_tag = parts

        print(f"Checking tag: {tag_str} (from {source})")
        if python_tag != expected_python:
            msg: str = (
                f"Python tag mismatch in {source}: "
                f"got '{python_tag}', expected '{expected_python}'"
            )
            raise RuntimeError(msg)

        if abi_tag != expected_abi:
            msg = (
                f"ABI tag mismatch in {source}: "
                f"got '{abi_tag}', expected '{expected_abi}'"
            )
            raise RuntimeError(msg)

        if not re.search(platform_pattern, platform_tag):
            msg = (
                f"Platform tag mismatch in {source}: "
                f"got '{platform_tag}', expected pattern matching "
                f"'{platform_pattern}' for TARGET_OS={target_os}"
            )
            raise RuntimeError(msg)

    print(f"OK: Wheel tag(s) valid for {source}: {', '.join(tags)}")


def _check_dylibs_minos(dylibs: list, expected_minos: str, source: str) -> None:
    mismatches = []
    for dylib in dylibs:
        try:
            result = subprocess.run(
                ["otool", "-l", str(dylib)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            continue

        minos = None
        lines = result.stdout.splitlines()
        for i, line in enumerate(lines):
            s = line.strip()
            if "LC_BUILD_VERSION" in s:
                for j in range(i + 1, min(i + 6, len(lines))):
                    if lines[j].strip().startswith("minos"):
                        minos = lines[j].strip().split()[1]
                        break
                break
            if "LC_VERSION_MIN_MACOSX" in s:
                for j in range(i + 1, min(i + 4, len(lines))):
                    if lines[j].strip().startswith("version"):
                        minos = lines[j].strip().split()[1]
                        break
                break

        # A dylib with a lower minos than the wheel tag is safe (forward compatible).
        # Only flag dylibs that require a *higher* macOS than the wheel claims to support.
        if minos and tuple(int(x) for x in minos.split(".")) > tuple(
            int(x) for x in expected_minos.split(".")
        ):
            mismatches.append(
                f"{dylib.name}: minos={minos}, expected<={expected_minos}"
            )

    if mismatches:
        raise RuntimeError(
            f"minos/platform tag mismatch in {len(mismatches)} dylib(s):\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )
    print(
        f"OK: All {len(dylibs)} dylib(s) have minos matching "
        f"platform tag ({expected_minos}) for {source}"
    )


def check_mac_wheel_minos() -> None:
    if sys.platform != "darwin":
        return

    wheel_dir = os.getenv("PYTORCH_FINAL_PACKAGE_DIR", "")

    if wheel_dir and os.path.isdir(wheel_dir):
        # Mode 1: extract dylibs from .whl file
        whls = list(Path(wheel_dir).glob("*.whl"))
        if not whls:
            print(f"No .whl files in {wheel_dir}, skipping wheel minos check")
            return

        macos_whl_re = re.compile(r"macosx_(\d+)_(\d+)_(\w+)\.whl$")
        for whl in whls:
            print(f"Checking wheel tag minos for: {whl.name}")
            m = macos_whl_re.search(whl.name)
            if not m:
                print(f"No macOS platform tag in {whl.name}, skipping")
                continue
            expected_minos = f"{m.group(1)}.{m.group(2)}"

            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(whl, "r") as zf:
                    dylib_names = [n for n in zf.namelist() if n.endswith(".dylib")]
                    if not dylib_names:
                        print("No .dylib files in wheel, skipping minos check")
                        continue
                    for name in dylib_names:
                        zf.extract(name, tmpdir)
                dylibs = list(Path(tmpdir).rglob("*.dylib"))
                _check_dylibs_minos(dylibs, expected_minos, whl.name)
    else:
        # Mode 2: read from installed torch package
        print("PYTORCH_FINAL_PACKAGE_DIR not set, checking installed torch dylibs")
        try:
            tags = _extract_installed_wheel_tags("torch")
        except Exception as e:
            print(f"Could not read installed torch metadata: {e}, skipping")
            return

        expected_minos = None
        for tag_str in tags:
            m = re.search(r"macosx_(\d+)_(\d+)_\w+", tag_str)
            if m:
                expected_minos = f"{m.group(1)}.{m.group(2)}"
                break

        if not expected_minos:
            print("No macOS platform tag found in installed torch metadata, skipping")
            return

        print(f"Expected minos from installed wheel tag: {expected_minos}")

        import torch

        torch_dir = Path(torch.__file__).parent
        dylibs = list(torch_dir.rglob("*.dylib"))
        if not dylibs:
            raise RuntimeError("No .dylib files found in installed torch")
        _check_dylibs_minos(dylibs, expected_minos, "installed torch")


if __name__ == "__main__":
    check_wheel_platform_tag()
    check_mac_wheel_minos()
