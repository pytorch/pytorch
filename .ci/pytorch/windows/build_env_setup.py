#!/usr/bin/env python3
"""GPU/toolchain environment setup for Windows manywheel/wheel CD builds.

Mirrors the Linux `.ci/manywheel/build_env_setup.py` pattern: install heavy
toolchain pieces (CUDA, MAGMA, MSVC env via vcvarsall, oneAPI for XPU),
then emit `export KEY=VALUE` lines to the file given by --env-out for the
parent bash wrapper to source. Without that handoff, the env we configure
here would die with this process.

Environment variables expected:
    GPU_ARCH_TYPE      - cpu, cuda, xpu (read from DESIRED_CUDA fallback)
    DESIRED_CUDA       - cpu, cu126, cu128, ..., xpu
    CUDA_VERSION       - 126, 128, 129, 130, 132 (CUDA builds only)
    VSDEVCMD_ARGS      - extra args for vcvarsall.bat (optional)
    VC_YEAR            - 2022 (or 2019)
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


# Directory containing this script (.ci/pytorch/windows). Used as the root
# for tmp_bin/, magma_*/, and the internal/*.bat installer scripts -- matches
# the legacy `%~dp0` convention so the installers' own path math keeps
# working.
WIN_CI_DIR = Path(__file__).resolve().parent


# Common env applied to every Windows wheel build. Mirrors the legacy
# build_pytorch.bat + internal/check_deps.bat.
COMMON_BUILD_ENV: dict[str, str] = {
    "PYTORCH_BINARY_BUILD": "1",
    "TH_BINARY_BUILD": "1",
    "INSTALL_TEST": "0",
    "MSSdk": "1",
    "DISTUTILS_USE_SDK": "1",
}


CPU_BUILD_ENV: dict[str, str] = {
    "USE_CUDA": "0",
}


# CUDA build flags shared across all CUDA versions. Mirrors the
# unconditional sets in cuda_config.bat plus the wheel-build defaults.
CUDA_BUILD_ENV_STATIC: dict[str, str] = {
    "USE_CUDA": "1",
    "TORCH_NVCC_FLAGS": "-Xfatbin -compress-all",
    "CMAKE_GENERATOR": "Ninja",
}


# Per-CUDA TORCH_CUDA_ARCH_LIST -- kept in sync with the CUDA_ARCH_LIST
# values in .ci/pytorch/windows/internal/cuda_config.bat. CUDA 13.x dropped
# sm_50/60/70, so the list cannot be left empty (CMake's defaults still
# include compute_50 which nvcc 13 rejects).
TORCH_CUDA_ARCH_LIST_TABLE: dict[str, str] = {
    "126": "5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0",
    "128": "7.5;8.0;8.6;9.0;10.0;12.0",
    "129": "7.5;8.0;8.6;9.0;10.0;12.0",
    "130": "7.5;8.0;8.6;9.0;10.0;12.0",
    "132": "7.5;8.0;8.6;9.0;10.0;12.0",
}


MAGMA_VERSION = "2.5.4"
MAGMA_URL_TEMPLATE = (
    "https://s3.amazonaws.com/ossci-windows/magma_{version}_{prefix}_{build_type}.7z"
)
SCCACHE_URL = "https://s3.amazonaws.com/ossci-windows/sccache.exe"
SCCACHE_CL_URL = "https://s3.amazonaws.com/ossci-windows/sccache-cl.exe"
RANDOMTEMP_URL = "https://github.com/peterjc123/randomtemp-rust/releases/download/v0.4/randomtemp.exe"


def find_vcvarsall(vc_year: str) -> Path:
    """Locate vcvarsall.bat via vswhere.exe.

    Mirrors the discovery in torch.utils.cpp_extension._jit._get_vc_env
    (gh-180707). Falls back to %VS15VCVARSALL% / %VS15INSTALLDIR% for parity
    with internal/check_deps.bat.
    """
    pre_set = os.environ.get("VS15VCVARSALL")
    if pre_set and Path(pre_set).is_file():
        return Path(pre_set)

    program_files = os.environ.get(
        "ProgramFiles(x86)",
        os.environ.get("ProgramFiles", r"C:\Program Files (x86)"),
    )
    vswhere = (
        Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    )
    if not vswhere.is_file():
        sys.exit(
            f"vswhere.exe not found at {vswhere}; Visual Studio {vc_year} "
            "C++ BuildTools is required to compile PyTorch on Windows"
        )

    vc_lower, vc_upper = ("16", "17") if vc_year == "2019" else ("17", "18")
    output = subprocess.check_output(
        [
            str(vswhere),
            "-legacy",
            "-products",
            "*",
            "-version",
            f"[{vc_lower},{vc_upper})",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    for line in output.splitlines():
        candidate = Path(line.strip()) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        if candidate.is_file():
            return candidate
    sys.exit(
        f"Visual Studio {vc_year} C++ BuildTools is required to compile PyTorch on Windows"
    )


def capture_vcvars_env(vcvarsall: Path, args: str) -> dict[str, str]:
    """Capture the env diff produced by `vcvarsall.bat <args>`.

    Mirrors torch.utils.cpp_extension._jit._get_vc_env (gh-180707). The
    `cmd /u /c` flag forces UTF-16LE output so non-ASCII paths in localized
    Windows installs round-trip intact. We then diff against the current
    environment so we only propagate variables vcvarsall actually changed.

    Note: the upstream _get_vc_env short-circuits on DISTUTILS_USE_SDK to
    pass through a pre-configured caller env. In CI we always run vcvarsall
    (and set DISTUTILS_USE_SDK ourselves as part of COMMON_BUILD_ENV), so we
    intentionally skip that check.
    """
    try:
        raw = subprocess.check_output(
            f'cmd /u /c "{vcvarsall}" {args} && set',
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(
            f"vcvarsall.bat {args} failed:\n"
            f"{exc.output.decode('utf-16le', errors='replace')}"
        )
    text = raw.decode("utf-16le", errors="replace")

    new_env: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        # Skip cmd's `=C:` / `=ExitCode` / `=::` pseudo-vars and banner lines.
        if not sep or not key or key.startswith("="):
            continue
        new_env[key] = value

    old_env = os.environ
    return {k: v for k, v in new_env.items() if old_env.get(k) != v}


def download(url: str, dest: Path, attempts: int = 5) -> None:
    """Stream `url` to `dest`, retrying with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading {url} -> {dest} (attempt {attempt}/{attempts})")
            with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            return
        except Exception as exc:
            if attempt == attempts:
                sys.exit(f"Failed to download {url}: {exc}")
            time.sleep(2**attempt)


def install_cuda_toolkit() -> None:
    """Run the legacy internal/cuda_install.bat which downloads + installs
    the CUDA toolkit, cuDNN, ZLIB, and the GPU driver DLLs.

    Kept as a .bat invocation rather than ported to Python: the installer
    logic is heavy (multiple version branches, NvToolsExt staging,
    setup.exe orchestration with Windows-only side effects) and the
    same pragmatic precedent exists in the Linux refactor for
    install_cuda.sh and friends.
    """
    bat = WIN_CI_DIR / "internal" / "cuda_install.bat"
    subprocess.run(["cmd", "/c", str(bat)], cwd=WIN_CI_DIR, check=True)


def find_cuda_path(dotted_version: str) -> Path:
    """Find the CUDA install root for the requested dotted version.

    Mirrors cuda_config.bat: respect CUDA_PATH_V{ver} if set, else fall
    back to the standard install location.
    """
    cuda_ver_key = f"CUDA_PATH_V{dotted_version.replace('.', '')}"
    candidate = os.environ.get(cuda_ver_key)
    if candidate:
        path = Path(candidate)
        if (path / "bin" / "nvcc.exe").is_file():
            return path
    standard = Path(
        rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{dotted_version}"
    )
    if (standard / "bin" / "nvcc.exe").is_file():
        return standard
    sys.exit(f"CUDA {dotted_version} not found at {standard} or via {cuda_ver_key}")


def find_nvtoolsext() -> Path:
    """Mirror internal/check_nvtx.bat. NVTX is required for CUDA wheels."""
    pre_set = os.environ.get("NVTOOLSEXT_PATH")
    if pre_set and (Path(pre_set) / "lib" / "x64" / "nvToolsExt64_1.lib").is_file():
        return Path(pre_set)
    standard = Path(r"C:\Program Files\NVIDIA Corporation\NvToolsExt")
    if (standard / "lib" / "x64" / "nvToolsExt64_1.lib").is_file():
        return standard
    sys.exit(
        "NVTX (Visual Studio Extension for CUDA) not installed; "
        f"checked {standard} and %NVTOOLSEXT_PATH%"
    )


def install_magma(cuda_prefix: str, build_type: str) -> Path:
    """Download + extract the pre-built MAGMA archive into WIN_CI_DIR.

    Mirrors the MAGMA block in the legacy build_pytorch.bat. Returns the
    extracted MAGMA root (set as MAGMA_HOME for the wheel build).
    """
    name = f"magma_{cuda_prefix}_{build_type}"
    archive = WIN_CI_DIR / f"{name}.7z"
    target = WIN_CI_DIR / name
    if target.is_dir():
        # Stale extract from a previous run; clear before re-downloading.
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(target)], check=False)
    if archive.is_file():
        archive.unlink()

    url = MAGMA_URL_TEMPLATE.format(
        version=MAGMA_VERSION, prefix=cuda_prefix, build_type=build_type
    )
    download(url, archive)
    subprocess.run(
        ["7z", "x", "-aoa", str(archive), f"-o{target}"], cwd=WIN_CI_DIR, check=True
    )
    return target


def install_sccache_binaries(tmp_bin: Path) -> None:
    """Download sccache, sccache-cl, and the randomtemp NVCC wrapper.

    randomtemp resolves an intermittent NVCC + sccache race that causes
    spurious CUDA build failures; see gh-25393.
    """
    tmp_bin.mkdir(parents=True, exist_ok=True)
    download(SCCACHE_URL, tmp_bin / "sccache.exe")
    download(SCCACHE_CL_URL, tmp_bin / "sccache-cl.exe")
    download(RANDOMTEMP_URL, tmp_bin / "randomtemp.exe")


def setup_nvcc_wrapper(tmp_bin: Path, cuda_path: Path) -> dict[str, str]:
    """Write the randomtemp+sccache NVCC wrapper batch file and return the
    env that points CMake's CUDA rule at it.
    """
    nvcc_bat = tmp_bin / "nvcc.bat"
    randomtemp = tmp_bin / "randomtemp.exe"
    sccache = tmp_bin / "sccache.exe"
    nvcc_exe = cuda_path / "bin" / "nvcc.exe"
    nvcc_bat.write_text(f'@"{randomtemp}" "{sccache}" "{nvcc_exe}" %*\n')
    print(f"Wrote NVCC wrapper at {nvcc_bat}:\n{nvcc_bat.read_text()}")
    # CMake doesn't accept backslashes in CMAKE_CUDA_COMPILER -- use the
    # mixed-path form the legacy bat produced via `cygpath -m`.
    return {
        "CUDA_NVCC_EXECUTABLE": str(nvcc_bat).replace("\\", "/"),
        "CMAKE_CUDA_COMPILER": str(nvcc_exe).replace("\\", "/"),
        "CMAKE_CUDA_COMPILER_LAUNCHER": f"{randomtemp};{sccache}",
    }


def prepend_path(*entries: Path | str) -> str:
    """Build a Windows PATH (`;`-separated) prepending entries to the current PATH."""
    current = os.environ.get("PATH", "")
    return ";".join((*[str(e) for e in entries], current))


def setup_cuda() -> dict[str, str]:
    cuda_version = os.environ.get("CUDA_VERSION", "")
    arch_list = TORCH_CUDA_ARCH_LIST_TABLE.get(cuda_version)
    if not arch_list:
        sys.exit(
            f"Unsupported CUDA_VERSION={cuda_version!r}; expected one of "
            f"{sorted(TORCH_CUDA_ARCH_LIST_TABLE)}"
        )
    dotted = f"{cuda_version[:-1]}.{cuda_version[-1]}"
    cuda_prefix = f"cuda{cuda_version}"
    build_type = "debug" if os.environ.get("DEBUG") == "1" else "release"
    print(f"CUDA {dotted} ({cuda_prefix}/{build_type})")

    install_cuda_toolkit()
    cuda_path = find_cuda_path(dotted)
    nvtools_ext = find_nvtoolsext()
    magma_home = install_magma(cuda_prefix, build_type)

    tmp_bin = WIN_CI_DIR / "tmp_bin"
    install_sccache_binaries(tmp_bin)
    nvcc_env = setup_nvcc_wrapper(tmp_bin, cuda_path)

    return {
        **CUDA_BUILD_ENV_STATIC,
        f"CUDA_PATH_V{cuda_version}": str(cuda_path),
        "CUDA_PATH": str(cuda_path),
        "TORCH_CUDA_ARCH_LIST": arch_list,
        "NVTOOLSEXT_PATH": str(nvtools_ext),
        "MAGMA_HOME": str(magma_home),
        "SCCACHE_IDLE_TIMEOUT": "1500",
        "CC": "sccache-cl",
        "CXX": "sccache-cl",
        "PATH": prepend_path(tmp_bin, cuda_path / "bin"),
        **nvcc_env,
    }


def shell_quote(value: str) -> str:
    if value and all(c.isalnum() or c in "_-./:=" for c in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def write_env_exports(env: dict[str, str], path: Path | None) -> None:
    """Write `export KEY=VALUE` lines for build.sh to source.

    Forward-slash-normalize PATH-like values so the parent bash sees them as
    POSIX paths; the build subprocess re-converts on Windows transparently
    via cygpath-aware tooling.
    """
    if path is None:
        return
    lines = [f"export {k}={shell_quote(v)}" for k, v in env.items()]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-out",
        type=Path,
        help="Write `export KEY=VALUE` lines here for build.sh to source.",
    )
    args = parser.parse_args()

    gpu_arch_type = os.environ.get("GPU_ARCH_TYPE") or os.environ.get(
        "DESIRED_CUDA", "cpu"
    )
    if gpu_arch_type.startswith("cu") and gpu_arch_type != "cpu":
        gpu_arch_type = "cuda"
    vc_year = os.environ.get("VC_YEAR", "2022")
    print(f"build_env_setup.py: GPU_ARCH_TYPE={gpu_arch_type} VC_YEAR={vc_year}")

    env_out: dict[str, str] = {**COMMON_BUILD_ENV}

    if gpu_arch_type == "cpu":
        env_out.update(CPU_BUILD_ENV)
        print("CPU environment configured")
    elif gpu_arch_type == "cuda":
        env_out.update(setup_cuda())
    else:
        # XPU branch is added in a follow-up commit; fail loudly so the CI
        # surface is unambiguous while the refactor is in progress.
        sys.exit(
            f"build_env_setup.py: GPU_ARCH_TYPE={gpu_arch_type!r} not yet ported "
            "from the legacy .bat scripts. Supported: cpu, cuda."
        )

    # Push our env into the current process so vcvarsall's PATH extension
    # layers on top of (rather than replaces) our additions when captured.
    os.environ.update(env_out)

    # vcvarsall env -- captured last so its PATH/LIB/etc. extensions stack
    # on top of whatever we just configured.
    vcvarsall = find_vcvarsall(vc_year)
    print(f"Sourcing {vcvarsall}")
    vsdevcmd_args = os.environ.get("VSDEVCMD_ARGS", "")
    env_out.update(capture_vcvars_env(vcvarsall, f"x64 {vsdevcmd_args}".strip()))

    write_env_exports(env_out, args.env_out)
    print("before-all setup complete")


if __name__ == "__main__":
    main()
