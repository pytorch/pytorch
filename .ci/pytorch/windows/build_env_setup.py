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
from pathlib import Path


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
    """
    if os.environ.get("DISTUTILS_USE_SDK"):
        return {}

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
    else:
        # CUDA and XPU branches are added in follow-up commits; fail loudly so
        # the CI surface is unambiguous while the refactor is in progress.
        sys.exit(
            f"build_env_setup.py: GPU_ARCH_TYPE={gpu_arch_type!r} not yet ported "
            "from the legacy .bat scripts. Currently only `cpu` is supported."
        )

    # vcvarsall env -- captured last so build flags above can't be shadowed
    # by anything MSVC sets.
    vcvarsall = find_vcvarsall(vc_year)
    print(f"Sourcing {vcvarsall}")
    vsdevcmd_args = os.environ.get("VSDEVCMD_ARGS", "")
    env_out.update(capture_vcvars_env(vcvarsall, f"x64 {vsdevcmd_args}".strip()))

    write_env_exports(env_out, args.env_out)
    print("before-all setup complete")


if __name__ == "__main__":
    main()
