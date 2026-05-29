#!/usr/bin/env python3
"""GPU/toolchain environment setup (runs once before any wheel is built).

When running on a standard manylinux container (no pre-installed GPU
toolkits), installs CUDA, cuDNN, NCCL, MAGMA, cuSPARSELt, etc. via the
.ci/docker/common/install_*.sh scripts.

Build-flag exports (USE_CUDA, TH_BINARY_BUILD, ...) are written to the file
given by --env-out; the caller (build.sh) sources it so the values reach
the wheel build subprocess. Without that handoff exports made here die
with this process.

Environment variables expected:
    GPU_ARCH_TYPE    - cpu, cuda, rocm, xpu
    DESIRED_CUDA     - cpu, cu126, cu128, rocm7.1, xpu, etc.
    GPU_ARCH_VERSION - 12.6, 12.8, 7.1, etc.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# CUDA build flags that don't depend on CUDA version or host arch.
# Values mirror the original build_cuda.sh; static linking is OFF (the
# nvidia/* pypi packages provide the shared libs, and static linking causes
# unresolved __cudaRegisterLinkedBinary_* RDC stubs when test executables
# link against libtorch_cuda.so).
CUDA_BUILD_ENV_STATIC: dict[str, str] = {
    # USE_CUDA=1 is also what repair_wheel.py reads to decide CUDA RPATHs
    # and to bundle NVPL on aarch64 -- keep it explicit even though CMake
    # would auto-detect.
    "USE_CUDA": "1",
    "NCCL_ROOT_DIR": "/usr/local/cuda",
    "CUDNN_ROOT_DIR": "/usr/local/cuda",
    "TH_BINARY_BUILD": "1",
    "INSTALL_TEST": "0",
    "USE_STATIC_CUDNN": "0",
    "USE_STATIC_NCCL": "0",
    "ATEN_STATIC_CUDA": "0",
    "USE_CUDA_STATIC_LINK": "0",
    "USE_CUPTI_SO": "1",
    "USE_SYSTEM_NCCL": "1",
    "NCCL_INCLUDE_DIR": "/usr/local/cuda/include/",
    "NCCL_LIB_DIR": "/usr/local/cuda/lib64/",
}

# Defaulted-to-1 like the original `${VAR:-1}` -- callers can override.
CUDA_BUILD_ENV_DEFAULTS: dict[str, str] = {
    "USE_CUSPARSELT": "1",
    "USE_CUFILE": "1",
}


# Compute capabilities each (cuda_version, host arch) wheel is built for, as
# the same {cc_int} representation used by torch/cuda/__init__.py:
# `50 == 5.0`, `120 == 12.0`. Kept in sync with PYTORCH_RELEASES_CODE_CC by
# validate_runtime_release_table_consistency() in
# .github/scripts/generate_binary_build_matrix.py.
TORCH_CUDA_ARCH_LIST_TABLE: dict[str, dict[str, set[int]]] = {
    "12.6": {
        "x86_64": {50, 60, 70, 75, 80, 86, 90},
        "aarch64": {80, 90},
    },
    "13.0": {
        "x86_64": {75, 80, 86, 90, 100, 120},
        "aarch64": {80, 90, 100, 110, 120},
    },
    "13.2": {
        "x86_64": {75, 80, 86, 90, 100, 120},
        "aarch64": {80, 90, 100, 110, 120},
    },
}

# Architectures we additionally emit PTX for (forward-compat for newer GPUs).
_PTX_ARCHES: set[int] = {120}


def torch_cuda_arch_list(cuda_version: str, arch: str) -> str:
    """Format TORCH_CUDA_ARCH_LIST for the wheel build (";"-separated).

    Returns e.g. "8.0;9.0;10.0;11.0;12.0+PTX" for cuda 13.x aarch64.

    CUDA 13.x dropped sm_50/60/70, so we must NOT leave this empty --
    CMake's defaults still include compute_50 which nvcc 13 rejects with
    "Unsupported gpu architecture 'compute_50'".
    """
    if cuda_version not in TORCH_CUDA_ARCH_LIST_TABLE:
        raise SystemExit(f"unknown cuda version {cuda_version}")
    archs = TORCH_CUDA_ARCH_LIST_TABLE[cuda_version].get(arch)
    if not archs:
        raise SystemExit(f"no TORCH_CUDA_ARCH_LIST for cuda {cuda_version} on {arch}")
    return ";".join(
        f"{cc // 10}.{cc % 10}" + ("+PTX" if cc in _PTX_ARCHES else "")
        for cc in sorted(archs)
    )


def cuda_build_env(cuda_version: str, arch: str) -> dict[str, str]:
    nvcc_flags = "-Xfatbin -compress-all --threads 2"
    if cuda_version.startswith("13."):
        nvcc_flags += " -compress-mode=size"
    env = {
        # Defaulted vars first so STATIC values still take precedence,
        # but caller-provided values for the defaults still win below.
        **{k: v for k, v in CUDA_BUILD_ENV_DEFAULTS.items() if not os.environ.get(k)},
        **CUDA_BUILD_ENV_STATIC,
        "TORCH_NVCC_FLAGS": nvcc_flags,
        "TORCH_CUDA_ARCH_LIST": torch_cuda_arch_list(cuda_version, arch),
    }
    if arch == "aarch64":
        # Pre-built MAGMA tarballs are x86-only.
        env["USE_MAGMA"] = "0"
    return env


CPU_BUILD_ENV: dict[str, str] = {
    "TH_BINARY_BUILD": "1",
    "USE_CUDA": "0",
}

PLATFORM_TAGS: dict[str, str] = {
    "x86_64": "manylinux_2_28_x86_64",
    "aarch64": "manylinux_2_28_aarch64",
}


def repo_root() -> Path:
    return Path(
        os.environ.get(
            "GITHUB_WORKSPACE",
            Path(__file__).resolve().parents[2],
        )
    )


def os_name() -> str:
    for line in Path("/etc/os-release").read_text().splitlines():
        if line.startswith("NAME="):
            return line.partition("=")[2].strip().strip('"')
    return ""


def install_os_packages() -> None:
    name = os_name()
    if any(distro in name for distro in ("AlmaLinux", "CentOS", "Red Hat")):
        subprocess.run(
            [
                "yum",
                "install",
                "-q",
                "-y",
                "zip",
                "openssl",
                "openssl-devel",
                "sudo",
                "wget",
                "curl",
                "perl",
                "util-linux",
                "xz",
                "bzip2",
                "git",
                "patch",
                "which",
                "zlib-devel",
            ],
            check=True,
        )
    elif "Ubuntu" in name:
        # Disable nvidia apt repos that may 404 on plain Ubuntu images
        for entry in Path("/etc/apt").rglob("*.list"):
            text = entry.read_text()
            new = "\n".join(
                f"# {line}" if "nvidia" in line else line for line in text.splitlines()
            )
            if new != text:
                entry.write_text(new + ("\n" if text.endswith("\n") else ""))
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(
            [
                "apt-get",
                "-y",
                "-qq",
                "install",
                "zip",
                "openssl",
                "wget",
                "curl",
                "git",
            ],
            check=True,
        )


def ensure_pip_on_path() -> None:
    """Manylinux images ship /usr/bin/python3 without pip; fall back to /opt/python/cp3*."""
    if (
        subprocess.run(
            ["python3", "-mpip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    ):
        return
    candidates = sorted(Path("/opt/python").glob("cp3*/bin"))
    if candidates:
        fallback = candidates[-1]
        os.environ["PATH"] = f"{fallback}{os.pathsep}{os.environ.get('PATH', '')}"
        print(f"Added {fallback} to PATH for pip access")


def cuda_version_from_env() -> str:
    """Match the original build_cuda.sh precedence: DESIRED_CUDA wins.

    GPU_ARCH_VERSION on aarch64 jobs comes through as e.g. "12.6-aarch64"
    which install_cuda.sh rejects, so we strip the "-aarch64" suffix when
    falling back to it.
    """
    desired = os.environ.get("DESIRED_CUDA", "")
    # "12.6" → keep; "cu126" → "12.6"
    if "." in desired and desired.replace(".", "").isdigit():
        return desired
    if len(desired) == 5 and desired.startswith("cu") and desired[2:].isdigit():
        return f"{desired[2:4]}.{desired[4]}"

    arch_version = os.environ.get("GPU_ARCH_VERSION", "")
    return arch_version.removesuffix("-aarch64")


def install_cuda_toolkit(cuda_version: str) -> None:
    """Stage install_cuda.sh + its required siblings, then run install_cuda + install_magma."""
    root = repo_root()
    docker_common = root / ".ci/docker/common"
    pins = root / ".ci/docker/ci_commit_pins"

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ("install_cuda.sh", "install_nccl.sh", "install_cusparselt.sh"):
            shutil.copy(docker_common / name, stage / name)
        (stage / "ci_commit_pins").mkdir()
        for nccl_pin in pins.glob("nccl*"):
            shutil.copy(nccl_pin, stage / "ci_commit_pins" / nccl_pin.name)

        subprocess.run(["bash", "install_cuda.sh", cuda_version], cwd=stage, check=True)

    subprocess.run(
        ["bash", str(docker_common / "install_magma.sh"), cuda_version], check=True
    )
    print(f"CUDA {cuda_version} toolkit installation complete")


def setup_cuda(cuda_version: str) -> None:
    arch = platform.machine()
    cuda_dir = Path(f"/usr/local/cuda-{cuda_version}")

    if not cuda_dir.is_dir():
        print(f"CUDA {cuda_version} not found, installing from scratch...")
        install_cuda_toolkit(cuda_version)
    else:
        print(f"CUDA {cuda_version} already installed, switching symlinks...")
        symlink = Path("/usr/local/cuda")
        if symlink.is_symlink() or symlink.exists():
            symlink.unlink()
        symlink.symlink_to(cuda_dir)

    if arch != "aarch64" and not (cuda_dir / "magma").is_dir():
        print("MAGMA not found, installing...")
        subprocess.run(
            [
                "bash",
                str(repo_root() / ".ci/docker/common/install_magma.sh"),
                cuda_version,
            ],
            check=True,
        )

    if arch != "aarch64":
        magma_link = Path("/usr/local/magma")
        if magma_link.is_symlink() or magma_link.exists():
            magma_link.unlink()
        magma_link.symlink_to(cuda_dir / "magma")

    create_cudnn_unversioned_symlinks()
    verify_cudnn()


def create_cudnn_unversioned_symlinks() -> None:
    """Some image builders drop libcudnn.so → libcudnn.so.9; CMake's find_library needs it."""
    cuda_lib = Path("/usr/local/cuda/lib64")
    if not cuda_lib.is_dir():
        return
    for sofile in cuda_lib.glob("libcudnn*.so.[0-9]*"):
        # libcudnn.so.9 → libcudnn.so
        base = sofile.name.split(".so.", 1)[0] + ".so"
        link = cuda_lib / base
        if not link.exists():
            print(f"Creating missing symlink: {link} -> {sofile.name}")
            link.symlink_to(sofile.name)


def verify_cudnn() -> None:
    print("cuDNN check:")
    libs = list(Path("/usr/local/cuda/lib64").glob("libcudnn*.so*"))
    if libs:
        for lib in libs:
            stat = lib.lstat()
            print(f"  {lib} ({stat.st_size} bytes)")
    else:
        print("WARNING: cuDNN libraries not found in /usr/local/cuda/lib64/")
    headers = list(Path("/usr/local/cuda/include").glob("cudnn*.h"))
    if headers:
        for h in headers:
            print(f"  {h}")
    else:
        print("WARNING: cuDNN headers not found in /usr/local/cuda/include/")


def cleanup_cuda_for_cpu_build() -> None:
    for entry in Path("/usr/local").glob("cuda*"):
        if entry.is_symlink():
            entry.unlink()
        elif entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)


def write_env_exports(env: dict[str, str], path: Path | None) -> None:
    """Write `export KEY=VALUE` lines for the parent shell to source."""
    if path is None:
        return
    lines = [f"export {k}={shell_quote(v)}" for k, v in env.items()]
    path.write_text("\n".join(lines) + "\n")


def shell_quote(value: str) -> str:
    if value and all(c.isalnum() or c in "_-./:=" for c in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-out",
        type=Path,
        help="Write `export KEY=VALUE` lines here for build.sh to source.",
    )
    args = parser.parse_args()

    arch = platform.machine()
    gpu_arch_type = os.environ.get("GPU_ARCH_TYPE", "cpu")
    print(
        f"Architecture: {arch}, GPU_ARCH_TYPE: {gpu_arch_type}, "
        f"REPO_ROOT: {repo_root()}"
    )

    install_os_packages()

    env_out: dict[str, str] = {}
    if arch in PLATFORM_TAGS:
        env_out["PLATFORM"] = PLATFORM_TAGS[arch]

    ensure_pip_on_path()

    # MKL: x86_64 only; aarch64 uses OpenBLAS/ACL from the builder image.
    if arch == "x86_64" and not Path("/opt/intel/lib").is_dir():
        print("MKL not found, installing...")
        subprocess.run(
            ["bash", str(repo_root() / ".ci/docker/common/install_mkl.sh")],
            check=True,
        )

    if gpu_arch_type in ("cuda", "cuda-aarch64"):
        cuda_version = cuda_version_from_env()
        if not cuda_version:
            sys.exit(
                "Could not determine CUDA version from GPU_ARCH_VERSION/DESIRED_CUDA"
            )
        setup_cuda(cuda_version)
        env_out.update(cuda_build_env(cuda_version, arch))
        print(f"CUDA {cuda_version} environment configured")
    elif gpu_arch_type in ("cpu", "cpu-aarch64", "cpu-s390x", "cpu-cxx11-abi"):
        cleanup_cuda_for_cpu_build()
        env_out.update(CPU_BUILD_ENV)
        print("CPU environment configured")
    # ROCm and XPU use the legacy Docker-in-Docker workflow and skip this script.

    write_env_exports(env_out, args.env_out)
    print("before-all setup complete")


if __name__ == "__main__":
    main()
