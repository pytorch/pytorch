#!/usr/bin/env python3
"""Repair a PyTorch wheel: bundle libgomp, set RPATHs, retag platform.

Uses the `wheel` Python package for unpack/pack/tags (not zip).

Usage: repair_wheel.py <input_dir> <output_dir>

Environment variables:
    DESIRED_CUDA     - cpu, cu126, cu130, etc.
    GPU_ARCH_TYPE    - cpu, cuda, cuda-aarch64, rocm, xpu
    GPU_ARCH_VERSION - 12.6, 13.0, 13.2, etc. (empty for CPU)
    USE_CUDA         - "0" or "1"
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PATCHELF = "/usr/local/bin/patchelf"


def detect_libgomp() -> Path:
    """libgomp lives in different places on Ubuntu vs RHEL-family images."""
    arch = platform.machine()
    os_release = Path("/etc/os-release").read_text()
    if "Ubuntu" in os_release:
        return Path(f"/usr/lib/{arch}-linux-gnu/libgomp.so.1")
    return Path("/usr/lib64/libgomp.so.1")


def cuda_rpaths(gpu_arch_version: str) -> str:
    """Build the colon-separated RPATH list for CUDA wheels.

    CUDA 13.x bundles all libs under nvidia/cu13/lib; CUDA 12.x ships them
    in per-component packages.
    """
    base = (
        "$ORIGIN/../../nvidia/cudnn/lib"
        ":$ORIGIN/../../nvidia/nvshmem/lib"
        ":$ORIGIN/../../nvidia/nccl/lib"
        ":$ORIGIN/../../nvidia/cusparselt/lib"
    )
    cuda_major = gpu_arch_version.split(".", 1)[0] if gpu_arch_version else ""
    if cuda_major == "13":
        return base + ":$ORIGIN/../../nvidia/cu13/lib"
    return (
        base
        + ":$ORIGIN/../../nvidia/cublas/lib"
        + ":$ORIGIN/../../nvidia/cuda_cupti/lib"
        + ":$ORIGIN/../../nvidia/cuda_nvrtc/lib"
        + ":$ORIGIN/../../nvidia/cuda_runtime/lib"
        + ":$ORIGIN/../../nvidia/cufft/lib"
        + ":$ORIGIN/../../nvidia/curand/lib"
        + ":$ORIGIN/../../nvidia/cusolver/lib"
        + ":$ORIGIN/../../nvidia/cusparse/lib"
        + ":$ORIGIN/../../cusparselt/lib"
        + ":$ORIGIN/../../nvidia/nvtx/lib"
        + ":$ORIGIN/../../nvidia/cufile/lib"
    )


def aarch64_extra_deps(use_cuda: bool) -> list[Path]:
    """Libraries to bundle into torch/lib/ on aarch64.

    CPU builds link against OpenBLAS + libgfortran; CUDA builds link against
    NVPL. Both pick up ARM Compute Library (ACL) for oneDNN acceleration.
    """
    deps: list[Path] = []
    candidates: list[Path] = [Path("/usr/lib64/libgfortran.so.5")]
    if Path("/acl/build").is_dir():
        candidates += [
            Path("/acl/build/libarm_compute.so"),
            Path("/acl/build/libarm_compute_graph.so"),
        ]
    if use_cuda:
        candidates += [
            Path(f"/usr/local/lib/{name}")
            for name in (
                "libnvpl_blas_lp64_gomp.so.0",
                "libnvpl_lapack_lp64_gomp.so.0",
                "libnvpl_blas_core.so.0",
                "libnvpl_lapack_core.so.0",
            )
        ]
    else:
        candidates.append(Path("/opt/OpenBLAS/lib/libopenblas.so.0"))
    deps = [p for p in candidates if p.is_file()]
    return deps


def patchelf(*args: str) -> None:
    subprocess.run([PATCHELF, *args], check=True)


def set_rpath(sofile: Path, rpath: str, force_rpath: bool) -> None:
    cmd = ["--set-rpath", rpath]
    if force_rpath:
        cmd.append("--force-rpath")
    cmd.append(str(sofile))
    patchelf(*cmd)


def unpack_wheel(wheel: Path, work: Path) -> Path:
    subprocess.run(["wheel", "unpack", str(wheel), "-d", str(work)], check=True)
    unpacked = next(work.glob("torch-*"), None)
    if unpacked is None:
        raise RuntimeError(f"wheel unpack produced no torch-* dir in {work}")
    return unpacked


def pack_wheel(unpacked: Path, output_dir: Path) -> None:
    subprocess.run(["wheel", "pack", str(unpacked), "-d", str(output_dir)], check=True)


def repair_wheel(
    wheel: Path,
    output_dir: Path,
    libgomp_path: Path,
    aarch64_deps: list[Path],
    c_so_rpath: str,
    lib_so_rpath: str,
    force_rpath: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        unpacked = unpack_wheel(wheel, work)
        torch_lib = unpacked / "torch" / "lib"

        # Bundle libgomp and rewrite NEEDED entries to point at our copy
        shutil.copy(libgomp_path, torch_lib / "libgomp.so.1")
        for sofile in (unpacked / "torch").glob("*.so*"):
            if sofile.is_file():
                patchelf(
                    "--replace-needed",
                    "libgomp.so.1",
                    "libgomp.so.1",
                    str(sofile),
                )

        # Bundle aarch64 BLAS/LAPACK/ACL dependencies (no-op on x86)
        for dep in aarch64_deps:
            shutil.copy(dep, torch_lib / dep.name)

        # Set RPATH on top-level (_C.so etc.) and lib/ shared objects
        for sofile in (unpacked / "torch").glob("*.so*"):
            if sofile.is_file():
                set_rpath(sofile, c_so_rpath, force_rpath)
        for sofile in torch_lib.glob("*.so*"):
            if sofile.is_file():
                set_rpath(sofile, lib_so_rpath, force_rpath)

        pack_wheel(unpacked, output_dir)


def retag_wheels(output_dir: Path, platform_tag: str) -> None:
    for whl in output_dir.glob("*.whl"):
        subprocess.run(
            ["wheel", "tags", "--platform-tag", platform_tag, "--remove", str(whl)],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    arch = platform.machine()
    use_cuda = os.environ.get("USE_CUDA", "0") == "1"
    gpu_arch_version = os.environ.get("GPU_ARCH_VERSION", "")

    libgomp_path = detect_libgomp()
    if not libgomp_path.exists():
        sys.exit(f"libgomp not found at {libgomp_path}")

    if use_cuda:
        rpaths = cuda_rpaths(gpu_arch_version)
        c_so_rpath = f"{rpaths}:$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = f"{rpaths}:$ORIGIN"
        force_rpath = True
    else:
        c_so_rpath = "$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = "$ORIGIN"
        force_rpath = False

    aarch64_deps = aarch64_extra_deps(use_cuda) if arch == "aarch64" else []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wheels = sorted(args.input_dir.glob("*.whl"))
    if not wheels:
        sys.exit(f"No wheels found in {args.input_dir}")

    for whl in wheels:
        repair_wheel(
            whl,
            args.output_dir,
            libgomp_path,
            aarch64_deps,
            c_so_rpath,
            lib_so_rpath,
            force_rpath,
        )

    retag_wheels(args.output_dir, f"manylinux_2_28_{arch}")
    repaired = list(args.output_dir.glob("*.whl"))
    print(f"Repaired {len(repaired)} wheel(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
