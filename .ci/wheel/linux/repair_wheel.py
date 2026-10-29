#!/usr/bin/env python3
"""Repair a PyTorch wheel: bundle runtime deps, set RPATHs, retag platform.

Uses auditwheel for unpack/repack (not the `wheel` CLI). `wheel pack`/`wheel tags`
emit an invalid ZIP64 header for archives over 4GB (pypa/wheel#692), which makes
large ROCm wheels fail to install under strict zip parsers such as uv
(pytorch#189748). auditwheel repacks via the stdlib `zipfile`, which writes a
correct ZIP64 record.

Usage: repair_wheel.py <input_dir> <output_dir>

Environment variables:
    DESIRED_CUDA       - cpu, cu126, cu130, xpu, rocm6.4.1, etc.
    GPU_ARCH_TYPE      - cpu, cuda, cuda-aarch64, rocm, xpu
    GPU_ARCH_VERSION   - 12.6, 13.0, 13.2, 6.4.1, etc. (empty for CPU)
    USE_CUDA           - "0" or "1"
    ROCM_HOME          - /opt/rocm (ROCm only)
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from auditwheel.wheeltools import add_platforms, InWheelCtx
from build_env_setup import PLATFORM_TAGS


PATCHELF = "/usr/local/bin/patchelf"


def wheel_platform_tags(wheel_name: str) -> list[str]:
    """Platform tags encoded in a wheel filename.

    The filename is ``{name}-{version}-{python}-{abi}-{platform}.whl``; the
    platform field is the last ``-``-delimited component and may itself be a
    ``.``-joined set of tags (e.g. ``linux_x86_64`` or
    ``manylinux1_x86_64.manylinux_2_17_x86_64``).
    """
    return wheel_name.removesuffix(".whl").rsplit("-", 1)[-1].split(".")


@dataclass
class BundledLib:
    """A compatibility library to copy into torch/lib."""

    src: Path
    dest_name: str  # final filename in torch/lib/


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
        + ":$ORIGIN/../../nvidia/cusolver/lib"
        + ":$ORIGIN/../../nvidia/cusparse/lib"
        + ":$ORIGIN/../../cusparselt/lib"
        + ":$ORIGIN/../../nvidia/nvtx/lib"
        + ":$ORIGIN/../../nvidia/cufile/lib"
    )


def rocm_rpaths(rocm_home: Path | None = None) -> str:
    """RPATH list for the TheRock wheel-based ROCm layout.

    ROCm packages are siblings of torch, so paths are ``$ORIGIN``-relative.
    Include flat and discovered per-target LLVM paths for ``libomp.so`` across
    TheRock package layouts. ``host-math/lib`` holds ``librocm-openblas.so.0``
    and ``rocm_sysdeps/lib`` holds ``librocm_sysdeps_numa.so.1``; devel images
    can place those same sonames under ``_rocm_sdk_devel`` instead of core.
    """
    rpaths = [
        "$ORIGIN/../../_rocm_sdk_core/lib",
        "$ORIGIN/../../_rocm_sdk_core/lib/rocm_sysdeps/lib",
        "$ORIGIN/../../_rocm_sdk_core/lib/host-math/lib",
        "$ORIGIN/../../_rocm_sdk_libraries/lib",
        "$ORIGIN/../../_rocm_sdk_core/lib/llvm/lib",
        "$ORIGIN/../../_rocm_sdk_devel/lib/host-math/lib",
        "$ORIGIN/../../_rocm_sdk_devel/lib/rocm_sysdeps/lib",
    ]
    if rocm_home is None:
        env_home = os.environ.get("ROCM_HOME")
        if env_home:
            rocm_home = Path(env_home)
    if rocm_home is not None:
        llvm_lib = rocm_home / "lib" / "llvm" / "lib"
        for libomp in sorted(llvm_lib.glob("*/libomp.so")):
            if libomp.is_file():
                rpaths.append(
                    f"$ORIGIN/../../_rocm_sdk_core/lib/llvm/lib/{libomp.parent.name}"
                )
    return ":".join(rpaths)


def arch_extra_deps(arch: str, use_cuda: bool) -> list[Path]:
    """
    CPU builds link against OpenBLAS + libgfortran
    CUDA builds link against NVPL.
    """
    candidates: list[Path] = [Path("/usr/lib64/libgfortran.so.5")]
    if arch == "aarch64":
        # Both CPU and CUDA builds pick up ARM Compute Library (ACL) for
        # oneDNN acceleration on AArch64.
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

    deps: list[Path] = [p for p in candidates if p.is_file()]
    return deps


def patchelf(*args: str) -> None:
    subprocess.run([PATCHELF, *args], check=True)


def set_rpath(sofile: Path, rpath: str, force_rpath: bool) -> None:
    cmd = ["--set-rpath", rpath]
    if force_rpath:
        cmd.append("--force-rpath")
    cmd.append(str(sofile))
    patchelf(*cmd)


def repair_wheel(
    wheel: Path,
    output_dir: Path,
    platform_tag: str,
    libgomp_path: Path,
    arch_deps: list[Path],
    bundled_libs: list[BundledLib],
    c_so_rpath: str,
    lib_so_rpath: str,
    force_rpath: bool,
) -> None:
    # InWheelCtx unpacks via auditwheel's zip2dir on enter and, once out_wheel is
    # set, regenerates RECORD and repacks via dir2zip on exit. dir2zip uses the
    # stdlib zipfile (correct ZIP64), unlike `wheel pack`/`wheel tags` which
    # corrupt >4GB ROCm wheels (pypa/wheel#692, pytorch#189748).
    with InWheelCtx(wheel, output_dir / wheel.name) as ctx:
        torch_dir = ctx.path / "torch"
        torch_lib = torch_dir / "lib"

        # Bundle libgomp and rewrite NEEDED entries to point at our copy
        shutil.copy(libgomp_path, torch_lib / "libgomp.so.1")
        for sofile in torch_dir.glob("*.so*"):
            if sofile.is_file():
                patchelf(
                    "--replace-needed",
                    "libgomp.so.1",
                    "libgomp.so.1",
                    str(sofile),
                )

        # Bundle aarch64 BLAS/LAPACK/ACL dependencies (no-op on x86)
        for dep in arch_deps:
            shutil.copy(dep, torch_lib / dep.name)

        # Keep only compatibility dependencies that are not supplied by the
        # ROCm SDK wheel packages (currently pre-10.0 rocSHMEM's libnuma).
        for lib in bundled_libs:
            shutil.copy(lib.src, torch_lib / lib.dest_name)
        # Set RPATH on top-level (_C.so etc.) and lib/ shared objects
        for sofile in torch_dir.glob("*.so*"):
            if sofile.is_file():
                set_rpath(sofile, c_so_rpath, force_rpath)
        for sofile in torch_lib.glob("*.so*"):
            if sofile.is_file():
                set_rpath(sofile, lib_so_rpath, force_rpath)

        # Retag linux_* -> manylinux_2_28_* in both the filename and the WHEEL
        # metadata. add_platforms updates ctx.out_wheel to the new name; RECORD
        # regeneration and repacking happen when the context exits.
        add_platforms(
            ctx, [platform_tag], remove_platforms=wheel_platform_tags(wheel.name)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    arch = platform.machine()
    use_cuda = os.environ.get("USE_CUDA", "0") == "1"
    gpu_arch_type = os.environ.get("GPU_ARCH_TYPE", "")
    gpu_arch_version = os.environ.get("GPU_ARCH_VERSION", "")
    is_rocm = gpu_arch_type == "rocm" or "rocm" in os.environ.get("DESIRED_CUDA", "")

    libgomp_path = detect_libgomp()
    if not libgomp_path.exists():
        sys.exit(f"libgomp not found at {libgomp_path}")

    bundled_libs: list[BundledLib] = []

    if use_cuda:
        rpaths = cuda_rpaths(gpu_arch_version)
        c_so_rpath = f"{rpaths}:$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = f"{rpaths}:$ORIGIN"
        force_rpath = True
    elif gpu_arch_type == "xpu":
        # XPU runtime libs come from pypi packages; set RPATHs like CUDA.
        xpu_rpaths = "$ORIGIN/../../../.."
        c_so_rpath = f"{xpu_rpaths}:$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = f"{xpu_rpaths}:$ORIGIN"
        force_rpath = True
    elif is_rocm:
        rocm_home = Path(os.environ.get("ROCM_HOME", "/opt/rocm"))
        # A classic install has no _rocm_sdk_* siblings, so SDK RPATHs
        # would not resolve. CI images always use the TheRock layout.
        if "_rocm_sdk" not in str(rocm_home):
            sys.exit(
                f"Only the TheRock SDK layout is supported (ROCM_HOME={rocm_home})"
            )
        rpaths = rocm_rpaths(rocm_home)
        c_so_rpath = f"{rpaths}:$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = f"{rpaths}:$ORIGIN"
        force_rpath = True
        # Pre-10.0 TheRock SDKs ship a rocSHMEM that dlopens the bare name
        # "libnuma.so". At wheel runtime that name resolves nowhere: the
        # SDK vendors numa only as librocm_sysdeps_numa.so.1, and on user
        # systems the bare dev name exists only if numactl-devel is
        # installed (#195670). The 10.0 SDK line links the vendored soname
        # directly instead (rocm-systems#6640), so bundle a bare-named
        # copy, reached via $ORIGIN on the RPATHs above, only for older
        # SDKs. The builder image installs numactl-libs for this.
        ver = gpu_arch_version
        if not ver or tuple(map(int, ver.split(".")[:2])) < (10, 0):
            is_ubuntu = "Ubuntu" in Path("/etc/os-release").read_text()
            libdir = "/usr/lib/x86_64-linux-gnu" if is_ubuntu else "/usr/lib64"
            libnuma = Path(libdir) / "libnuma.so.1"
            if not libnuma.is_file():
                sys.exit(f"libnuma to bundle for rocSHMEM not found: {libnuma}")
            bundled_libs.append(BundledLib(src=libnuma, dest_name="libnuma.so"))
    else:
        c_so_rpath = "$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = "$ORIGIN"
        force_rpath = False

    arch_deps = arch_extra_deps(arch, use_cuda)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wheels = sorted(args.input_dir.glob("*.whl"))
    if not wheels:
        sys.exit(f"No wheels found in {args.input_dir}")

    if arch not in PLATFORM_TAGS:
        sys.exit(f"Unknown arch {arch}")
    platform_tag = PLATFORM_TAGS[arch]
    for whl in wheels:
        repair_wheel(
            whl,
            args.output_dir,
            platform_tag,
            libgomp_path,
            arch_deps,
            bundled_libs,
            c_so_rpath,
            lib_so_rpath,
            force_rpath,
        )

    repaired = list(args.output_dir.glob("*.whl"))
    print(f"Repaired {len(repaired)} wheel(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
