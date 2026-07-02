#!/usr/bin/env python3
"""Repair a PyTorch wheel: bundle libgomp + GPU libs, set RPATHs, retag platform.

Uses the `wheel` Python package for unpack/pack/tags (not zip).

Usage: repair_wheel.py <input_dir> <output_dir>

Environment variables:
    DESIRED_CUDA       - cpu, cu126, cu130, xpu, rocm6.4.1, etc.
    GPU_ARCH_TYPE      - cpu, cuda, cuda-aarch64, rocm, xpu
    GPU_ARCH_VERSION   - 12.6, 13.0, 13.2, 6.4.1, etc. (empty for CPU)
    USE_CUDA           - "0" or "1"
    PYTORCH_ROCM_ARCH  - ;-separated gfx targets (ROCm only)
    ROCM_HOME          - /opt/rocm (ROCm only)
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PATCHELF = "/usr/local/bin/patchelf"


@dataclass
class BundledLib:
    """A shared library to copy into torch/lib/ and (optionally) patchelf-rewrite NEEDED entries for."""

    src: Path
    dest_name: str  # final filename in torch/lib/
    needed_alias: str | None = None  # original SONAME to replace in NEEDED entries


@dataclass
class AuxFile:
    """Auxiliary content (e.g. MIOpen db, RCCL algos, gfx kernel files) copied into the wheel."""

    src: Path
    rel_dest: str  # relative path under torch/, e.g. "lib/rocblas/library/Tensile..."


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


# ROCm shared libs to bundle. Discovered under $ROCM_HOME/{lib,lib64}.
ROCM_SO_FILES: list[str] = [
    "libMIOpen.so",
    "libamdhip64.so",
    "libhipblas.so",
    "libhipfft.so",
    "libhiprand.so",
    "libhipsolver.so",
    "libhipsparse.so",
    "libhsa-runtime64.so",
    "libamd_comgr.so",
    "libmagma.so",
    "librccl.so",
    "librocblas.so",
    "librocfft.so",
    "librocm_smi64.so",
    "librocrand.so",
    "librocsolver.so",
    "librocsparse.so",
    "libroctracer64.so",
    "libroctx64.so",
    "libhipblaslt.so",
    "libhipsparselt.so",
    "libhiprtc.so",
    "librocprofiler-sdk.so",
    "librocprofiler-register.so",
    "libhsa-amd-aqlprofile64.so",
    "librocm-core.so",
    "librocroller.so",
]


def rocm_os_deps() -> list[Path]:
    """OS-side runtime deps that must travel with ROCm wheels."""
    os_release = Path("/etc/os-release").read_text()
    if "Ubuntu" in os_release:
        prefix = "/usr/lib/x86_64-linux-gnu"
        libtinfo = "/lib/x86_64-linux-gnu/libtinfo.so.6"
        libdrm_dir = "/usr/lib/x86_64-linux-gnu"
    else:  # AlmaLinux / CentOS / RHEL
        prefix = "/usr/lib64"
        # CentOS Linux had libtinfo.so.5; AlmaLinux ships .6.
        libtinfo_5 = Path("/usr/lib64/libtinfo.so.5")
        libtinfo = (
            str(libtinfo_5) if libtinfo_5.exists() else "/usr/lib64/libtinfo.so.6"
        )
        libdrm_dir = "/opt/amdgpu/lib64"
    return [
        Path(f"{prefix}/libnuma.so.1"),
        Path(f"{prefix}/libelf.so.1"),
        Path(libtinfo),
        Path(f"{prefix}/libdw.so.1"),
        Path(f"{libdrm_dir}/libdrm.so.2"),
        Path(f"{libdrm_dir}/libdrm_amdgpu.so.1"),
    ]


def find_rocm_lib(rocm_home: Path, basename: str) -> Path | None:
    """Locate a ROCm library, falling back from lib/ to lib64/ to a wider search."""
    for sub in ("lib", "lib64"):
        for hit in (rocm_home / sub).rglob(basename + "*"):
            if hit.is_file() and hit.name.startswith(basename):
                return hit
    for hit in rocm_home.rglob(basename + "*"):
        if hit.is_file() and hit.name.startswith(basename):
            return hit
    return None


def rocm_arch_filter(arch_list: str) -> list[str]:
    return [a for a in arch_list.split(";") if a]


def rocm_lib_kernels(
    rocm_home: Path, lib_subdir: str, archs: list[str]
) -> list[AuxFile]:
    """Per-gfx kernel files under $ROCM_HOME/lib/<lib_subdir>/library/ plus the non-gfx common files."""
    src_dir = rocm_home / "lib" / lib_subdir / "library"
    if not src_dir.is_dir():
        return []
    files: list[AuxFile] = []
    for entry in sorted(src_dir.iterdir()):
        if not entry.is_file():
            continue
        # Pick gfx-specific files matching the arch set, plus common (non-gfx) files.
        name = entry.name
        if "gfx" in name and not any(a in name for a in archs):
            continue
        files.append(AuxFile(src=entry, rel_dest=f"lib/{lib_subdir}/library/{name}"))
    return files


def rocm_bundle(rocm_home: Path) -> tuple[list[BundledLib], list[AuxFile]]:
    """Build the ROCm bundle spec: shared libs and auxiliary kernel/db files.

    Versioned ROCm sonames (libfoo.so.6) get renamed to bare .so to match the
    NEEDED entries that hipcc emits, mirroring the original build_rocm.sh
    fname_without_so_number behaviour. NEEDED entries inside the wheel are
    rewritten to the renamed copies via patchelf in repair_wheel().
    """
    libs: list[BundledLib] = []
    for stem in ROCM_SO_FILES:
        path = find_rocm_lib(rocm_home, stem)
        if path is None:
            sys.exit(f"Required ROCm library not found: {stem}")
        # Strip the SO version: libfoo.so.6.1 -> libfoo.so. The ROCm-built
        # binaries in the wheel link against the bare .so SONAME.
        libs.append(BundledLib(src=path, dest_name=stem, needed_alias=stem))
    for os_lib in rocm_os_deps():
        if os_lib.is_file():
            libs.append(BundledLib(src=os_lib, dest_name=os_lib.name))

    archs = rocm_arch_filter(os.environ.get("PYTORCH_ROCM_ARCH", ""))
    aux: list[AuxFile] = []
    for sub in ("rocblas", "hipblaslt", "hipsparselt"):
        aux += rocm_lib_kernels(rocm_home, sub, archs)
    miopen_db = rocm_home / "share/miopen/db"
    if miopen_db.is_dir():
        for entry in sorted(miopen_db.iterdir()):
            if entry.is_file() and any(a in entry.name for a in archs):
                aux.append(AuxFile(src=entry, rel_dest=f"share/miopen/db/{entry.name}"))
    rccl_dir = rocm_home / "share/rccl/msccl-algorithms"
    if rccl_dir.is_dir():
        for entry in sorted(rccl_dir.iterdir()):
            if entry.is_file():
                aux.append(
                    AuxFile(
                        src=entry,
                        rel_dest=f"share/rccl/msccl-algorithms/{entry.name}",
                    )
                )
    amdgpu_ids = Path("/opt/amdgpu/share/libdrm/amdgpu.ids")
    if amdgpu_ids.is_file():
        aux.append(AuxFile(src=amdgpu_ids, rel_dest="share/libdrm/amdgpu.ids"))
    return libs, aux


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


def replace_needed(unpacked_torch: Path, original: str, replacement: str) -> None:
    """Rewrite NEEDED entries that match `original*` to `replacement` across the wheel."""
    for sofile in unpacked_torch.rglob("*.so*"):
        if not sofile.is_file():
            continue
        try:
            needed = subprocess.check_output(
                [PATCHELF, "--print-needed", str(sofile)], text=True
            ).splitlines()
        except subprocess.CalledProcessError:
            continue
        for entry in needed:
            if entry == original or entry.startswith(original + "."):
                patchelf("--replace-needed", entry, replacement, str(sofile))


def repair_wheel(
    wheel: Path,
    output_dir: Path,
    libgomp_path: Path,
    aarch64_deps: list[Path],
    bundled_libs: list[BundledLib],
    aux_files: list[AuxFile],
    c_so_rpath: str,
    lib_so_rpath: str,
    force_rpath: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        unpacked = unpack_wheel(wheel, work)
        torch_dir = unpacked / "torch"
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
        for dep in aarch64_deps:
            shutil.copy(dep, torch_lib / dep.name)

        # TODO: Remove when switching to ROCm wheels
        # Bundle GPU-specific shared libs (currently only ROCm uses this).
        # Copy follows symlinks so versioned sonames become real files we can
        # rename to their bare .so form to match what the wheel links against.
        for lib in bundled_libs:
            dest = torch_lib / lib.dest_name
            shutil.copy(lib.src, dest)
            if lib.needed_alias:
                replace_needed(torch_dir, lib.needed_alias, lib.dest_name)

        # Copy auxiliary content (gfx kernel files, MIOpen db, RCCL algos, ...)
        for aux in aux_files:
            dest = torch_dir / aux.rel_dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(aux.src, dest)

        # Set RPATH on top-level (_C.so etc.) and lib/ shared objects
        for sofile in torch_dir.glob("*.so*"):
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
    gpu_arch_type = os.environ.get("GPU_ARCH_TYPE", "")
    gpu_arch_version = os.environ.get("GPU_ARCH_VERSION", "")
    is_rocm = gpu_arch_type == "rocm" or "rocm" in os.environ.get("DESIRED_CUDA", "")

    libgomp_path = detect_libgomp()
    if not libgomp_path.exists():
        sys.exit(f"libgomp not found at {libgomp_path}")

    bundled_libs: list[BundledLib] = []
    aux_files: list[AuxFile] = []

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
        bundled_libs, aux_files = rocm_bundle(rocm_home)
        c_so_rpath = "$ORIGIN:$ORIGIN/lib"
        lib_so_rpath = "$ORIGIN"
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
            bundled_libs,
            aux_files,
            c_so_rpath,
            lib_so_rpath,
            force_rpath,
        )

    retag_wheels(args.output_dir, f"manylinux_2_28_{arch}")
    repaired = list(args.output_dir.glob("*.whl"))
    print(f"Repaired {len(repaired)} wheel(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
