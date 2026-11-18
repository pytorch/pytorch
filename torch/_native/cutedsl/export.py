"""Ahead-of-time export of CuTeDSL kernels (``cute.compile``, then ``export_to_c``),
shared by native AOT kernels (tools/native_aot) and by user-defined kernels in the
C++ wrapper (torch/_inductor/codegen/cutedsl/user_kernel_aot.py).

An export writes a header and an object that holds host code (the module load and
the launch wrapper) next to the kernels' cubin; the object calls shims of the DSL's
runtime archive (runtime_archive), which is linked after it.

Nothing from torch is imported at module level: tools/native_aot/build_stage2.py
loads this file by path, without importing torch.
"""

import platform
from collections.abc import Callable, Iterable
from pathlib import Path


__all__ = [
    "HOST_TARGETS",
    "RUNTIME_ARCHIVE",
    "host_target",
    "runtime_archive",
    "warm_up_exporter",
]

# --host-target values. Unpinned, the DSL compiles the host code for the build
# machine's CPU, so what an AVX-512 machine exports fails with SIGILL on CPUs
# without AVX-512.
HOST_TARGETS = {
    "x86_64": "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64",
    # No -mcpu: an explicit triple alone is that triple's generic CPU.
    "aarch64": "llvm -mtriple=aarch64-unknown-linux-gnu",
}

RUNTIME_ARCHIVE = "libcuda_dialect_runtime_static.a"


def host_target() -> str:
    """The --host-target for this machine's architecture: its generic CPU."""
    machine = platform.machine()
    if machine not in HOST_TARGETS:
        raise RuntimeError(
            f"CuTeDSL export: no pinned host target for {machine!r}. Falling back "
            "would target this machine's CPU; add an entry to "
            "torch._native.cutedsl.export.HOST_TARGETS."
        )
    return HOST_TARGETS[machine]


def runtime_archive(
    roots: Iterable[str], cuda_major: int | None, warn: Callable[[str], object]
) -> str:
    """The DSL's static runtime archive, whose shims every exported object calls,
    for CUDA ``cuda_major``, from the first of ``roots`` (directories that hold the
    nvidia_cutlass_dsl package, or the root of an fbcode par's tree) that has one.

    From nvidia-cutlass-dsl 4.6.1, wheels (and fbcode pars) ship one per CUDA major
    installed, in nvidia_cutlass_dsl/cu<N>/lib/. Only the cu12 one is a hard
    dependency; the others come with an extra (nvidia-cutlass-dsl[cu13]), so a CUDA
    13 environment may have only cu12. Before, wheels shipped one in
    nvidia_cutlass_dsl/lib/, and fbcode pars put the libs-base one in lib/ and the
    libs-cu13 one, renamed libcuda_dialect_cu13_runtime_static.a, in
    nvidia_cutlass_dsl/lib/.

    Takes the archive for ``cuda_major``, else one that is not per CUDA major, else
    the one for the highest CUDA major there, after passing ``warn`` a message that
    says what to install (the extra, or for CUDA 12 nvidia-cutlass-dsl again); raises
    FileNotFoundError, saying it too, when there is none.
    """
    roots = list(dict.fromkeys(roots))
    if cuda_major == 12:
        # nvidia-cutlass-dsl itself requires the cu12 one: there is no cu12 extra.
        install = "reinstall nvidia-cutlass-dsl"
    else:
        install = f"install 'nvidia-cutlass-dsl[cu{cuda_major or '<N>'}]'"
    for root in map(Path, roots):
        dsl = root / "nvidia_cutlass_dsl"
        # archive -> its CUDA major (from its cu<N> directory or its name, never
        # from a directory above the package, say an environment named cu12),
        # None for one that is not per CUDA major
        found: dict[Path, int | None] = {}
        for path in sorted(dsl.glob(f"cu*/lib/{RUNTIME_ARCHIVE}")):
            if (major := path.parent.parent.name[2:]).isdigit():
                found[path] = int(major)
        for path in sorted(dsl.glob("lib/libcuda_dialect_cu*_runtime_static.a")):
            if (major := path.name.split("_")[2][2:]).isdigit():
                found[path] = int(major)
        for path in (root / "lib" / RUNTIME_ARCHIVE, dsl / "lib" / RUNTIME_ARCHIVE):
            if path.is_file():
                found[path] = None
        if not found:
            continue
        for wanted in (cuda_major, None):
            for path, major in found.items():
                if major == wanted:
                    return str(path)
        fallback = max(found, key=lambda path: found[path] or 0)
        warn(
            f"nvidia-cutlass-dsl has no CuTeDSL runtime archive for CUDA "
            f"{cuda_major}; linking {fallback} instead (found: "
            f"{', '.join(map(str, found))}). Only the cu12 one is a hard dependency "
            f"of nvidia-cutlass-dsl: {install} to link the matching one"
        )
        return str(fallback)
    where = f" in {roots[0]}" if roots else ""
    raise FileNotFoundError(
        f"CuTeDSL runtime archive {RUNTIME_ARCHIVE} not found next to the "
        f"nvidia-cutlass-dsl package{where}: {install}"
    )


_warmed_up = False


def warm_up_exporter() -> None:
    """Build one JIT engine in this process, so that export_to_c works for any GPU
    architecture.

    export_to_c needs LLVM state that the DSL sets up only when it creates a JIT
    engine, which it does for a compile without kernels or for the architecture it
    runs on (CUTE_DSL_ARCH, else the first GPU's). Without this, an export for
    another architecture as the first compile in a process fails with "Failed to
    dump object file with PIC relocation". A compile without kernels sets it up in
    ~0.2s, once per process, with no CUDA device needed.
    """
    from torch._native.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK

    global _warmed_up
    with CUTEDSL_COMPILE_LOCK:
        if _warmed_up:
            return
        import cutlass.cute as cute

        @cute.jit
        def no_kernel():
            pass

        cute.compile(no_kernel)
        _warmed_up = True
