"""Ahead-of-time export of CuTeDSL kernels (``cute.compile``, then ``export_to_c``),
shared by native AOT kernels (tools/native_aot/toolchains.py) and by user-defined
kernels in the C++ wrapper (torch/_inductor/codegen/cutedsl/user_kernel_aot.py).

An export writes a header and an object that holds host code (the module load and
the launch wrapper) next to the kernels' cubin.
"""

import platform

from torch._native.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK


__all__ = ["HOST_TARGETS", "host_target", "warm_up_exporter"]

# --host-target values. Unpinned, the DSL compiles the host code for the build
# machine's CPU, so what an AVX-512 machine exports fails with SIGILL on CPUs
# without AVX-512.
HOST_TARGETS = {
    "x86_64": "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64",
    # No -mcpu: an explicit triple alone is that triple's generic CPU.
    "aarch64": "llvm -mtriple=aarch64-unknown-linux-gnu",
}


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
