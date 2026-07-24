"""Stage-2 driver for the standard torch build: export AOT kernels,
generate the stub sources, and relink torch_cuda with them embedded.

Invoked by tools/build_pytorch_libs.py after the main cmake build (the
kernel builders package-import torch, so torch must be built first;
the source tree plus the freshly installed torch/lib is importable at
that point, for both editable and wheel builds).

Skips -- leaving a normal artifacts-free build -- when any prerequisite
is missing, so the standard build NEVER hard-depends on the DSL stack:

  * NATIVE_AOT=0 in the environment (explicit opt-out)
  * no CUDA build (USE_CUDA off / no nvcc toolchain in the build)
  * DSL runtime not importable (nvidia_cutlass_dsl not installed)
  * TORCH_CUDA_ARCH_LIST contains no exportable arch (Blackwell only,
    for now -- see export.EXPORT_SMS); on-device export runs when the
    arch list is unset and a supported GPU is present

A failure AFTER the skip checks is a real error and fails the build:
silently shipping a wheel without the kernels it was asked to embed is
worse than failing loudly (pass NATIVE_AOT=0 to bypass).
"""

import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))


def _report(msg: str) -> None:
    print(f"-- native-AOT stage 2: {msg}", flush=True)


def _torch_importable() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def should_run() -> bool:
    if os.getenv("NATIVE_AOT", "1") == "0":
        _report("disabled (NATIVE_AOT=0)")
        return False
    try:
        import nvidia_cutlass_dsl  # noqa: F401
    except ImportError:
        _report("skipped (nvidia_cutlass_dsl not installed)")
        return False
    if not _torch_importable():
        _report("skipped (built torch not importable)")
        return False
    import torch

    if not torch.backends.cuda.is_built():
        _report("skipped (torch built without CUDA)")
        return False
    arch_list = os.getenv("TORCH_CUDA_ARCH_LIST")
    if arch_list:
        sys.path.insert(0, HERE)
        import export as export_mod

        if not export_mod.archs_from_cuda_arch_list(arch_list):
            _report(
                f"skipped (TORCH_CUDA_ARCH_LIST={arch_list!r} has no "
                f"exportable arch; supported: {' '.join(export_mod.EXPORT_SMS)})"
            )
            return False
    elif not torch.cuda.is_available():
        _report("skipped (no TORCH_CUDA_ARCH_LIST and no local GPU to detect from)")
        return False
    return True


def main() -> int:
    if not should_run():
        return 0
    py = sys.executable
    _report("exporting kernels")
    subprocess.check_call([py, os.path.join(HERE, "export.py")], cwd=REPO)
    _report("generating stub sources")
    subprocess.check_call([py, os.path.join(HERE, "gen_aot_lib.py")], cwd=REPO)
    # Relink JUST torch_cuda: the embedded glob (CONFIGURE_DEPENDS)
    # picks up the new artifacts. A full `--target install` would also
    # work but walks the whole install manifest (~15 min); the targeted
    # relink is seconds. The hand copy into torch/lib (what both wheel
    # packaging and editable installs ship from) is exactly what
    # install would do for this file: verified byte-identical, no
    # RPATH/install-time fixup applies to torch_cuda.
    _report("relinking torch_cuda with embedded kernels")
    build_dir = os.path.join(REPO, "build")
    subprocess.check_call(
        ["cmake", "--build", ".", "--target", "torch_cuda"], cwd=build_dir
    )
    import shutil

    built = os.path.join(build_dir, "lib", "libtorch_cuda.so")
    installed = os.path.join(REPO, "torch", "lib", "libtorch_cuda.so")
    if not os.path.exists(built):
        raise RuntimeError(f"expected relinked library at {built}")
    shutil.copy2(built, installed)
    _report(f"done ({os.path.getsize(built) >> 20} MiB relinked into torch/lib)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
