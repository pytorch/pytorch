"""Stage-2 driver for the standard torch build: export AOT kernels,
generate the stub sources, and relink torch_cuda with them embedded.

The kernel builders package-import torch, so stage 2 needs a BUILT,
INSTALLED torch -- and scikit-build-core has no post-build hook inside
the PEP 517 backend (the wheel is assembled before torch is ever
importable). Stage 2 is therefore a post-install step:

  * CI: .ci/pytorch/build.sh runs it after installing the built wheel,
    passing --wheel so the relinked library is patched back into that
    wheel (see patch_wheel) before it ships to test jobs
  * dev: `spin develop` / `spin install` chain it after the pip
    install; after a raw `pip install -e .`, run it manually:
    python tools/native_aot/build_stage2.py

Skips -- leaving a normal artifacts-free build -- when AOT kernels are
not applicable to this build:

  * TORCH_NATIVE_AOT=0 in the environment (explicit opt-out)
  * no CUDA build (USE_CUDA off / no nvcc toolchain in the build)
  * no toolchain targets this build's backend (Toolchain.BACKENDS); a
    ROCm build skips here today, and gains AOT support by adding a
    toolchain class rather than by editing this gate
  * TORCH_CUDA_ARCH_LIST contains no exportable arch (Blackwell only,
    for now -- see export.EXPORTABLE_ARCHES); on-device export runs when
    arch list is unset and a supported GPU is present

Past those checks the DSL runtimes are REQUIRED, not optional: a
toolchain that targets this backend was asked for declared kernels, and
a wheel missing some of them underperforms silently instead of failing.
So a missing runtime -- or any later failure -- fails the build. Set
TORCH_NATIVE_AOT=0 to build without embedded DSL kernels.
"""

import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
BUILD_DIR = os.path.join(REPO, "build")
# Must match the NATIVE_AOT_ARTIFACTS_DIR cache var of caffe2/CMakeLists.txt,
# which defaults to ${CMAKE_BINARY_DIR}/native_aot: the relink below only sees
# artifacts the embedded glob picks up.
NATIVE_AOT_ARTIFACTS_DIR = os.path.join(BUILD_DIR, "native_aot")
# See export.py: as a script sys.path[0] is this directory, so the repo root
# has to go on the path for `tools.native_aot` to import from any cwd.
sys.path.insert(0, REPO)


def _report(msg: str) -> None:
    print(f"-- native-AOT stage 2: {msg}", flush=True)


def _torch_probe(expr: str) -> bool:
    """Evaluate a torch expression in a SUBPROCESS and return its truth.
    Never imports torch in this process: a torch that hard-crashes on
    import (e.g. an ASan build aborting because the sanitizer runtime
    is not LD_PRELOADed into plain python) must degrade to a skip, not
    kill the build script -- an in-process ImportError-only check
    misses those, and any crash would also poison the later checks.

    cwd=HERE: `python -c` puts the cwd on sys.path (unlike `python
    script.py`, which uses the script's dir), so probing from the repo
    root would import the SOURCE torch/ tree instead of the installed
    wheel and always fail.

    The verdict travels via stdout, not the exit code: a CUDA torch
    can segfault in interpreter-shutdown teardown on GPU-less
    machines AFTER the expression evaluated (observed on the B200
    build job), which would corrupt an exit-code verdict."""
    code = f"import torch; print('PROBE_OK' if ({expr}) else 'PROBE_NO', flush=True)"
    probe = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE
    )
    return "PROBE_OK" in probe.stdout


def should_run() -> bool:
    if os.getenv("TORCH_NATIVE_AOT", "1") == "0":
        _report("disabled (TORCH_NATIVE_AOT=0)")
        return False
    if not _torch_probe("True"):
        _report("skipped (built torch not importable)")
        return False
    if not _torch_probe("torch.backends.cuda.is_built()"):
        _report("skipped (torch built without CUDA)")
        return False

    # Platform BEFORE runtimes: which toolchains are even candidates is a
    # property of the build, so a backend with no AOT toolchain is "not
    # applicable", not "missing a wheel". torch.version.hip is the
    # discriminator -- backends.cuda.is_built() is True on ROCm too.
    from tools.native_aot import toolchains

    backend = "rocm" if _torch_probe("torch.version.hip is not None") else "cuda"
    usable = toolchains.for_backend(backend)
    if not usable:
        _report(f"skipped (no AOT toolchain targets {backend})")
        return False

    # Past here the backend HAS toolchains, so their runtimes are REQUIRED,
    # not optional: shipping a wheel with only some of the declared kernels
    # is the silent-partial-artifact failure the sidecar and orphan checks
    # exist to prevent, and it would show up as a performance regression
    # rather than an error. Set TORCH_NATIVE_AOT=0 to opt out of AOT
    # kernels entirely -- that is the supported way to build without the
    # DSL wheels.
    gaps = {
        k: tc.missing_runtimes() for k, tc in usable.items() if tc.missing_runtimes()
    }
    if gaps:
        detail = "; ".join(
            f"{k} needs {', '.join(ms)}" for k, ms in sorted(gaps.items())
        )
        raise RuntimeError(
            f"native-AOT stage 2: this {backend} build has AOT toolchains whose "
            f"runtimes are not installed ({detail}). Install them, or set "
            f"TORCH_NATIVE_AOT=0 to build without embedded DSL kernels."
        )
    arch_list = os.getenv("TORCH_CUDA_ARCH_LIST")
    if arch_list:
        from tools.native_aot import export as export_mod

        if not export_mod.archs_from_cuda_arch_list(arch_list):
            _report(
                f"skipped (TORCH_CUDA_ARCH_LIST={arch_list!r} has no "
                f"exportable arch; exportable: "
                f"{' '.join(export_mod.EXPORTABLE_ARCHES)})"
            )
            return False
    elif not _torch_probe("torch.cuda.is_available()"):
        _report("skipped (no TORCH_CUDA_ARCH_LIST and no local GPU to detect from)")
        return False
    return True


def _installed_lib_dir() -> str:
    """The lib/ directory of the INSTALLED torch package. Anchored on
    the compiled _C extension rather than torch.__file__: editable
    redirect installs serve Python from the source tree while compiled
    artifacts live in site-packages (same trick as _load_dll_libraries
    in torch/__init__.py). Wheel installs resolve both to the same dir.

    Resolved in a subprocess like every other torch touch in this
    driver: find_spec("torch._C") imports the parent torch package,
    and a CUDA torch can segfault in interpreter-shutdown teardown on
    GPU-less build machines -- which would fail the build AFTER stage
    2 finished (observed on the B200 build job). The probe prints the
    path BEFORE its interpreter exits, so trust non-empty output even
    if the probe process then dies in teardown."""
    code = (
        "import importlib.util, os\n"
        "spec = importlib.util.find_spec('torch._C')\n"
        "if spec is not None and spec.origin:\n"
        "    print(os.path.dirname(spec.origin), flush=True)\n"
    )
    probe = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE
    )
    out = probe.stdout.strip()
    if not out:
        raise RuntimeError("cannot locate installed torch._C")
    return os.path.join(out, "lib")


def _wheel_hash_and_size(path: str) -> tuple[str, int]:
    import base64
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()
    return f"sha256={digest}", os.path.getsize(path)


def patch_wheel(wheel_path: str, lib_path: str) -> None:
    """Replace torch/lib/libtorch_cuda.so inside an already-built wheel
    and fix its RECORD entry. The zip CLI updates members in place
    without recompressing the rest of the archive (~30s vs several
    minutes for a full `python -m build` reassembly, which would also
    re-walk the whole cmake install manifest)."""
    import shutil
    import tempfile
    import zipfile

    if shutil.which("zip") is None:
        raise RuntimeError("--wheel requires the zip CLI (in-place member update)")
    lib_rel = "torch/lib/libtorch_cuda.so"
    with zipfile.ZipFile(wheel_path) as zf:
        records = [n for n in zf.namelist() if n.endswith(".dist-info/RECORD")]
        if lib_rel not in zf.namelist() or len(records) != 1:
            raise RuntimeError(f"{wheel_path}: not a torch wheel ({lib_rel}/RECORD)")
        record_rel = records[0]
        record_lines = zf.read(record_rel).decode().splitlines()

    digest, size = _wheel_hash_and_size(lib_path)
    for i, line in enumerate(record_lines):
        if line.startswith(lib_rel + ","):
            record_lines[i] = f"{lib_rel},{digest},{size}"
            break
    else:
        raise RuntimeError(f"{wheel_path}: RECORD has no entry for {lib_rel}")

    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, os.path.dirname(lib_rel)))
        os.makedirs(os.path.join(td, os.path.dirname(record_rel)))
        shutil.copy2(lib_path, os.path.join(td, lib_rel))
        with open(os.path.join(td, record_rel), "w") as f:
            f.write("\n".join(record_lines) + "\n")
        subprocess.check_call(
            ["zip", "-q", os.path.abspath(wheel_path), lib_rel, record_rel], cwd=td
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        default=None,
        help="also embed the relinked libtorch_cuda into this built wheel "
        "(CI: the dist/*.whl handed to test jobs must carry the kernels)",
    )
    args = parser.parse_args(argv)
    if not should_run():
        return 0
    py = sys.executable
    art = NATIVE_AOT_ARTIFACTS_DIR
    _report("exporting kernels")
    export = [py, os.path.join(HERE, "export.py"), "--out-dir", art]
    subprocess.check_call(export, cwd=REPO)
    _report("generating stub sources")
    gen = [py, os.path.join(HERE, "gen_aot_lib.py"), "--artifacts-dir", art]
    subprocess.check_call(gen, cwd=REPO)
    # Relink JUST torch_cuda: the embedded glob (CONFIGURE_DEPENDS)
    # picks up the new artifacts. A full `--target install` would also
    # work but walks the whole install manifest (~15 min); the targeted
    # relink is seconds. The hand copy into the installed torch/lib is
    # exactly what install would do for this file: verified
    # byte-identical, no RPATH/install-time fixup applies to torch_cuda.
    _report("relinking torch_cuda with embedded kernels")
    subprocess.check_call(
        ["cmake", "--build", ".", "--target", "torch_cuda"], cwd=BUILD_DIR
    )
    import shutil

    built = os.path.join(BUILD_DIR, "lib", "libtorch_cuda.so")
    if not os.path.exists(built):
        raise RuntimeError(f"expected relinked library at {built}")
    installed = os.path.join(_installed_lib_dir(), "libtorch_cuda.so")
    shutil.copy2(built, installed)
    _report(f"{os.path.getsize(built) >> 20} MiB relinked into {installed}")
    if args.wheel:
        _report(f"embedding into {args.wheel}")
        patch_wheel(args.wheel, built)
    _report("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
