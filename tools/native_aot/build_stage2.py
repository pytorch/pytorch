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
  * TORCH_CUDA_ARCH_LIST contains no exportable arch (Hopper and
    Blackwell today -- see export.EXPORTABLE_ARCHES); on-device export
    runs when the arch list is unset and a supported GPU is present.
    Several exportable arches export one tree per arch, and the
    generated stub selects per compute capability at runtime.

Only once every one of those checks has passed are the DSL runtimes
REQUIRED, not optional: a toolchain that targets this backend was asked
for declared kernels, and a wheel missing some of them underperforms
silently instead of failing. So a missing runtime -- or any later
failure -- fails the build, while a build that was going to skip anyway
never demands them. Set TORCH_NATIVE_AOT=0 to build without embedded DSL
kernels.
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
# has to go on the path for `tools.native_aot` to import from any cwd. Appended,
# not inserted, so the source torch/ tree never shadows the installed wheel.
sys.path.append(REPO)


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
    ok = "PROBE_OK" in probe.stdout
    if not ok and "PROBE_NO" not in probe.stdout and probe.stderr.strip():
        # The expression never evaluated, so torch itself failed to import or
        # crashed. That degrades to a skip by design, but silently swallowing
        # the reason makes a real import regression look like an absent CUDA.
        _report(f"probe {expr!r} produced no verdict; stderr follows")
        print(probe.stderr.rstrip(), flush=True)
    return ok


def _artifact_size(art: str) -> str:
    """Summarize what is about to be linked in, as "N object(s), M.M MiB".

    Reported because nothing else states it: these bytes land in
    libtorch_cuda, and they scale with declarations x precompile points x
    arches -- so a wheel can grow tens of MiB from an arch added to
    TORCH_CUDA_ARCH_LIST, with no line in the build log that says so."""
    from tools.native_aot import toolchains

    # Every artifact ext except headers: a .o is linked in and a .cubin is
    # embedded as bytes by its launcher, but an ABI header only feeds the
    # compiler and contributes nothing to the shipped library.
    exts = toolchains.all_artifact_exts() - {".h"}
    sizes = [
        os.path.getsize(os.path.join(root, fn))
        for root, _, files in os.walk(art)
        for fn in files
        if os.path.splitext(fn)[1] in exts
    ]
    return f"{len(sizes)} object(s), {sum(sizes) / (1 << 20):.1f} MiB"


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

    arch_list = os.getenv("TORCH_CUDA_ARCH_LIST")
    if arch_list:
        from tools.native_aot import export as export_mod

        archs = export_mod.archs_from_cuda_arch_list(arch_list)
        if not archs:
            _report(
                f"skipped (TORCH_CUDA_ARCH_LIST={arch_list!r} has no "
                f"exportable arch; exportable: "
                f"{' '.join(export_mod.EXPORTABLE_ARCHES)})"
            )
            return False
        if len(archs) > 1:
            # Supported: export nests one tree per arch, gen_aot_lib walks both
            # depths and emits a per-capability selector, and the embedded-link
            # globs cover both. Reported because it multiplies embedded kernel
            # bytes -- one full set per arch.
            _report(f"multi-arch: {' '.join(archs)}")
    elif not _torch_probe("torch.cuda.is_available()"):
        _report("skipped (no TORCH_CUDA_ARCH_LIST and no local GPU to detect from)")
        return False
    else:
        # On-device export compiles for whatever GPU is present, so check it is
        # one we can export for BEFORE committing. Without this, a dev box
        # outside EXPORTABLE_ARCHES (sm_86, sm_120) exports for its own arch and
        # then fails in generation -- after a successful build, and leaving a
        # tree that cannot even configure until build/native_aot is removed.
        from tools.native_aot import export as export_mod

        local = export_mod._detected_arch()
        if local not in export_mod.EXPORTABLE_ARCHES:
            _report(
                f"skipped (local GPU is {local or 'undetectable'}; exportable: "
                f"{' '.join(export_mod.EXPORTABLE_ARCHES)})"
            )
            return False

    # LAST of the checks, deliberately: every skip above means stage 2 exports
    # nothing, and demanding the DSL wheels for a build that was going to skip
    # anyway just fails builds that never wanted them. Only once we know we
    # WILL export are the runtimes required -- shipping a wheel with some of
    # the declared kernels missing is the silent-partial failure the sidecar
    # and orphan checks exist to prevent, and it surfaces as a performance
    # regression rather than an error. TORCH_NATIVE_AOT=0 is the supported way
    # to build without the DSL wheels.
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
    """Replace torch/lib/libtorch_cuda.so inside an already-built wheel and fix
    its RECORD entry.

    Rewritten member-by-member with zipfile rather than shelling out: the zip
    CLI is absent from the manywheel images (they install unzip only), and
    .ci/manywheel/repair_wheel.py already had to move off `wheel pack` because
    it emitted invalid ZIP64 above 4GB (pytorch#189748) -- a CUDA wheel with
    embedded kernels is squarely in that range. Copying members preserves each
    entry's existing compression instead of recompressing the archive.
    """
    import shutil
    import zipfile

    lib_rel = "torch/lib/libtorch_cuda.so"
    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()
        records = [n for n in names if n.endswith(".dist-info/RECORD")]
        if lib_rel not in names or len(records) != 1:
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
    record_text = "\n".join(record_lines) + "\n"

    # Rebuild beside the original and rename over it, so an interrupted rewrite
    # cannot leave a half-written wheel where a valid one used to be.
    tmp_whl = wheel_path + ".naot.tmp"
    replaced = {lib_rel, record_rel}
    try:
        with (
            zipfile.ZipFile(wheel_path) as src,
            zipfile.ZipFile(tmp_whl, "w", allowZip64=True) as dst,
        ):
            for info in src.infolist():
                if info.filename in replaced:
                    continue
                dst.writestr(info, src.read(info.filename), info.compress_type)
            dst.write(lib_path, lib_rel, zipfile.ZIP_STORED)
            dst.writestr(record_rel, record_text)
        shutil.move(tmp_whl, wheel_path)
    finally:
        if os.path.exists(tmp_whl):
            os.remove(tmp_whl)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        default=None,
        help="also embed the relinked libtorch_cuda into this built wheel "
        "(CI: the dist/*.whl handed to test jobs must carry the kernels)",
    )
    parser.add_argument(
        "--print-verdict",
        action="store_true",
        help="print RUN or SKIP and exit; the shell callers install the DSL "
        "runtimes only when this says RUN, so the decision lives in one place "
        "(their own `python -c` probe disagreed with _torch_probe on GPU-less "
        "CUDA builders, where a CUDA torch can segfault in teardown)",
    )
    args = parser.parse_args(argv)
    if args.print_verdict:
        print("RUN" if should_run() else "SKIP", flush=True)
        return 0
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
    _report(f"embedding {_artifact_size(art)}")
    # Relink JUST torch_cuda: the embedded glob (CONFIGURE_DEPENDS)
    # picks up the new artifacts. A full `--target install` would also
    # work but walks the whole install manifest (~15 min); the targeted
    # relink is seconds. The hand copy into the installed torch/lib is
    # exactly what install would do for this file: verified
    # byte-identical, no RPATH/install-time fixup applies to torch_cuda.
    _report("relinking torch_cuda with embedded kernels")
    if not os.path.isdir(BUILD_DIR):
        # BUILD_DIR mirrors pyproject.toml's build-dir; an override there (or
        # a build run from another directory) lands here as a bare
        # FileNotFoundError out of subprocess.
        raise RuntimeError(
            f"native-AOT stage 2: build directory {BUILD_DIR} does not exist. "
            f"It must match pyproject.toml's build-dir; re-run the build from "
            f"the repo root, or set TORCH_NATIVE_AOT=0 to skip stage 2."
        )
    subprocess.check_call(
        ["cmake", "--build", ".", "--target", "torch_cuda"], cwd=BUILD_DIR
    )
    import shutil

    built = os.path.join(BUILD_DIR, "lib", "libtorch_cuda.so")
    if not os.path.exists(built):
        raise RuntimeError(f"expected relinked library at {built}")
    installed = os.path.join(_installed_lib_dir(), "libtorch_cuda.so")
    if not os.path.exists(installed):
        # Refuse to create it: _installed_lib_dir found *a* torch, and writing a
        # library into a layout that never had one means we are pointed at the
        # wrong environment.
        raise RuntimeError(
            f"native-AOT stage 2: {installed} does not exist, so the torch on "
            f"sys.path is not the one this tree built. Install the wheel from "
            f"this build first, or set TORCH_NATIVE_AOT=0."
        )
    # Temp file + rename: copying in place truncates the library other processes
    # may be mapping, and a failure part-way (ENOSPC, EPERM on a root-owned
    # site-packages) would leave a torch that cannot import at all.
    staged = installed + ".naot.tmp"
    try:
        shutil.copy2(built, staged)
        os.replace(staged, installed)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    _report(f"{os.path.getsize(built) >> 20} MiB relinked into {installed}")
    # S2: size is not evidence. The artifacts dir is a CMake CACHE PATH while
    # this script hardcodes one, and CONFIGURE_DEPENDS is Ninja/Makefile only --
    # both let the relink succeed while embedding nothing, which would ship a
    # kernel-free wheel with a green build.
    if not _torch_probe("torch._native._native_aot_embedded()"):
        raise RuntimeError(
            "native-AOT stage 2: relinked libtorch_cuda reports no embedded "
            "kernels (torch._native._native_aot_embedded() is False). The "
            f"artifacts in {art} were not linked in -- check that CMake's "
            f"NATIVE_AOT_ARTIFACTS_DIR matches, and that the generator honors "
            f"CONFIGURE_DEPENDS."
        )
    if args.wheel:
        _report(f"embedding into {args.wheel}")
        patch_wheel(args.wheel, built)
    _report("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
