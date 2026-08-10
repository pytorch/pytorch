"""Stage-2 driver for the standard torch build: export AOT kernels,
generate the stub sources, and relink torch_cuda with them embedded.

The kernel builders package-import torch, so stage 2 needs a BUILT,
INSTALLED torch -- and scikit-build-core has no post-build hook inside
the PEP 517 backend (the wheel is assembled before torch is ever
importable). Stage 2 is therefore a post-install step:

  * CI: .ci/pytorch/build.sh runs it after installing the built wheel,
    then reassembles the wheel (incremental, via the persistent build
    dir) so the artifact shipped to test jobs embeds the kernels
  * dev: `spin develop` / `spin install` chain it after the pip
    install; after a raw `pip install -e .`, run it manually:
    python tools/native_aot/build_stage2.py

Skips -- leaving a normal artifacts-free build -- when any prerequisite
is missing, so the standard build NEVER hard-depends on the DSL stack:

  * NATIVE_AOT=0 in the environment (explicit opt-out)
  * no CUDA build (USE_CUDA off / no nvcc toolchain in the build)
  * DSL runtime not importable (nvidia_cutlass_dsl or tvm_ffi, the
    same pair torch/_native/cutedsl_utils.py gates the JIT layer on)
  * TORCH_CUDA_ARCH_LIST contains no exportable arch (Blackwell only,
    for now -- see export.EXPORTABLE_ARCHES); on-device export runs when
    arch list is unset and a supported GPU is present

A failure AFTER the skip checks is a real error and fails the build:
silently shipping a wheel without the kernels it was asked to embed is
worse than failing loudly (pass NATIVE_AOT=0 to bypass).
"""

import importlib.util
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))


def _load_export():
    # By file path (like the sibling tools): `import export` after a
    # sys.path insert works at runtime but is opaque to type checkers.
    spec = importlib.util.spec_from_file_location(
        "export", os.path.join(HERE, "export.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load tools/native_aot/export.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    if os.getenv("NATIVE_AOT", "1") == "0":
        _report("disabled (NATIVE_AOT=0)")
        return False
    for dist in ("nvidia_cutlass_dsl", "tvm_ffi"):
        if importlib.util.find_spec(dist) is None:
            _report(f"skipped ({dist} not installed)")
            return False
    if not _torch_probe("True"):
        _report("skipped (built torch not importable)")
        return False
    if not _torch_probe("torch.backends.cuda.is_built()"):
        _report("skipped (torch built without CUDA)")
        return False
    arch_list = os.getenv("TORCH_CUDA_ARCH_LIST")
    if arch_list:
        export_mod = _load_export()
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
    _report("exporting kernels")
    subprocess.check_call([py, os.path.join(HERE, "export.py")], cwd=REPO)
    _report("generating stub sources")
    subprocess.check_call([py, os.path.join(HERE, "gen_aot_lib.py")], cwd=REPO)
    # Relink JUST torch_cuda: the embedded glob (CONFIGURE_DEPENDS)
    # picks up the new artifacts. A full `--target install` would also
    # work but walks the whole install manifest (~15 min); the targeted
    # relink is seconds. The hand copy into the installed torch/lib is
    # exactly what install would do for this file: verified
    # byte-identical, no RPATH/install-time fixup applies to torch_cuda.
    _report("relinking torch_cuda with embedded kernels")
    build_dir = os.path.join(REPO, "build")
    subprocess.check_call(
        ["cmake", "--build", ".", "--target", "torch_cuda"], cwd=build_dir
    )
    import shutil

    built = os.path.join(build_dir, "lib", "libtorch_cuda.so")
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
