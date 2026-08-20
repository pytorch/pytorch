"""Stage-2 driver for the standard torch build: export AOT kernels,
generate the stub sources, and relink torch_cuda with them embedded.

A post-install step because the kernel builders package-import torch, and
scikit-build-core has no post-build hook inside the PEP 517 backend (the wheel
is assembled before torch is importable):

  * CI: .ci/pytorch/build.sh, after installing the built wheel, with --wheel so
    the relinked library is patched back into it before test jobs get it
  * dev: chained by `spin develop` / `spin install`; after a raw
    `pip install -e .`, run `python tools/native_aot/build_stage2.py`

Skips -- leaving a normal artifacts-free build -- when AOT kernels are not
applicable. Keep this list in sync with should_run(), which reports each one,
and with CONTRIBUTING.md, which states them for users:

  * TORCH_NATIVE_AOT=0 (the environment, else the CMake cache)
  * not Linux: everything downstream is ELF
  * the built torch does not import, or was built without CUDA
  * no toolchain targets this backend (Toolchain.BACKENDS), e.g. ROCm
  * CUDA older than _MIN_CUDA_MAJOR, or a version it cannot determine
  * the interpreter has no published DSL wheel and none is installed
  * a static torch_cuda, which cannot take the version script
  * nothing declares kernels (no torch/_native/ops/*/aot.py)
  * TORCH_CUDA_ARCH_LIST names no exportable arch (export.EXPORTABLE_ARCHES);
    with it unset, on-device export runs if a supported GPU is present

Past those gates the DSL runtimes are REQUIRED, and any failure fails the build:
a wheel missing declared kernels underperforms silently instead of failing.
TORCH_NATIVE_AOT=0 builds without them.
"""

import glob
import os
import shutil
import subprocess
import sys
import sysconfig
import zipfile


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
BUILD_DIR = os.path.join(REPO, "build")
# Where caffe2/CMakeLists.txt include()s native_aot.cmake from, spelled there as
# ${CMAKE_BINARY_DIR}/native_aot: the generator writes the file here, so a change
# on either side has to move both.
NATIVE_AOT_ARTIFACTS_DIR = os.path.join(BUILD_DIR, "native_aot")
# See export.py: as a script sys.path[0] is this directory, so the repo root
# has to go on the path for `tools.native_aot` to import from any cwd. Appended,
# not inserted, so the source torch/ tree never shadows the installed wheel.
sys.path.append(REPO)


def _report(msg: str) -> None:
    """Progress/diagnostics, on STDERR.

    Not stdout: --print-verdict writes a machine-read word there and the CI
    shells compare it with ==, so a report on the same stream corrupts the
    verdict of the one case that reports AND proceeds (multi-arch)."""
    print(f"-- native-AOT stage 2: {msg}", file=sys.stderr, flush=True)


def _torch_probe(expr: str) -> bool:
    """Evaluate a torch expression in a SUBPROCESS and return its truth.

    Three deliberate details, each for a failure an in-process check has:
      * subprocess: a torch that hard-crashes on import (an ASan build without
        the sanitizer LD_PRELOADed) must degrade to a skip, not kill the build
      * cwd=HERE: `python -c` puts the cwd on sys.path, so probing from the repo
        root imports the SOURCE torch/ tree instead of the installed wheel
      * verdict via stdout, not the exit code: a CUDA torch can segfault in
        interpreter teardown AFTER the expression evaluated (seen on B200)"""
    code = f"import torch; print('PROBE_OK' if ({expr}) else 'PROBE_NO', flush=True)"
    probe = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE
    )
    ok = "PROBE_OK" in probe.stdout
    if not ok and "PROBE_NO" not in probe.stdout:
        # torch failed to import or crashed. Degrading to a skip is by
        # design; doing it silently makes an import regression look like
        # an absent CUDA.
        _report_probe_failure(expr, probe.stderr, probe.returncode)
    return ok


def _torch_value(expr: str) -> str | None:
    """Evaluate a torch expression in a SUBPROCESS and return its str value.

    Out of process for _torch_probe's reasons: reading the device capability in
    here would initialize CUDA in the driver process."""
    code = f"import torch; print('NAOT_VALUE:' + str({expr}), flush=True)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE
    )
    for line in out.stdout.splitlines():
        if line.startswith("NAOT_VALUE:"):
            return line[len("NAOT_VALUE:") :]
    # Report WHY: swallowed, a real CUDA error (driver mismatch, mangled
    # CUDA_VISIBLE_DEVICES) reads as "local GPU is undetectable".
    _report_probe_failure(expr, out.stderr, out.returncode)
    return None


def _report_probe_failure(expr: str, stderr: str, returncode: int) -> None:
    """Say why a subprocess probe produced no verdict, on stderr.

    The returncode as well as the stderr: a child killed by a signal (OOM,
    import-time segfault) writes neither, and the caller then logs a confident
    wrong reason with nothing explaining it."""
    _report(f"probe {expr!r} produced no verdict (exit {returncode})")
    if stderr.strip():
        print(stderr.rstrip(), file=sys.stderr, flush=True)
    elif returncode < 0:
        _report(f"probe was killed by signal {-returncode} and said nothing")


def _arch_list() -> str:
    """The arch list THIS build targets: the environment, else the CMake cache.

    The cache half matters because CMake resolves the variable first and the
    environment only as its default (cmake/Dependencies.cmake), so a
    -DTORCH_CUDA_ARCH_LIST=... build has no such environment variable and used
    to export for the local GPU instead."""
    return os.getenv("TORCH_CUDA_ARCH_LIST") or (
        _cmake_cache_value("TORCH_CUDA_ARCH_LIST") or ""
    )


def _cmake_cache_value(name: str) -> str | None:
    """One entry from this build's CMakeCache.txt, or None if the key is absent.

    None rather than "": CMake calls a DEFINED-but-EMPTY entry false, so the two
    cannot be collapsed without the opt-out disagreeing with the build."""
    cache = os.path.join(BUILD_DIR, "CMakeCache.txt")
    try:
        with open(cache, errors="replace") as f:
            for line in f:
                key, sep, value = line.partition("=")
                if sep and key.split(":")[0] == name:
                    return value.strip()
    except OSError:
        # should_run() promises never to raise and this is the first thing it
        # reads. A root-created build/ gives PermissionError, a directory named
        # CMakeCache.txt gives IsADirectoryError; either turned --print-verdict
        # into a traceback, so the shell read "" != RUN and skipped the install
        # that the real invocation then demanded.
        _report(f"could not read {cache}; treating {name} as unset")
    return None


# What CMake treats as TRUE for a QUOTED if() argument, which is the form the
# generated native_aot.cmake uses. Both sides read the same variable, and a spelling
# only one honours means the build and stage 2 disagree about whether kernels were
# asked for.
_CMAKE_TRUE_VALUES = frozenset({"1", "on", "yes", "true", "y"})


def _cmake_false(value: str) -> bool:
    """Whether the generated native_aot.cmake reads this value as "do not embed".

    Stated as "not a TRUE constant", because that is what CMake does with the QUOTED
    argument that file uses: only 1/ON/YES/TRUE/Y and non-zero numbers are true, and
    everything else is false -- OFF, `00`, `xyz-NOTFOUND`, and junk like `disable`
    alike. An allowlist of falsy spellings instead left the two sides disagreeing on
    anything outside it, so the build skipped while stage 2 exported and relinked,
    then failed its own post-relink check complaining about the wrong thing."""
    v = value.strip().lower()
    if v in _CMAKE_TRUE_VALUES:
        return False
    try:
        return float(v) == 0.0
    except ValueError:
        return True


def _opted_out() -> bool:
    """Whether this build asked for no embedded kernels."""
    # An EMPTY value reads as absent, matching CMake (where "" is falsy) and
    # _arch_list: `TORCH_NATIVE_AOT=` is how a shell blanks a variable without
    # unsetting it, and counting that as "set" hid the cached opt-out below.
    env = os.getenv("TORCH_NATIVE_AOT") or ""
    if env:
        return _cmake_false(env)
    # The cache too (where -DTORCH_NATIVE_AOT=0 lands), so a manual stage-2 run in
    # a tree configured that way does not export, relink a library the generated
    # CMake embeds nothing into, and then fail its own post-relink check.
    cached = _cmake_cache_value("TORCH_NATIVE_AOT")
    # `is not None`, not truthiness: CMake calls a DEFINED-but-empty entry FALSE, so
    # `-DTORCH_NATIVE_AOT=` embeds nothing -- and reading that as "unset" had stage 2
    # export and relink for a build that wanted neither.
    if cached is not None and _cmake_false(cached):
        _report(
            f"disabled (TORCH_NATIVE_AOT={cached} in {BUILD_DIR}/CMakeCache.txt, "
            f"not in this environment; pass TORCH_NATIVE_AOT=1 to re-enable)"
        )
        return True
    return False


# The oldest CUDA major that gets embedded kernels. CUDA 12 tops out at sm_90
# (.ci/manywheel/build_env_setup.py's arch table) and every 13.x config builds
# sm_90 too, so a 12.x export is a strict subset of what the 13.x wheels already
# ship. In should_run() rather than per-toolchain because the CI shells install
# the DSL wheels only when it says RUN, so 12.x also skips the ~440MB install.
_MIN_CUDA_MAJOR = 13


def _cuda_major() -> int | None:
    """This build's CUDA major, or None if it cannot be determined.

    The cache first -- it is what the build was configured with -- then the
    installed torch, out of process.

    LAZILY, one probe at a time: as a tuple every probe ran, so the torch
    subprocess fired even when the cache had answered -- and where torch is not
    importable it reported a failed probe, which is noise in every build and broke a
    test that (correctly) expected a matching major to say nothing."""
    probes = (
        lambda: _cmake_cache_value("CUDAToolkit_VERSION_MAJOR") or "",
        lambda: (_cmake_cache_value("CUDA_VERSION") or "").split(".")[0],
        lambda: (_torch_value("torch.version.cuda or ''") or "").split(".")[0],
    )
    for probe in probes:
        value = probe()
        if value.strip().isdigit():
            return int(value)
    return None


def _archive_major(parts: list[str]) -> int:
    """The CUDA major a split archive's path claims, or -1 for an unsuffixed one."""
    for p in parts:
        if p.startswith("cu") and p[2:].isdigit():
            return int(p[2:])
    return -1


def _dsl_runtime_archive() -> str | None:
    """The DSL dialect runtime archive the CuTeDSL kernel objects need, preferring
    this build's CUDA major.

    4.5.x shipped one archive at <root>/lib/; 4.6.x splits it per major (cu12/lib/,
    cu13/lib/) and only cu12 is a hard dependency, so a CUDA 13 environment often
    holds cu12 alone. A mismatch warns rather than failing: in 4.6.2 the archives
    are the same objects (`ar p | md5sum` equal, differing only in ar timestamps)
    and a CUDA 13.2 build linked against cu12 passed the AOT suite. Still preferred
    and reported, since the per-major split says they may diverge."""
    import importlib.util

    spec = importlib.util.find_spec("nvidia_cutlass_dsl")
    if spec is None or not spec.submodule_search_locations:
        return None
    archive = "libcuda_dialect_runtime_static.a"
    found, split = [], False
    for root in spec.submodule_search_locations:
        for dirpath, _dirs, files in os.walk(root):
            if archive not in files:
                continue
            # Components RELATIVE to the package root. The absolute path also holds
            # the venv or conda directory, so an environment named cu12/cu13 was read
            # as the wheel's own per-major layout: it matched both archives and the
            # walk order picked one, or it made an unsplit pre-4.6 wheel look split
            # and be refused.
            parts = os.path.relpath(dirpath, root).split(os.sep)
            found.append((os.path.join(dirpath, archive), parts))
            split = split or _archive_major(parts) >= 0
    if not found:
        return None
    if not split:
        return found[0][0]
    major = _cuda_major()
    for path, parts in found:
        if f"cu{major}" in parts:
            return path
    # Highest major present, sorted first so the pick is deterministic.
    fallback = max(sorted(found), key=lambda f: _archive_major(f[1]))
    _report(
        f"the CuTeDSL wheel ships no dialect runtime for CUDA {major}; linking "
        f"{fallback[0]} instead (found: {', '.join(sorted(p for p, _ in found))}). "
        f"Only cu12 is a hard dependency of nvidia-cutlass-dsl, so install the "
        f"matching extra to be sure -- `pip install "
        f"'nvidia-cutlass-dsl[cu{major}]==<pinned version>'`, see install_cutlass_dsl "
        f"in .ci/pytorch/common_utils.sh."
    )
    return fallback[0]


def _artifact_size(art: str) -> str:
    """Summarize what is about to be linked in, as "N object(s), M.M MiB".

    Reported because nothing else states it, and these bytes scale with
    declarations x precompile points x arches: a wheel can grow tens of MiB from
    one added arch. Read from the generated CMake, NOT by walking the tree, which
    also holds artifacts that lost the arch tie-break (8.2 MiB where 4.7 shipped)."""
    from tools.native_aot.gen_aot_lib import CMAKE_INCLUDE

    listed = os.path.join(art, CMAKE_INCLUDE)
    if not os.path.exists(listed):
        return "nothing (no native_aot.cmake written)"
    # The target_sources() block ONLY: the objects also appear in the
    # set_source_files_properties() call above it, so scanning the whole file
    # counted them twice.
    with open(listed) as f:
        text = f.read()
    block = text.partition("target_sources(torch_cuda PRIVATE")[2].partition(")")[0]
    objects, sources = [], []
    for line in block.splitlines():
        line = line.strip()
        if not (line.startswith('"') and line.endswith('"')):
            continue
        path = line[1:-1]
        # The inverse of gen_aot_lib._cmake_str.
        for char in (";", "$", '"', "\\"):
            path = path.replace("\\" + char, char)
        (objects if path.endswith(".o") else sources).append(path)
    # Sources as well as objects: a kind that embeds its artifact in the generated
    # source (Triton's cubin bytes) contributes no object, so counting objects
    # alone reported 0.0 MiB for a build embedding megabytes.
    total = sum(os.path.getsize(p) for p in objects + sources if os.path.exists(p))
    return (
        f"{len(objects)} object(s) + {len(sources)} generated source(s), "
        f"{total / (1 << 20):.1f} MiB"
    )


def should_run() -> bool:
    """Whether stage 2 will export for this build. Answers from build
    properties ONLY, and never raises.

    Separate from require_runtimes() because the CI shells ask this to decide
    whether to INSTALL the DSL wheels (--print-verdict): a verdict that demanded
    them could only ever say "no" on a fresh image."""
    if _opted_out():
        _report("disabled (TORCH_NATIVE_AOT=0)")
        return False
    # Everything downstream is ELF: the relink targets libtorch_cuda.so, and the
    # version script plus --exclude-libs are GNU-ld options. Without this arm a
    # Windows or macOS CUDA build demanded wheels that do not exist for it.
    if sys.platform != "linux":
        _report(f"skipped (native-AOT kernels are Linux-only; this is {sys.platform})")
        return False
    if not _torch_probe("True"):
        _report("skipped (built torch not importable)")
        return False
    if not _torch_probe("torch.backends.cuda.is_built()"):
        _report("skipped (torch built without CUDA)")
        return False

    # Backend before runtimes: a backend with no AOT toolchain is "not
    # applicable", not "missing a wheel".
    from tools.native_aot import toolchains

    backend = _backend()
    usable = toolchains.for_backend(backend)
    if not usable:
        _report(f"skipped (no AOT toolchain targets {backend})")
        return False

    # Ahead of every runtime demand, reusing the backend probe above: a 12.x build
    # is "not applicable" like a non-Linux one. An undeterminable major counts as
    # too old, since _dsl_runtime_archive() cannot pick a runtime without it.
    if backend == "cuda":
        major = _cuda_major()
        if major is None or major < _MIN_CUDA_MAJOR:
            _report(
                f"skipped (CUDA {major if major is not None else 'version undetermined'};"
                f" embedded kernels need CUDA {_MIN_CUDA_MAJOR} or newer, since 12.x"
                f" tops out at sm_90 and every 13.x build already ships it)"
            )
            return False

    # Also "not applicable": the DSL wheels are cp-tagged and do not exist for
    # every interpreter the release matrix builds. For the pinned 4.6.2 the
    # -libs-* packages publish cp310-cp314 plus cp314t, manylinux only, no sdist,
    # while the matrix also builds 3.13t/3.15/3.15t. Those wheels keep the JIT
    # path; demanding an unresolvable tag would fail the build on nothing anyone
    # can fix.
    #
    # cp314t EXISTS, so the gate is NOT "free-threaded": that spelling skipped
    # 3.14t and shipped a kernel-free 3.14t CUDA wheel. What has no wheel is
    # free-threaded below 3.14, and anything past 3.14 either way.
    #
    # Py_GIL_DISABLED, not sys._is_gil_enabled(): the question is which wheel TAG
    # pip must find, an ABI property, and a free-threaded build still needs cp3XXt
    # with the GIL re-enabled -- where _is_gil_enabled() reports True and would
    # vote RUN. It is also what pip's own tag matching reads, and exists on 3.10.
    #
    # ANY, not all, and only when ABSENT: a kind needing no runtimes always
    # reports none missing, so `all` went permanently False once a second
    # toolchain registered.
    if any(tc.missing_runtimes() for tc in usable.values()):
        free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        ver = sys.version_info[:2]
        if ver > (3, 14) or (free_threaded and ver < (3, 14)):
            _report(
                f"skipped (no DSL wheel for python "
                f"{sys.version_info.major}.{sys.version_info.minor}"
                f"{'t' if free_threaded else ''}; installed ones are used if present)"
            )
            return False

    # A STATIC torch_cuda cannot take the version script that keeps the DSL's
    # entry points out of the public ABI (CMake silently discards
    # target_link_options for an archive), so they would join it with no warning.
    # Answered here rather than after a full export.
    # The same CMake-truthiness predicate as the opt-out, not a hand-rolled uppercase
    # tuple: CMakeLists.txt declares this with option(), EnvVarForwarding passes
    # BUILD_* through verbatim, and `BUILD_SHARED_LIBS=off` therefore produced a
    # static torch_cuda that this arm waved through.
    shared = _cmake_cache_value("BUILD_SHARED_LIBS")
    if shared is not None and _cmake_false(shared):
        _report(
            "skipped (BUILD_SHARED_LIBS=OFF; kernels cannot be embedded in a static torch_cuda)"
        )
        return False

    from tools.native_aot import export as export_mod

    if not glob.glob(os.path.join(export_mod.OPS_DIR, "*", "aot.py")):
        _report("skipped (no declarations under torch/_native/ops)")
        return False

    arch_list = _arch_list()
    if arch_list:
        archs = export_mod.archs_from_cuda_arch_list(arch_list)
        if not archs:
            _report(
                f"skipped (TORCH_CUDA_ARCH_LIST={arch_list!r} has no "
                f"exportable arch; exportable: "
                f"{' '.join(export_mod.EXPORTABLE_ARCHES)})"
            )
            return False
        if len(archs) > 1:
            # Supported (export nests one tree per arch and the generated stub
            # selects per capability); reported because it multiplies embedded
            # bytes, one full set per arch.
            _report(f"multi-arch: {' '.join(archs)}")
    elif not _torch_probe("torch.cuda.is_available()"):
        _report("skipped (no TORCH_CUDA_ARCH_LIST and no local GPU to detect from)")
        return False
    else:
        # On-device export compiles for whatever GPU is present, so check it
        # BEFORE committing: a dev box outside EXPORTABLE_ARCHES exported for its
        # own arch and then failed in generation, after a successful build.
        # Through a subprocess, not export._detected_arch(), which would
        # initialize CUDA here -- what _torch_probe exists to avoid.
        local = _torch_value("'sm_%d%d' % torch.cuda.get_device_capability()")
        if local not in export_mod.EXPORTABLE_ARCHES:
            _report(
                f"skipped (local GPU is {local or 'undetectable'}; exportable: "
                f"{' '.join(export_mod.EXPORTABLE_ARCHES)})"
            )
            return False

    return True


def require_runtimes() -> None:
    """Fail unless every toolchain that targets this build has its runtime.

    Called only once stage 2 is actually running, never from the verdict: every
    should_run() skip exports nothing, so demanding the wheels there fails builds
    that never wanted them. A build that WILL export and cannot would instead
    ship a wheel missing declared kernels, which surfaces as a performance
    regression rather than an error. TORCH_NATIVE_AOT=0 opts out."""
    from tools.native_aot import toolchains

    backend = _backend()
    usable = toolchains.for_backend(backend)
    gaps = {
        k: tc.missing_runtimes() for k, tc in usable.items() if tc.missing_runtimes()
    }
    if gaps:
        # DISTRIBUTION names, not REQUIRED_RUNTIMES' module names: they differ
        # (module `cutlass` ships in nvidia-cutlass-dsl), and this text used to
        # say `pip install cutlass tvm_ffi`, which installs unrelated packages.
        dists = sorted(
            {d for k in gaps for d in toolchains.get_toolchain(k).RUNTIME_DISTS}
        )
        detail = "; ".join(
            f"{k} needs {', '.join(ms)}" for k, ms in sorted(gaps.items())
        )
        raise RuntimeError(
            f"native-AOT stage 2: this {backend} build has AOT toolchains whose "
            f"runtimes are not installed ({detail}). Install the distributions "
            f"that provide them -- {', '.join(dists)} -- as "
            f"install_cutlass_dsl in .ci/pytorch/common_utils.sh does (it holds "
            f"the pinned versions), or set TORCH_NATIVE_AOT=0 to build without "
            f"embedded DSL kernels."
        )


def _backend() -> str:
    """ "cuda" or "rocm", as torch reports it. backends.cuda.is_built() is True on
    ROCm too, so torch.version.hip is the discriminator."""
    return "rocm" if _torch_probe("torch.version.hip is not None") else "cuda"


def _installed_lib_dir() -> str:
    """The lib/ directory of the INSTALLED torch package.

    Anchored on the compiled _C extension, not torch.__file__: an editable
    redirect install serves Python from the source tree while the compiled
    artifacts live in site-packages (as _load_dll_libraries does). In a
    subprocess like every torch touch here, and trusted on non-empty output even
    if that process then dies in teardown -- the path is printed first."""
    code = (
        "import importlib.util, os\n"
        "spec = importlib.util.find_spec('torch._C')\n"
        "if spec is not None and spec.origin:\n"
        "    print('NAOT_VALUE:' + os.path.dirname(spec.origin), flush=True)\n"
    )
    probe = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE
    )
    # Marked rather than taking stdout whole: this is joined into a path, so any
    # other line the child writes (a warning, a site hook) becomes part of it.
    for line in probe.stdout.splitlines():
        if line.startswith("NAOT_VALUE:"):
            return os.path.join(line[len("NAOT_VALUE:") :], "lib")
    _report_probe_failure("find_spec('torch._C')", probe.stderr, probe.returncode)
    raise RuntimeError(
        f"native-AOT stage 2: cannot locate the installed torch._C, so there is no "
        f"torch/lib to copy the relinked library into. The probe exited "
        f"{probe.returncode} having printed no path"
        + (" (its stderr is above)" if probe.stderr.strip() else " and nothing at all")
        + ". Install the wheel from this build first."
    )


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

    Member-by-member with zipfile rather than shelling out: the zip CLI is absent
    from the manywheel images, and repair_wheel.py already had to move off
    `wheel pack` for emitting invalid ZIP64 above 4GB (pytorch#189748), which a
    CUDA wheel with embedded kernels exceeds.
    """

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
                # Streamed, not read()+writestr(): several members run to
                # hundreds of MiB, and the read form holds each whole in
                # memory twice, on a machine already running a link.
                with src.open(info) as s, dst.open(info, "w") as d:
                    shutil.copyfileobj(s, d)
            # Both rewritten members keep their ORIGINAL compression, and the
            # RECORD needs its own ZipInfo to do so: writestr() with a plain name
            # takes the ZipFile default, and dst has none, so the two members this
            # function touches were the only ones stored uncompressed (~630 KiB
            # added to the artifact every test shard downloads).
            dst.write(lib_path, lib_rel, src.getinfo(lib_rel).compress_type)
            record_info = src.getinfo(record_rel)
            dst.writestr(record_info, record_text)
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
        "runtimes only when this says RUN, so one place owns the decision (their "
        "own probe disagreed with this one on GPU-less CUDA builders)",
    )
    args = parser.parse_args(argv)
    if args.print_verdict:
        print("RUN" if should_run() else "SKIP", flush=True)
        return 0
    # An EXPLICIT opt-out first, and only that: it is read from the environment and
    # the CMake cache, so it needs no torch, and honouring it keeps
    # TORCH_NATIVE_AOT=0 a real kill switch on the binary-build path -- which no
    # pull-request CI exercises, so a failure there first appears in the nightly
    # matrix. Every other gate stays behind the refusal below.
    if _opted_out():
        _report("TORCH_NATIVE_AOT is falsy; skipping")
        return 0
    # --wheel means the caller installed torch on the line above, so "not
    # importable" is a broken build rather than "not applicable". Degrading it to
    # a skip shipped a green, kernel-free release wheel.
    if args.wheel and not _torch_probe("True"):
        raise RuntimeError(
            "native-AOT stage 2: --wheel was given, so torch was just installed, "
            "but it does not import (stderr above). Refusing to patch a wheel "
            "without kernels. Fix the install, or do not pass --wheel for a build "
            "where a plain `import torch` cannot work -- the ASan and TSan images "
            "need LD_PRELOAD or a different interpreter, which is why the CI "
            "shells guard this call with a CUDA check. TORCH_NATIVE_AOT=0 does "
            "exempt it, and nothing else does: every other gate needs torch to "
            "decide, so this refusal stays ahead of them."
        )
    if not should_run():
        return 0
    require_runtimes()
    py = sys.executable
    art = NATIVE_AOT_ARTIFACTS_DIR
    _report("exporting kernels")
    export = [py, os.path.join(HERE, "export.py"), "--out-dir", art]
    # Read ONCE and used by both children below, so they cannot be told different
    # arch lists. Into the export child's ENVIRONMENT even when this build only had
    # it as a cache entry, or export compiles for a list generation never hears of.
    arch_list = _arch_list()
    env = dict(os.environ)
    if arch_list:
        env["TORCH_CUDA_ARCH_LIST"] = arch_list
    subprocess.check_call(export, cwd=REPO, env=env)
    _report("generating stub sources")
    gen = [py, os.path.join(HERE, "gen_aot_lib.py"), "--artifacts-dir", art]
    # The archive the generator names in the CMake it emits.
    if archive := _dsl_runtime_archive():
        gen += ["--dsl-runtime", archive]
    # Name the arches THIS build targets, so a tree left by a build with a
    # different TORCH_CUDA_ARCH_LIST is ignored rather than shipped. Omitted for an
    # on-device export, where export and generation resolve the arch identically.
    if arch_list:
        from tools.native_aot import export as export_mod

        # Both: --archs filters the trees, --arch-list is the raw value recorded
        # in the emitted CMake. Only this caller knows they are one request.
        gen += ["--archs", *export_mod.archs_from_cuda_arch_list(arch_list)]
        gen += ["--arch-list", arch_list]
    subprocess.check_call(gen, cwd=REPO)
    # Nothing generated is legitimate: no declaration ships kernels for the arch
    # this build targets. Stop rather than relink an unchanged library and then
    # assert kernels are in it, which would fail a build that did what was asked.
    if not glob.glob(os.path.join(art, "*", "aot_*.cpp")):
        _report("no declaration ships kernels for this build; nothing embedded")
        return 0
    _report(f"embedding {_artifact_size(art)}")
    # RECONFIGURE before relinking, explicitly: the generated file registers
    # itself in CMAKE_CONFIGURE_DEPENDS, but only from the reconfigure that first
    # READS it, and the build.ninja on disk predates this generation. Without it
    # the relink silently omits the kernels (417 MiB where 423 ships).
    # BEFORE the reconfigure, and keyed on the CACHE, not the directory: `cmake -B`
    # on a directory that does not exist exits 0 and configures FROM SCRATCH -- with
    # none of scikit-build-core's -D flags and none of the cache the shipped libtorch
    # was built from -- and the relink and copy below would then put that library
    # over the installed torch. Checked here rather than after the reconfigure, where
    # the directory always exists by construction and the guard could never fire.
    if not os.path.exists(os.path.join(BUILD_DIR, "CMakeCache.txt")):
        raise RuntimeError(
            f"native-AOT stage 2: {BUILD_DIR} holds no CMakeCache.txt, so it is not "
            f"the build this torch came from. It must match pyproject.toml's "
            f"build-dir; re-run the build from the repo root, or set "
            f"TORCH_NATIVE_AOT=0 to skip stage 2."
        )
    _report("reconfiguring to pick up the generated CMake")
    # Output CAPTURED, for two reasons. CMake prints most of its failure context on
    # stdout, so DEVNULL left a failing configure with nothing to read; and the STATUS
    # line the generated file emits is the only pre-relink evidence that the build
    # agrees it should embed. Requiring it here keeps every state where the two sides
    # disagree from reaching the relink -- and therefore from reaching the copy over
    # the installed torch, which used to happen BEFORE the check that catches it.
    # --log-level, because the marker below is a message(STATUS): EnvVarForwarding
    # forwards every CMAKE_* environment variable into the cache with FORCE, so
    # CMAKE_MESSAGE_LOG_LEVEL=WARNING (quietening configure output) hides it -- and
    # then persists in the cache after the variable is gone. Stage 2 would fail the
    # build advising -DTORCH_NATIVE_AOT=1, which is not the problem. The flag wins
    # over the cached value, and this output is captured, so nothing a user reads
    # gets noisier.
    configure = subprocess.run(
        ["cmake", "--log-level=STATUS", "-S", REPO, "-B", BUILD_DIR],
        capture_output=True,
        text=True,
    )
    if configure.returncode != 0:
        print(configure.stdout.rstrip(), file=sys.stderr, flush=True)
        print(configure.stderr.rstrip(), file=sys.stderr, flush=True)
        raise RuntimeError(
            f"native-AOT stage 2: reconfiguring {BUILD_DIR} failed (exit "
            f"{configure.returncode}, output above)."
        )
    from tools.native_aot.gen_aot_lib import EMBED_STATUS

    if EMBED_STATUS not in configure.stdout:
        raise RuntimeError(
            f"native-AOT stage 2: generation wrote "
            f"{os.path.join(art, 'native_aot.cmake')}, but the reconfigure did not "
            f"report embedding it. The build has declined the kernels this run just "
            f"exported -- most often TORCH_NATIVE_AOT is falsy in "
            f"{BUILD_DIR}/CMakeCache.txt (a -D from an earlier configure) while stage "
            f"2 was told to run. Reconfigure with -DTORCH_NATIVE_AOT=1, or set "
            f"TORCH_NATIVE_AOT=0 to skip stage 2 entirely."
        )
    # Relink JUST torch_cuda: `--target install` walks the whole install manifest
    # (~15 min) where this is seconds. The hand copy below is then byte-identical
    # to what install writes, but ONLY because the generated CMake sets
    # BUILD_WITH_INSTALL_RPATH -- otherwise install rewrites RPATH and the copy
    # ships the builder's build/lib instead of $ORIGIN. One mechanism; do not move
    # either half alone.
    _report("relinking torch_cuda with embedded kernels")
    # pyproject.toml aliases MAX_JOBS only inside scikit-build-core's own
    # subprocess, so this relink otherwise ran at ninja's default (ncpu+2) while
    # each generated .cpp pulls the whole ATen/CUDA header set.
    relink = ["cmake", "--build", ".", "--target", "torch_cuda"]
    jobs = os.getenv("MAX_JOBS") or os.getenv("CMAKE_BUILD_PARALLEL_LEVEL")
    if jobs and jobs.isdigit():
        relink += ["--parallel", jobs]
    subprocess.check_call(relink, cwd=BUILD_DIR)

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
    backup = installed + ".naot.bak"
    try:
        shutil.copy2(built, staged)
        # The displaced library is kept until the verification below passes, because
        # this overwrites an INSTALLED torch and a relink that embedded nothing would
        # otherwise stay there after the build failed.
        #
        # A HARDLINK, not a rename: `installed` must never stop existing. Renaming it
        # away first made the swap two steps, and a kill between them (Ctrl-C during
        # the `spin develop` that chains this, an OOM after writing ~400 MiB, ENOSPC
        # on the rename) left the environment with NO library and the `finally` below
        # deleting the only other copy -- strictly worse than the un-verified library
        # the backup exists to prevent. The link shares the inode, so it also costs
        # nothing, where the rename left ~400 MiB behind on every successful build.
        if os.path.exists(backup):
            os.remove(backup)  # a stale one, from a run that died before its cleanup
        try:
            os.link(installed, backup)
        except OSError:
            # No hardlinks here (an exotic site-packages filesystem). Proceed without
            # a restore rather than fail the build: the swap below is still atomic, so
            # the worst case is the un-verified library staying put, which is what
            # this whole block improves on rather than depends on.
            _report(f"cannot hardlink {installed}; proceeding without a restore copy")
            backup = None
        os.replace(staged, installed)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    _report(f"{os.path.getsize(built) >> 20} MiB relinked into {installed}")
    # Size is not evidence: this script's artifacts dir must agree with the one
    # caffe2/CMakeLists.txt include()s, and a mismatch lets the relink succeed
    # while embedding nothing -- a kernel-free wheel with a green build.
    if not _torch_probe("torch._native._native_aot_embedded()"):
        # Put the previous library back before failing: leaving site-packages holding
        # a library this run has just declared unusable makes every later `import
        # torch` in that environment the build's problem too.
        if backup:
            os.replace(backup, installed)
        raise RuntimeError(
            "native-AOT stage 2: relinked libtorch_cuda reports no embedded "
            "kernels (torch._native._native_aot_embedded() is False). The "
            f"artifacts in {art} were not linked in -- check that this build's "
            f"CMAKE_BINARY_DIR is {BUILD_DIR} (it must match pyproject.toml's "
            f"build-dir), and that {art}/native_aot.cmake exists and is the file "
            f"caffe2/CMakeLists.txt includes."
        )
    if backup:
        # Verified, so drop the restore copy. It is a hardlink, so this frees the
        # displaced library's blocks rather than a second copy of them -- but leaving
        # it would still keep the OLD inode alive in site-packages after every build,
        # invisible to `pip uninstall` because it is in no RECORD.
        os.remove(backup)
    if args.wheel:
        _report(f"embedding into {args.wheel}")
        patch_wheel(args.wheel, built)
    _report("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
