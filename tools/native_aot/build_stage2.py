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

  * TORCH_NATIVE_AOT=0 (explicit opt-out); read from the environment,
    else from the CMake cache, which is where a configure recorded it
  * not Linux: everything downstream is ELF (the relink targets
    libtorch_cuda.so, and the version script is a GNU-ld option)
  * the built torch does not import, or was built without CUDA
  * no toolchain targets this build's backend (Toolchain.BACKENDS); a
    ROCm build skips here today, and gains AOT support by adding a
    toolchain class rather than by editing this gate
  * the interpreter has no published DSL wheel and none is installed
    (free-threaded below 3.14, or newer than the published cp tags)
  * nothing declares kernels (no torch/_native/ops/*/aot.py)
  * TORCH_CUDA_ARCH_LIST contains no exportable arch (Hopper and
    Blackwell today -- see export.EXPORTABLE_ARCHES); on-device export
    runs when the arch list is unset and a supported GPU is present.
    Several exportable arches export one tree per arch, and the
    generated stub selects per compute capability at runtime.

Keep this list in sync with should_run() and with CONTRIBUTING.md.

Only once every one of those checks has passed are the DSL runtimes
REQUIRED, not optional: a toolchain that targets this backend was asked
for declared kernels, and a wheel missing some of them underperforms
silently instead of failing. So a missing runtime -- or any later
failure -- fails the build, while a build that was going to skip anyway
never demands them. Set TORCH_NATIVE_AOT=0 to build without embedded DSL
kernels.
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
# Must match NATIVE_AOT_ARTIFACTS_DIR in caffe2/CMakeLists.txt, which is a plain
# (deliberately non-cache) ${CMAKE_BINARY_DIR}/native_aot: the relink below only
# embeds what the manifest in THAT directory names.
NATIVE_AOT_ARTIFACTS_DIR = os.path.join(BUILD_DIR, "native_aot")
# See export.py: as a script sys.path[0] is this directory, so the repo root
# has to go on the path for `tools.native_aot` to import from any cwd. Appended,
# not inserted, so the source torch/ tree never shadows the installed wheel.
sys.path.append(REPO)


def _report(msg: str) -> None:
    """Progress/diagnostics, on STDERR.

    Not stdout: --print-verdict writes a machine-read word there, and the CI
    shells compare it with ==. A report line landing on the same stream made
    the multi-arch build (the one case that reports AND proceeds) answer
    "-- native-AOT stage 2: multi-arch: sm_90 sm_100\\nRUN", which matches
    nothing. Build logs interleave both streams, so nothing else changes."""
    print(f"-- native-AOT stage 2: {msg}", file=sys.stderr, flush=True)


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
    if not ok and "PROBE_NO" not in probe.stdout:
        # The expression never evaluated, so torch itself failed to import or
        # crashed. That degrades to a skip by design, but silently swallowing
        # the reason makes a real import regression look like an absent CUDA.
        _report_probe_failure(expr, probe.stderr, probe.returncode)
    return ok


def _torch_value(expr: str) -> str | None:
    """Evaluate a torch expression in a SUBPROCESS and return its str value.

    The value-returning sibling of _torch_probe, and required for the same
    reason: reading e.g. the local device capability by importing torch HERE
    would initialize CUDA in the driver process and let an import that
    hard-crashes take the build script down with it -- the exact failure mode
    _torch_probe was introduced for."""
    code = f"import torch; print('NAOT_VALUE:' + str({expr}), flush=True)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE
    )
    for line in out.stdout.splitlines():
        if line.startswith("NAOT_VALUE:"):
            return line[len("NAOT_VALUE:") :]
    # No value: report WHY. Swallowing it turned a real CUDA error (driver/runtime
    # mismatch, cudaErrorNoDevice, a mangled CUDA_VISIBLE_DEVICES) into
    # "local GPU is undetectable" -- on a dev box that is the difference between
    # "install a driver" and "your hardware is unsupported".
    _report_probe_failure(expr, out.stderr, out.returncode)
    return None


def _report_probe_failure(expr: str, stderr: str, returncode: int) -> None:
    """Say why a subprocess probe produced no verdict, on stderr.

    The returncode is reported as well as the stderr because a child killed by a
    SIGNAL writes neither: an OOM kill or an import-time segfault produced
    rc=-9/-11 with both streams empty, so the caller logged a confident wrong
    reason ("torch built without CUDA") with nothing at all explaining it."""
    _report(f"probe {expr!r} produced no verdict (exit {returncode})")
    if stderr.strip():
        print(stderr.rstrip(), file=sys.stderr, flush=True)
    elif returncode < 0:
        _report(f"probe was killed by signal {-returncode} and said nothing")


def _arch_list() -> str:
    """The arch list THIS build targets: the environment, else the CMake cache.

    The cache half matters because CMake resolves the variable first and the
    environment only as its default (cmake/Dependencies.cmake), so a build
    configured with -DTORCH_CUDA_ARCH_LIST=... has no such environment variable.
    Reading only the environment made stage 2 record ARCH_LIST_ABSENT for those
    builds, which caffe2/CMakeLists.txt treats as "no claim, take the artifacts as
    they are" -- so the staleness guard was dead exactly there, and a -D build
    embedded kernels for whatever arch the previous build or the local GPU
    produced."""
    return os.getenv("TORCH_CUDA_ARCH_LIST") or _cmake_cache_value(
        "TORCH_CUDA_ARCH_LIST"
    )


def _cmake_cache_value(name: str) -> str:
    """One entry from this build's CMakeCache.txt, or "" if there is none."""
    cache = os.path.join(BUILD_DIR, "CMakeCache.txt")
    try:
        with open(cache, errors="replace") as f:
            for line in f:
                key, sep, value = line.partition("=")
                if sep and key.split(":")[0] == name:
                    return value.strip()
    except OSError:
        # should_run() promises never to raise, and this is the FIRST thing it
        # reads: a build/ created by a root or container run and then rebuilt as a
        # normal user gives PermissionError, a directory named CMakeCache.txt gives
        # IsADirectoryError. Either turned --print-verdict into a traceback, so the
        # shell read "" != RUN, skipped installing the runtimes, and the real
        # invocation then failed demanding them.
        _report(f"could not read {cache}; treating {name} as unset")
    return ""


# What CMake treats as false for an if(<var>) test, lower-cased. The two sides
# have to agree: a spelling one honours and the other ignores means the build
# embeds kernels while stage 2 thinks it opted out, or the reverse.
_OPT_OUT_VALUES = frozenset({"0", "off", "false", "no", "n", "ignore", "notfound"})


def _opted_out() -> bool:
    """Whether this build asked for no embedded kernels.

    Environment first, then the CMake cache, because caffe2/CMakeLists.txt caches
    the variable so the opt-out survives a reconfigure with no environment. If
    only the environment were read here, a manual stage-2 run in a tree
    configured with TORCH_NATIVE_AOT=0 would export, relink a library CMake had
    already declined to embed anything into, and then fail its own post-relink
    "reports no embedded kernels" check."""
    # An EMPTY value reads as absent, matching the CMake side (where "" is falsy)
    # and _arch_list here: otherwise TORCH_NATIVE_AOT="" -- which is how a shell
    # blanks a variable it does not want to unset -- would count as "set, and not
    # 0", so the cached opt-out below could never be reached.
    env = os.getenv("TORCH_NATIVE_AOT") or ""
    if env:
        return env.strip().lower() in _OPT_OUT_VALUES
    # The cache, so a manual stage-2 run in a tree configured with the opt-out does
    # not export, relink a library CMake declined to embed anything into, and then
    # fail its own post-relink "reports no embedded kernels" check.
    cached = _cmake_cache_value("TORCH_NATIVE_AOT")
    if cached and cached.strip().lower() in _OPT_OUT_VALUES:
        _report(
            f"disabled (TORCH_NATIVE_AOT={cached} in {BUILD_DIR}/CMakeCache.txt, "
            f"not in this environment; pass TORCH_NATIVE_AOT=1 to re-enable)"
        )
        return True
    return False


def _dsl_runtime_archive() -> str | None:
    """The DSL dialect runtime archive the CuTeDSL kernel objects need, matched to
    this build's CUDA major.

    In Python because it is a lookup, not a policy: CMake globbed for the file,
    matched the major with a regex on the path, and had a FATAL_ERROR arm -- for
    something importlib answers directly.

    The major MUST match. 4.5.x shipped one archive at <root>/lib/; 4.6.x splits it
    per CUDA major (<root>/cu12/lib/, <root>/cu13/lib/) and -- the trap -- only
    cu12 is a hard dependency of nvidia-cutlass-dsl, with cu13 behind an extra. So
    a plain install on a CUDA 13 build leaves ONLY the cu12 archive present, and
    taking whatever is there would quietly link a cu12-built runtime into a CUDA 13
    libtorch_cuda. Refusing names the fix instead. The unsuffixed path is accepted
    only when the wheel has no split at all (pre-4.6)."""
    import importlib.util

    spec = importlib.util.find_spec("nvidia_cutlass_dsl")
    if spec is None or not spec.submodule_search_locations:
        return None
    # The cache first (it is what the build was configured with), then torch out of
    # process. Unknown must not become an empty match: the first version of this
    # raised "ships no dialect runtime for CUDA " and told the user to install
    # `nvidia-cutlass-dsl[cu]`, for a tree whose cache simply had no CUDA entry.
    major = (
        _cmake_cache_value("CUDAToolkit_VERSION_MAJOR")
        or (_cmake_cache_value("CUDA_VERSION").split(".")[0])
    )
    if not major:
        reported = _torch_value("torch.version.cuda or ''") or ""
        major = reported.split(".")[0]
    found, split = [], False
    for root in spec.submodule_search_locations:
        for dirpath, _dirs, files in os.walk(root):
            if "libcuda_dialect_runtime_static.a" in files:
                path = os.path.join(dirpath, "libcuda_dialect_runtime_static.a")
                found.append(path)
                if os.sep + "cu" in path and any(
                    part.startswith("cu") and part[2:].isdigit()
                    for part in path.split(os.sep)
                ):
                    split = True
    if not found:
        return None
    if split and not major:
        raise RuntimeError(
            f"native-AOT stage 2: the CuTeDSL wheel ships per-CUDA-major dialect "
            f"runtimes ({', '.join(sorted(found))}) but this build's CUDA major "
            f"could not be determined -- neither {BUILD_DIR}/CMakeCache.txt nor the "
            f"installed torch reports one. Linking an arbitrary one is not safe; "
            f"configure the build first, or set TORCH_NATIVE_AOT=0."
        )
    if split:
        want = f"{os.sep}cu{major}{os.sep}"
        for path in found:
            if want in path:
                return path
        raise RuntimeError(
            f"native-AOT stage 2: the CuTeDSL wheel ships no dialect runtime for "
            f"CUDA {major} (found: {', '.join(sorted(found))}). Only cu12 is a hard "
            f"dependency of nvidia-cutlass-dsl; install the matching extra, e.g. "
            f"`pip install 'nvidia-cutlass-dsl[cu{major}]==<pinned version>'` (see "
            f"install_cutlass_dsl in .ci/pytorch/common_utils.sh), or set "
            f"TORCH_NATIVE_AOT=0 to build without embedded kernels. Linking the "
            f"cu12 runtime into a CUDA {major} library instead is not safe."
        )
    return found[0]


def _artifact_size(art: str) -> str:
    """Summarize what is about to be linked in, as "N object(s), M.M MiB".

    Reported because nothing else states it: these bytes land in
    libtorch_cuda, and they scale with declarations x precompile points x
    arches -- so a wheel can grow tens of MiB from an arch added to
    TORCH_CUDA_ARCH_LIST, with no line in the build log that says so.

    Read from the generator's manifest, NOT by walking the tree: the tree also
    holds artifacts that lost the arch tie-break or belong to an earlier build's
    arch list, and those are not linked. Walking it reported 8.2 MiB where
    4.7 MiB shipped."""
    from tools.native_aot.gen_aot_lib import CMAKE_INCLUDE

    listed = os.path.join(art, CMAKE_INCLUDE)
    if not os.path.exists(listed):
        return "nothing (no manifest written)"
    # The emitted target_sources() block ONLY: the objects are also listed in the
    # set_source_files_properties() call above it, so scanning every quoted line in
    # the file counted them twice ("4 object(s)" for two).
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
    # SOURCES as well as objects: a kind that embeds its artifact in the generated
    # source (Toolchain.link_exts empty -- Triton's cubin bytes) contributes no
    # OBJECT= line at all, so counting objects alone reported "0 object(s),
    # 0.0 MiB" for a build embedding megabytes, which is exactly what this line
    # exists to make visible.
    total = sum(os.path.getsize(p) for p in objects + sources if os.path.exists(p))
    return (
        f"{len(objects)} object(s) + {len(sources)} generated source(s), "
        f"{total / (1 << 20):.1f} MiB"
    )


def should_run() -> bool:
    """Whether stage 2 will export for this build. Answers from build
    properties ONLY, and never raises.

    Separate from require_runtimes() because of who asks and when: the CI
    shells ask this to decide whether to INSTALL the DSL wheels
    (--print-verdict), so a verdict that itself demanded those wheels could
    only ever say "no" on a fresh image -- and did, by raising, leaving the
    shells to skip the install and the real invocation to fail on the very
    runtimes the verdict was meant to request."""
    if _opted_out():
        _report("disabled (TORCH_NATIVE_AOT=0)")
        return False
    # Linux only, and stated here rather than assumed. Everything downstream is
    # ELF: the relink targets build/lib/libtorch_cuda.so (Windows produces
    # torch_cuda.dll in build/bin), and the version script plus --exclude-libs are
    # GNU-ld options. Without this arm a Windows or macOS CUDA build with a GPU and
    # a declaration reached require_runtimes() and demanded wheels that do not
    # exist for it.
    if sys.platform != "linux":
        _report(f"skipped (native-AOT kernels are Linux-only; this is {sys.platform})")
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

    backend = _backend()
    usable = toolchains.for_backend(backend)
    if not usable:
        _report(f"skipped (no AOT toolchain targets {backend})")
        return False

    # Same "not applicable" category: the DSL wheels are cp-tagged and simply do
    # not exist for every interpreter the release matrix builds. Verified against
    # PyPI for the pinned 4.6.2 (2026-08-18): the -libs-* packages publish
    # cp310-cp314 plus cp314-cp314t, manylinux only, with no sdist -- while the
    # matrix also builds 3.13t, 3.15 and 3.15t. pip cannot resolve them there
    # however it tries, so asking would fail the wheel build on something no one
    # can fix; not-applicable is the honest answer, and those wheels keep the JIT
    # path.
    #
    # cp314t EXISTS, so free-threaded alone is not the question -- gating on it
    # skipped 3.14t and shipped a kernel-free 3.14t CUDA wheel, which is the
    # silent underperformance this module exists to prevent. What has no wheel is
    # free-threaded BELOW 3.14 (no cp313t anywhere in the chain, base included)
    # and anything past 3.14, threaded or not.
    #
    # Only when they are ABSENT: if they import, they are installable here
    # whatever PyPI publishes today, so a newer interpreter needs no edit here.
    #
    # Py_GIL_DISABLED, not sys._is_gil_enabled(). The question is which wheel TAG
    # pip must find, which is an ABI property: a free-threaded build still needs
    # a cp3XXt wheel even when the GIL has been re-enabled (PYTHON_GIL=1, or any
    # extension without Py_mod_gil), and _is_gil_enabled() reports True there --
    # so on 3.14t it would say "not free-threaded", vote RUN, and hand pip a tag
    # that does not exist. This config var is what pip's own tag matching reads
    # (packaging/tags.py), it is public, and it exists on 3.10 -- the matrix
    # floor -- while _is_gil_enabled() is 3.13+.
    # ANY, not all: a kind that needs no runtimes at all (Triton compiles with
    # the wheel torch already depends on) always reports none missing, so `all`
    # became permanently False the moment a second toolchain was registered and
    # this gate stopped firing entirely. The question is "does this build have to
    # install something?", and one missing runtime is enough.
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

    # Nothing DECLARES kernels: also "not applicable", and it must be decided
    # before the runtimes are demanded. Otherwise a tree with no aot.py -- an
    # earlier commit of this stack, or a bisect -- fails a build for want of
    # ~190MB of wheels it would then use to export exactly nothing.
    # A STATIC torch_cuda cannot take the version script that keeps the DSL's
    # entry points out of the public ABI (CMake silently discards
    # target_link_options for an archive), so caffe2/CMakeLists.txt refuses to
    # embed there. Skipping HERE as well makes it the same "not applicable" answer
    # as the Linux check, instead of a full export followed by a configure error
    # that every later build of that tree repeats.
    if _cmake_cache_value("BUILD_SHARED_LIBS") in ("OFF", "0", "FALSE", "NO"):
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
        #
        # Asked through a subprocess, not export._detected_arch(): that imports
        # torch and initializes CUDA in THIS process, which is what _torch_probe
        # exists to avoid -- and a teardown segfault there (seen on the b200
        # builder) would fail a stage 2 that had otherwise succeeded.
        from tools.native_aot import export as export_mod

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

    Called ONLY once stage 2 is actually running, never from the verdict:
    every skip in should_run() means stage 2 exports nothing, so demanding
    the DSL wheels there fails builds that never wanted them. But a build
    that WILL export and cannot is the silent-partial failure the sidecar and
    orphan checks exist to prevent -- it ships a wheel missing some declared
    kernels, which surfaces as a performance regression rather than an error.
    TORCH_NATIVE_AOT=0 is the supported way to build without the DSL wheels."""
    from tools.native_aot import toolchains

    backend = _backend()
    usable = toolchains.for_backend(backend)
    gaps = {
        k: tc.missing_runtimes() for k, tc in usable.items() if tc.missing_runtimes()
    }
    if gaps:
        # DISTRIBUTION names, not the importable module names in
        # REQUIRED_RUNTIMES: this is the only error text a developer sees for the
        # stack's one hard failure, and it told them to `pip install cutlass
        # tvm_ffi`, which installs unrelated packages. The two genuinely differ --
        # module `cutlass` ships in nvidia-cutlass-dsl.
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
        "    print('NAOT_VALUE:' + os.path.dirname(spec.origin), flush=True)\n"
    )
    probe = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE
    )
    # Marked like the other probes rather than taking stdout whole: this value is
    # joined into a filesystem path, so any other line the child writes (a
    # deprecation warning, a site hook) would become part of it, and the failure
    # then surfaces as the misleading "not the torch this tree built".
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

    Rewritten member-by-member with zipfile rather than shelling out: the zip
    CLI is absent from the manywheel images (they install unzip only), and
    .ci/manywheel/repair_wheel.py already had to move off `wheel pack` because
    it emitted invalid ZIP64 above 4GB (pytorch#189748) -- a CUDA wheel with
    embedded kernels is squarely in that range. Copying members preserves each
    entry's existing compression instead of recompressing the archive.
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
                # Streamed, not read()+writestr(): a CUDA wheel has several
                # members in the hundreds of MiB (libtorch_cpu.so, bundled
                # cuBLAS/cuDNN), and the read form holds each one whole in
                # memory, twice, on a build machine already running a link.
                with src.open(info) as s, dst.open(info, "w") as d:
                    shutil.copyfileobj(s, d)
            # Keep the member's ORIGINAL compression. Storing it uncompressed
            # grows the wheel by the whole ratio of a ~1GB .so -- and in CI that
            # wheel is the artifact every test shard downloads.
            dst.write(lib_path, lib_rel, src.getinfo(lib_rel).compress_type)
            # The RECORD too, via its own ZipInfo: writestr() with a plain NAME
            # takes the ZipFile's default compression, and dst was opened without
            # one -- so the one member this function rewrites was the one member
            # stored uncompressed. Measured on a real torch RECORD: 1,087,255 B
            # stored where it deflates to 439,747 B, i.e. ~630 KiB added to the
            # artifact every test shard downloads.
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
        "runtimes only when this says RUN, so the decision lives in one place "
        "(their own `python -c` probe disagreed with _torch_probe on GPU-less "
        "CUDA builders, where a CUDA torch can segfault in teardown)",
    )
    args = parser.parse_args(argv)
    if args.print_verdict:
        print("RUN" if should_run() else "SKIP", flush=True)
        return 0
    # --wheel means a CI caller that installed torch on the line before this one,
    # so "not importable" there is a broken build, not "not applicable". Degrading
    # it to a skip shipped a green, kernel-free release wheel -- and in the
    # manywheel container, where the wheel is installed unrepaired and
    # --no-deps, that is a plausible state rather than a hypothetical one.
    if args.wheel and not _torch_probe("True"):
        raise RuntimeError(
            "native-AOT stage 2: --wheel was given, so torch was just installed, "
            "but it does not import (stderr above). Refusing to patch a wheel "
            "without kernels. Fix the install, or do not pass --wheel for a build "
            "where a plain `import torch` cannot work -- the ASan and TSan images "
            "need LD_PRELOAD or a different interpreter, which is why the CI "
            "shells guard this call with a CUDA check. TORCH_NATIVE_AOT=0 does NOT "
            "exempt it: this refusal is deliberately ahead of every applicability "
            "gate, since deciding them requires importing torch."
        )
    if not should_run():
        return 0
    require_runtimes()
    py = sys.executable
    art = NATIVE_AOT_ARTIFACTS_DIR
    _report("exporting kernels")
    export = [py, os.path.join(HERE, "export.py"), "--out-dir", art]
    # TORCH_CUDA_ARCH_LIST in the child's environment even when this build only
    # ever had it as a CMake cache entry: export.py translates it into the arches
    # it compiles for, and letting it read a bare environment made the export
    # target a different arch list than the one recorded in the manifest.
    env = dict(os.environ)
    if arch_list := _arch_list():
        env["TORCH_CUDA_ARCH_LIST"] = arch_list
    subprocess.check_call(export, cwd=REPO, env=env)
    _report("generating stub sources")
    gen = [py, os.path.join(HERE, "gen_aot_lib.py"), "--artifacts-dir", art]
    # The archive the generator names in the CMake it emits. Found HERE rather
    # than in CMake, which used to glob for it and match the CUDA major by regex.
    if archive := _dsl_runtime_archive():
        gen += ["--dsl-runtime", archive]
    # Name the arches THIS build targets, so a tree from a build with a
    # different TORCH_CUDA_ARCH_LIST is ignored instead of shipped. Omitted for
    # an on-device export, where the local GPU is the whole arch list and
    # export resolves it the same way generation would.
    arch_list = _arch_list()
    if arch_list:
        from tools.native_aot import export as export_mod

        # Both: --archs is what generation filters trees by, --arch-list is the
        # raw value it records in the manifest for CMake to compare against. Only
        # this caller knows they describe the same request.
        gen += ["--archs", *export_mod.archs_from_cuda_arch_list(arch_list)]
        gen += ["--arch-list", arch_list]
    subprocess.check_call(gen, cwd=REPO)
    # Nothing generated means nothing to embed, and that is legitimate: no
    # declaration ships kernels for the arch this build targets (or none
    # declares any yet). Stop here rather than relinking an unchanged library
    # and then asserting kernels are in it -- that assertion would fail a build
    # that did exactly what was asked.
    if not glob.glob(os.path.join(art, "*", "aot_*.cpp")):
        _report("no declaration ships kernels for this build; nothing embedded")
        return 0
    _report(f"embedding {_artifact_size(art)}")
    # Relink JUST torch_cuda: the manifest (CONFIGURE_DEPENDS) makes the
    # reconfigure ahead of this relink pick up the new artifacts. A full
    # `--target install` would also work but walks the whole install manifest
    # (~15 min); the targeted relink is seconds. The hand copy into the installed
    # torch/lib is then byte-identical to what install writes -- but only BECAUSE
    # caffe2/CMakeLists.txt sets BUILD_WITH_INSTALL_RPATH on torch_cuda in the
    # embedding branch. Without it install rewrites RPATH and the copy ships the
    # builder's own build/lib instead of $ORIGIN, so that property and this copy
    # are one mechanism; do not move either alone.
    # RECONFIGURE first, explicitly. caffe2/CMakeLists.txt include()s the file the
    # generator just wrote, and the generated file registers itself in
    # CMAKE_CONFIGURE_DEPENDS -- but only from the reconfigure that first READS it.
    # The build.ninja on disk was generated before the file existed (or before this
    # generation rewrote it), so `cmake --build --target torch_cuda` alone would
    # relink WITHOUT the kernels: measured, a 417 MiB library where 423 MiB ships,
    # and the post-relink check below then fails the build. Owning the sequencing
    # here is the point of moving the logic out of CMake.
    _report("reconfiguring to pick up the generated CMake")
    subprocess.check_call(
        ["cmake", "-S", REPO, "-B", BUILD_DIR], stdout=subprocess.DEVNULL
    )
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
    # Honour the build's parallelism knob. pyproject.toml aliases MAX_JOBS to
    # CMAKE_BUILD_PARALLEL_LEVEL only inside scikit-build-core's own subprocess,
    # so this relink otherwise ran at ninja's default (ncpu+2) -- and each
    # generated .cpp pulls the whole ATen/CUDA header set, which is why PyTorch
    # already carries knobs like FLASH_ATTENTION_MAX_JOBS for OOM.
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
    try:
        shutil.copy2(built, staged)
        os.replace(staged, installed)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    _report(f"{os.path.getsize(built) >> 20} MiB relinked into {installed}")
    # Size is not evidence. This script's artifacts dir must agree with the one
    # caffe2/CMakeLists.txt derives from CMAKE_BINARY_DIR, and CONFIGURE_DEPENDS
    # is Ninja/Makefile only -- both let the relink succeed while embedding
    # nothing, which would ship a kernel-free wheel with a green build.
    if not _torch_probe("torch._native._native_aot_embedded()"):
        raise RuntimeError(
            "native-AOT stage 2: relinked libtorch_cuda reports no embedded "
            "kernels (torch._native._native_aot_embedded() is False). The "
            f"artifacts in {art} were not linked in -- check that this build's "
            f"CMAKE_BINARY_DIR is {BUILD_DIR} (it must match pyproject.toml's "
            f"build-dir), and that {art}/native_aot.cmake exists and is the file "
            f"caffe2/CMakeLists.txt includes."
        )
    if args.wheel:
        _report(f"embedding into {args.wheel}")
        patch_wheel(args.wheel, built)
    _report("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
