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

Two modes, split by the skip list.

  * Before: no stage-2 work, so a normal wheel with the generated stubs in place
    but no kernels behind them
  * After: stage 2 exports and embeds, runtimes required to progress, failures
    fail the build.

TORCH_NATIVE_AOT=0 opts out, and every entry point honours it; it is expected to
stay constant across a build's phases.

Assumes the torch it finds installed is the one this tree just built, which every
caller above arranges, and verifies that rather than repairing it.
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
# Where the configure records the arch list it resolved, relative to the build
# directory: a contract with cmake/Codegen.cmake, so the path is spelled once.
# Joined onto BUILD_DIR at call time, like _cmake_cache_value, so a test can move it.
ARCH_LIST_RECORD = os.path.join("native_aot", "arch_list.txt")
# Where caffe2/CMakeLists.txt include()s native_aot.cmake from, as
# ${CMAKE_BINARY_DIR}/native_aot: a change on either side has to move both.
NATIVE_AOT_ARTIFACTS_DIR = os.path.join(BUILD_DIR, "native_aot")
# See export.py: as a script sys.path[0] is this directory, so the repo root
# has to go on the path for `tools.native_aot` to import from any cwd. Appended,
# not inserted, so the source torch/ tree never shadows the installed wheel.
sys.path.append(REPO)


def _report(msg: str) -> None:
    """Progress and diagnostics on stderr, since --print-verdict owns stdout."""
    print(f"-- native-AOT stage 2: {msg}", file=sys.stderr, flush=True)


# Long enough for a cold `import torch` on a machine already running a build,
# short enough that a wedged one fails rather than holding the job to its step
# limit with nothing on either stream.
_PROBE_TIMEOUT_SECONDS = 300


def _run_probe(code: str, expr: str) -> subprocess.CompletedProcess[str] | None:
    """One probe subprocess, or None if it could not be run at all.

    Bounded and guarded because should_run() answers from build properties and
    degrades rather than raising: the SPAWN fails on its own (a fork returning
    EAGAIN at the end of a MAX_JOBS build, an interpreter missing from the image),
    and `import torch` against a wedged driver never returns. Either way
    --print-verdict wrote nothing, the CI shell read that as neither RUN nor SKIP,
    skipped the DSL install, and the real stage-2 run then demanded it."""
    try:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=HERE,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as e:
        _report(f"probe {expr!r} could not run ({type(e).__name__}: {e})")
        return None


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
    probe = _run_probe(code, expr)
    if probe is None:
        return False
    ok = "PROBE_OK" in probe.stdout
    if not ok and "PROBE_NO" not in probe.stdout:
        # torch failed to import or crashed. Degrading to a skip is deliberate, but
        # reported: silently, a broken import reads as an absent CUDA.
        _report_probe_failure(expr, probe.stderr, probe.returncode)
    return ok


def _torch_value(expr: str) -> str | None:
    """Evaluate a torch expression in a SUBPROCESS and return its str value.

    Out of process for _torch_probe's reasons: reading the device capability in
    here would initialize CUDA in the driver process."""
    code = f"import torch; print('NAOT_VALUE:' + str({expr}), flush=True)"
    out = _run_probe(code, expr)
    if out is None:
        return None
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
    """The arch list THIS build targets, as the CONFIGURE resolved it.

    Read from the file cmake/Codegen.cmake records, because that is the only place
    the resolved value exists: EnvVarForwarding forwards the environment into the
    cache only when the variable is undefined (so -D wins), and Dependencies.cmake
    then shadows the cache with the environment for the rest of the configure. Either
    source read alone is therefore wrong for one of the two kinds of build. Reading
    the record also means a developer who changes the environment without
    reconfiguring gets the list the build COMPILED for, not the one they now intend.

    Recorded-but-empty is not the same as no record: it means the configure resolved
    no arch list, and the on-device path takes over. Absent means nothing configured
    this tree (a hand run, an older build dir), so fall back to the env-then-cache
    reading the configure itself performs."""
    try:
        with open(os.path.join(BUILD_DIR, ARCH_LIST_RECORD)) as f:
            return f.read().strip()
    except OSError:
        return os.getenv("TORCH_CUDA_ARCH_LIST") or (
            _cmake_cache_value("TORCH_CUDA_ARCH_LIST") or ""
        )


def _cmake_cache_value(name: str) -> str | None:
    """One entry from this build's CMakeCache.txt, or None if the key is absent.

    None rather than "": CMake calls a DEFINED-but-EMPTY entry false, so the two
    cannot be collapsed without the opt-out disagreeing with the build."""
    cache = os.path.join(BUILD_DIR, "CMakeCache.txt")
    found = None
    try:
        with open(cache, errors="replace") as f:
            for line in f:
                key, sep, value = line.partition("=")
                if sep and key.split(":")[0] == name:
                    # The LAST assignment, as CMake itself takes it: appending a
                    # line to flip a setting without reconfiguring is a normal
                    # thing to do, and taking the first read the opposite value
                    # from the build that wrote it.
                    found = value.strip()
    except OSError:
        # should_run() promises never to raise and this is the first thing it
        # reads. A root-created build/ gives PermissionError, a directory named
        # CMakeCache.txt gives IsADirectoryError; either turned --print-verdict
        # into a traceback, so the shell read "" != RUN and skipped the install
        # that the real invocation then demanded.
        _report(f"could not read {cache}; treating {name} as unset")
    return found


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
    # Matched EXACTLY, no strip(): re-derived from `cmake -P` over the spellings
    # that reach here, where " 1" is true and " y" is false.
    if value.lower() in _CMAKE_TRUE_VALUES:
        return False
    # Everything else is CMake's number parse, which tolerates LEADING whitespace
    # and not trailing ("1 " and "1\n" are both false, and a value out of a
    # $(grep ...) or a folded YAML scalar arrives exactly that way), reads hex as a
    # C literal ("0x1" true, "0x0" false), and has never heard of float()'s
    # python-only "1_0".
    v = value.lstrip()
    if v != v.rstrip() or "_" in v:
        return True
    try:
        return float(v) == 0.0
    except ValueError:
        pass
    body = v[1:] if v[:1] in "+-" else v
    if body[:2].lower() == "0x":
        try:
            return int(v, 16) == 0
        except ValueError:
            pass
    return True


def _opted_out() -> bool:
    """Whether this build asked for no embedded kernels."""
    # An EMPTY value reads as absent, matching CMake (where "" is falsy) and
    # _arch_list: `TORCH_NATIVE_AOT=` is how a shell blanks a variable without
    # unsetting it, and counting that as "set" hid the cached opt-out below.
    env = os.getenv("TORCH_NATIVE_AOT") or ""
    if env:
        if not _cmake_false(env):
            return False
        # Reported HERE, once, naming the value and its source. Two callers each
        # printed their own line afterwards, and both were wrong for the cache case:
        # "TORCH_NATIVE_AOT is falsy" where the environment holds nothing, and
        # "TORCH_NATIVE_AOT=0" for a cache entry that says OFF.
        _report(f"disabled (TORCH_NATIVE_AOT={env} in this environment)")
        return True
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


def should_run() -> bool:
    """Whether stage 2 will export for this build.

    Answers from build properties ONLY, and every step that touches the machine --
    the CMake cache, the two torch subprocesses -- reports and returns a default
    instead of raising: --print-verdict's caller compares stdout with ==, so a
    traceback here is neither RUN nor SKIP.

    Separate from require_runtimes() because the CI shells ask this to decide
    whether to INSTALL the DSL wheels (--print-verdict): a verdict that demanded
    them could only ever say "no" on a fresh image."""
    if _opted_out():
        return False  # _opted_out() reports the value and where it came from
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
    should_run() skip exports nothing, so demanding the wheels there would fail builds
    that never wanted them. A build that will export and cannot would otherwise ship a
    wheel missing its declared kernels, which shows up as slowness rather than an
    error. TORCH_NATIVE_AOT=0 opts out."""
    from tools.native_aot import toolchains

    backend = _backend()
    usable = toolchains.for_backend(backend)
    gaps = {
        k: tc.missing_runtimes() for k, tc in usable.items() if tc.missing_runtimes()
    }
    if gaps:
        # Distribution names, not REQUIRED_RUNTIMES' module names: they differ (module
        # `cutlass` ships in nvidia-cutlass-dsl), and a pip line naming the modules
        # installs unrelated packages.
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


def _cache_entries() -> dict[str, tuple[str, str]]:
    """This build's cache as name -> (TYPE=value, the doc line above it).

    Last assignment wins, as in _cmake_cache_value. The doc line names the source of an
    entry EnvVarForwarding wrote ("From environment", "From env <NAME>").
    """
    entries: dict[str, tuple[str, str]] = {}
    doc = ""
    try:
        with open(os.path.join(BUILD_DIR, "CMakeCache.txt"), errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("//"):
                    doc = line[2:]
                    continue
                key, sep, value = line.partition("=")
                if sep and not line.startswith("#") and ":" in key:
                    name, _, kind = key.partition(":")
                    entries[name] = (f"{kind}={value}", doc)
                doc = ""
    except OSError:
        return {}
    return entries


# The doc lines EnvVarForwarding writes above an entry it sourced from the environment.
_ENV_SOURCED = ("From environment", "From env ")


def _refuse_cache_drift(before: dict[str, tuple[str, str]]) -> None:
    """Refuse a reconfigure that changed this build's configuration.

    EnvVarForwarding writes every BUILD_*/USE_*/CMAKE_* environment variable (and the
    names in its alias lists, TORCH_CUDA_ARCH_LIST among them) into the cache when it
    differs from the cached value, so a stage-2 run in another environment than the
    build silently relinks torch_cuda against other settings. It CREATES the entry when
    the build had none, so an added env-sourced one is drift too; additions CMake makes
    itself are not.
    """
    drift = []
    for name, (value, doc) in _cache_entries().items():
        if name in before:
            if before[name][0] != value:
                drift.append((name, before[name][0], value, doc))
        elif doc.startswith(_ENV_SOURCED):
            drift.append((name, "absent", value, doc))
    if not drift:
        return
    changed = "\n".join(
        f"  {name}: {old} -> {new}" + (f" ({doc})" if doc else "")
        for name, old, new, doc in drift
    )
    raise RuntimeError(
        f"native-AOT stage 2: reconfiguring {BUILD_DIR} changed this build's "
        f"configuration, so torch_cuda would be relinked against different settings "
        f"than the rest of this install:\n{changed}\n"
        f"Run stage 2 in the same environment as the build (the CI shells and "
        f"`spin develop` do), or re-run the build with the settings you want."
    )


def _editable_rebuild_finder() -> object | None:
    """The installed editable finder, if it rebuilds torch on import.

    scikit-build-core's ``editable.rebuild`` (also SKBUILD_EDITABLE_REBUILD) installs a
    meta-path finder that runs `cmake --build` and `cmake --install` on the first
    `import torch`. Read from sys.meta_path rather than from pyproject.toml or the
    environment: the finder is what the torch on this path actually does.
    """
    for finder in sys.meta_path:
        if type(finder).__name__.startswith("ScikitBuild") and getattr(
            finder, "rebuild_flag", False
        ):
            return finder
    return None


def _refuse_editable_rebuild() -> None:
    """Refuse to run under an import-time rebuild.

    Every probe here imports torch in a subprocess, so each one would trigger that
    build and install -- overwriting the library this script relinked, and making the
    final embedded-kernels check describe whatever the probe just built.
    """
    if _editable_rebuild_finder() is None:
        return
    raise RuntimeError(
        "native-AOT stage 2: this torch was installed with scikit-build-core's "
        "editable.rebuild enabled, so importing torch rebuilds and reinstalls it. "
        "Stage 2 relinks torch_cuda and copies it over that install, and its probes "
        "import torch, so the two would race and the result would not be the library "
        "this script verified. Reinstall without editable.rebuild (unset "
        "SKBUILD_EDITABLE_REBUILD and the pyproject setting), or set "
        "TORCH_NATIVE_AOT=0 to skip stage 2."
    )


def _lib_snapshot() -> dict[str, tuple[int, int]]:
    """(mtime, size) for every file in the build tree's lib/ except torch_cuda's.

    Stage 2 installs libtorch_cuda.so and nothing else, so these are exactly the
    libraries a relink may rebuild but this script will not ship.
    """
    lib = os.path.join(BUILD_DIR, "lib")
    out = {}
    with os.scandir(lib) as entries:
        for e in entries:
            if e.name == "libtorch_cuda.so" or not e.is_file(follow_symlinks=False):
                continue
            st = e.stat(follow_symlinks=False)
            out[e.name] = (st.st_mtime_ns, st.st_size)
    return out


def _refuse_unshipped_rebuilds(before: dict[str, tuple[int, int]]) -> None:
    """Refuse a relink that rebuilt a library stage 2 does not install.

    `--target torch_cuda` builds its DEPENDENCIES too, so a source edited since the
    last full build lands in libtorch_cpu.so (or c10) as well -- and only
    libtorch_cuda.so is copied over the installed torch, leaving the two mismatched.
    Every supported caller builds everything immediately before this, so a difference
    here means stage 2 was run by hand against an edited tree.
    """
    after = _lib_snapshot()
    rebuilt = sorted(n for n, v in after.items() if before.get(n, v) != v)
    if not rebuilt:
        return
    raise RuntimeError(
        f"native-AOT stage 2: relinking torch_cuda also rebuilt "
        f"{', '.join(rebuilt[:4])}{' and others' if len(rebuilt) > 4 else ''}, which "
        f"stage 2 does not install -- the tree has sources newer than the last full "
        f"build, and installing libtorch_cuda.so alone would leave the two "
        f"mismatched. Re-run the build (`spin develop`, or `pip install -e .` "
        f"followed by this script), which installs them together."
    )


def _cmake_for_this_build() -> str:
    """The cmake that configured this build tree, else cmake from PATH.

    Recorded as CMAKE_COMMAND, and not necessarily on PATH: a build uses whatever cmake
    its environment had, often a pip wheel's rather than /usr/local/bin/cmake.
    """
    cached = _cmake_cache_value("CMAKE_COMMAND")
    if cached and os.path.exists(cached):
        return cached
    if cached:
        _report(f"{cached} configured this build but is gone; using cmake from PATH")
    return "cmake"


def _run_child(cmd: list[str], what: str, **kw: object) -> None:
    """Run one of stage 2's own steps, naming the step and a signal death on failure,
    neither of which check_call reports. The child's own output is inherited."""
    code = subprocess.call(cmd, **kw)  # type: ignore[arg-type]
    if code == 0:
        return
    raise RuntimeError(
        f"native-AOT stage 2: {what} "
        + (f"was killed by signal {-code}" if code < 0 else f"exited {code}")
        + " (its output is above). Command: "
        + " ".join(cmd)
    )


def _installed_lib_dir() -> str:
    """The lib/ directory of the INSTALLED torch package.

    Anchored on the compiled _C extension, not torch.__file__: an editable install
    serves Python from the source tree while the artifacts live in site-packages."""
    code = (
        "import importlib.util, os\n"
        "spec = importlib.util.find_spec('torch._C')\n"
        "if spec is not None and spec.origin:\n"
        "    print('NAOT_VALUE:' + os.path.dirname(spec.origin), flush=True)\n"
    )
    # find_spec imports the PARENT package, so this is a full `import torch`: bounded
    # and guarded through _run_probe like every other one here.
    probe = _run_probe(code, "find_spec('torch._C')")
    if probe is not None:
        # Marked rather than taking stdout whole: this is joined into a path, so any
        # other line the child writes (a warning, a site hook) becomes part of it.
        for line in probe.stdout.splitlines():
            if line.startswith("NAOT_VALUE:"):
                return os.path.join(line[len("NAOT_VALUE:") :], "lib")
        _report_probe_failure("find_spec('torch._C')", probe.stderr, probe.returncode)
        why = f"exited {probe.returncode} having printed no path" + (
            " (its stderr is above)" if probe.stderr.strip() else " and nothing at all"
        )
    else:
        why = "could not be run at all (reported above)"
    raise RuntimeError(
        f"native-AOT stage 2: cannot locate the installed torch._C, so there is no "
        f"torch/lib to copy the relinked library into. The probe {why}. Install the "
        f"wheel from this build first."
    )


def _copied_member(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """A ZipInfo for a member copied into a new archive.

    Field by field, not the source's ZipInfo: `info.extra` carries its ZIP64 header
    offset, so above zipfile's ZIP64_LIMIT (2 GiB) a copy inherits stale offsets."""
    out = zipfile.ZipInfo(info.filename, info.date_time)
    out.compress_type = info.compress_type
    out.external_attr = info.external_attr
    out.internal_attr = info.internal_attr
    out.create_system = info.create_system
    out.comment = info.comment
    # file_size decides whether open(..., "w") writes a ZIP64 header, so left at 0 it
    # raises on exactly the members above the limit.
    out.file_size = info.file_size
    return out


def _write_hashed(dst: zipfile.ZipFile, info: zipfile.ZipInfo, path: str) -> str:
    """Stream `path` into `dst` as `info`; return its RECORD `sha256=...,size` fields.

    Hashed from the bytes actually archived, in the same pass: a second read would
    double a ~400 MiB read."""
    import base64
    import hashlib

    info.file_size = os.path.getsize(path)  # see _copied_member: sizes the header
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f, dst.open(info, "w") as d:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
            d.write(chunk)
    digest = base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()
    return f"sha256={digest},{size}"


def patch_wheel(wheel_path: str, lib_path: str) -> None:
    """Replace torch/lib/libtorch_cuda.so inside an already-built wheel and fix its
    RECORD entry.

    zipfile, not the zip CLI (absent from the manywheel images) or `wheel pack`, which
    emits invalid ZIP64 above 4GB (pytorch#189748).
    """

    lib_rel = "torch/lib/libtorch_cuda.so"
    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()
        records = [n for n in names if n.endswith(".dist-info/RECORD")]
        if lib_rel not in names or len(records) != 1:
            raise RuntimeError(f"{wheel_path}: not a torch wheel ({lib_rel}/RECORD)")
        record_rel = records[0]
        record_lines = zf.read(record_rel).decode().splitlines()

    # Located before a byte is written, so a wheel this cannot patch costs nothing.
    for lib_line, line in enumerate(record_lines):
        if line.startswith(lib_rel + ","):
            break
    else:
        raise RuntimeError(f"{wheel_path}: RECORD has no entry for {lib_rel}")

    # Rebuilt beside the original and renamed over it, so an interrupted rewrite
    # cannot replace a valid wheel.
    tmp_whl = f"{wheel_path}.naot.{os.getpid()}.tmp"
    try:
        with (
            zipfile.ZipFile(wheel_path) as src,
            zipfile.ZipFile(tmp_whl, "w", allowZip64=True) as dst,
        ):
            entry = ""
            for info in src.infolist():
                # The RECORD last, once the digest is known; every other member
                # keeps its position, so .dist-info stays where archivers expect it.
                if info.filename == record_rel:
                    continue
                out = _copied_member(info)
                if info.filename == lib_rel:
                    entry = _write_hashed(dst, out, lib_path)
                    continue
                # Streamed, not read()+writestr(): several members run to hundreds
                # of MiB and the read form holds each twice.
                with src.open(info) as s, dst.open(out, "w") as d:
                    shutil.copyfileobj(s, d)
            record_lines[lib_line] = f"{lib_rel},{entry}"
            # The RECORD keeps its original compression: writestr() with a plain
            # name would take dst's default, which is none, and store it uncompressed.
            dst.writestr(
                _copied_member(src.getinfo(record_rel)),
                "\n".join(record_lines) + "\n",
            )
        shutil.move(tmp_whl, wheel_path)
    finally:
        if os.path.exists(tmp_whl):
            os.remove(tmp_whl)


def _invalidate_stale_include() -> None:
    """Stop this build tree from embedding what a PREVIOUS run wired up.

    caffe2/CMakeLists.txt include()s the generated file unconditionally, and
    .ci/manywheel/build_all.sh shares one build/ across eight interpreters. OVERWRITTEN
    rather than deleted: CMake registers an include()d file as a configure dependency
    only if it existed at configure time."""
    from tools.native_aot.gen_aot_lib import CMAKE_INCLUDE, write_nothing_to_embed

    art = NATIVE_AOT_ARTIFACTS_DIR
    if not os.path.exists(os.path.join(art, CMAKE_INCLUDE)):
        return
    write_nothing_to_embed(art)
    _report(f"a previous run left kernels wired up in {art}; disabled them")


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
    # The explicit opt-out first, and only that: it needs no torch, so
    # TORCH_NATIVE_AOT=0 is a kill switch even on the binary-build path.
    if _opted_out():
        return 0  # _opted_out() reports the value and where it came from
    # After the opt-out, like every other refusal, and before the first probe.
    _refuse_editable_rebuild()
    # --wheel means torch was installed on the line above, so "not importable" is a
    # broken build, not "not applicable"; hence ahead of the gates that need torch.
    if args.wheel and not _torch_probe("True"):
        raise RuntimeError(
            "native-AOT stage 2: --wheel was given, so torch was just installed, "
            "but it does not import (stderr above). Refusing to patch a wheel "
            "without kernels. Fix the install, or drop --wheel for a build where "
            "`import torch` cannot work (the ASan and TSan images need LD_PRELOAD "
            "or a different interpreter). TORCH_NATIVE_AOT=0 also exempts it."
        )
    # ...and only then that it exists, so a typo costs a second rather than a full
    # export, reconfigure, relink and copy.
    if args.wheel and not os.path.exists(args.wheel):
        raise RuntimeError(f"native-AOT stage 2: --wheel {args.wheel} does not exist")
    if not should_run():
        _invalidate_stale_include()
        return 0
    require_runtimes()
    py = sys.executable
    art = NATIVE_AOT_ARTIFACTS_DIR
    _report("exporting kernels")
    export = [py, os.path.join(HERE, "export.py"), "--out-dir", art]
    # Read ONCE for both children below, so they cannot be told different arch lists,
    # and into the export child's environment even if this build only cached it.
    arch_list = _arch_list()
    env = dict(os.environ)
    if arch_list:
        env["TORCH_CUDA_ARCH_LIST"] = arch_list
    _run_child(export, "exporting kernels", cwd=REPO, env=env)
    _report("generating stub sources")
    gen = [py, os.path.join(HERE, "gen_aot_lib.py"), "--artifacts-dir", art]
    # The archive the generator names in the CMake it emits.
    if archive := _dsl_runtime_archive():
        gen += ["--dsl-runtime", archive]
    # Name the arches THIS build targets, so a tree left by a build with a different
    # TORCH_CUDA_ARCH_LIST is ignored. Omitted for an on-device export.
    if arch_list:
        from tools.native_aot import export as export_mod

        # Both: --archs filters the trees, --arch-list is the raw value recorded
        # in the emitted CMake. Only this caller knows they are one request.
        gen += ["--archs", *export_mod.archs_from_cuda_arch_list(arch_list)]
        gen += ["--arch-list", arch_list]
    _run_child(gen, "generating stub sources", cwd=REPO)
    # Nothing generated is legitimate: no declaration ships kernels for this arch.
    # Stop rather than relink unchanged and then assert kernels are in it.
    sources = glob.glob(os.path.join(art, "*", "aot_*.cpp"))
    if not sources:
        _report("no declaration ships kernels for this build; nothing embedded")
        return 0
    # The count, and the size delta after the relink, rather than parsing the generated
    # CMake: these bytes scale with declarations x precompile points x arches.
    _report(f"embedding kernels from {len(sources)} generated source(s)")
    # Reconfigure explicitly: the generated file registers itself in
    # CMAKE_CONFIGURE_DEPENDS only from the reconfigure that first reads it.
    # Keyed on the CACHE, not the directory: `cmake -B` on a missing directory exits 0
    # and configures FROM SCRATCH, without scikit-build-core's -D flags.
    if not os.path.exists(os.path.join(BUILD_DIR, "CMakeCache.txt")):
        raise RuntimeError(
            f"native-AOT stage 2: {BUILD_DIR} holds no CMakeCache.txt, so it is not "
            f"the build this torch came from. It must match pyproject.toml's "
            f"build-dir; re-run the build from the repo root, or set "
            f"TORCH_NATIVE_AOT=0 to skip stage 2."
        )
    _report("reconfiguring to pick up the generated CMake")
    # The cmake that configured this tree, and its cache as it stands, so the
    # reconfigure can be held to both (see _refuse_cache_drift).
    cmake_exe = _cmake_for_this_build()
    cache_before = _cache_entries()
    # Captured because CMake prints its failure context on stdout, and the STATUS line
    # the generated file emits is the only pre-relink evidence that it will embed.
    # --log-level, because that marker is a message(STATUS) and a cached
    # CMAKE_MESSAGE_LOG_LEVEL=WARNING (EnvVarForwarding FORCEs it) would hide it.
    configure = subprocess.run(
        [cmake_exe, "--log-level=STATUS", "-S", REPO, "-B", BUILD_DIR],
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
    # Ahead of the relink, so a drifting configure cannot reach the installed torch.
    _refuse_cache_drift(cache_before)
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
    # Relink JUST torch_cuda: `--target install` walks the whole manifest (~15 min).
    # The copy below matches it only because the CMake sets BUILD_WITH_INSTALL_RPATH.
    _report("relinking torch_cuda with embedded kernels")
    # pyproject.toml aliases MAX_JOBS only inside scikit-build-core's subprocess.
    relink = [cmake_exe, "--build", ".", "--target", "torch_cuda"]
    jobs = os.getenv("MAX_JOBS") or os.getenv("CMAKE_BUILD_PARALLEL_LEVEL")
    if jobs and jobs.isdigit():
        relink += ["--parallel", jobs]
    build_lib = os.path.join(BUILD_DIR, "lib", "libtorch_cuda.so")
    # Taken across the relink, for the size delta reported below.
    before = os.path.getsize(build_lib) if os.path.exists(build_lib) else 0
    siblings = _lib_snapshot()
    _run_child(relink, "relinking torch_cuda", cwd=BUILD_DIR)
    # Before the copy, so a mismatched pair never reaches the installed torch.
    _refuse_unshipped_rebuilds(siblings)

    if not os.path.exists(build_lib):
        raise RuntimeError(f"expected relinked library at {build_lib}")
    installed = os.path.join(_installed_lib_dir(), "libtorch_cuda.so")
    if not os.path.exists(installed):
        # Refuse to create it: _installed_lib_dir found *a* torch, so a layout that
        # never held the library means we are pointed at the wrong environment.
        raise RuntimeError(
            f"native-AOT stage 2: {installed} does not exist, so the torch on "
            f"sys.path is not the one this tree built. Install the wheel from "
            f"this build first, or set TORCH_NATIVE_AOT=0."
        )
    # Temp file + rename: copying in place truncates a library others may be mapping,
    # and one os.replace means `installed` never stops existing.
    staged = f"{installed}.naot.{os.getpid()}.tmp"
    try:
        shutil.copy2(build_lib, staged)
        os.replace(staged, installed)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    grew = (os.path.getsize(build_lib) - before) >> 20
    _report(
        f"{os.path.getsize(build_lib) >> 20} MiB relinked into {installed} "
        f"({grew:+d} MiB of embedded kernels)"
    )
    # Size is not evidence: this script's artifacts dir must agree with the one
    # caffe2/CMakeLists.txt include()s, or the relink embeds nothing and still exits 0.
    if not _torch_probe("torch._native._native_aot_embedded()"):
        raise RuntimeError(
            "native-AOT stage 2: relinked libtorch_cuda reports no embedded "
            "kernels (torch._native._native_aot_embedded() is False). The "
            f"artifacts in {art} were not linked in -- check that this build's "
            f"CMAKE_BINARY_DIR is {BUILD_DIR} (it must match pyproject.toml's "
            f"build-dir), and that {art}/native_aot.cmake exists and is the file "
            f"caffe2/CMakeLists.txt includes. {installed} now holds that library, "
            f"so reinstall the wheel from this build to get back to a known state."
        )
    if args.wheel:
        _report(f"embedding into {args.wheel}")
        patch_wheel(args.wheel, build_lib)
    _report("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
