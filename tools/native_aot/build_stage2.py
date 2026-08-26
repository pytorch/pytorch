"""Stage-2 driver for the standard torch build: export AOT kernels,
generate the stub sources, and relink torch_cuda with them embedded.

A post-install step because the kernel builders package-import torch, and
scikit-build-core has no post-build hook inside the PEP 517 backend (the wheel
is assembled before torch is importable). This module is the VERDICT half:
it answers whether a build should export at all, and nothing here runs a
step -- the CLI and the relink arrive with the rest of the driver.

Skips -- leaving a normal artifacts-free build -- when AOT kernels are not
applicable. Keep this list in sync with should_run(), which reports each one:

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
import subprocess
import sys
import sysconfig


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
BUILD_DIR = os.path.join(REPO, "build")
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


# Long enough for a cold `import torch` on a machine already running a build,
# short enough that a wedged one fails rather than holding the job to its step
# limit with nothing on either stream.
_PROBE_TIMEOUT = 300


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
            timeout=_PROBE_TIMEOUT,
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
