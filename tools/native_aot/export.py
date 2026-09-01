"""AOT-export DSL kernels for native-AOT declarations.

Stage 2 of the two-stage build, with the built torch importable, so kernel builder
modules are ordinary package imports (``torch._native.ops.<op>.<module>``) and share
code with their JIT wrappers. Only the aot.py DECLARATION modules stay torch-free at
module scope, since torchgen loads those during stage 1.

For each ``torch/_native/ops/<op>/aot.py``, expands the spec grid (list
fields cross-multiply) and for every grid point runs the toolchain's
compile + export into ``<out-dir>/<op>/``, writing:

    <prefix>.h / <prefix>.o    C-ABI header + kernel object
    <prefix>.json              marshalling sidecar {spec, arch, tensor_args}

Prefixes carry their arch (``topk_..._det__sm100a``): every exported C symbol derives
from the prefix, so two arches sharing one would collide in libtorch_cuda.

The builder module exposes ``build(spec)`` returning a dict whose ``kind`` selects
the toolchain; tools/native_aot/toolchains.py holds the per-kind contracts. Existing
artifacts are skipped unless --force.

Spec points compile on a forkserver pool, one job per (point, arch), so results do
not depend on --jobs, which follows the torch build's parallelism (MAX_JOBS, then
CMAKE_BUILD_PARALLEL_LEVEL, then half the CPU count). Plain fork is unusable: the
parent may have initialized CUDA, and forked workers inherit a dead context silently.

With --arch (one or more sm strings) export never touches the CUDA driver, so kernels
build on GPU-less machines, and the arch is per-compile rather than per-process --
CuTeDSL takes --gpu-arch, which outranks CUTE_DSL_ARCH. CuTeDSL needs one warmup
compile per process for that; see tools/native_aot/cutedsl_warmup.py.

With --arch (one or more sm strings) export never touches the CUDA
driver, so kernels build on GPU-less machines. The arch is per-COMPILE
state: CuTeDSL takes a --gpu-arch option, which outranks CUTE_DSL_ARCH
(base_dsl/dsl.py prefers compile_options.gpu_arch over envar.arch), and
Triton gets a fixed-target driver per export. So a single pool serves
every (point, arch) job. A multi-arch fan-out nests each arch under
<out-dir>/<arch>/ to keep the trees independently linkable; a single arch
still lands flat in <out-dir>. The next commit of this stack makes that
uniform.
CuTeDSL needs one warmup compile per process for this to work; see
tools/native_aot/cutedsl_warmup.py.

Usage (from the repo root, venv with torch built and the DSL wheel
active):
    python tools/native_aot/export.py [--out-dir build/native_aot]
                                        [--ops topk] [--force] [--jobs 8]
                                        [--arch sm_90a sm_100a]
"""

import argparse
import importlib
import json
import os
import sys


REPO = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
# Run as a script, sys.path[0] is this directory, so `tools.native_aot` imports only
# when the cwd is the repo root; put the root on the path instead.
#
# APPEND, never insert(0): stage 2 runs against the INSTALLED wheel, and the repo root
# holds a torch/ source tree with no compiled extension. Ahead of site-packages it
# shadows the real torch and every worker's `import torch` fails.
sys.path.append(REPO)

# torchgen is pure Python and imports without a built torch, which this module scope
# needs: stage-1 codegen and the linter image that runs the tools tests both lack it.
from tools.native_aot import toolchains

from torchgen import native_aot_decl as decl
from torchgen.native_aot_spec_grid import expand_specs


OPS_DIR = os.path.join(REPO, "torch", "_native", "ops")

# Bump on any change to the sidecar layout or to the launcher-generation contract
# that reads it; gen_aot_lib refuses mismatched sidecars.
SIDECAR_VERSION = 1

# forkserver, never "fork": a fork parent that has initialized CUDA gives workers a
# dead context, silently. forkserver is as safe as spawn and pays the torch import
# once rather than per worker.
# TODO(native-aot): forkserver does not exist on Windows; fall back to "spawn" when
# Windows CUDA builds start exporting.
POOL_START_METHOD = "forkserver"

# Preloaded in the forkserver's server process, so workers inherit it imported. Only
# modules safe in a fork PARENT belong here: importing torch neither initializes CUDA
# nor builds DSL state.
POOL_PRELOAD = ("torch",)


def load_builder(op: str, kernel_module: str):
    # A package import, not a file-path load: builders may use relative imports and
    # torch machinery.
    name = f"torch._native.ops.{op}.{kernel_module.removesuffix('.py')}"
    return importlib.import_module(name).build


_HERE = os.path.dirname(os.path.abspath(__file__))


def _file_hash(path: str) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# Loaded modules whose contents decide what an artifact means, and which the
# tools/*.py glob below cannot see: torch._native holds the builders,
# torchgen.native_aot the declaration machinery, torch._vendor the vendored DSL
# packages and their kernel bodies.
_CLOSURE_PREFIXES = ("torch._native", "torch._vendor", "torchgen.native_aot")

# Tool sources that cannot change what an artifact means, so hashing them would
# re-export every kernel for nothing: gen_aot_lib.py only consumes sidecars, and
# build_stage2.py passes no kernel-affecting option.
_CLOSURE_EXCLUDED = frozenset({"gen_aot_lib.py", "build_stage2.py"})


def source_closure(decl_path: str | None = None) -> dict[str, str]:
    """{repo-relative path: content hash} for every loaded source that can change
    what an artifact means: the builder's import closure, the declaration machinery
    (_CLOSURE_PREFIXES), this tool's own sources, and the op's aot.py.

    Recorded per sidecar; gen_aot_lib re-hashes from disk and refuses to pair edited
    sources with stale artifacts, so this over-approximates on purpose.

    Only imported modules appear in sys.modules, hence the glob for tools/ sources
    and the explicit decl_path: declarations load by file path and never enter
    sys.modules, so a KERNEL_MODULE or grid edit would otherwise go unnoticed."""
    import glob

    out = {}
    # A snapshot, because hashing can trigger imports and mutating sys.modules
    # mid-iteration raises.
    for name, mod in list(sys.modules.items()):
        if not name.startswith(_CLOSURE_PREFIXES):
            continue
        f = getattr(mod, "__file__", None)
        if f and os.path.exists(f):
            out[os.path.relpath(f, REPO)] = _file_hash(f)
    for f in glob.glob(os.path.join(_HERE, "*.py")):
        if os.path.basename(f) in _CLOSURE_EXCLUDED:
            continue
        out[os.path.relpath(f, REPO)] = _file_hash(f)
    if decl_path and os.path.exists(decl_path):
        out[os.path.relpath(decl_path, REPO)] = _file_hash(decl_path)
    return dict(sorted(out.items()))


def _json_normal(value):
    """The spec as a sidecar reads it back: tuples become lists, JSON having no tuple
    type. Skip detection compares a live grid point against a recorded spec, so
    without this a tuple-valued field never matches and re-exports every run."""
    if isinstance(value, (tuple, list)):
        return [_json_normal(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_normal(v) for k, v in value.items()}
    return value


def _detected_arch() -> str | None:
    """The local device as an sm string ("sm_100"), or None without CUDA.

    Recorded in the sidecar so an on-device export's artifacts carry an arch identity;
    without one the generated gate would fall back to the declaration's ARCHS and
    advertise hardware nothing was compiled for. No "a" suffix: the gate compares
    major.minor, which both spellings share."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    return f"sm_{major * 10 + minor}"


def _arch_tag(arch: str) -> str:
    """Arch as an artifact-name tag: "sm_100a" -> "sm100a". Short because it lands in
    every exported C symbol, and those are already long."""
    return arch.replace("_", "", 1)


def _effective_arch(arch: str | None) -> str | None:
    """The arch the artifacts are really compiled for: the explicit one if
    given, else whatever a toolchain's own env var selects
    (Toolchain.ARCH_ENV_VAR, e.g. CUTE_DSL_ARCH), else the local device.

    One resolver for both the sidecar and the directory name, so a tree cannot
    disagree with its own sidecars -- the runtime gate comes from the sidecar, and a
    directory saying otherwise would make the layout lie. The skip check recomputes
    it, so two runs differing only in the environment are told apart.

    Takes no toolchain by design: with a kind's arch variable refused rather than
    honoured, resolution is --arch or the device, neither of which varies by kind."""
    if arch:
        return arch
    # A toolchain's arch variable without --arch is refused rather than honoured: it
    # is per-kind, so it answers only for kinds that declare one, while the tree it
    # would name holds every kind's artifacts.
    named = {
        k.ARCH_ENV_VAR: os.getenv(k.ARCH_ENV_VAR)
        for k in toolchains.TOOLCHAINS.values()
        if k.ARCH_ENV_VAR and os.getenv(k.ARCH_ENV_VAR)
    }
    if named:
        listed = ", ".join(f"{k}={v}" for k, v in sorted(named.items()))
        raise RuntimeError(
            f"{listed} {'is' if len(named) == 1 else 'are'} set but --arch is not. "
            f"A toolchain's arch variable is per-kind, so it cannot name the arch "
            f"for every kind in one export; pass --arch (e.g. --arch "
            f"{named[min(named)]}) to state it once."
        )
    return _detected_arch()


def export_point(
    op_pkg: str, kernel_module: str, point: dict, out_dir: str, arch: str | None = None
) -> str:
    """Compile and export one spec point, and write its sidecar.

    Module-level with picklable arguments, so it runs identically inline and as a pool
    job, and holds no process-global state, so one process serves any mix of arches.

    A missing DSL runtime is fatal here rather than a skip: a declaration reaching
    this point targets this build's backend, so its kernels were asked for, and
    exporting only some of them ships a wheel that silently underperforms. Use
    TORCH_NATIVE_AOT=0 to build without them. The ImportError arm exists because a
    builder cannot be asked its kind without importing its runtime."""
    try:
        build = load_builder(op_pkg, kernel_module)
        b = build(point)
    except ImportError as e:
        raise RuntimeError(
            f"{op_pkg}: cannot export, DSL runtime not installed "
            f"({e.name or e}). Install it, or set TORCH_NATIVE_AOT=0 to "
            f"build without embedded DSL kernels."
        ) from e
    # Builder dicts may omit kind (CuTeDSL is the default); sidecars always
    # carry it, written below.
    tc = toolchains.get_toolchain(b.get("kind", "cutedsl"))
    missing = tc.missing_runtimes()
    if missing:
        raise RuntimeError(
            f"{op_pkg}: cannot export, {tc.kind} needs {', '.join(missing)}. "
            f"Install them, or set TORCH_NATIVE_AOT=0 to build without "
            f"embedded DSL kernels."
        )
    tc.validate_build_result(b)
    # Arch-qualifying the prefix is what lets several arches ship in one library:
    # every exported C symbol derives from it, so two arches sharing one are
    # duplicate definitions at link time.
    effective_arch = _effective_arch(arch)
    if effective_arch:
        b["prefix"] = f"{b['prefix']}__{_arch_tag(effective_arch)}"
    prefix = b["prefix"]
    extra = tc.export(b, out_dir, arch=arch)
    sidecar = {
        "version": SIDECAR_VERSION,
        "prefix": prefix,
        "kind": tc.kind,
        "spec": point,
        "arch": effective_arch,
        # The declaration lives at a path fixed by construction, so it needs
        # no threading through the job tuple.
        "sources": source_closure(os.path.join(OPS_DIR, op_pkg, "aot.py")),
        # The compiler, which no source file names (see runtimes_current).
        "runtimes": runtime_versions(tc.kind),
        **extra,
    }
    with open(os.path.join(out_dir, prefix + ".json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    return prefix


def _collect_jobs(ops_filter, out_root: str, archs):
    """(op_pkg, kernel_module, point, out_dir, arch) per spec point per
    arch across every declaration; grids expand here (cheap,
    torch-light), skip detection is _job_needed's sidecar scan. A
    single arch (or None = detect from the local device) keeps the flat
    <out-root>/<decl_id>/ layout; a multi-arch fan-out nests
    <out-root>/<arch>/<decl_id>/ so per-arch artifact trees stay
    independently gen-able and linkable."""
    jobs = []
    multi = len(archs) > 1
    for entry in sorted(os.listdir(OPS_DIR)):
        op_dir = os.path.join(OPS_DIR, entry)
        if not os.path.exists(os.path.join(op_dir, "aot.py")):
            continue
        for d in decl.load_declarations(os.path.join(op_dir, "aot.py")):
            did = decl.decl_id(d)
            if ops_filter and entry not in ops_filter and d.ATEN_OP not in ops_filter:
                continue
            for arch in archs:
                # Declaration-level arch support: skip (declaration x
                # arch) pairs the op's kernels are not valid on. An
                # on-device export (arch None) is not filtered -- the
                # builder machine is the target by construction.
                if arch is not None and arch not in decl.archs_of(d):
                    continue
                # `multi` implies explicit sm strings (a [None] arch
                # list is always length 1); the arch check narrows for
                # the type checker.
                root = os.path.join(out_root, arch) if multi and arch else out_root
                out_dir = os.path.join(root, did)
                os.makedirs(out_dir, exist_ok=True)
                points = expand_specs(d.kernel_precompile_grid())
                _check_no_orphan_artifacts(out_dir, points)
                for point in points:
                    jobs.append((entry, d.KERNEL_MODULE, point, out_dir, arch))
    return jobs


# Every "this tree is inconsistent" error ends the same way. `spin clean` clears the
# default --out-dir too, but takes the whole build tree with it, so name the surgical
# command first.
_CLEAN_HINT = (
    "run `rm -rf {d}` and re-export (`spin clean` also clears it, "
    "along with the rest of the build tree)"
)


def _read_sidecar(path: str) -> dict:
    """A sidecar's JSON. Unreadable sidecars are fatal.

    The sidecar is written last, so its presence marks a completed export; one that
    exists but will not parse means the .o/.h beside it are of unknown provenance.
    Reached even under --force, via the orphan scan, so generation never links
    artifacts nothing validated.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"{path}: sidecar exists but could not be read ({e}). The "
            f"artifacts beside it cannot be trusted; "
            f"{_CLEAN_HINT.format(d=os.path.dirname(path))}."
        ) from e


def _check_no_orphan_artifacts(out_dir: str, specs=None) -> None:
    """Fail if a directory holds kernel artifacts no current grid point
    claims.

    Two ways that happens. Artifacts with NO sidecar at all: the sidecar is
    the commit marker, so they mean an interrupted or hand-edited export.
    Artifacts whose sidecar records a spec that is no longer in the grid:
    dropping a point from kernel_precompile_grid() generates no job for it
    and nothing prunes it. Either way the CMake globs link *.o by pattern,
    so the object ships with no launcher referencing it -- exactly what this
    check exists to prevent. An EMPTY directory is fine (clean build, or a
    new spec point).

    ``specs`` is the expanded grid for this (declaration, arch); None skips
    the stale-point half, for callers that do not have the grid in hand.
    """
    exts = {e for tc in toolchains.TOOLCHAINS.values() for e in tc.artifact_exts}
    names = os.listdir(out_dir)
    if not any(n.endswith(".json") for n in names):
        orphans = sorted(n for n in names if os.path.splitext(n)[1] in exts)
        if orphans:
            raise RuntimeError(
                f"{out_dir}: kernel artifacts with no sidecar "
                f"({', '.join(orphans[:4])}{', ...' if len(orphans) > 4 else ''}). "
                f"The sidecar is written last, so this is an interrupted or "
                f"hand-edited export; {_CLEAN_HINT.format(d=out_dir)}."
            )
        return
    if specs is None:
        return
    live = [_json_normal(p) for p in specs]
    stale = sorted(
        fn
        for fn in names
        if fn.endswith(".json")
        and _read_sidecar(os.path.join(out_dir, fn)).get("spec") not in live
    )
    if stale:
        raise RuntimeError(
            f"{out_dir}: sidecars for spec points no longer in the grid "
            f"({', '.join(stale[:4])}{', ...' if len(stale) > 4 else ''}). "
            f"Their kernel objects would still be linked with no launcher "
            f"referencing them; {_CLEAN_HINT.format(d=out_dir)}."
        )


def runtime_versions(kind: str) -> dict[str, str]:
    """{distribution: version} for the runtimes that COMPILE this kind.

    Metadata only, no import of the DSL itself. An uninstalled distribution is
    recorded as absent rather than omitted, so a sidecar compiled without the wheel is
    distinguishable from one predating this record."""
    import importlib.metadata as md

    out = {}
    for dist in toolchains.get_toolchain(kind).RUNTIME_DISTS:
        try:
            out[dist] = md.version(dist)
        except md.PackageNotFoundError:
            out[dist] = "absent"
    return dict(sorted(out.items()))


def runtimes_current(sidecar: dict) -> bool:
    """True if the sidecar was compiled by the DSL versions installed now.

    The compiler is not in the source closure -- no file on disk changes when the
    wheel is upgraded -- so without this an upgrade re-exports nothing and the tree
    mixes artifacts from two compilers. A sidecar predating this record counts stale.

    Ignorance is not staleness: with none of the kind's distributions installed this
    returns True, which is the generation-only run on a machine without the DSL
    wheels, where re-exporting is impossible anyway."""
    # sidecar["kind"] rather than a default: both callers reach here past a schema
    # check that proves the field, and guessing would judge one toolchain's artifact
    # by another's compiler versions.
    tc = toolchains.get_toolchain(sidecar["kind"])
    current = runtime_versions(tc.kind)
    # all() over an empty dict is True, so a kind with no RUNTIME_DISTS takes
    # this arm too: nothing whose version could have changed.
    if all(v == "absent" for v in current.values()):
        return True
    return sidecar.get("runtimes") == current


def sources_current(sidecar: dict) -> bool:
    """True if every source file recorded in the sidecar's closure
    still hashes the same on disk. Sidecars without a closure or from
    a different schema version count as stale (re-export)."""
    if sidecar.get("version") != SIDECAR_VERSION:
        return False
    sources = sidecar.get("sources")
    if not sources:
        return False
    for rel, digest in sources.items():
        path = os.path.join(REPO, rel)
        if not os.path.exists(path) or _file_hash(path) != digest:
            return False
    return True


def _job_needed(job, force: bool) -> bool:
    """Cheap skip check without compiling: skip only when the sidecar's spec, its arch
    and every file in its source closure still match, so an edited kernel module
    re-exports without --force.

    The arch comparison covers a recorded arch differing from what this run resolves
    to -- artifacts predating arch identity, or a tree carried between machines. Both
    must re-export, since the recorded arch is what the runtime gate is built from.
    Compared through _effective_arch, so both sides resolve the same way."""
    if force:
        return True
    _, _, point, out_dir, arch = job
    spec = _json_normal(point)
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".json"):
            continue
        sc = _read_sidecar(os.path.join(out_dir, fn))
        # Schema first, because every field below is read by name: a sidecar written
        # by another version re-exports here rather than raising a KeyError that names
        # neither the file nor a remedy.
        if sc.get("version") != SIDECAR_VERSION or "kind" not in sc:
            return True
        tc = toolchains.get_toolchain(sc["kind"])
        if sc.get("spec") == spec and sc.get("arch") == _effective_arch(arch):
            # The sidecar is the skip marker, but it is not proof the
            # artifacts it describes are still on disk: anything that
            # removes a .o/.h without its .json (a partial clean, an
            # over-eager prune) would otherwise be skipped here and fail
            # much later as a missing include at compile time.
            prefix = sc.get("prefix", "")
            if any(
                not os.path.exists(os.path.join(out_dir, prefix + e))
                for e in tc.artifact_exts
            ):
                return True
            return not (sources_current(sc) and runtimes_current(sc))
    return True


def _run_job(job) -> str:
    return export_point(*job)


def archs_from_cuda_arch_list(arch_list: str) -> list[str]:
    """TORCH_CUDA_ARCH_LIST -> the sm strings in it that are EXPORTABLE_ARCHES,
    order-preserving and deduplicated. "9.0a;10.0a" (or space-separated) ->
    ["sm_90a", "sm_100a"].

    "9.0a;10.0a" (or space-separated) -> ["sm_100a"]. A +PTX suffix is
    stripped; named entries ("Hopper") are not translated -- callers
    should pass numeric lists (CI does). Dedup matters: "10.0;10.0+PTX" names
    one arch twice, and a repeated entry would otherwise read as
    multi-arch downstream (nested artifact layout, --jobs > 1)."""
    out = []
    for entry in arch_list.replace(";", " ").split():
        entry = entry.removesuffix("+PTX")
        parts = entry.split(".")
        if len(parts) != 2 or not parts[0].isdigit():
            continue  # named arch ("Hopper") or malformed: skip
        minor = parts[1]
        suffix = "a" if minor.endswith("a") else ""
        minor_num = minor.removesuffix("a")
        if not minor_num.isdigit():
            continue
        sm = f"sm_{int(parts[0]) * 10 + int(minor_num)}{suffix}"
        if sm in EXPORTABLE_ARCHES and sm not in out:
            out.append(sm)
    return out


# Which TORCH_CUDA_ARCH_LIST entries are ELIGIBLE for AOT kernels on the
# automatic export path. A filter, never a build list: it cannot cause an
# export, only permit one, and an explicit --arch bypasses it. So a list
# with no eligible entry exports nothing and stage 2 skips, printing why.
#
# Distinct from a declaration's ARCHS (what the KERNELS support, sm_90+);
# this says what the standard build SHIPS. Both spellings of a CC are
# listed because they are distinct nvcc targets used by different builds
# for the same hardware -- "10.0a" (arch-conditional, needed by
# tcgen05/wgmma) in b200-native-aot.yml, plain "10.0" elsewhere and in the
# manywheel lists. Omitting either silently exports nothing there.
#
# sm_103/sm_103a are deliberately absent: no release or CI arch list names
# 10.3, and sm_100 SASS is forward-compatible to it, so nothing is lost by
# leaving it out. Selection is by full capability (major AND minor), so a
# 10.3 device declines sm_100 kernels rather than loading them -- adding
# 10.3 is a line here plus hardware to test it on.
EXPORTABLE_ARCHES = ("sm_100", "sm_100a")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(REPO, "build", "native_aot"))
    parser.add_argument(
        "--ops", nargs="*", help="restrict to these ops/<dir> names or ATEN_OPs"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-export existing artifacts"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="parallel compile processes (forkserver). Default follows "
        "the torch build's parallelism: MAX_JOBS, then "
        "CMAKE_BUILD_PARALLEL_LEVEL, then half the CPU count -- the same "
        "pair pyproject.toml's [tool.scikit-build.env] hands to cmake.",
    )
    parser.add_argument(
        "--arch",
        nargs="*",
        default=None,
        metavar="SM",
        help="target architecture(s), e.g. --arch sm_90a sm_100a. With an "
        "explicit arch, export never touches the CUDA driver and runs on "
        "GPU-less machines (CuTeDSL via --gpu-arch; Triton via an "
        "explicit GPUTarget). Default: detect from the local device. "
        "Multiple archs nest artifacts under <out-dir>/<arch>/.",
    )
    args = parser.parse_args()
    if args.jobs is None:
        env_jobs = os.getenv("MAX_JOBS") or os.getenv("CMAKE_BUILD_PARALLEL_LEVEL")
        # Half the CPU count, not all of it: os.cpu_count() reports SMT
        # siblings, and one compile per virtual thread oversubscribes.
        args.jobs = int(env_jobs) if env_jobs else max(1, (os.cpu_count() or 2) // 2)
    if args.arch is None and os.getenv("TORCH_CUDA_ARCH_LIST"):
        # Standard-build integration: export for the Blackwell subset of
        # the architectures the main build compiled for (the wheel may
        # run on machines unlike the builder). Explicit --arch wins.
        args.arch = archs_from_cuda_arch_list(os.environ["TORCH_CUDA_ARCH_LIST"])
        if args.arch:
            print(f"arch from TORCH_CUDA_ARCH_LIST: {' '.join(args.arch)}")
        else:
            print(
                "TORCH_CUDA_ARCH_LIST contains no AOT-exportable arch "
                f"(exportable: {' '.join(EXPORTABLE_ARCHES)}); nothing to export"
            )
            return
    archs = args.arch if args.arch else [None]
    jobs = _collect_jobs(args.ops, args.out_dir, archs)
    todo = [j for j in jobs if _job_needed(j, args.force)]
    if len(todo) < len(jobs):
        print(f"{len(jobs) - len(todo)} points already exported, skipped")

    total = 0
    if args.jobs <= 1 or len(todo) <= 1:
        for job in todo:
            prefix = _run_job(job)
            print(f"  {prefix}: exported")
            total += 1
    else:
        # ONE pool over every (point, arch) job: each toolchain takes its
        # arch per compile (CuTeDSL --gpu-arch, Triton a fixed GPUTarget),
        # so no process is pinned to an arch and mixed jobs pack freely.
        import multiprocessing
        from concurrent.futures import as_completed, ProcessPoolExecutor

        ctx = multiprocessing.get_context(POOL_START_METHOD)
        # Import torch once in the server; workers inherit it by fork.
        ctx.set_forkserver_preload(list(POOL_PRELOAD))
        n = min(args.jobs, len(todo))
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
            futs = {pool.submit(_run_job, job): job for job in todo}
            for fut in as_completed(futs):
                prefix = fut.result()  # re-raises worker failures
                arch = futs[fut][4]
                print(f"  {prefix}{f' [{arch}]' if len(archs) > 1 else ''}: exported")
                total += 1

    print(f"exported {total} kernels")


if __name__ == "__main__":
    main()
