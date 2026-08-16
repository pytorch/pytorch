"""AOT-export DSL kernels for native-AOT declarations.

Stage 2 of the two-stage build (build torch -> build the AOT lib): runs
with the BUILT torch importable, so kernel builder modules are ordinary
package imports (``torch._native.ops.<op>.<module>``) and may freely
share code with their JIT wrappers -- the same-kernel property is then
by construction, not by parallel restatement. Only the aot.py
DECLARATION modules stay torch-free at module scope (torchgen loads
those during stage 1, before torch exists).

For each ``torch/_native/ops/<op>/aot.py``, expands the spec grid (list
fields cross-multiply) and for every grid point runs the toolchain's
compile + export into ``<out-dir>/<op>/``, writing:

    <prefix>.h / <prefix>.o    C-ABI header + kernel object
    <prefix>.json              marshalling sidecar {spec, tensor_args}

The builder module must expose ``build(spec)`` returning a dict whose
``kind`` selects the toolchain; see tools/native_aot/toolchains.py for
the per-toolchain contracts (required keys, emitted artifacts). Builder
results are validated up front so a malformed one fails with a message
naming the missing keys.

Idempotent: existing artifacts are skipped unless --force.

Spec points compile on a forkserver process pool. Each point is
independent, so results do not depend on --jobs. --jobs follows the
torch build's parallelism (MAX_JOBS, then CMAKE_BUILD_PARALLEL_LEVEL,
then half the CPU count); --jobs 1 forces serial. Plain fork is
unusable: the parent has initialized CUDA and forked workers inherit a
dead context, silently -- they report is_initialized() False and cannot
allocate. forkserver forks from a pre-CUDA server process instead, and
pays the torch import once there rather than per worker. Only torch is
preloaded; cutlass or triton would build state in that fork parent.

With --arch (one or more sm strings) export never touches the CUDA
driver, so kernels build on GPU-less machines. The arch is per-COMPILE
state: CuTeDSL takes a --gpu-arch option, which outranks CUTE_DSL_ARCH
(base_dsl/dsl.py prefers compile_options.gpu_arch over envar.arch), and
Triton gets a fixed-target driver per export. So a single pool serves
every (point, arch) job. Multiple archs still nest under
<out-dir>/<arch>/ to keep each tree independently linkable.
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
# Run as a script (`python tools/native_aot/export.py`), sys.path[0] is this
# directory, so `tools.native_aot` is importable only when the cwd happens to
# be the repo root. Put the root on the path explicitly instead of loading
# siblings by file path, which also keeps the imports legible to type checkers.
#
# APPEND, never insert(0): stage 2 runs against the INSTALLED wheel, and the
# repo root contains a torch/ source tree with no compiled extension. Ahead of
# site-packages it shadows the real torch, so every worker's `import torch`
# dies with "loaded the torch/_C folder of the PyTorch repository". Invisible
# in a `pip install -e .` checkout, where that tree IS the installed torch.
sys.path.append(REPO)

# torchgen is pure Python and imports with no built torch, which this
# module scope requires: stage-1 codegen and the linter image that runs
# the tools tests both lack torch.
from tools.native_aot import toolchains

from torchgen import native_aot_decl as decl
from torchgen.native_aot_spec_grid import expand_specs


OPS_DIR = os.path.join(REPO, "torch", "_native", "ops")

# Sidecar schema version. Bump on any change to the sidecar layout or
# to the launcher-generation contract that reads it; gen_aot_lib refuses
# mismatched sidecars (re-export rather than debugging a garbled .cpp).
SIDECAR_VERSION = 1

# Pool start method: forkserver, never "fork" (the parent has initialized
# CUDA, and forked workers inherit a dead context that fails silently).
# forkserver forks from a pre-CUDA server process, so it is as safe as
# spawn while paying the torch import once instead of per worker.
# TODO(native-aot): forkserver does not exist on Windows. Nothing calls
# this from the build there yet (stage 2 targets libtorch_cuda.so only),
# so fall back to "spawn" when Windows CUDA builds start exporting.
POOL_START_METHOD = "forkserver"

# Preloaded in the forkserver's server process, so every worker inherits
# it already imported. Only modules that are safe in a fork PARENT belong
# here: importing torch neither initializes CUDA nor builds any DSL state.
POOL_PRELOAD = ("torch",)


def load_builder(op: str, kernel_module: str):
    # Package import (not file-path load): builders may use relative
    # imports and torch machinery -- stage 2 runs against built torch.
    name = f"torch._native.ops.{op}.{kernel_module.removesuffix('.py')}"
    return importlib.import_module(name).build


_HERE = os.path.dirname(os.path.abspath(__file__))


def _file_hash(path: str) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# Loaded modules under these prefixes belong in the source closure:
# torch._native is the builder's own closure, torchgen.native_aot the
# shared declaration machinery (expand_specs picks the grid points, the
# validating loader decides what a declaration means). Neither is caught
# by the tools/*.py glob below -- they live outside tools/.
_CLOSURE_PREFIXES = ("torch._native", "torchgen.native_aot")


def source_closure(decl_path: str | None = None) -> dict[str, str]:
    """{repo-relative path: content hash} for every loaded source file
    that can change what an artifact MEANS: the builder's import closure,
    the shared declaration machinery (_CLOSURE_PREFIXES), this tool's own
    sources (a launcher-template edit in toolchains.py counts), and the
    op's own aot.py.

    Recorded per sidecar; gen_aot_lib re-hashes from disk and refuses to
    pair edited sources with stale artifacts. Deliberately
    over-approximates -- staleness must err toward re-export.

    Only IMPORTED modules appear in sys.modules, which is why the tools/
    sources are globbed from disk instead -- and why decl_path is passed
    explicitly: declarations are loaded by file path and never enter
    sys.modules, so KERNEL_MODULE or kernel_precompile_grid() edits would
    otherwise reuse artifacts built from the old grid."""
    import glob
    import sys

    out = {}
    # Snapshot: hashing can trigger imports, and mutating sys.modules
    # mid-iteration raises "dictionary changed size during iteration".
    for name, mod in list(sys.modules.items()):
        if not name.startswith(_CLOSURE_PREFIXES):
            continue
        f = getattr(mod, "__file__", None)
        if f and os.path.exists(f):
            out[os.path.relpath(f, REPO)] = _file_hash(f)
    for f in glob.glob(os.path.join(_HERE, "*.py")):
        # gen_aot_lib.py only CONSUMES sidecars (its edits are picked up
        # by re-running generation); hashing it would re-export every
        # kernel on a generation-only change.
        if os.path.basename(f) == "gen_aot_lib.py":
            continue
        out[os.path.relpath(f, REPO)] = _file_hash(f)
    if decl_path and os.path.exists(decl_path):
        out[os.path.relpath(decl_path, REPO)] = _file_hash(decl_path)
    return dict(sorted(out.items()))


def _json_normal(value):
    """The spec as a sidecar reads it back: tuples become lists, since
    JSON has no tuple type. Skip detection compares a live grid point
    against a recorded spec, so a tuple-valued field must be converted
    or it never matches its own sidecar and re-exports on every run."""
    if isinstance(value, (tuple, list)):
        return [_json_normal(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_normal(v) for k, v in value.items()}
    return value


def _effective_arch(arch: str | None, tc: toolchains.Toolchain) -> str | None:
    """The arch the artifacts are really compiled for: the explicit one if
    given, else whatever the toolchain's own env var selects
    (Toolchain.ARCH_ENV_VAR, e.g. CUTE_DSL_ARCH).

    Resolving it on BOTH sides -- recorded in the sidecar, recomputed by
    the skip check -- is what makes a flat --out-dir safe across runs that
    set only that variable:

        CUTE_DSL_ARCH=sm_90a  export.py --out-dir build/native_aot
        CUTE_DSL_ARCH=sm_100a export.py --out-dir build/native_aot

    Comparing the raw --arch value would leave both runs at None, so the
    second matches on spec alone, skips every point, and the sm_90a
    objects stay on disk behind a sidecar the caller reads as sm_100a."""
    return arch or (os.getenv(tc.ARCH_ENV_VAR) if tc.ARCH_ENV_VAR else None)


def export_point(
    op_pkg: str, kernel_module: str, point: dict, out_dir: str, arch: str | None = None
) -> str:
    """Compile + export ONE spec point and write its sidecar. Self-
    contained (module-level function, picklable args) so it runs
    identically inline and as a pool job. ``arch`` is an explicit sm
    string, passed through to the toolchain (CuTeDSL takes it as
    --gpu-arch, Triton as a fixed GPUTarget); no process-global state, so
    one process may serve any mix of arches.

    A missing DSL runtime is FATAL here, not a skip: a declaration that
    reaches this point targets this build's backend (build_stage2 filtered
    on Toolchain.BACKENDS), so its kernels were asked for, and exporting
    only some of them would ship a wheel that silently underperforms.
    Build without the DSL wheels via TORCH_NATIVE_AOT=0 instead. The
    ImportError arm exists because a builder cannot be asked its kind
    without importing its runtime -- build() constructs the kernel."""
    try:
        build = load_builder(op_pkg, kernel_module)
        b = build(point)
    except ImportError as e:
        raise RuntimeError(
            f"{op_pkg}: cannot export, DSL runtime not installed "
            f"({e.name or e}). Install it, or set TORCH_NATIVE_AOT=0 to "
            f"build without embedded DSL kernels."
        ) from e
    tc = toolchains.get_toolchain(b.get("kind", "cutedsl"))
    missing = tc.missing_runtimes()
    if missing:
        raise RuntimeError(
            f"{op_pkg}: cannot export, {tc.kind} needs {', '.join(missing)}. "
            f"Install them, or set TORCH_NATIVE_AOT=0 to build without "
            f"embedded DSL kernels."
        )
    tc.validate_build_result(b)
    prefix = b["prefix"]
    extra = tc.export(b, out_dir, arch=arch)
    sidecar = {
        "version": SIDECAR_VERSION,
        "prefix": prefix,
        "kind": tc.kind,
        "spec": point,
        "arch": _effective_arch(arch, tc),
        # The declaration lives at a path fixed by construction, so it needs
        # no threading through the job tuple.
        "sources": source_closure(os.path.join(OPS_DIR, op_pkg, "aot.py")),
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


# Every "this tree is inconsistent" error ends the same way. `spin clean`
# does clear the default --out-dir (build/ sits above .gitignore's
# NOT-CLEAN-FILES marker), but it takes the whole build tree with it, so
# name the surgical command first.
_CLEAN_HINT = (
    "run `rm -rf {d}` and re-export (`spin clean` also clears it, "
    "along with the rest of the build tree)"
)


def _read_sidecar(path: str) -> dict:
    """A sidecar's JSON. Unreadable sidecars are fatal.

    The sidecar is written LAST, so its presence marks a completed
    export. One that exists but will not parse means the tree is
    corrupted and the .o/.h beside it are of unknown provenance.
    Re-exporting would just make the directory look consistent again.
    Reached even under --force, via the orphan scan in _collect_jobs, so
    gen_aot_lib never links artifacts nothing validated.
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
    """Cheap skip check without compiling: an exported point's sidecar
    records its spec, its arch and its source closure; skip only when
    all three match -- the spec, the arch the artifacts were compiled
    for, and every recorded source file unchanged on disk (so an edited
    kernel module re-exports without --force).

    The arch check is load-bearing for SEQUENTIAL single-arch runs into
    one --out-dir: those keep the flat <out-root>/<decl_id>/ layout (only
    a multi-arch run nests per-arch), so `--arch sm_100` followed by
    `--arch sm_100a` lands in the same directory. Without comparing arch,
    the second run matches the first run's sidecar on spec alone and
    skips every point, leaving sm_100 objects behind a sidecar that
    claims sm_100a. The comparison goes through _effective_arch so runs
    that set only the toolchain's arch env var are caught the same way."""
    if force:
        return True
    _, _, point, out_dir, arch = job
    spec = _json_normal(point)
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".json"):
            continue
        sc = _read_sidecar(os.path.join(out_dir, fn))
        tc = toolchains.get_toolchain(sc.get("kind", "cutedsl"))
        if sc.get("spec") == spec and sc.get("arch") == _effective_arch(arch, tc):
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
            return not sources_current(sc)
    return True


def _run_job(job) -> str:
    op_pkg, kernel_module, point, out_dir, arch = job
    return export_point(op_pkg, kernel_module, point, out_dir, arch)


def archs_from_cuda_arch_list(arch_list: str) -> list[str]:
    """TORCH_CUDA_ARCH_LIST -> the sm strings from it that are
    EXPORTABLE_ARCHES, order-preserving and deduplicated.

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
# sm_103/sm_103a are deliberately absent: nothing names 10.3, sm_100 SASS
# is forward-compatible to it, and _arch_gate compares only the CUDA
# major -- so an sm_103 artifact would pass the gate on a 10.0 device and
# then fail the module load instead of declining. Re-add with a
# major+minor gate.
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
