"""AOT-export DSL kernels for native-AOT declarations.

Stage 2 of the two-stage build (build torch -> build the AOT lib): runs
with the BUILT torch importable, so kernel builder modules are ordinary
package imports (``torch._native.ops.<op>.<module>``) and may freely
share code with their JIT wrappers -- the same-kernel property is then
by construction, not by parallel restatement. Only the aot.py
DECLARATION modules stay stdlib-only at module scope (torchgen loads
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

Spec points compile on a spawn-context process pool (Inductor's
compile_worker lesson: never fork after CUDA init -- the parent
initializes CUDA through the first cute.compile, and Triton export
queries device capability). Each worker imports torch + the DSL
runtime once and then serves (op, kernel_module, spec, out_dir) jobs;
results are byte-identical to a serial run (verified by test). --jobs
defaults to the torch build's own parallelism (MAX_JOBS, then
CMAKE_BUILD_PARALLEL_LEVEL, then CPU count); --jobs 1 forces serial.

With --arch (one or more sm strings) export never touches the CUDA
driver: CuTeDSL takes the arch via CUTE_DSL_ARCH and Triton via a
fixed-target driver, so kernels build on GPU-less machines. Multiple
archs fan out under <out-dir>/<arch>/ with one worker pool per arch
(both DSLs pin the arch per process).

Usage (from the repo root, venv with torch built and the DSL wheel
active):
    python tools/native_aot/export.py [--out-dir build/native_aot]
                                        [--ops topk] [--force] [--jobs 8]
                                        [--arch sm_90a sm_100a]
"""

import argparse
import importlib.util
import json
import os


REPO = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
OPS_DIR = os.path.join(REPO, "torch", "_native", "ops")

# Sidecar schema version. Bump on any change to the sidecar layout or
# to the launcher-generation contract that reads it; gen_aot_lib refuses
# mismatched sidecars (re-export rather than debugging a garbled .cpp).
SIDECAR_VERSION = 1


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# By file path, not `from torch._native...`: a package import executes
# torch/__init__.py, but grid expansion (and the tools test suite) must
# work on a checkout where torch is not built. _spec_grid is torch-free
# by contract (see its docstring).
expand_specs = _load_by_path(
    "_spec_grid", os.path.join(REPO, "torch", "_native", "_spec_grid.py")
).expand_specs


def load_builder(op: str, kernel_module: str):
    # Package import (not file-path load): builders may use relative
    # imports and torch machinery -- stage 2 runs against built torch.
    name = f"torch._native.ops.{op}.{kernel_module.removesuffix('.py')}"
    return importlib.import_module(name).build


_HERE = os.path.dirname(os.path.abspath(__file__))
toolchains = _load_by_path("toolchains", os.path.join(_HERE, "toolchains.py"))
decl = _load_by_path("decl", os.path.join(_HERE, "decl.py"))


def _file_hash(path: str) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def source_closure() -> dict[str, str]:
    """{repo-relative path: content hash} for every torch._native
    source file currently loaded -- the builder's live import closure
    (kernel module plus everything it pulled in: traits, launch glue,
    shared tables) -- plus this tool's own sources (a launcher-template
    edit in toolchains.py changes what a sidecar MEANS, so it must
    invalidate artifacts like a kernel edit does). Recorded per sidecar
    at export; gen_aot_lib re-hashes the files from disk and refuses to
    pair edited kernel sources with stale artifacts (the
    touch-and-forget --force footgun otherwise). The closure
    over-approximates (editing an unrelated loaded _native module
    invalidates too); staleness must err toward re-export."""
    import glob
    import sys

    out = {}
    for name, mod in sys.modules.items():
        if not name.startswith("torch._native"):
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
    return dict(sorted(out.items()))


def _json_normal(point: dict) -> dict:
    """The spec as it reads back from a sidecar (tuples -> lists,
    dict-key ordering). Skip detection compares specs across a JSON
    round-trip, so both sides must be in this form -- a tuple-valued
    grid field would otherwise never match its own sidecar and
    re-export on every run."""
    return json.loads(json.dumps(point))


def export_point(
    op_pkg: str, kernel_module: str, point: dict, out_dir: str, arch: str | None = None
) -> str:
    """Compile + export ONE spec point and write its sidecar. Self-
    contained (module-level function, picklable args) so it runs
    identically inline and as a spawn-pool job. ``arch``: explicit sm
    string; also mirrored into CUTE_DSL_ARCH (the DSL caches it at the
    FIRST read, so it must be in the env before any cutlass import --
    safe here because workers are fresh spawn processes and the inline
    path sets it in main() before builders import)."""
    if arch:
        os.environ.setdefault("CUTE_DSL_ARCH", arch)
    build = load_builder(op_pkg, kernel_module)
    b = build(point)
    tc = toolchains.get_toolchain(b.get("kind", "cutedsl"))
    tc.validate_build_result(b)
    prefix = b["prefix"]
    extra = tc.export(b, out_dir, arch=arch)
    sidecar = {
        "version": SIDECAR_VERSION,
        "prefix": prefix,
        "kind": tc.kind,
        "spec": point,
        "arch": arch,
        "sources": source_closure(),
        **extra,
    }
    with open(os.path.join(out_dir, prefix + ".json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    return prefix


def _collect_jobs(ops_filter, out_root: str, force: bool, archs):
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
                if arch is not None and arch not in d.ARCHS:
                    continue
                # `multi` implies explicit sm strings (a [None] arch
                # list is always length 1); the arch check narrows for
                # the type checker.
                root = os.path.join(out_root, arch) if multi and arch else out_root
                out_dir = os.path.join(root, did)
                os.makedirs(out_dir, exist_ok=True)
                for point in expand_specs(d.kernel_precompile_grid()):
                    jobs.append((entry, d.KERNEL_MODULE, point, out_dir, arch))
    return jobs


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
    records its spec and source closure; skip only when the spec
    matches AND every recorded source file is unchanged on disk --
    an edited kernel module re-exports without --force."""
    if force:
        return True
    _, _, point, out_dir, _arch = job
    for fn in os.listdir(out_dir):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(out_dir, fn)) as f:
                sc = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if sc.get("spec") == _json_normal(point):
            return not sources_current(sc)
    return True


def _run_job(job) -> str:
    op_pkg, kernel_module, point, out_dir, arch = job
    return export_point(op_pkg, kernel_module, point, out_dir, arch)


def archs_from_cuda_arch_list(arch_list: str) -> list[str]:
    """TORCH_CUDA_ARCH_LIST -> the sm strings from it that are
    EXPORTABLE_ARCHES, order-preserving and deduplicated.

    "9.0a;10.0a" (or space-separated) -> ["sm_100a"]. Named entries
    ("Hopper") and +PTX suffixes are not translated -- callers should
    pass numeric lists (CI does). Dedup matters: "10.0;10.0+PTX" names
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


# Arch allow-list for the AUTOMATIC export path: which entries of the main
# build's TORCH_CUDA_ARCH_LIST are ELIGIBLE for AOT kernels. Blackwell only,
# for now. A filter, never a build list -- it cannot cause an export, only
# permit one, and an explicit `--arch` bypasses it entirely (so a dev can
# hand-export for any arch a declaration claims). Consequences:
#
#   * an arch list with no eligible entry ("7.5 9.0") exports NOTHING and
#     stage 2 skips silently, leaving a normal artifact-free build;
#   * a mixed list ("7.5 9.0a 10.0a") exports just the eligible subset.
#
# Both spellings of a CC are listed because they are distinct nvcc targets
# and different builds use different ones for the SAME hardware: "10.0a"
# (arch-conditional, what tcgen05/wgmma need) in b200-native-aot.yml, plain
# "10.0" in every other Blackwell job and in the shipped manywheel lists.
# Omitting either makes those builds silently export nothing.
#
# NOT the same set as decl.py's _DEFAULT_ARCHS or a declaration's ARCHS,
# which say what the KERNELS support (sm_90+); this says what the standard
# build SHIPS. Keeping it to one selected arch per build also preserves the
# flat artifacts layout the embedded CMake globs walk (multi-arch nests
# per-arch trees, which they do not).
#
# sm_103/sm_103a are deliberately absent: no CI or release arch list names
# 10.3, sm_100 SASS is forward-compatible to CC 10.3, and _arch_gate only
# compares the CUDA major -- so an sm_103 artifact would pass the gate on a
# 10.0 device and then fail the module load instead of declining. Re-add
# them together with a major+minor gate.
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
        help="parallel compile processes (spawn). Default follows the "
        "torch build's parallelism: MAX_JOBS, then "
        "CMAKE_BUILD_PARALLEL_LEVEL, then the CPU count -- the same "
        "precedence tools/setup_helpers/cmake.py hands to ninja.",
    )
    parser.add_argument(
        "--arch",
        nargs="*",
        default=None,
        metavar="SM",
        help="target architecture(s), e.g. --arch sm_90a sm_100a. With an "
        "explicit arch, export never touches the CUDA driver and runs on "
        "GPU-less machines (CuTeDSL via CUTE_DSL_ARCH; Triton via an "
        "explicit GPUTarget). Default: detect from the local device. "
        "Multiple archs nest artifacts under <out-dir>/<arch>/.",
    )
    args = parser.parse_args()
    if args.jobs is None:
        env_jobs = os.getenv("MAX_JOBS") or os.getenv("CMAKE_BUILD_PARALLEL_LEVEL")
        args.jobs = int(env_jobs) if env_jobs else os.cpu_count() or 1
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
    if len(archs) > 1 and args.jobs <= 1:
        # Multi-arch REQUIRES the pool: CUTE_DSL_ARCH is cached per
        # process at first read, so one inline process cannot compile
        # two archs. Spawn workers are fresh processes per job.
        raise SystemExit("--arch with multiple targets requires --jobs > 1")

    jobs = _collect_jobs(args.ops, args.out_dir, args.force, archs)
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
        # Spawn context: the parent has initialized CUDA (grid expansion
        # may import torch); forked children would inherit a broken
        # context. Same reasoning as Inductor's compile_worker pool.
        # One pool PER ARCH, sequentially: CUTE_DSL_ARCH is cached per
        # process at first read, so a worker that compiled arch A would
        # silently compile arch B's cutedsl kernels for A. Workers die
        # with their pool, so each arch gets fresh processes.
        import multiprocessing
        from concurrent.futures import as_completed, ProcessPoolExecutor

        ctx = multiprocessing.get_context("spawn")
        for arch in archs:
            arch_todo = [j for j in todo if j[4] == arch]
            if not arch_todo:
                continue
            if len(archs) > 1:
                print(f"[{arch}] {len(arch_todo)} points")
            n = min(args.jobs, len(arch_todo))
            with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
                futs = {pool.submit(_run_job, job): job for job in arch_todo}
                for fut in as_completed(futs):
                    prefix = fut.result()  # re-raises worker failures
                    print(f"  {prefix}: exported")
                    total += 1

    print(f"exported {total} kernels")


if __name__ == "__main__":
    main()
