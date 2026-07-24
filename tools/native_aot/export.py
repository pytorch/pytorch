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

Usage (from the repo root, venv with torch built and the DSL wheel
active):
    python tools/native_aot/export.py [--out-dir build/native_aot]
                                        [--ops topk] [--force] [--jobs 8]
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
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from torch._native._spec_grid import expand_specs


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


def export_point(op_pkg: str, kernel_module: str, point: dict, out_dir: str) -> str:
    """Compile + export ONE spec point and write its sidecar. Self-
    contained (module-level function, picklable args) so it runs
    identically inline and as a spawn-pool job."""
    build = load_builder(op_pkg, kernel_module)
    b = build(point)
    tc = toolchains.get_toolchain(b.get("kind", "cutedsl"))
    tc.validate_build_result(b)
    prefix = b["prefix"]
    extra = tc.export(b, out_dir)
    sidecar = {
        "version": SIDECAR_VERSION,
        "prefix": prefix,
        "kind": tc.kind,
        "spec": point,
        "sources": source_closure(),
        **extra,
    }
    with open(os.path.join(out_dir, prefix + ".json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    return prefix


def _collect_jobs(ops_filter, out_root: str, force: bool):
    """(op_pkg, kernel_module, point, out_dir) per spec point across
    every declaration; grids expand here (cheap, torch-light), skip
    detection is _job_needed's sidecar scan."""
    jobs = []
    for entry in sorted(os.listdir(OPS_DIR)):
        op_dir = os.path.join(OPS_DIR, entry)
        if not os.path.exists(os.path.join(op_dir, "aot.py")):
            continue
        for d in decl.load_declarations(os.path.join(op_dir, "aot.py")):
            did = decl.decl_id(d)
            if ops_filter and entry not in ops_filter and d.ATEN_OP not in ops_filter:
                continue
            out_dir = os.path.join(out_root, did)
            os.makedirs(out_dir, exist_ok=True)
            for point in expand_specs(d.kernel_precompile_grid()):
                jobs.append((entry, d.KERNEL_MODULE, point, out_dir))
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
    _, _, point, out_dir = job
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
    op_pkg, kernel_module, point, out_dir = job
    return export_point(op_pkg, kernel_module, point, out_dir)


def archs_from_cuda_arch_list(arch_list: str) -> list[str]:
    """TORCH_CUDA_ARCH_LIST -> sm strings the DSL toolchains accept,
    restricted to EXPORT_SMS.

    "9.0a;10.0a" (or space-separated) -> ["sm_100a"]. Named entries
    ("Hopper") and +PTX suffixes are not translated -- callers should
    pass numeric lists (CI does)."""
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
        if sm in EXPORT_SMS:
            out.append(sm)
    return out


# Architectures the standard build exports AOT kernels for: Blackwell
# only, for now. Entries outside this set are skipped (not failed), so
# a mixed arch list ("7.5 9.0a 10.0a") exports just the Blackwell
# subset and other builds proceed without artifacts. Single-arch also
# keeps the flat artifacts layout the embedded link globs (multi-arch
# nests per-arch trees, which the link does not walk).
EXPORT_SMS = ("sm_100", "sm_100a", "sm_103", "sm_103a")


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
    args = parser.parse_args()
    if args.jobs is None:
        env_jobs = os.getenv("MAX_JOBS") or os.getenv("CMAKE_BUILD_PARALLEL_LEVEL")
        args.jobs = int(env_jobs) if env_jobs else os.cpu_count() or 1

    jobs = _collect_jobs(args.ops, args.out_dir, args.force)
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
        import multiprocessing
        from concurrent.futures import as_completed, ProcessPoolExecutor

        ctx = multiprocessing.get_context("spawn")
        n = min(args.jobs, len(todo))
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
            futs = {pool.submit(_run_job, job): job for job in todo}
            for fut in as_completed(futs):
                prefix = fut.result()  # re-raises worker failures
                print(f"  {prefix}: exported")
                total += 1

    print(f"exported {total} kernels")


if __name__ == "__main__":
    main()
