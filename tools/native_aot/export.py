"""AOT-export DSL kernels for native-AOT declarations.

Stage 2 of the two-stage build, with the built torch importable, so kernel builder
modules are ordinary package imports (``torch._native.ops.<op>.<module>``) and share
code with their JIT wrappers. Only the aot.py DECLARATION modules stay torch-free at
module scope, since torchgen loads those during stage 1.

For each ``torch/_native/ops/<op>/aot.py`` this expands the spec grid (list fields
cross-multiply) and, per grid point, runs the toolchain's compile and export into
``<out-dir>/<target>/<op>/`` -- one tree per selected compile target -- writing:

    <prefix>.h / <prefix>.o    C-ABI header + kernel object
    <prefix>.json              marshalling sidecar {spec, arch, tensor_args}

Prefixes carry their target (``my_kernel__sm100a``): every exported C symbol derives
from the prefix, so two targets sharing one would collide in libtorch_cuda.

The builder module exposes ``build(spec)`` returning a dict whose ``kind`` selects
the toolchain; tools/native_aot/toolchains.py holds the per-kind contracts. Existing
artifacts are skipped unless --force.

Spec points compile on a forkserver pool, one job per (point, target), so results do
not depend on --jobs, which follows the torch build's parallelism (MAX_JOBS, then
CMAKE_BUILD_PARALLEL_LEVEL, then half the CPU count). Plain fork is unusable: the
parent may have initialized CUDA, and forked workers inherit a dead context silently.

With --arch (one or more sm strings) export never touches the CUDA driver, so kernels
build on GPU-less machines. Each value names a device the containing build supports;
the exporter chooses the widest compatible target from each op's ARCHS. The selected
target is per-compile rather than per-process -- CuTeDSL takes --gpu-arch, which
outranks CUTE_DSL_ARCH. CuTeDSL needs one warmup compile per process for that; see
tools/native_aot/cutedsl_warmup.py.

Usage (from the repo root, in a venv with torch built and the DSL wheel active):
    python tools/native_aot/export.py [--out-dir build/native_aot]
                                        [--ops my_op] [--force] [--jobs 8]
                                        [--arch sm_90 sm_100]
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
# dead context, silently. Forkserver is as safe as spawn and, on Python versions
# with correct preload path handling, pays the torch import once rather than per worker.
# TODO(native-aot): forkserver does not exist on Windows; fall back to "spawn" when
# Windows CUDA builds start exporting.
POOL_START_METHOD = "forkserver"


# CPython gh-117378 fixed forkserver preload's sys.path inheritance in 3.12.8 and
# 3.13.1. On older versions, skip preloading so workers import torch after their
# parent sys.path is restored. Importing torch is otherwise safe in a fork parent:
# it initializes neither CUDA nor DSL state.
def _pool_preload(version: tuple[int, ...]) -> tuple[str, ...]:
    fixed = (
        version >= (3, 14)
        or (version[:2] == (3, 13) and version >= (3, 13, 1))
        or (version[:2] == (3, 12) and version >= (3, 12, 8))
    )
    return ("torch",) if fixed else ()


POOL_PRELOAD = _pool_preload(sys.version_info[:3])


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
    """The local device as a plain sm string ("sm_100"), or None without CUDA.

    This is a supported-device input to declaration target selection, not necessarily
    the compile target recorded in the sidecar."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    return f"sm_{major * 10 + minor}"


def _arch_tag(arch: str) -> str:
    """Compile target as an artifact-name tag: "sm_100a" -> "sm100a"."""
    return arch.replace("_", "", 1)


def _effective_arch(arch: str | None) -> str | None:
    """Resolve an explicit sm spelling, or detect the local device when absent.

    A toolchain-specific arch variable cannot describe a build containing several
    kinds, so it is refused rather than used. Declaration target selection happens
    after this function resolves the supported device."""
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
            f"for every kind in one export; remove it and pass --arch (e.g. "
            f"--arch {named[min(named)]}) to state the supported device once. "
            f"Each declaration's ARCHS will select the compile target."
        )
    return _detected_arch()


def export_point(
    op_pkg: str, kernel_module: str, point: dict, out_dir: str, arch: str | None = None
) -> str:
    """Compile and export one spec point, and write its sidecar.

    Module-level with picklable arguments, so it runs identically inline and as a pool
    job, and holds no process-global state, so one process serves any mix of targets.

    A missing DSL runtime is fatal here rather than a skip: a declaration reaching
    this point targets this build's backend, so its kernels were asked for, and
    exporting only some of them ships a wheel that silently underperforms. Use
    TORCH_NATIVE_AOT=0 to build without them. The ImportError arm exists because a
    builder cannot be asked its kind without importing its runtime."""
    try:
        build = load_builder(op_pkg, kernel_module)
        b = build(point)
    except ImportError as e:
        runtimes = {
            runtime
            for tc in toolchains.TOOLCHAINS.values()
            for runtime in tc.REQUIRED_RUNTIMES
        }
        if e.name and any(
            e.name == runtime or e.name.startswith(runtime + ".")
            for runtime in runtimes
        ):
            raise RuntimeError(
                f"{op_pkg}: cannot export, DSL runtime not installed "
                f"({e.name}). Install it, or set TORCH_NATIVE_AOT=0 to "
                f"build without embedded DSL kernels."
            ) from e
        raise RuntimeError(
            f"{op_pkg}: cannot export because its builder import failed ({e})."
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
    # Target-qualifying the prefix lets several targets ship in one library:
    # every exported C symbol derives from it, so two targets sharing one are
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
    """One job tuple per spec point and selected compile target.

    Grids expand here, which is cheap and torch-light; skip detection is
    _job_needed's sidecar scan. Targets always use
    <out-root>/<target>/<decl_id>/, while the generated .cpp sits at
    <out-root>/<decl_id>/ and covers every selected target."""
    jobs = []
    build_archs = _resolved_build_arches(archs)
    for entry, d in _iter_declarations(ops_filter):
        did = decl.decl_id(d)
        targets = _targets_for_declaration(d, build_archs)
        if not targets:
            print(
                f"{did}: declares kernels but none for this build -- supported "
                f"devices {' '.join(build_archs)}, and the declaration's ARCHS "
                f"({' '.join(decl.archs_of(d))}) has no compatible target, so this "
                f"op falls back to aten."
            )
            continue
        points = expand_specs(d.kernel_precompile_grid())
        for target in targets:
            out_dir = os.path.join(out_root, target, did)
            os.makedirs(out_dir, exist_ok=True)
            _check_no_orphan_artifacts(out_dir, points)
            for point in points:
                jobs.append((entry, d.KERNEL_MODULE, point, out_dir, target))
    return jobs


def _iter_declarations(ops_filter):
    """Yield ``(op package, declaration)`` in stable filesystem order."""
    for entry in sorted(os.listdir(OPS_DIR)):
        path = os.path.join(OPS_DIR, entry, "aot.py")
        if not os.path.exists(path):
            continue
        for d in decl.load_declarations(path):
            if ops_filter and entry not in ops_filter and d.ATEN_OP not in ops_filter:
                continue
            yield entry, d


def _resolved_build_arches(archs) -> dict[str, tuple[int, int]]:
    """Resolve supported-device spellings to compute capabilities."""
    out: dict[str, tuple[int, int]] = {}
    for arch in archs:
        resolved = _effective_arch(arch)
        if not resolved:
            raise RuntimeError(
                "cannot determine the arch to export for: no --arch given and no "
                "local GPU to detect from. Pass --arch (e.g. --arch sm_100a), "
                "which also lets export run on a machine without a GPU."
            )
        cc = decl.cc_of(resolved)
        out.setdefault(resolved, cc)
    return out


def _targets_for_declaration(d, build_archs: dict[str, tuple[int, int]]) -> list[str]:
    """One declaration's widest compatible targets for the supported devices."""
    out = []
    candidates = decl.archs_of(d)
    for device_cc in build_archs.values():
        target = decl.widest_compatible_target(candidates, device_cc)
        if target is not None and target not in out:
            out.append(target)
    return out


def targets_for_arches(archs, ops_filter=None) -> list[str]:
    """All per-op AOT targets selected for supported device architectures."""
    build_archs = _resolved_build_arches(archs)
    out = []
    for _, d in _iter_declarations(ops_filter):
        for target in _targets_for_declaration(d, build_archs):
            if target not in out:
                out.append(target)
    return out


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


def _invalidate_generation(out_dir: str) -> None:
    """Drop the previous generation, so the tree reads as not-generated-yet.

    Overwritten with the nothing-to-embed include rather than deleted: CMake keeps a
    configure dependency on an include()d file only if it existed at configure time,
    so unlinking it makes a later generation invisible to a plain `cmake --build`,
    which relinks without the kernels and reports success. See
    gen_aot_lib.write_nothing_to_embed."""
    from tools.native_aot.gen_aot_lib import CMAKE_INCLUDE, write_nothing_to_embed

    stale = os.path.join(out_dir, CMAKE_INCLUDE)
    if os.path.exists(stale):
        write_nothing_to_embed(out_dir)
        print(f"invalidated {stale}; regenerate after this export")


def _check_no_orphan_artifacts(out_dir: str, specs) -> None:
    """Report or refuse kernel artifacts no current grid point claims.

    Artifacts NO SIDECAR CLAIMS are reported, not fatal: they come from an export
    interrupted between writing them and writing its sidecar, nothing links an
    artifact no sidecar names, and re-exporting that point overwrites it. Reported
    whether or not another point in the directory committed, so the first export of a
    tree is not a special case.

    Artifacts whose sidecar records a spec no longer in the grid are fatal: dropping a
    point from kernel_precompile_grid() generates no job for it and nothing prunes it,
    while the sidecar still makes it look exported.

    An empty directory is fine. ``specs`` is the expanded grid for this
    (declaration, compile target).
    """
    exts = toolchains.all_artifact_exts()
    names = os.listdir(out_dir)
    # Per artifact, not per directory: an interrupt lands among points that already
    # committed, so "does this directory hold any sidecar" would see nothing wrong.
    claimed = {os.path.splitext(n)[0] for n in names if n.endswith(".json")}
    orphans = sorted(
        n
        for n in names
        if os.path.splitext(n)[1] in exts and os.path.splitext(n)[0] not in claimed
    )
    if orphans:
        listed = f"{', '.join(orphans[:4])}{', ...' if len(orphans) > 4 else ''}"
        print(
            f"{out_dir}: {len(orphans)} artifact(s) no sidecar claims ({listed}); an "
            f"export died before committing them, or they were copied in by hand. "
            f"Nothing links an artifact no sidecar names, so the cost is disk: a "
            f"re-export of that point overwrites it WHILE THE POINT IS STILL IN THE "
            f"GRID, and otherwise {_CLEAN_HINT.format(d=out_dir)}."
        )
    live = [_json_normal(p) for p in specs]

    def _is_stale(fn: str) -> bool:
        sc = _read_sidecar(os.path.join(out_dir, fn))
        # SCHEMA FIRST, as everywhere: "spec" is read by name, and a bump that
        # changed its representation would make the first export in an existing
        # tree demand `rm -rf` rather than re-export. Another schema is not
        # stale, it is unreadable.
        if sc.get("version") != SIDECAR_VERSION:
            return False
        return sc.get("spec") not in live

    stale = sorted(fn for fn in names if fn.endswith(".json") and _is_stale(fn))
    if stale:
        raise RuntimeError(
            f"{out_dir}: {len(stale)} sidecar(s) for spec points no longer in the "
            f"grid ({', '.join(stale[:4])}{', ...' if len(stale) > 4 else ''}). "
            f"Generation emits one dispatch branch per sidecar and takes the spec "
            f"from the sidecar itself, so these would ship a wired-up kernel for a "
            f"point the grid no longer has; {_CLEAN_HINT.format(d=out_dir)}."
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
    """Cheap skip check without compiling: skip only when the sidecar's spec, target
    and every file in its source closure still match, so an edited kernel module
    re-exports without --force.

    The target comparison catches artifacts predating target identity or a tree
    carried between configurations. Both must re-export, since the recorded target
    is what the runtime gate is built from.
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


def build_arches_from_cuda_arch_list(arch_list: str) -> list[str]:
    """TORCH_CUDA_ARCH_LIST -> supported device architectures.

    Feature-set suffixes do not constrain an op's compiler target: ARCHS does. A
    +PTX suffix is stripped and named entries ("Hopper") are not translated; CI
    passes numeric lists."""
    out = []
    for entry in arch_list.replace(";", " ").split():
        major, _, minor = entry.removesuffix("+PTX").partition(".")
        suffix = minor[-1:] if minor.endswith(("a", "f")) else ""
        minor = minor.removesuffix(suffix) if suffix else minor
        # isascii too: str.isdigit is Unicode-aware, so an Arabic-Indic or
        # full-width digit satisfies it AND converts, which torchgen's sm parsing
        # rejects for the same reason.
        if not all(p.isascii() and p.isdigit() for p in (major, minor)):
            continue  # named arch ("Hopper") or malformed: skip
        arch = f"sm_{int(major) * 10 + int(minor)}"
        try:
            cc = decl.cc_of(arch)
        except RuntimeError:
            continue
        if cc not in decl.known_device_capabilities():
            continue
        if arch not in out:
            out.append(arch)
    return out


def archs_from_cuda_arch_list(arch_list: str) -> list[str]:
    """TORCH_CUDA_ARCH_LIST -> per-op-selected native-AOT compile targets."""
    return targets_for_arches(build_arches_from_cuda_arch_list(arch_list))


def main(argv: list[str] | None = None) -> None:
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
        help="parallel compile processes (forkserver). Default follows the "
        "torch build: MAX_JOBS, then CMAKE_BUILD_PARALLEL_LEVEL, then half the "
        "CPU count.",
    )
    parser.add_argument(
        "--arch",
        nargs="*",
        default=None,
        metavar="SM",
        help="device architecture(s) supported by the build, e.g. --arch sm_90 "
        "sm_100. Each op selects a compatible compiler target from its ARCHS. "
        "With an explicit arch, export never touches the CUDA driver and runs "
        "on GPU-less machines. Default: detect from the local device.",
    )
    args = parser.parse_args(argv)
    # Checked before anything reads the disk, so a typo (`--arch sm100a`) is refused
    # rather than matching no declaration and exiting 0.
    #
    # TORCH_CUDA_ARCH_LIST is filtered below rather than refused: a release list names
    # arches AOT does not ship (7.5, 8.6) and must not fail the build for it.
    for named_arch in args.arch or ():
        try:
            cc = decl.cc_of(named_arch)
        except RuntimeError as e:
            raise RuntimeError(
                f"--arch {named_arch} is not an arch this tooling knows"
            ) from e
        if cc not in decl.known_device_capabilities():
            raise RuntimeError(
                f"--arch {named_arch} is not an arch this tooling knows. To support "
                f"another device, add it to an architecture family or add a compile "
                f"target for it."
            )
    if args.jobs is None:
        env_jobs = os.getenv("MAX_JOBS") or os.getenv("CMAKE_BUILD_PARALLEL_LEVEL")
        # Half the CPU count, not all of it: os.cpu_count() reports SMT
        # siblings, and one compile per virtual thread oversubscribes.
        args.jobs = int(env_jobs) if env_jobs else max(1, (os.cpu_count() or 2) // 2)
    from_arch_list = args.arch is None and bool(os.getenv("TORCH_CUDA_ARCH_LIST"))
    if from_arch_list:
        # Standard-build integration: the main build supplies devices; declarations
        # supply compiler targets. Explicit --arch wins.
        args.arch = build_arches_from_cuda_arch_list(os.environ["TORCH_CUDA_ARCH_LIST"])
    archs = args.arch if args.arch is not None else [None]
    try:
        jobs = _collect_jobs(args.ops, args.out_dir, archs)
    except RuntimeError:
        # Every refusal in there tells the user to remove a target tree, and the
        # previous generation names every object in it -- so following the advice made
        # the NEXT main build fail in CMake on a missing source, inside a @generated
        # file that says nothing about native-AOT. Invalidate before re-raising, the
        # same rule as below.
        _invalidate_generation(args.out_dir)
        raise
    if from_arch_list:
        selected = list(dict.fromkeys(job[4] for job in jobs))
        if not selected:
            print(
                "TORCH_CUDA_ARCH_LIST matches no native-AOT declaration target; "
                "nothing to export"
            )
            return
        print(f"AOT targets from TORCH_CUDA_ARCH_LIST: {' '.join(selected)}")
    todo = [j for j in jobs if _job_needed(j, args.force)]
    if len(todo) < len(jobs):
        print(f"{len(jobs) - len(todo)} points already exported, skipped")
    if todo:
        # Invalidate the previous generation before touching any artifact. Artifacts
        # are direct link inputs in build.ninja while generation is not a build step,
        # so an interrupted export followed by a plain `cmake --build` would relink a
        # library mixing objects from two revisions, described by launchers generated
        # for the older ones. Stage 2 and the by-hand flow are unaffected: generation
        # rewrites it.
        _invalidate_generation(args.out_dir)

    total = 0
    if args.jobs <= 1 or len(todo) <= 1:
        for job in todo:
            prefix = _run_job(job)
            print(f"  {prefix}: exported")
            total += 1
    else:
        # ONE pool over every (point, target) job: each toolchain takes its
        # target per compile (CuTeDSL --gpu-arch, Triton a fixed GPUTarget),
        # so no process is pinned to a target and mixed jobs pack freely.
        import multiprocessing
        from concurrent.futures import as_completed, ProcessPoolExecutor

        ctx = multiprocessing.get_context(POOL_START_METHOD)
        # Fixed Pythons import torch once in the server; affected ones import per worker.
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
