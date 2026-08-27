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
compile + export into ``<out-dir>/<arch>/<op>/`` -- one tree per arch,
whatever the arch count -- writing:

    <prefix>.h / <prefix>.o    C-ABI header + kernel object
    <prefix>.json              marshalling sidecar {spec, arch, tensor_args}

Prefixes carry their arch (``topk_..._det__sm100a``): every exported C
symbol derives from the prefix, so two arches sharing one would be
duplicate definitions once both link into libtorch_cuda.

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
every (point, arch) job. Every arch nests under <out-dir>/<arch>/ to
keep each tree independently linkable.
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


# Loaded modules under these prefixes belong in the source closure and are not
# caught by the tools/*.py glob below: torch._native is the builder's own
# closure, torchgen.native_aot the shared declaration machinery, and
# torch._vendor the vendored DSL packages -- which hold real kernel bodies, so
# without them an edited body left every closure unchanged and a relink shipped
# kernels compiled from the old source, unreported.
_CLOSURE_PREFIXES = ("torch._native", "torch._vendor", "torchgen.native_aot")

# Tool sources that cannot change what an artifact MEANS, so hashing them would
# re-export every kernel for an edit that could not have changed one:
# gen_aot_lib.py only consumes sidecars, and build_stage2.py only decides whether
# stage 2 runs -- it passes no kernel-affecting option, the arch list being read
# here (see main()).
_CLOSURE_EXCLUDED = ("gen_aot_lib.py", "build_stage2.py")


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
        if os.path.basename(f) in _CLOSURE_EXCLUDED:
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


def _detected_arch() -> str | None:
    """The local device as an sm string ("sm_100"), or None without CUDA.

    Without this in the sidecar an on-device export's artifacts carry no arch
    identity, and the generated gate falls back to the declaration's ARCHS --
    advertising hardware nothing was compiled for (a B200 build offering major 9
    because ARCHS lists sm_90). Recorded without the "a" suffix: the gate compares
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
    """Arch as an artifact-name tag: "sm_100a" -> "sm100a".

    Short because it lands in every exported C symbol, and those are already
    long (cute_dsl_<prefix>_wrapper over a spec-derived prefix)."""
    return arch.replace("_", "", 1)


def _effective_arch(arch: str | None) -> str | None:
    """The arch the artifacts are really compiled for: the explicit one if
    given, else whatever a toolchain's own env var selects
    (Toolchain.ARCH_ENV_VAR, e.g. CUTE_DSL_ARCH), else the local device.

    ONE resolver for both the sidecar and the directory name, so a tree can never
    disagree with its own sidecars -- the runtime gate comes from the sidecar, and
    a directory saying otherwise would make the layout lie.

    Recorded and recomputed by the skip check, so two runs differing only in that
    variable are told apart; comparing the raw --arch value left both at None and
    the second skipped every point.

    Takes no toolchain, and that is the invariant rather than an omission: once a
    kind's arch variable is refused rather than honoured, resolution is --arch or
    the device, neither of which varies by kind. A resolver that consulted one
    could answer differently for two kinds in a single export, which is exactly how
    a tree came to disagree with its own sidecars."""
    if arch:
        return arch
    # An arch variable with no --arch is REFUSED, not honoured: it is per-kind, so
    # it answers only for kinds that declare one. With CUTE_DSL_ARCH=sm_90a set, a
    # tree named sm_90a held a sidecar recording the DETECTED sm_100 for a kind
    # with no variable -- and generation filters by directory while the shipped
    # gate comes from the sidecar. Refusing leaves one arch=None path that every
    # kind answers identically.
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


def _claimed_spelling(arch: str, claimed: tuple[str, ...]) -> str | None:
    """The spelling in ``claimed`` for ``arch``'s capability, or None.

    Prefers the arch-conditional spelling when a declaration lists both, matching
    the generator's tie-break: it is what the kernels were written against."""
    want = decl.cc_of(arch)
    same_cc = [a for a in claimed if decl.cc_of(a) == want]
    return min(same_cc, key=lambda a: (not a.endswith("a"), a)) if same_cc else None


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
    torch-light), skip detection is _job_needed's sidecar scan.

    ONE layout whatever the arch count: <out-root>/<arch>/<decl_id>/, so adding
    an arch is another directory rather than a different shape. A per-count layout
    left the other shape's tree behind when a build switched between one arch and
    several, and the same arch then appeared twice for one declaration
    (gen_aot_lib's duplicate-prefix check stays as a backstop).

    The generated .cpp sits at <out-root>/<decl_id>/, covering all of them."""
    jobs = []
    # Declarations that matched NO requested arch, reported at the end: an ARCHS
    # of only conditional spellings ships nothing for a release list of plain ones,
    # and with several declarations the result is partial but looks healthy -- the
    # matched ops embed and pass the post-relink check while the rest are simply
    # absent, with no tree for generation to complain about.
    skipped: dict[str, list[str]] = {}
    declared: dict[str, tuple[str, ...]] = {}
    for entry in sorted(os.listdir(OPS_DIR)):
        op_dir = os.path.join(OPS_DIR, entry)
        if not os.path.exists(os.path.join(op_dir, "aot.py")):
            continue
        for d in decl.load_declarations(os.path.join(op_dir, "aot.py")):
            did = decl.decl_id(d)
            if ops_filter and entry not in ops_filter and d.ATEN_OP not in ops_filter:
                continue
            for arch in archs:
                # No unnamed layout: an artifact whose arch nobody can state
                # cannot be matched to hardware by the runtime gate.
                layout_arch = _effective_arch(arch)
                if not layout_arch:
                    raise RuntimeError(
                        "cannot determine the arch to export for: no --arch "
                        "given and no local GPU to detect from. Pass --arch "
                        "(e.g. --arch sm_100a), which also lets export run on "
                        "a machine without a GPU."
                    )
                # TWO paths, because this name is also what generation filters
                # trees by (--archs, from the same list stage 2 passed here):
                #
                #   * an EXPLICIT arch is used verbatim, and a declaration not
                #     claiming it is skipped. Resolving it to another spelling
                #     named the tree something generation was never told about, so
                #     it embedded nothing -- silently, since no sources reads as
                #     "no declaration ships kernels". Every plain spelling in a
                #     release arch list hit this.
                #   * an ON-DEVICE arch adopts the spelling the declaration claims
                #     for the detected capability, as the generator's tie-break
                #     does. It passes no --archs, so it cannot desynchronize, and
                #     it is the path that needed resolving: the device reports the
                #     plain spelling, so a declaration pinning ('sm_100a',) got a
                #     tree it disowned.
                if arch is not None:
                    # Validated even though the comparison below is by string:
                    # cc_of is the only thing that rejects a malformed sm string,
                    # and without it `--arch sm100a` matched no declaration and
                    # exported nothing at exit 0 -- a typo that looked like success.
                    decl.cc_of(layout_arch)
                    claims = decl.archs_of(d)
                    if layout_arch not in claims:
                        # Claiming this CAPABILITY under another spelling is always a
                        # mistake, so it is reported HERE, per arch. The misses
                        # collected below are suppressed once the declaration ships
                        # for any arch, which hid exactly this case: the matched
                        # arches embed and pass the post-relink check while every
                        # device of the missed capability falls back to aten, with
                        # nothing in the build log.
                        other = _claimed_spelling(layout_arch, claims)
                        if other:
                            print(
                                f"{did}: declares kernels but none for this build -- "
                                f"requested {layout_arch}, and the declaration's "
                                f"ARCHS ({' '.join(claims)}) names that capability "
                                f"only as {other}, so this op falls back to aten on "
                                f"{layout_arch} hardware. The spellings must match "
                                f"exactly."
                            )
                        else:
                            skipped.setdefault(did, []).append(layout_arch)
                            declared[did] = claims
                        continue
                else:
                    claimed = _claimed_spelling(layout_arch, decl.archs_of(d))
                    if claimed is None:
                        # The explicit path's miss, on the automatic path: without
                        # this an on-device run that ships nothing for the local
                        # device says only `exported 0 kernels`, naming no op.
                        skipped.setdefault(did, []).append(layout_arch)
                        declared[did] = decl.archs_of(d)
                        continue
                    layout_arch = claimed
                out_dir = os.path.join(out_root, layout_arch, did)
                os.makedirs(out_dir, exist_ok=True)
                points = expand_specs(d.kernel_precompile_grid())
                _check_no_orphan_artifacts(out_dir, points)
                for point in points:
                    # The arch is named for the COMPILE too, so artifacts are built
                    # for what the sidecar records, not what the toolchain picks.
                    jobs.append((entry, d.KERNEL_MODULE, point, out_dir, layout_arch))
    shipped = {os.path.basename(j[3]) for j in jobs}
    for did, missed in sorted(skipped.items()):
        if did not in shipped:
            # The declaration's OWN ARCHS, not an illustration: a fixed example
            # contradicted the case at hand (it read "an ARCHS of ('sm_100a',)" for a
            # declaration claiming sm_100), which sends the reader the wrong way.
            print(
                f"{did}: declares kernels but none for this build -- requested "
                f"{' '.join(missed)}, and the declaration's ARCHS "
                f"({' '.join(declared[did])}) names none of them, so this op falls back "
                f"to aten. The spellings must match exactly."
            )
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


def _check_no_orphan_artifacts(out_dir: str, specs) -> None:
    """Report or refuse kernel artifacts no current grid point claims.

    Two ways that happens, and they differ in how recoverable they are.

    Artifacts NO SIDECAR CLAIMS, beside points that did commit: an export that
    died between writing them and writing its sidecar (a Ctrl-C, an OOM-killed
    worker, a compile failure -- the DSL writes the .h before the .o, so even the
    last of those strands one). REPORTED, not fatal: nothing links an artifact no
    sidecar names, so the cost is disk, and a re-export of that point overwrites
    it. As a refusal it turned every transient failure in a 48-point grid into a
    hand-delete of the directory, which --force could not clear either -- this scan
    runs before that flag is read.

    Reported whether or not anything else in the directory committed: keying it on
    "did some point commit" left the FIRST export of a tree fatal, which is the
    shape every interrupted new arch has, and diagnosed it as a partial copy.

    Artifacts whose sidecar records a spec no longer in the grid: dropping a point
    from kernel_precompile_grid() generates no job for it and nothing prunes it.
    Fatal, because the sidecar makes it look exported.

    An EMPTY directory is fine (clean build, or a new spec point).

    ``specs`` is the expanded grid for this (declaration, arch).
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
    recorded as absent rather than omitted, so "compiled where the wheel was
    missing" differs from "compiled before this was recorded"."""
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
    mixes artifacts from two compilers while the build reports one. A sidecar
    predating this record counts as stale.

    Ignorance is not staleness: with none of the kind's distributions installed
    this returns True, which is the generation-only run on a machine without the
    DSL wheels, where re-exporting is impossible anyway."""
    tc = toolchains.get_toolchain(sidecar.get("kind", "cutedsl"))
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
    """Cheap skip check without compiling: skip only when the sidecar's spec, its
    arch and every file in its source closure all still match, so an edited kernel
    module re-exports without --force.

    Per-arch directories stop two arches colliding in one tree, so the arch
    comparison guards the cases where the RECORDED arch differs from what this run
    resolves to: artifacts predating arch identity, and a tree carried between
    machines. Both must re-export, since the recorded arch is what the runtime gate
    is built from. Compared through _effective_arch, so the recorded value and this
    run's are resolved the same way -- --arch, else the local device."""
    if force:
        return True
    _, _, point, out_dir, arch = job
    spec = _json_normal(point)
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".json"):
            continue
        sc = _read_sidecar(os.path.join(out_dir, fn))
        # SCHEMA FIRST: every field below is read by name. A version dropping or
        # renaming "kind" used to raise a bare KeyError naming no file and no
        # remedy, where the version field exists so the point re-exports.
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
    """TORCH_CUDA_ARCH_LIST -> the sm strings from it that are
    EXPORTABLE_ARCHES, order-preserving and deduplicated.

    "9.0a;10.0a" (or space-separated) -> ["sm_100a"]. A +PTX suffix is stripped;
    named entries ("Hopper") are not translated, and CI passes numeric lists.
    Dedup matters because "10.0;10.0+PTX" names one arch twice, which would read
    as multi-arch downstream.

    ONE arch per compute capability, preferring the arch-conditional spelling:
    "10.0;10.0a" is one piece of hardware, and exporting both builds two full sets
    of which generation uses one (_by_arch prefers the conditional, what the
    kernels were written against). The loser is more than wasted compile time --
    it used to ship inside libtorch_cuda with no launcher (54 objects / 3.5 MiB
    measured). CUDA 13.x manywheel lists reach here, so this is the common case."""
    out = []
    for entry in arch_list.replace(";", " ").split():
        major, _, minor = entry.removesuffix("+PTX").partition(".")
        suffix = "a" if minor.endswith("a") else ""
        minor = minor.removesuffix("a")
        # str.isdigit accepts a non-ASCII digit that int() then rejects, so an
        # entry like "\N{SUPERSCRIPT TWO}.0" raises rather than being skipped. Kept
        # as-is: no release arch list holds one, and \d would behave identically.
        if not (major.isdigit() and minor.isdigit()):
            continue  # named arch ("Hopper") or malformed: skip
        sm = f"sm_{int(major) * 10 + int(minor)}{suffix}"
        if sm in EXPORTABLE_ARCHES and sm not in out:
            out.append(sm)
    # Collapse per capability, keeping the conditional spelling wherever the
    # list named it. Order-preserving on the survivors.
    conditional = {a.removesuffix("a") for a in out if a.endswith("a")}
    return [a for a in out if a.endswith("a") or a not in conditional]


# Which TORCH_CUDA_ARCH_LIST entries are ELIGIBLE for AOT kernels on the
# automatic export path. A filter, never a build list: it cannot cause an
# export, only permit one, and an explicit --arch bypasses it. So a list
# with no eligible entry exports nothing and stage 2 skips, printing why.
#
# Distinct from a declaration's ARCHS (what the KERNELS support, sm_90+);
# this says what the standard build SHIPS. Both spellings of a capability are
# listed because they are distinct nvcc targets used by different builds for
# the same hardware -- "10.0a" (arch-conditional, needed by tcgen05/wgmma) in
# b200-native-aot.yml, plain "10.0" elsewhere and in the manywheel lists.
# Omitting either silently exports nothing there.
#
# WHICH capabilities, and what admitting one costs: every entry a build's arch
# list names is another full set of compiled kernels inside its libtorch_cuda
# (one duplicate arch measured 54 objects / 3.5 MiB), and past this filter the
# DSL runtimes stop being optional -- build_stage2.should_run reads this tuple to
# decide whether stage 2 runs at all, so a box whose GPU is listed here and whose
# wheels are missing fails the build instead of skipping. Hopper is listed
# because every release arch list names 9.0 and the default ARCHS already covers
# it; no CI job exercises AOT kernels there yet -- the one functional AOT job in
# this stack runs on B200 and arrives with the first declaration -- so Hopper's
# coverage is this suite plus that Blackwell job. sm_103
# stays absent because no arch list we see names 10.3, and adding it would only
# grow wheels.
EXPORTABLE_ARCHES = ("sm_90", "sm_90a", "sm_100", "sm_100a")


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
        help="target architecture(s), e.g. --arch sm_90a sm_100a. With an "
        "explicit arch, export never touches the CUDA driver and runs on "
        "GPU-less machines (CuTeDSL via --gpu-arch; Triton via an "
        "explicit GPUTarget). Default: detect from the local device.",
    )
    args = parser.parse_args(argv)
    if args.jobs is None:
        env_jobs = os.getenv("MAX_JOBS") or os.getenv("CMAKE_BUILD_PARALLEL_LEVEL")
        # Half the CPU count, not all of it: os.cpu_count() reports SMT
        # siblings, and one compile per virtual thread oversubscribes.
        args.jobs = int(env_jobs) if env_jobs else max(1, (os.cpu_count() or 2) // 2)
    if args.arch is None and os.getenv("TORCH_CUDA_ARCH_LIST"):
        # Standard-build integration: export for the exportable subset of what
        # the main build compiled for. Explicit --arch wins.
        args.arch = archs_from_cuda_arch_list(os.environ["TORCH_CUDA_ARCH_LIST"])
        if not args.arch:
            print(
                "TORCH_CUDA_ARCH_LIST contains no AOT-exportable arch "
                f"(exportable: {' '.join(EXPORTABLE_ARCHES)}); nothing to export"
            )
            return
        print(f"arch from TORCH_CUDA_ARCH_LIST: {' '.join(args.arch)}")
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
