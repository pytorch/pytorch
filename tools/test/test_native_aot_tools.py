from __future__ import annotations

import contextlib
import glob
import importlib
import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
import unittest.mock as mock

# Ordinary package imports, like every other tools test (CI runs
# `PYTHONPATH=$(pwd) pytest tools/test`). These modules keep their module
# scope torch-free precisely so this works: the Test tools job runs in the
# linter image, which has no built torch.
from tools.native_aot import export, gen_aot_lib, toolchains

from torchgen import native_aot_decl


_TOOLS_FILE = os.path.abspath(toolchains.__file__)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_TOOLS_FILE)))


# The DSL versions this environment would compile with; sidecars must carry
# them or the skip check treats them as built by a different compiler.
_RUNTIMES = export.runtime_versions("cutedsl")

SIDECAR = {
    "prefix": "fakeop_f32_n1024_k8",
    # Every sidecar records the arch it was compiled for and the kind that
    # built it; generation reads both rather than defaulting either, so a
    # fixture missing them is not a sidecar export could have written.
    "arch": "sm_100a",
    "kind": "cutedsl",
    "spec": {"dtype": "float32", "N": 1024, "K": 8, "deterministic": False},
    "tensor_args": [
        {"name": "mX", "dynamic_sizes": [0], "dynamic_strides": [0]},
        {"name": "mOut", "dynamic_sizes": [0, 1], "dynamic_strides": [0]},
    ],
}


def _no_ambient_arch(device=None):
    """Take the ambient environment out of arch resolution.

    ``device`` is what _detected_arch should report -- None for "no GPU", or an
    sm string to test the device fallback, which only answers once no arch env
    var does.

    _effective_arch resolves an unspecified arch from the builder's GPU or from
    a toolchain's ARCH_ENV_VAR, so a sidecar written without one is legitimately
    stale when either answers. Tests about spec/source matching say nothing
    about arch and must not depend on the runner's hardware -- nor on the
    ambient environment: with CUTE_DSL_ARCH exported (the invocation export.py's
    own docstring documents) patching only the device left six of them failing."""
    env = {
        tc.ARCH_ENV_VAR: "" for tc in toolchains.TOOLCHAINS.values() if tc.ARCH_ENV_VAR
    }
    stack = contextlib.ExitStack()
    stack.enter_context(
        mock.patch.object(export, "_detected_arch", return_value=device)
    )
    # patch.dict cannot unset, and os.getenv("") is falsy, which is what
    # _effective_arch tests -- so an empty value is equivalent to absent here.
    stack.enter_context(mock.patch.dict(os.environ, env, clear=False))
    return stack


def _manifest(artifacts_dir):
    """What the generator told CMake, parsed back: {"arch_list": str|None,
    "sources": [...], "objects": [...]}.

    The generator emits native_aot.cmake -- the CMake CALLS, not data -- so this
    reads the quoted paths out of its target_sources() and the arch list out of its
    header comment. Tests assert the same invariants as when it was a manifest
    file; only the artifact changed."""
    out = {"arch_list": None, "sources": [], "objects": []}
    path = os.path.join(artifacts_dir, gen_aot_lib.CMAKE_INCLUDE)
    with open(path) as f:
        text = f.read()
    if "TORCH_CUDA_ARCH_LIST=" in text:
        m = re.search(r"TORCH_CUDA_ARCH_LIST='([^']*)'", text)
        if m is None:
            raise AssertionError(f"{path}: arch list present but unquoted:\n{text}")
        out["arch_list"] = m.group(1)
    elif "without an arch list" in text:
        # The generator makes no claim -- a hand run or an on-device export. A
        # DISTINCT value from None, which means "the file said nothing at all".
        out["arch_list"] = "absent"
    block = text.partition("target_sources(torch_cuda PRIVATE")[2].partition(")")[0]
    for line in block.splitlines():
        line = line.strip()
        if not (line.startswith('"') and line.endswith('"')):
            continue
        raw = line[1:-1]
        for char in (";", "$", '"', "\\"):
            raw = raw.replace("\\" + char, char)
        (out["objects"] if raw.endswith(".o") else out["sources"]).append(raw)
    return out


_DECL_REL = os.path.relpath(
    os.path.join(os.path.dirname(__file__), "..", "native_aot", "decl.py"), export.REPO
)


def _current_sources():
    """A source closure that matches this tree, for sidecars meant to look
    current."""
    return {_DECL_REL: export._file_hash(os.path.join(export.REPO, _DECL_REL))}


def _write_sidecar(d, point, prefix="x", exts=(".o", ".h"), **over):
    """Write into ``d`` the artifacts and the sidecar a current export would have
    left for ``point``, so _job_needed sees a skippable job. ``over`` replaces or
    adds fields (arch, sources, ...) for the cases that are about one of them."""
    _touch_artifacts(d, prefix, exts=exts)
    sc = {
        "version": export.SIDECAR_VERSION,
        "prefix": prefix,
        "kind": "cutedsl",
        "spec": point,
        "sources": _current_sources(),
        "runtimes": _RUNTIMES,
    }
    sc.update(over)
    with open(os.path.join(d, prefix + ".json"), "w") as f:
        json.dump(sc, f)


def _write_fake_decl(ops_dir, archs_line="", grid="[{'N': 1}]"):
    """One declaration under ops_dir/fakeop/, the minimum _collect_jobs loads:
    one grid point and the two C++ hooks, whose text nothing here reads."""
    os.makedirs(os.path.join(ops_dir, "fakeop"), exist_ok=True)
    with open(os.path.join(ops_dir, "fakeop", "aot.py"), "w") as f:
        f.write(
            'ATEN_OP = "fakeop"\nDISPATCH_KEY = "CUDA"\nKERNEL_MODULE = "k.py"\n'
            + archs_line
            + f"def kernel_precompile_grid():\n    return {grid}\n"
            "def covered_axes(self):\n    return {}\n"
            "def cpp_dispatch(spec):\n    return 'true'\n"
            "def cpp_launch(spec, launch_fn):\n    return launch_fn\n"
        )


def _touch_artifacts(out_dir, prefix, exts=(".o", ".h"), tensor_args=None):
    """Create the files a sidecar claims. _job_needed verifies they exist, so a
    fixture that writes only the .json would always re-export.

    The .h is DERIVED from the sidecar's own tensor_args -- one struct per tensor,
    named <prefix>_Tensor_<name>_t, with each array bound equal to the number of
    dims that tensor claims. validate_abi requires that equality (an over-claim
    stores past the end of the struct; an under-claim leaves the kernel reading an
    uninitialized slot), so a header with hand-picked bounds described an ABI no
    export could produce, and every generation fixture built on it was exercising
    the guard against an impossible input."""
    args = SIDECAR["tensor_args"] if tensor_args is None else tensor_args
    for e in exts:
        body = ""
        if e == ".h":
            body = "#pragma once\n#include <stdint.h>\n"
            for a in args:
                fields = ["  void* data;"]
                if a.get("dynamic_sizes"):
                    fields.append(
                        f"  int32_t dynamic_shapes[{len(a['dynamic_sizes'])}];"
                    )
                if a.get("dynamic_strides"):
                    fields.append(
                        f"  int64_t dynamic_strides[{len(a['dynamic_strides'])}];"
                    )
                body += (
                    "typedef struct {\n"
                    + "\n".join(fields)
                    + f"\n}} {prefix}_Tensor_{a['name']}_t;\n"
                )
        with open(os.path.join(out_dir, prefix + e), "w") as f:
            f.write(body)


class TestExportJobs(unittest.TestCase):
    def test_job_skip_matches_on_spec(self):
        # Skip detection matches the sidecar's recorded spec AND a
        # current source closure; a spec match alone (no/mismatched
        # sources) re-exports.
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32", "N": 4096}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            self.assertTrue(export._job_needed(job, force=False))
            _write_sidecar(d, point)
            with _no_ambient_arch():
                self.assertFalse(export._job_needed(job, force=False))
                self.assertTrue(export._job_needed(job, force=True))
                other = ("fakeop", "aot_kernel.py", {"dtype": "bfloat16"}, d, None)
                self.assertTrue(export._job_needed(other, force=False))

    def test_job_skip_survives_json_round_trip(self):
        # Tuple-valued grid fields read back from the sidecar as lists;
        # skip detection must normalize both sides or such points
        # re-export on every run (the pointwise family's in_dtypes hit
        # this).
        with tempfile.TemporaryDirectory() as d:
            point = {"aten": "add.Tensor", "in_dtypes": ("float32", "bfloat16")}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point, spec=export._json_normal(point))
            with _no_ambient_arch():
                self.assertFalse(export._job_needed(job, force=False))

    def test_run_job_is_module_level(self):
        # The pool pickles the job function by qualified name; a closure
        # or nested function would break only at --jobs > 1.
        # (A real pickle round-trip needs export importable by module
        # name, which this by-path test harness can't provide.)
        self.assertEqual(export._run_job.__qualname__, "_run_job")
        self.assertEqual(export.export_point.__qualname__, "export_point")

    def test_pool_never_forks_after_cuda_init(self):
        # Plain "fork" gives workers a dead CUDA context (measured:
        # is_initialized() False, allocation fails, no exception) because
        # the parent initializes CUDA before the pool starts.
        # Pinned to forkserver, not just "not fork": main() calls
        # set_forkserver_preload unconditionally, which a spawn context
        # would silently ignore.
        self.assertEqual(export.POOL_START_METHOD, "forkserver")

    def test_cutedsl_export_passes_gpu_arch(self):
        # The arch must reach the compiler as a --gpu-arch OPTION, not as
        # process state: that is what lets one worker serve several
        # arches. Appended to any builder-supplied options, never
        # replacing them. Stubs cute.compile/export_to_c because this
        # suite runs in the linter image, which has no DSL installed.
        import sys
        import types
        from typing import Any, cast

        seen = {}

        def fake_compile(fn, *args, **kwargs):
            seen["options"] = kwargs.get("options")
            return types.SimpleNamespace(export_to_c=lambda **kw: None)

        fake_cute = types.ModuleType("cutlass.cute")
        # cast: a ModuleType instance has no declared attributes, so plain
        # assignment fails type checking (idiom from test/test_utils.py).
        cast(Any, fake_cute).compile = fake_compile
        fake_cutlass = types.ModuleType("cutlass")
        cast(Any, fake_cutlass).cute = fake_cute
        tc = toolchains.CuteDslToolchain()
        with (
            mock.patch.dict(
                sys.modules, {"cutlass": fake_cutlass, "cutlass.cute": fake_cute}
            ),
            mock.patch.object(toolchains.CuteDslToolchain, "_warm_up_exporter"),
        ):
            for opts, want in (
                (None, "--gpu-arch sm_90a"),
                ("--enable-assertions", "--enable-assertions --gpu-arch sm_90a"),
            ):
                b = {"prefix": "p", "fn": None, "fake_args": (), "tensor_args": []}
                if opts:
                    b["options"] = opts
                tc.export(b, "/tmp", arch="sm_90a")
                self.assertEqual(seen["options"], want)
            # No arch: builder options pass through untouched (the
            # detect-from-device path must not inject --gpu-arch).
            tc.export(
                {"prefix": "p", "fn": None, "fake_args": (), "tensor_args": []}, "/tmp"
            )
            self.assertIsNone(seen["options"])

    def test_toolchains_declare_their_backend(self):
        # Which build backends a kind can emit for is DATA, not a platform
        # check buried in the gate: that is what lets a ROCm build skip
        # cleanly today and what makes a future ROCm DSL a new class with
        # BACKENDS = ("rocm",) rather than an edit to build_stage2.
        for kind, tc in toolchains.TOOLCHAINS.items():
            self.assertTrue(tc.BACKENDS, f"{kind} declares no BACKENDS")
        self.assertEqual(sorted(toolchains.for_backend("rocm")), [])
        self.assertEqual(
            sorted(toolchains.for_backend("cuda")),
            sorted(toolchains.TOOLCHAINS),
        )

    def test_missing_runtime_is_fatal_not_skipped(self):
        # A declaration whose toolchain targets this backend was ASKED for, so a
        # missing runtime must fail rather than ship a wheel with fewer kernels
        # than declared (TORCH_NATIVE_AOT=0 is the way to build without them).
        #
        # export is STUBBED on the runtime-present half: the real one imports
        # cutlass into the test process, leaks CUTE_DSL_LIBS into os.environ, and
        # with CUTE_DSL_ARCH set made the asserted exception come from arch
        # resolution rather than the gate.
        reached = []
        with (
            mock.patch.object(
                toolchains.CuteDslToolchain,
                "missing_runtimes",
                classmethod(lambda cls: []),
            ),
            mock.patch.object(
                toolchains.CuteDslToolchain,
                "export",
                lambda self, b, out_dir, arch=None: reached.append(arch) or {},
            ),
            _no_ambient_arch(device="sm_100a"),
        ):
            b = {"prefix": "p", "fn": None, "fake_args": (), "tensor_args": []}
            with mock.patch.object(export, "load_builder", lambda *a: lambda p: b):
                with tempfile.TemporaryDirectory() as d:
                    export.export_point("fakeop", "aot_kernel.py", {}, d)
        # Runtime present: it got PAST the gate and into the toolchain's export.
        self.assertEqual(len(reached), 1)

        with mock.patch.object(
            toolchains.CuteDslToolchain,
            "missing_runtimes",
            classmethod(lambda cls: ["cutlass"]),
        ):
            b = {"prefix": "p", "fn": None, "fake_args": (), "tensor_args": []}
            with mock.patch.object(export, "load_builder", lambda *a: lambda p: b):
                with self.assertRaisesRegex(RuntimeError, "cannot export"):
                    export.export_point("fakeop", "aot_kernel.py", {}, "/tmp")

    def test_pool_preload_stays_fork_safe(self):
        # The forkserver's server process is the fork PARENT for every
        # worker, so only modules that are inert in a fork parent may be
        # preloaded. Importing torch initializes no CUDA state; cutlass
        # and triton would build DSL/driver state in the parent that every
        # worker then inherits, which is what this pins down.
        self.assertEqual(export.POOL_PRELOAD, ("torch",))

    def test_json_normal_matches_sidecar_round_trip(self):
        # _json_normal replaces a json.dumps/loads pair, so it must agree
        # with one exactly: any divergence makes a spec mismatch its own
        # sidecar and re-export forever.
        point = {
            "dtype": "float32",
            "in_dtypes": ("float32", "bfloat16"),
            "N": 4096,
            "tma": True,
        }
        self.assertEqual(export._json_normal(point), json.loads(json.dumps(point)))
        self.assertEqual(
            export._json_normal(point)["in_dtypes"], ["float32", "bfloat16"]
        )


class TestArch(unittest.TestCase):
    def test_effective_arch_answers_identically_for_every_kind(self):
        # The invariant the refusal above buys: with no --arch, resolution is
        # device detection, which does not vary by kind -- so a tree can never
        # disagree with its own sidecars. This is what a per-kind env var broke.
        cutedsl = toolchains.get_toolchain("cutedsl")
        no_env = toolchains.Toolchain()
        self.assertIsNone(no_env.ARCH_ENV_VAR)
        with _no_ambient_arch(device="sm_100"):
            self.assertEqual(export._effective_arch(None, cutedsl), "sm_100")
            self.assertEqual(export._effective_arch(None, no_env), "sm_100")
            self.assertEqual(export._effective_arch(None), "sm_100")
        # An explicit arch wins for every kind alike.
        with mock.patch.dict(os.environ, {"CUTE_DSL_ARCH": "sm_90a"}):
            self.assertEqual(export._effective_arch("sm_100a", cutedsl), "sm_100a")
            self.assertEqual(export._effective_arch("sm_100a", no_env), "sm_100a")

    def test_arch_tag_is_short(self):
        # The tag lands in every exported C symbol, so its shape is part of
        # the artifact ABI: one underscore dropped, nothing else.
        self.assertEqual(export._arch_tag("sm_100a"), "sm100a")
        self.assertEqual(export._arch_tag("sm_90"), "sm90")

    def test_an_arch_env_var_without_explicit_arch_is_refused(self):
        # An arch variable is PER KIND, so it answers only for kinds that declare
        # one: with CUTE_DSL_ARCH=sm_90a set, a tree named sm_90a held a sidecar
        # recording the DETECTED sm_100 for a kind with no variable -- and
        # generation filters by directory while the gate comes from the sidecar.
        # Refusing leaves one arch=None path that every kind answers identically.
        class _Other(toolchains.Toolchain):
            kind = "other"
            ARCH_ENV_VAR = "OTHER_DSL_ARCH"

        registry = dict(toolchains.TOOLCHAINS, other=_Other())
        with (
            mock.patch.dict(toolchains.TOOLCHAINS, registry, clear=True),
            mock.patch.dict(
                os.environ, {"CUTE_DSL_ARCH": "sm_90a", "OTHER_DSL_ARCH": "sm_100a"}
            ),
            mock.patch.object(export, "_detected_arch", return_value=None),
        ):
            # Named in the message so the user knows WHICH variable to unset, and
            # both are named when both are set.
            with self.assertRaisesRegex(RuntimeError, "CUTE_DSL_ARCH=sm_90a"):
                export._effective_arch(None)
            with self.assertRaisesRegex(RuntimeError, "OTHER_DSL_ARCH=sm_100a"):
                export._effective_arch(None)
            # Agreeing values are refused too: the point is not a conflict
            # between them, it is that neither can speak for the other kinds.
            with mock.patch.dict(os.environ, {"OTHER_DSL_ARCH": "sm_90a"}):
                with self.assertRaisesRegex(RuntimeError, "--arch"):
                    export._effective_arch(None)
            # ...and through the overload the production callers use: export_point
            # and _job_needed both pass a toolchain, and honouring the variable for
            # THOSE while refusing it here reinstates the whole bug -- which stayed
            # green, because every assertion above omits the toolchain.
            for tc in (toolchains.get_toolchain("cutedsl"), _Other()):
                with self.subTest(kind=tc.kind):
                    with self.assertRaisesRegex(RuntimeError, "CUTE_DSL_ARCH=sm_90a"):
                        export._effective_arch(None, tc)
            # An explicit --arch is the way to say it, for every kind at once.
            self.assertEqual(export._effective_arch("sm_100a"), "sm_100a")

    def test_export_prefix_is_arch_qualified(self):
        # Two arches must not produce the same prefix: the exported symbols
        # (cute_dsl_<prefix>_wrapper, <prefix>_Kernel_Module_Load) are derived
        # from it, so an unqualified prefix is a duplicate definition when both
        # arches link into one libtorch_cuda.
        seen = []

        class _FakeTc(toolchains.Toolchain):
            kind = "cutedsl"
            artifact_exts = (".o", ".h")
            # So the sidecar's recorded compiler versions are non-empty below.
            RUNTIME_DISTS = ("nvidia-cutlass-dsl",)

            def missing_runtimes(self):
                return []

            def validate_build_result(self, b):
                pass

            def export(self, b, out_dir, arch=None):
                # The PAIR, not the prefix alone: the prefix NAMES an arch and this
                # is the arch the kernel is really compiled for, so recording only
                # the prefix let "compile for no arch while the prefix claims one"
                # pass, and the sidecar then labels a kernel it did not describe.
                seen.append((b["prefix"], arch))
                _touch_artifacts(out_dir, b["prefix"])
                return {"tensor_args": []}

        fake = _FakeTc()
        with tempfile.TemporaryDirectory() as d:
            with (
                mock.patch.object(
                    export,
                    "load_builder",
                    return_value=lambda p: {"prefix": "k", "kind": "cutedsl"},
                ),
                mock.patch.object(toolchains, "get_toolchain", return_value=fake),
            ):
                for arch in ("sm_90a", "sm_100a"):
                    export.export_point("fakeop", "aot_kernel.py", {"n": 1}, d, arch)
                want = [("k__sm90a", "sm_90a"), ("k__sm100a", "sm_100a")]
                self.assertEqual(seen, want)
                # The sidecar WRITER, against what the readers require. It had
                # no test: deleting "version" and "arch" from what export_point
                # writes left the suite green, since every reader passes against
                # hand-written fixtures. Asserted inside the patch, so the same
                # toolchain answers here as when the file was written.
                with open(os.path.join(d, "k__sm100a.json")) as f:
                    written = json.load(f)
                self.assertEqual(written["version"], export.SIDECAR_VERSION)
                self.assertEqual(written["arch"], "sm_100a")
                self.assertEqual(written["kind"], "cutedsl")
                self.assertEqual(written["prefix"], "k__sm100a")
                self.assertEqual(written["spec"], {"n": 1})
                self.assertTrue(written["sources"], "no source closure recorded")
                self.assertEqual(
                    written["runtimes"], export.runtime_versions("cutedsl")
                )
                # ...and what it wrote is what the readers accept.
                self.assertTrue(export.sources_current(written))
                self.assertTrue(export.runtimes_current(written))

                # The ON-DEVICE path, where the RESOLVED arch is the only one that
                # can reach the prefix and the sidecar. Recording the raw argument
                # left "arch": null -- which generation refuses -- behind an
                # unqualified prefix, and nothing noticed: every case above passes
                # --arch, where the raw and resolved values are the same string.
                with _no_ambient_arch(device="sm_100"):
                    export.export_point("fakeop", "aot_kernel.py", {"n": 2}, d, None)
                with open(os.path.join(d, "k__sm100.json")) as f:
                    on_device = json.load(f)
                self.assertEqual(on_device["prefix"], "k__sm100")
                self.assertEqual(on_device["arch"], "sm_100")
                # The compile is still told the RAW arch (None = let the DSL take
                # the local device, which is what was just detected); pinned so
                # passing something else has to be a deliberate change here.
                self.assertEqual(seen[-1], ("k__sm100", None))

    def test_job_skip_compares_the_arch(self):
        # Two exports into ONE --out-dir differing only in arch: comparing spec
        # alone would skip the second and leave the first arch's objects behind a
        # sidecar the caller reads as the second arch.
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32", "N": 4096}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point, arch="sm_90a")
            # _collect_jobs resolves the arch once and puts it in the job, so the
            # comparison is job-vs-sidecar and needs no ambient variable.
            with _no_ambient_arch():
                same = ("fakeop", "aot_kernel.py", point, d, "sm_90a")
                other = ("fakeop", "aot_kernel.py", point, d, "sm_100a")
                self.assertFalse(export._job_needed(same, force=False))
                self.assertTrue(export._job_needed(other, force=False))
                # An on-device job (arch resolved from the device) is not the
                # sm_90a one either.
                with mock.patch.object(export, "_detected_arch", return_value="sm_100"):
                    self.assertTrue(export._job_needed(job, force=False))

    def test_job_skip_re_exports_when_the_compiler_changed(self):
        # The compiler appears in no file the source closure hashes, so without
        # this an upgraded DSL wheel invalidates nothing and one tree mixes
        # artifacts from two compilers while the build reports one. Every other
        # skip fixture records the CURRENT versions, so dropping runtimes_current
        # from the skip check entirely left the suite green.
        #
        # runtime_versions is patched rather than read: on a machine with no DSL
        # wheels every version is "absent", which is deliberately NOT staleness,
        # and this assertion would then invert. That is the linter image, where
        # this suite runs.
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32", "N": 4096}
            job = ("fakeop", "aot_kernel.py", point, d, "sm_100a")
            with (
                _no_ambient_arch(),
                mock.patch.object(
                    export, "runtime_versions", return_value={"a-dsl": "9.9.9"}
                ),
            ):
                _write_sidecar(d, point, arch="sm_100a", runtimes={"a-dsl": "9.9.9"})
                self.assertFalse(export._job_needed(job, force=False))
                _write_sidecar(d, point, arch="sm_100a", runtimes={"a-dsl": "0.0.1"})
                self.assertTrue(export._job_needed(job, force=False))

    def test_job_skip_rejects_arch_less_sidecar_when_env_set(self):
        # The reported case, from the other side: a sidecar that recorded no
        # arch at all (an on-device export) must NOT satisfy a run whose
        # CUTE_DSL_ARCH names a target, or the env-var run silently inherits
        # objects built for whatever the builder's GPU was.
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32", "N": 4096}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point, arch=None)
            with _no_ambient_arch():
                named = ("fakeop", "aot_kernel.py", point, d, "sm_100a")
                self.assertTrue(export._job_needed(named, force=False))
                # ...and where an arch IS resolvable it does not satisfy an
                # on-device run either: that run knows its arch, the sidecar names
                # none. _detected_arch patched rather than trusted, so the claim
                # does not depend on the runner having a GPU.
                with mock.patch.object(export, "_detected_arch", return_value="sm_100"):
                    self.assertTrue(export._job_needed(job, force=False))
                # Only where no arch can be resolved at all is it a match.
                self.assertFalse(export._job_needed(job, force=False))

    def test_cc_of_reads_the_capability_both_spellings_name(self):
        # The exporter matches a declaration's ARCHS by STRING while the generator
        # groups sidecars by capability, so the two spellings of one piece of
        # hardware have to parse EQUAL -- a declaration pinning ('sm_100a',)
        # disowned the 'sm_100' its own on-device export produced.
        self.assertEqual(native_aot_decl.cc_of("sm_90"), (9, 0))
        self.assertEqual(native_aot_decl.cc_of("sm_103a"), (10, 3))
        self.assertEqual(
            native_aot_decl.cc_of("sm_100a"), native_aot_decl.cc_of("sm_100")
        )

    def test_cc_of_refuses_what_it_cannot_read(self):
        # Each would otherwise compute a plausible capability and emit a gate no
        # device satisfies ("sm_9" -> (0, 9), "sm_1000" -> (100, 0)), so the op
        # ships, links and declines every call unreported.
        #
        # assertRaisesRegex, not assertRaises: cc_of raises two different
        # RuntimeErrors, so a bare check passed with the digit-length guard
        # removed -- "sm_9" then tripped the RANGE error instead.
        #
        # The last four are what str.isdigit() let through: full-width digits,
        # Arabic-Indic digits and a leading zero each parsed as capability 9.0,
        # and the superscript reached int(), whose ValueError the loader's
        # `except RuntimeError` could not wrap.
        for bad in (
            "sm_9",
            "sm_1000",
            "sm_100f",
            "sm_",
            "",
            "100a",
            "sm_10a0",
            "sm_\uff19\uff10",
            "sm_\u0669\u0660",
            "sm_090",
            "sm_\u00b2\u00b2",
        ):
            with self.subTest(arch=bad):
                with self.assertRaisesRegex(
                    RuntimeError, "cannot read a compute capability"
                ):
                    native_aot_decl.cc_of(bad)

    def test_cc_of_refuses_a_capability_outside_the_known_range(self):
        # Parses fine, but no such hardware: a gate for it is dead code.
        with self.assertRaisesRegex(RuntimeError, "outside the known range"):
            native_aot_decl.cc_of("sm_130")

    def test_the_detected_arch_is_the_local_capability(self):
        # An on-device export records this in the sidecar; without it the generated
        # gate falls back to the declaration's ARCHS and advertises hardware nothing
        # was compiled for. Every other test patches this function out, so it is the
        # only place the mapping itself runs -- and it drops the "a" suffix on
        # purpose, since the gate compares major.minor, which both spellings share.
        fake = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True, get_device_capability=lambda: (10, 3)
            )
        )
        with mock.patch.dict(sys.modules, {"torch": fake}):
            self.assertEqual(export._detected_arch(), "sm_103")
        # No CUDA is "no local arch", not a crash in the middle of a build.
        fake.cuda.is_available = lambda: False
        with mock.patch.dict(sys.modules, {"torch": fake}):
            self.assertIsNone(export._detected_arch())

    def test_archs_from_cuda_arch_list(self):
        # TORCH_CUDA_ARCH_LIST -> the EXPORTABLE_ARCHES subset; named,
        # malformed, +PTX and non-exportable entries drop out.
        f = export.archs_from_cuda_arch_list
        self.assertEqual(f("7.5 8.9"), [])
        # Hopper and Blackwell together: one tree per arch, selected at
        # runtime by capability.
        self.assertEqual(f("9.0a;10.0a"), ["sm_90a", "sm_100a"])
        self.assertEqual(f("8.0 9.0 10.0+PTX"), ["sm_90", "sm_100"])
        # Both spellings of a CC are separate nvcc targets, and both are
        # exportable: CI passes "10.0a", the wheel builds pass "10.0".
        self.assertEqual(f("10.0"), ["sm_100"])
        # 10.3 stays unexportable -- nothing names it, so shipping it would
        # only grow wheels (the gate itself is major.minor and would be safe).
        self.assertEqual(f("Hopper 10.3a"), [])

    def test_archs_from_cuda_arch_list_dedups(self):
        # "10.0;10.0+PTX" names one arch twice; a repeated entry would read as
        # multi-arch downstream and export a second full set of kernels.
        f = export.archs_from_cuda_arch_list
        self.assertEqual(f("10.0;10.0+PTX"), ["sm_100"])
        self.assertEqual(f("10.0a 10.0a"), ["sm_100a"])

    def test_archs_from_cuda_arch_list_collapses_one_capability(self):
        # Both spellings of a capability are the same hardware, and generation
        # can only use one of them (_by_arch prefers the arch-conditional
        # build). Exporting both compiles a second full set of kernels whose
        # objects then ship inside libtorch_cuda with no launcher referencing
        # them. CUDA 13.x manywheel lists reach this, so it is the common case.
        f = export.archs_from_cuda_arch_list
        self.assertEqual(f("10.0;10.0a"), ["sm_100a"])
        self.assertEqual(f("10.0a;10.0"), ["sm_100a"])
        # Different capabilities are untouched, and order is preserved.
        self.assertEqual(f("9.0a;10.0;10.0a"), ["sm_90a", "sm_100a"])
        self.assertEqual(f("10.0a;9.0"), ["sm_100a", "sm_90"])

    def test_collect_jobs_respects_declaration_archs(self):
        # A declaration pinning ARCHS gets no jobs for other arches; an
        # on-device export (arch None) is never filtered.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops, 'ARCHS = ("sm_100a",)\n')
            # _detected_arch patched: the on-device call below resolves the
            # directory from it, and an unpatched run would pass only on a
            # machine with a GPU -- this suite must also pass in the linter
            # image, which has no built torch at all.
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device="sm_100a"),
            ):
                blackwell = export._collect_jobs(None, out, ["sm_100a"])
                hopper = export._collect_jobs(None, out, ["sm_90a"])
                on_device = export._collect_jobs(None, out, [None])
        self.assertEqual(len(blackwell), 1)
        self.assertEqual(len(hopper), 0)
        self.assertEqual(len(on_device), 1)

    def test_multi_arch_jobs_nest_per_arch(self):
        # Every job nests under <out>/<arch>/<decl_id>, one arch or several:
        # adding an arch to an op is then just another directory. There is no
        # second (flat) layout -- the assertions below pin that for the
        # single-arch and on-device cases too.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops)
            # _detected_arch patched, not read from the runner: the layout must
            # be the same shape everywhere, and an unpatched call would make
            # this test depend on whether the machine has a GPU.
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                # Device-only resolution: an ambient CUTE_DSL_ARCH would outrank
                # it and put the on-device job in a different arch directory.
                _no_ambient_arch(device="sm_100"),
            ):
                multi = export._collect_jobs(None, out, ["sm_90a", "sm_100a"])
                single = export._collect_jobs(None, out, [None])
        self.assertEqual(len(multi), 2)
        dirs = sorted(os.path.basename(os.path.dirname(j[3])) for j in multi)
        self.assertEqual(dirs, ["sm_100a", "sm_90a"])
        self.assertEqual({j[4] for j in multi}, {"sm_90a", "sm_100a"})
        # ONE layout: a single arch nests under its own directory too, so
        # adding an arch to an op is just another directory.
        (sj,) = single
        self.assertEqual(os.path.basename(sj[3]), "fakeop")
        # sm_100a, not the detected sm_100: an on-device export adopts the
        # spelling the DECLARATION claims, as the generator's tie-break does. So
        # the tree cannot be one the declaration disowns -- a declaration pinning
        # ("sm_100a",) got an sm_100 tree, and generation's "delete and re-export"
        # remedy rebuilt the identical one -- and the job carries that arch
        # explicitly, so kernels match what is recorded.
        self.assertEqual(os.path.basename(os.path.dirname(sj[3])), "sm_100a")
        self.assertEqual(sj[4], "sm_100a")

    def test_by_arch_groups_and_orders_by_capability(self):
        scs = [
            {"prefix": "k__sm100a", "arch": "sm_100a"},
            {"prefix": "k__sm90a", "arch": "sm_90a"},
            {"prefix": "k2__sm90a", "arch": "sm_90a"},
        ]
        groups = gen_aot_lib._by_arch(scs)
        self.assertEqual(list(groups), [(9, 0), (10, 0)])
        self.assertEqual(
            [s["prefix"] for s in groups[(9, 0)]], ["k__sm90a", "k2__sm90a"]
        )

    def test_by_arch_prefers_the_arch_conditional_build(self):
        # Both are valid on 10.0 hardware; the conditional build is the one the
        # kernels were written against, and shipping both would otherwise let
        # directory order decide.
        for order in (
            [("sm_100", "p"), ("sm_100a", "c")],
            [("sm_100a", "c"), ("sm_100", "p")],
        ):
            scs = [{"prefix": n, "arch": a} for a, n in order]
            groups = gen_aot_lib._by_arch(scs)
            self.assertEqual(list(groups), [(10, 0)])
            self.assertEqual([s["prefix"] for s in groups[(10, 0)]], ["c"])

    def test_by_arch_rejects_an_arch_less_sidecar(self):
        # Export names the arch of everything it writes, so this is a tree from
        # before it did. Rejected rather than grouped: a capability nothing
        # matches would emit a branch that declines every call in silence.
        with self.assertRaisesRegex(RuntimeError, "records no arch"):
            gen_aot_lib._by_arch([{"prefix": "old", "arch": None}])

    def test_dropped_tie_break_candidate_gets_no_launcher(self):
        # Both spellings of a capability are exportable ("10.0;10.0a"), so the
        # plain build loses the tie-break -- and a launcher emitted for it would
        # be defined and never called, i.e. -Wunused-function in an anonymous
        # namespace, fatal under CI's WERROR.
        class _Decl:
            ATEN_OP = "fakeop"
            DISPATCH_KEY = "CUDA"
            ARCHS = ("sm_100", "sm_100a")

            @staticmethod
            def cpp_dispatch(spec):
                return "true"

            @staticmethod
            def cpp_launch(spec, launch_fn):
                return f"{launch_fn}(self, out, at::cuda::getCurrentCUDAStream());"

        def sc(arch):
            return dict(
                SIDECAR, prefix=f"fakeop_p__{arch.replace('_', '', 1)}", arch=arch
            )

        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _Decl,
            [sc("sm_100"), sc("sm_100a")],
            "const at::Tensor & self, const at::Tensor & out",
        )
        self.assertIn("launch_fakeop_p__sm100a(", src)
        self.assertNotIn("launch_fakeop_p__sm100(", src)

    def test_cc_of_and_device_match(self):
        self.assertEqual(gen_aot_lib._cc_of("sm_90"), (9, 0))
        self.assertEqual(gen_aot_lib._cc_of("sm_103a"), (10, 3))
        m = gen_aot_lib._device_match(10, 3)
        self.assertIn("major == 10", m)
        self.assertIn("minor == 3", m)


class TestSidecarIntegrity(unittest.TestCase):
    """The sidecar is written after the artifacts, so it is the commit
    marker: absent means not-yet-exported, corrupt or orphaned means the
    tree cannot be trusted."""

    def test_empty_dir_is_fine(self):
        # A clean build (or a newly added spec point) has no sidecar and
        # no artifacts; it must export, not fail.
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                export._check_no_orphan_artifacts(d, [])
        self.assertEqual(out.getvalue(), "", "a clean directory says nothing")

    def test_artifacts_without_a_sidecar_are_reported_not_fatal(self):
        # An export that died between compiling and writing the sidecar. Generation
        # names artifacts from sidecars and there is no glob anywhere in the link
        # path, so an undescribed orphan is disk rather than payload; refusing it
        # only forced a hand-delete.
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "k_f32.o"), "w").close()
            with contextlib.redirect_stdout(io.StringIO()) as out:
                export._check_no_orphan_artifacts(d, [])
        self.assertIn("no sidecar claims", out.getvalue())
        self.assertIn("k_f32.o", out.getvalue())

    def test_an_orphan_beside_a_committed_point_is_reported_not_fatal(self):
        # Where an interrupt actually lands: among points that already committed.
        # Keyed per DIRECTORY the check saw a sidecar and asked nothing further, so
        # this pair survived every later export unreported. Keyed per artifact and
        # FATAL it went too far the other way: the DSL writes the .h before the .o, so
        # a single failed compile in a 48-point grid made every later export refuse the
        # directory -- including --force, which is read after this scan.
        with tempfile.TemporaryDirectory() as d:
            for name in ("k_n1.o", "k_n1.h", "k_n2.h"):
                open(os.path.join(d, name), "w").close()
            with open(os.path.join(d, "k_n1.json"), "w") as f:
                # version, so the stale half runs rather than short-circuiting.
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "k_n1",
                        "kind": "cutedsl",
                        "spec": {"N": 1},
                    },
                    f,
                )
            with contextlib.redirect_stdout(io.StringIO()) as out:
                export._check_no_orphan_artifacts(d, [{"N": 1}, {"N": 2}])
        self.assertIn("k_n2.h", out.getvalue())
        self.assertIn("no sidecar claims", out.getvalue())

    def test_a_fresh_trees_failed_first_export_is_not_called_corruption(self):
        # Nothing committed is what the FIRST export of an arch looks like when it
        # dies: the DSL writes the .h before the .o, so a failed compile strands one
        # per point and no sidecar exists yet. Diagnosing that as "a partial copy or
        # a hand-edited export" was both wrong and fatal, and --force could not
        # clear it -- this scan runs before that flag is read.
        with tempfile.TemporaryDirectory() as d:
            for name in ("k_n1.o", "k_n1.h"):
                open(os.path.join(d, name), "w").close()
            with contextlib.redirect_stdout(io.StringIO()) as out:
                export._check_no_orphan_artifacts(d, [{"N": 1}])
        self.assertIn("no sidecar claims", out.getvalue())
        self.assertNotIn("partial copy", out.getvalue())

    def test_artifacts_with_sidecar_are_fine(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "k.o"), "w").close()
            with open(os.path.join(d, "k.json"), "w") as f:
                # version, or the stale half short-circuits at the schema gate and
                # the spec comparison below never runs.
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "k",
                        "kind": "cutedsl",
                        "spec": {"N": 1},
                    },
                    f,
                )
            with contextlib.redirect_stdout(io.StringIO()) as out:
                export._check_no_orphan_artifacts(d, [{"N": 1}])
        # SILENT, not merely non-fatal: claiming per ARTIFACT is what stops a healthy
        # directory reporting its own kernels, and asserting "does not raise" left
        # that unpinned -- reporting every artifact as an orphan passed.
        self.assertEqual(out.getvalue(), "")

    def test_unreadable_sidecar_is_fatal(self):
        # Present but unparsable: corruption, not an interrupted run.
        # Re-exporting would paper over it (and --force skips the check).
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "k.json")
            with open(path, "w") as f:
                f.write("{truncated")
            with self.assertRaisesRegex(RuntimeError, "could not be read"):
                export._read_sidecar(path)

    def test_cross_product(self):
        pts = export.expand_specs(
            [{"dtype": ["float32", "bfloat16"], "N": [1024, 2048], "K": 8}]
        )
        self.assertEqual(len(pts), 4)
        self.assertIn({"dtype": "float32", "N": 2048, "K": 8}, pts)
        self.assertIn({"dtype": "bfloat16", "N": 1024, "K": 8}, pts)

    def test_multiple_blocks_concatenate(self):
        pts = export.expand_specs([{"N": [1, 2]}, {"N": 3, "extra": True}])
        self.assertEqual(pts, [{"N": 1}, {"N": 2}, {"N": 3, "extra": True}])

    def test_scalar_only_spec(self):
        self.assertEqual(
            export.expand_specs([{"N": 4096, "K": 64}]), [{"N": 4096, "K": 64}]
        )


class TestLauncherGeneration(unittest.TestCase):
    def test_marshalling_from_sidecar(self):
        src = gen_aot_lib.gen_launcher(SIDECAR)
        # Struct fill: one dynamic size per listed dim, in slot order.
        self.assertIn("mX_s.dynamic_shapes[0] = static_cast<int32_t>(mX.size(0));", src)
        self.assertIn("mX_s.dynamic_strides[0] = mX.stride(0);", src)
        self.assertIn(
            "mOut_s.dynamic_shapes[1] = static_cast<int32_t>(mOut.size(1));", src
        )
        # Lazy module load + wrapper call with all tensor structs.
        self.assertIn("c10::call_once(fakeop_f32_n1024_k8_loaded", src)
        self.assertIn(
            "cute_dsl_fakeop_f32_n1024_k8_wrapper(&fakeop_f32_n1024_k8_module, &mX_s, &mOut_s,",
            src,
        )


class TestScalarArgs(unittest.TestCase):
    def test_launcher_forwards_scalars_by_value(self):
        sidecar = dict(
            SIDECAR,
            tensor_args=[{"name": "mX", "dynamic_sizes": [0]}],
            scalar_args=[{"name": "alpha", "ctype": "float"}],
        )
        src = gen_aot_lib.gen_launcher(sidecar)
        # Scalars appear after tensors in both the params and the call,
        # matching the exported wrapper's argument order.
        self.assertIn("const at::Tensor& mX, float alpha, c10::Stream stream", src)
        self.assertIn("(&fakeop_f32_n1024_k8_module, &mX_s, alpha,", src)
        # The shared contract carries a device-agnostic c10::Stream; the
        # body narrows it to the raw handle the C ABI takes.
        self.assertIn("c10::cuda::CUDAStream(stream).stream()", src)

    def test_no_scalar_args_unchanged(self):
        src = gen_aot_lib.gen_launcher(SIDECAR)
        self.assertIn(
            "const at::Tensor& mX, const at::Tensor& mOut, c10::Stream stream", src
        )


class _FakeDecl:
    """Minimal declaration object for gen_op tests (the real contract
    lives in decl.py; gen_op only touches these attributes)."""

    ATEN_OP = "fakeop"
    DISPATCH_KEY = "CUDA"
    # Set explicitly: fixtures bypass the validating loader, which is
    # what normalizes ARCHS on real declarations. Annotated so
    # subclasses can narrow it without a bad-override.
    ARCHS: tuple[str, ...] = ("sm_90", "sm_90a", "sm_100", "sm_100a")

    @staticmethod
    def cpp_dispatch_prelude():
        return (
            "if (self.scalar_type() != at::kFloat) return false;\n"
            "const int64_t N = self.size(-1);"
        )

    @staticmethod
    def cpp_dispatch(spec):
        return f"N == {spec['N']} && k == {spec['K']}"

    @staticmethod
    def cpp_launch(spec, launch_fn):
        return f"{launch_fn}(self, out, at::cuda::getCurrentCUDAStream());"


@contextlib.contextmanager
def _patched_generation(ops, declarations=(_FakeDecl,)):
    """What every gen_aot_lib.main() test patches: the ops dir discovery walks,
    the declarations it loads (None to use the real loader on a written aot.py,
    or a callable to answer per path), and the two native_functions.yaml lookups
    -- "fakeop" is not a real op, so the real ones raise."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(gen_aot_lib, "OPS_DIR", ops))
        if declarations is not None:
            load = (
                declarations
                if callable(declarations)
                else lambda path: list(declarations)
            )
            stack.enter_context(
                mock.patch.object(gen_aot_lib.decl, "load_declarations", load)
            )
        stack.enter_context(
            mock.patch.object(
                gen_aot_lib,
                "impl_signature_params",
                lambda op: "const at::Tensor & self, int64_t k",
            )
        )
        stack.enter_context(
            mock.patch.object(gen_aot_lib, "precomputed_args", lambda op: [])
        )
        yield


class TestInt32SizeGate(unittest.TestCase):
    # The DSL's exported ABI carries int32_t shape slots while aten sizes
    # are int64_t, so the generated gate must DECLINE oversized dims
    # rather than let the launcher's static_cast truncate them.
    def test_gate_covers_plain_and_optional_tensors(self):
        gate = gen_aot_lib._int32_size_gate(
            "const at::Tensor & self, int64_t k, "
            "const std::optional<at::Tensor>& values"
        )
        # Plain tensor: unconditional scan.
        self.assertIn("self.sizes().begin()", gate)
        # Optional tensor: has_value() guarded, arrow deref.
        self.assertIn("values.has_value()", gate)
        self.assertIn("values->sizes().begin()", gate)
        # Declines, never truncates -- and does so as ONE guarded statement,
        # so an oversized dim in any tensor leaves via the same early return.
        self.assertEqual(gate.count("return false;"), 1)
        self.assertIn("if (C10_UNLIKELY(", gate)
        self.assertTrue(gate.rstrip().endswith(")) return false;"), gate)
        # Non-tensor params contribute nothing: a wrongly-included scalar
        # would emit its name in a sizes() probe.
        self.assertNotIn("k.sizes()", gate)

    def test_the_gate_expression_is_exact(self):
        # The WHOLE expression, because substrings cannot tell it from three silent
        # miscomputes that each left the suite green: the sense inverted (== for !=,
        # so it declines every normal call and accepts every oversized one), the
        # probes joined with && (declines only when EVERY tensor is oversized), and
        # only the first plain tensor gated. Two plain tensors, so the last shows.
        gate = gen_aot_lib._int32_size_gate(
            "const at::Tensor & self, int64_t k, const at::Tensor & out, "
            "const std::optional<at::Tensor>& weight"
        )
        want = (
            "  if (C10_UNLIKELY("
            "self.sizes().end() != std::find_if(self.sizes().begin(), "
            "self.sizes().end(), _naot_dim_too_big)"
            " || out.sizes().end() != std::find_if(out.sizes().begin(), "
            "out.sizes().end(), _naot_dim_too_big)"
            " || (weight.has_value() && weight->sizes().end() != "
            "std::find_if(weight->sizes().begin(), weight->sizes().end(), "
            "_naot_dim_too_big))"
            ")) return false;"
        )
        self.assertEqual(gate.splitlines()[-1], want)

    def test_gate_empty_without_tensors(self):
        self.assertEqual(gen_aot_lib._int32_size_gate("int64_t k, bool largest"), "")

    def test_gate_emitted_into_the_stub(self):
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k",
        )
        self.assertIn("inline bool _naot_dim_too_big", src)
        # The COMPARISON, not just the signature line: a helper that can never fire
        # (int64's max, say) leaves every oversized dim to truncate through the
        # launcher's static_cast, and asserting the signature alone missed it.
        self.assertIn("return d > std::numeric_limits<int32_t>::max();", src)
        # _FakeDecl declares no cpp_covers, so there is exactly one gate site.
        self.assertEqual(src.count("// Size gate:"), 1)
        self.assertIn("self.sizes().begin()", src)

    def test_covers_gets_the_same_gates_as_the_stub(self):
        # The stated invariant: coverage must be no wider than the stub's
        # acceptance. The stub declines on an unsupported device and on an
        # oversized dim, so covers must decline there too -- otherwise the
        # router hands those calls to a stub that refuses them and they lose
        # their JIT route. Nothing asserted this: the previous test used a
        # declaration with no cpp_covers and admitted as much in a comment.
        class _WithCovers(_FakeDecl):
            @staticmethod
            def cpp_covers():
                return "return true;"

        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _WithCovers,
            [sidecar],
            "const at::Tensor & self, int64_t k",
            covers=(
                "const at::Tensor & self, int64_t k",
                "covers_fakeop(...)-> bool",
                "return true;",
            ),
        )
        # Two of each: once in the kernel, once in the covers predicate.
        self.assertEqual(src.count("// Size gate:"), 2)
        self.assertEqual(src.count("// Device gate:"), 2)
        # ...and the covers copy reads the TENSOR's device, not the current one:
        # the router calls it before any device guard, so on a mixed-capability
        # host the current device need not be the one the op will run on.
        self.assertIn(
            "const auto* _naot_props = at::cuda::getDeviceProperties(self.device().index());",
            src,
        )
        self.assertIn(
            "const auto* _naot_props = at::cuda::getCurrentDeviceProperties();", src
        )


class TestAotSourceGeneration(unittest.TestCase):
    def test_full_source(self):
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k, const at::Tensor & out",
        )
        # Prelude verbatim (indented) ahead of the chain.
        self.assertIn("if (self.scalar_type() != at::kFloat) return false;", src)
        # Dispatch branch wraps the launch with the point's launcher name.
        self.assertIn("if (N == 1024 && k == 8) {", src)
        self.assertIn(
            "launch_fakeop_f32_n1024_k8(self, out, at::cuda::getCurrentCUDAStream());",
            src,
        )
        # Falls through to false: the LAST statement of the kernel body, which
        # is what routes an unmatched call to op.impl. Asserted positionally --
        # the device gate and the size gate each emit a "return false;" too, so
        # a substring check here passes even with no fallback at all.
        body = src.split("bool fakeop_cuda_aot_kernel(", 1)[1]
        body = body[: body.index("\n}\n")]
        self.assertTrue(body.rstrip().endswith("return false;"), body[-200:])
        # Registers on the generated DispatchStub.
        self.assertIn(
            "REGISTER_CUDA_DISPATCH(fakeop_aot_stub, &::fakeop_cuda_aot_kernel)", src
        )
        self.assertIn('#include "fakeop_f32_n1024_k8.h"', src)

    def test_branches_in_sidecar_order(self):
        s1 = dict(SIDECAR, spec={"N": 1024, "K": 8})
        s2 = dict(SIDECAR, prefix="fakeop_f32_n2048_k8", spec={"N": 2048, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop", "CUDA", _FakeDecl, [s1, s2], "const at::Tensor & self, int64_t k"
        )
        self.assertLess(src.index("N == 1024"), src.index("N == 2048"))

    def test_prelude_optional(self):
        class NoPrelude(_FakeDecl):
            # The contract's "absent prelude" case; the deliberate
            # callable -> None override trips bad-override.
            cpp_dispatch_prelude = None  # pyrefly: ignore [bad-override]

        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop", "CUDA", NoPrelude, [sidecar], "const at::Tensor & self, int64_t k"
        )
        self.assertNotIn("scalar_type() != at::kFloat", src)
        self.assertIn("if (N == 1024 && k == 8) {", src)

    def test_the_branch_launches_inside_its_condition(self):
        # The BLOCK, not its three pieces: asserting the cond line, the launch call
        # and a trailing `return false` separately is satisfied by a branch that
        # launches BEFORE the if (a kernel run with its precompile precondition
        # unmet) and by one that returns false after launching (the kernel runs and
        # aten then recomputes over its own output). Both left the suite green.
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k, const at::Tensor & out",
        )
        self.assertIn(
            "    if (N == 1024 && k == 8) {\n"
            "      launch_fakeop_f32_n1024_k8(self, out, "
            "at::cuda::getCurrentCUDAStream());\n"
            "      return true;\n"
            "    }",
            src,
        )

    def test_cpp_covers_emission(self):
        # cpp_covers -> a bool fn over the schema params + a
        # TORCH_LIBRARY_FRAGMENT registration in the _native_aot ns.
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        covers = (
            "const at::Tensor & self, int64_t k",
            "covers_fakeop(Tensor self, int k) -> bool",
            "return self.scalar_type() == at::kFloat && k == 8;",
        )
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k",
            covers,
        )
        self.assertIn(
            "bool fakeop_cuda_covers(const at::Tensor & self, int64_t k) {", src
        )
        self.assertIn("TORCH_LIBRARY_FRAGMENT(_native_aot, m) {", src)
        self.assertIn(
            'm.def("covers_fakeop(Tensor self, int k) -> bool", &::fakeop_cuda_covers);',
            src,
        )
        # The declaration's BODY, which nothing asserted: emitting an empty one
        # passed, and the is_cuda() a reader might look for comes from the
        # generated guard, not from here. A covers predicate reduced to the
        # generated guards claims coverage the stub declines.
        self.assertIn("return self.scalar_type() == at::kFloat && k == 8;", src)
        # ...and it ends by declining, so a body that answers only some paths does
        # not fall off the end of a bool function.
        fn = src.split("bool fakeop_cuda_covers(")[1].split("\n}")[0]
        self.assertTrue(
            fn.rstrip().endswith(
                "return false;  // undecided above: not covered, "
                "the same default the stub takes"
            ),
            fn,
        )

    def test_cpp_covers_absent_no_registration(self):
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop", "CUDA", _FakeDecl, [sidecar], "const at::Tensor & self, int64_t k"
        )
        self.assertNotIn("TORCH_LIBRARY_FRAGMENT", src)
        self.assertNotIn("fakeop_cuda_covers", src)

    def test_covers_signature_from_schema(self):
        # Functional schema args + out-variant outputs as trailing
        # optionals; SymInt degrades to int.
        params, schema = gen_aot_lib.covers_signature("topk")
        self.assertIn("int64_t k", params)
        self.assertIn("const std::optional<at::Tensor>& values", params)
        self.assertIn("const std::optional<at::Tensor>& indices", params)
        self.assertEqual(
            schema,
            "covers_topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, Tensor? values=None, Tensor? indices=None) -> bool",
        )

    def test_covers_signature_kwarg_only_and_defaults(self):
        # Pin the tricky schema shapes the per-argument rendering must
        # handle: a kwarg-only section (the '*' marker), a list default
        # ('int[1] dim=[]'), and Scalar defaults -- surgery on the whole
        # schema string mangles these.
        _, schema = gen_aot_lib.covers_signature("sum.dim_IntList")
        self.assertEqual(
            schema,
            "covers_sum_dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor? out=None) -> bool",
        )
        _, schema = gen_aot_lib.covers_signature("amax")
        self.assertEqual(
            schema,
            "covers_amax(Tensor self, int[1] dim=[], bool keepdim=False, Tensor? out=None) -> bool",
        )
        _, schema = gen_aot_lib.covers_signature("add.Tensor")
        self.assertEqual(
            schema,
            "covers_add_Tensor(Tensor self, Tensor other, *, Scalar alpha=1, Tensor? out=None) -> bool",
        )


class TestStructuredIntrospection(unittest.TestCase):
    """Read off native_functions.yaml, so they are tested against REAL ops the
    way covers_signature already is. Every generation test patches these two out,
    which left `precomputed_args` returning [] and impl_signature_params
    returning nonsense both invisible to the suite."""

    def test_precomputed_args_reports_what_meta_replaces(self):
        # The commit's headline reason for the generated header note: index_add's
        # dim arrives already wrapped, sum.dim_IntList's arrives RAW, and a prelude
        # comparing a raw negative dim against self.dim()-1 silently declines every
        # dim=-1 call.
        self.assertEqual(gen_aot_lib.precomputed_args("index_add"), ["dim"])
        self.assertEqual(gen_aot_lib.precomputed_args("sum.dim_IntList"), [])

    def test_impl_signature_is_the_structured_impl_signature(self):
        params = gen_aot_lib.impl_signature_params("topk")
        # Outputs last, because meta() allocated them before the stub runs.
        self.assertTrue(params.startswith("const at::Tensor & self, int64_t k"))
        self.assertIn("const at::Tensor & values", params)
        self.assertIn("const at::Tensor & indices", params)

    def test_precompute_note_states_which_case_applies(self):
        # The note goes into every generated file, where an author debugging a
        # silently-declining dim reads it.
        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        wrapped = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k",
            None,
            ["dim"],
        )
        self.assertIn("precomputes (impl receives them wrapped/replaced): dim", wrapped)
        raw = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [sidecar],
            "const at::Tensor & self, int64_t k",
            None,
            [],
        )
        self.assertIn("precomputes NOTHING", raw)
        self.assertIn("arrive RAW", raw)


class TestAtomicWrites(unittest.TestCase):
    def test_a_failed_write_leaves_neither_a_partial_file_nor_a_tmp(self):
        # CMake reads the emitted file as authoritative, so a half-written one is worse
        # than none. Only the final content was asserted before, which a plain
        # open(path, "w") satisfies just as well.
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "manifest.txt")
            with open(target, "w") as f:
                f.write("PREVIOUS\n")
            real_replace = os.replace
            with mock.patch.object(
                gen_aot_lib.os, "replace", side_effect=OSError("simulated failure")
            ):
                with self.assertRaises(OSError):
                    gen_aot_lib._write_atomic(target, "NEW\n")
            self.assertIs(real_replace, os.replace)
            with open(target) as f:
                self.assertEqual(f.read(), "PREVIOUS\n", "target was clobbered")
            self.assertEqual(
                [f for f in os.listdir(d) if f.endswith(".tmp")], [], "left a .tmp"
            )


class TestShippedVsDeclaredArchs(unittest.TestCase):
    def test_shipping_an_arch_the_declaration_disowns_is_fatal(self):
        # Called out as a guarantee ("shipped arches outside ARCHS fail
        # generation") and reached by no test, because the shared fixture
        # declares every arch. A packaging bug here would otherwise emit a gate
        # for hardware the op never claimed to support.
        class _Narrow(_FakeDecl):
            ARCHS = ("sm_100a",)

        s90 = dict(SIDECAR, prefix="fakeop_p__sm90a", arch="sm_90a")
        with self.assertRaisesRegex(RuntimeError, "declaration supports only"):
            gen_aot_lib.gen_op(
                "fakeop", "CUDA", _Narrow, [s90], "const at::Tensor & self, int64_t k"
            )

    def test_declaring_more_than_was_shipped_is_fine(self):
        # The other direction is normal: a build targeting one arch ships one
        # tree for an op whose kernels support several.
        s100 = dict(SIDECAR, prefix="fakeop_p__sm100a", arch="sm_100a")
        src = gen_aot_lib.gen_op(
            "fakeop", "CUDA", _FakeDecl, [s100], "const at::Tensor & self, int64_t k"
        )
        self.assertIn("major == 10", src)
        self.assertNotIn("major == 9", src)


# mX with one size and one stride slot, and with none at all: the two claims the
# per-argument checks are made against, with SIDECAR's own mOut beside them.
_MX_ONE_SLOT = [
    {"name": "mX", "dynamic_sizes": [0], "dynamic_strides": [0]},
    SIDECAR["tensor_args"][1],
]
_MX_NO_SLOTS = [{"name": "mX"}, SIDECAR["tensor_args"][1]]


class TestAbiValidation(unittest.TestCase):
    """The launcher is a fixed template and the ABI comes from the DSL, so a width
    or slot-count mismatch is a wrong value at runtime, not a compile error.

    Every fixture here is in the layout the DSL actually emits -- `typedef struct
    { ...fields... } <prefix>_Tensor_<name>_t;`, one struct PER TENSOR ARGUMENT.
    Fixtures shaped like `struct x { ... }` cannot exercise a per-argument check
    at all."""

    def _struct(self, name, shapes=1, strides=1, stride_t="int64_t", shape_t="int32_t"):
        p = SIDECAR["prefix"]
        fields = ["  void* data;"]
        if shapes:
            fields.append(f"  {shape_t} dynamic_shapes[{shapes}];")
        if strides:
            fields.append(f"  {stride_t} dynamic_strides[{strides}];")
        return "typedef struct {\n" + "\n".join(fields) + f"\n}} {p}_Tensor_{name}_t;\n"

    def _header(self, mx=None, mout=None):
        # Defaults match SIDECAR's claims: mX one shape + one stride, mOut two
        # shapes + one stride. A fixture that does NOT match them describes an ABI
        # no export could produce.
        return (
            "#pragma once\n#include <stdint.h>\n"
            + (mx if mx is not None else self._struct("mX", shapes=1, strides=1))
            + (mout if mout is not None else self._struct("mOut", shapes=2, strides=1))
        )

    def _mx_header(self, member):
        """A whole header whose mX struct declares `member` where its
        dynamic_strides go, with SIDECAR's mOut struct beside it."""
        p = SIDECAR["prefix"]
        return (
            "#pragma once\n#include <stdint.h>\n"
            "typedef struct {\n  void* data;\n"
            f"  int32_t dynamic_shapes[1];\n  {member}\n}} {p}_Tensor_mX_t;\n"
            + self._struct("mOut", shapes=2, strides=1)
        )

    def _sidecar(self, tmpdir, header: str, **over) -> dict:
        with open(os.path.join(tmpdir, SIDECAR["prefix"] + ".h"), "w") as f:
            f.write(header)
        return dict(SIDECAR, _dir=tmpdir, **over)

    def _refuses(self, header, pattern, **over):
        tc = toolchains.CuteDslToolchain()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, pattern):
                tc.validate_abi(self._sidecar(d, header, **over))

    def _accepts(self, header, **over):
        tc = toolchains.CuteDslToolchain()
        with tempfile.TemporaryDirectory() as d:
            tc.validate_abi(self._sidecar(d, header, **over))

    def test_a_matching_header_is_accepted(self):
        # int32 SHAPE slots are expected and handled (explicit cast + size gate);
        # only the stride WIDTH and both slot COUNTS are the subject.
        self._accepts(self._header())

    def test_int32_stride_slots_are_refused(self):
        # aten strides are int64 and the launcher assigns them straight across, so
        # int32 slots truncate silently -- an implicit conversion, no warning.
        self._refuses(
            self._header(mx=self._struct("mX", stride_t="int32_t")),
            "must be declared 64-bit",
        )

    def test_the_width_is_checked_per_argument(self):
        # use_32bit_stride is a per-argument kwarg of the DSL's fake-tensor
        # constructor, so a whole-file search for one "int64_t dynamic_strides"
        # passed a header declaring int64 for mX and int32 for mOut, and the
        # launcher then narrowed mOut's stride.
        self._refuses(
            self._header(mout=self._struct("mOut", shapes=2, stride_t="int32_t")),
            "mOut",
        )

    def test_a_longer_argument_name_does_not_shadow_a_shorter_one(self):
        # header.find("<prefix>_Tensor_mX_t") also matched inside
        # "<prefix>_Tensor_mX_tile_t", so with the longer name declared FIRST the
        # check read the wrong tensor's struct -- accepting a truncating stride
        # and an out-of-bounds store, depending only on argument order.
        for label, header in (
            (
                "longer first",
                self._struct("mX_tile", shapes=4, strides=3)
                + self._header(mx=self._struct("mX", stride_t="int32_t")),
            ),
            (
                "longer last",
                self._header(mx=self._struct("mX", stride_t="int32_t"))
                + self._struct("mX_tile", shapes=4, strides=3),
            ),
        ):
            with self.subTest(order=label):
                self._refuses(header, "mX's dynamic_strides|must be declared")

    def test_a_slot_count_must_match_exactly(self):
        # The bound and the sidecar list are independent statements about one
        # number -- the bound from the DSL's fake args, the list hand-written in
        # the builder -- and the launcher fills exactly the slots the sidecar
        # lists, into an UNINITIALIZED local. Over-claiming stores past the end;
        # under-claiming leaves the kernel reading an indeterminate value. Only
        # the over-claim was checked, so the silent one got through.
        for label, header in (
            (
                "header declares more",
                self._header(mx=self._struct("mX", shapes=1, strides=2)),
            ),
            (
                "header declares fewer",
                self._header(mx=self._struct("mX", shapes=3, strides=1)),
            ),
        ):
            with self.subTest(case=label):
                self._refuses(header, r"claims \d+ dynamic_\w+ slot")

    def test_a_missing_member_must_match_a_zero_claim(self):
        # The DSL omits the array entirely at zero slots, so absent means zero --
        # which still has to equal what the sidecar claims.
        self._refuses(
            self._header(mx=self._struct("mX", shapes=1, strides=0)),
            "declares no dynamic_strides",
        )
        # ...and a tensor claiming nothing is fine with no arrays at all.
        self._accepts(
            self._header(mx=self._struct("mX", shapes=0, strides=0)),
            tensor_args=_MX_NO_SLOTS,
        )

    def test_whitespace_variants_are_accepted(self):
        # `int64_t  dynamic_strides [ 1 ]` is the same declaration. Exact
        # single-space matching refused a perfect export -- and the DSL's own
        # C-type table stores these spellings WITH a trailing space, so one
        # upstream refactor would have failed every build.
        for label, decl in (
            ("two spaces", "int64_t  dynamic_strides[1];"),
            ("space before bracket", "int64_t dynamic_strides [1];"),
            ("spaces inside", "int64_t dynamic_strides[ 1 ];"),
            ("newline separated", "int64_t\n  dynamic_strides[1];"),
        ):
            with self.subTest(spelling=label):
                self._accepts(self._mx_header(decl))

    def test_a_non_literal_bound_is_refused(self):
        # Accepting it skipped the count check in silence, which is the mode this
        # guard exists to rule out.
        for bound in ("MX_NDYN", "1 + 1", ""):
            with self.subTest(bound=bound):
                member = f"int64_t dynamic_strides[{bound}];"
                self._refuses(self._mx_header(member), "not a literal count")

    def test_a_struct_declared_twice_is_refused(self):
        # An #ifdef'd 32-bit variant: which widths the compiler sees would depend
        # on the preprocessor, so there is no single answer to give.
        self._refuses(
            self._header() + self._struct("mX", stride_t="int32_t"),
            "declared 2 times",
        )

    def test_an_unparsable_header_is_refused_not_skipped(self):
        # Fails CLOSED on a layout this code cannot read. Skipping there is how the
        # guard would switch off silently after an upstream header change. The
        # anchored parse also means a mere MENTION of the type name (a banner
        # comment naming the wrapper signature) is no longer mistaken for it.
        self._refuses(
            "struct x { int64_t dynamic_strides[1]; };\n", "no `typedef struct"
        )
        self._refuses(
            f"/* ABI: {SIDECAR['prefix']}_Tensor_mX_t* */\n" + self._header(mx=""),
            "no `typedef struct",
        )

    def test_a_malformed_sidecar_is_refused_with_a_useful_message(self):
        # These used to surface as a bare KeyError/TypeError out of a build step.
        self._refuses(
            self._header(), "names no tensor", tensor_args=[{"dynamic_strides": [0]}]
        )
        self._refuses(self._header(), "not a list", tensor_args={"mX": {}})
        self._refuses(
            self._header(),
            "not a list of",
            tensor_args=[dict(SIDECAR["tensor_args"][0], dynamic_strides="01")],
        )

    def test_a_kernel_with_no_stride_slots_is_not_checked_for_width(self):
        # Nothing to get wrong: the launcher emits no stride assignment at all.
        self._accepts(
            self._header(mx=self._struct("mX", shapes=1, strides=0)),
            tensor_args=[
                {"name": "mX", "dynamic_sizes": [0]},
                SIDECAR["tensor_args"][1],
            ],
        )

    def test_a_comment_is_not_read_as_the_declaration(self):
        # re.search over the raw body took the FIRST textual match, so a
        # commented-out declaration above the real one was read instead: the
        # int32 case then shipped a narrowing assignment, and the over-claim case
        # ASan-faulted where the compiler said nothing. Members are parsed whole,
        # between semicolons, with comments stripped first.
        p = SIDECAR["prefix"]

        def hdr(comment, member):
            return (
                "#pragma once\n#include <stdint.h>\n"
                "typedef struct {\n  void* data;\n  int32_t dynamic_shapes[1];\n"
                f"  {comment}\n  {member}\n}} {p}_Tensor_mX_t;\n"
                + self._struct("mOut", shapes=2, strides=1)
            )

        one = _MX_ONE_SLOT
        for label, comment, member, targs in (
            (
                "hides int32",
                "/* int64_t dynamic_strides[1]; */",
                "int32_t dynamic_strides[1];",
                one,
            ),
            (
                "hides an under-claim",
                "/* int64_t dynamic_strides[1]; */",
                "int64_t dynamic_strides[3];",
                one,
            ),
            (
                "hides an over-claim",
                "/* room for int64_t dynamic_strides[3]; */",
                "int64_t dynamic_strides[1];",
                [dict(one[0], dynamic_strides=[0, 1, 2]), one[1]],
            ),
        ):
            with self.subTest(case=label):
                self._refuses(
                    hdr(comment, member), "dynamic_strides", tensor_args=targs
                )
        # ...while a comment BETWEEN the type and the member name is still the
        # same declaration, and must be accepted.
        self._accepts(
            hdr("", "int64_t /* 64-bit */ dynamic_strides[1];"), tensor_args=one
        )

    def test_a_terminator_this_parser_cannot_read_does_not_swallow_the_next_struct(
        self,
    ):
        # `} *Ptr_t;` is ordinary C, and the body regex used to run past it into
        # the NEXT struct -- so tensor B was checked against tensor A's
        # declarations. A brace-free body cannot reach beyond its own closing
        # brace, so the unreadable declaration is simply not registered.
        p = SIDECAR["prefix"]
        self._refuses(
            "#pragma once\n#include <stdint.h>\n"
            "typedef struct {\n  void* data;\n  int64_t dynamic_strides[1];\n"
            f"}} *{p}_TensorPtr_mA_t;\n"
            "typedef struct {\n  void* data;\n  int32_t dynamic_shapes[1];\n"
            f"  int32_t dynamic_strides[1];\n}} {p}_Tensor_mX_t;\n"
            + self._struct("mOut", shapes=2, strides=1),
            "dynamic_strides",
            tensor_args=_MX_ONE_SLOT,
        )

    def test_an_unreadable_struct_is_refused_even_when_nothing_is_claimed(self):
        # The "claims nothing, so nothing to check" escape kept the under-claim
        # direction open: claiming nothing is exactly the state that leaves every
        # slot the header DOES declare unwritten in an uninitialized local. Proved
        # by running the generated launcher: the unfilled stride read as
        # 140461696592624.
        p = SIDECAR["prefix"]
        self._refuses(
            "#pragma once\n#include <stdint.h>\n"
            f"typedef struct {p}_Tensor_mX_s {{\n  void* data;\n"
            "  int32_t dynamic_shapes[2];\n  int64_t dynamic_strides[2];\n"
            f"}} {p}_Tensor_mX_t;\n" + self._struct("mOut", shapes=2, strides=1),
            # The tag form IS readable now, so this refuses on the real problem --
            # the header declares slots the sidecar claims none of.
            r"claims 0 dynamic_\w+ slot",
            tensor_args=_MX_NO_SLOTS,
        )
        # ...and where the declaration genuinely cannot be parsed, a zero claim is
        # refused too, rather than skipped: that is the state which leaves every
        # declared slot unwritten.
        self._refuses(
            "#pragma once\n#include <stdint.h>\n"
            "typedef struct {\n  void* data;\n  int64_t dynamic_strides[2];\n"
            f"}} *{p}_TensorPtr_mX_t;\n" + self._struct("mOut", shapes=2, strides=1),
            "no `typedef struct",
            tensor_args=_MX_NO_SLOTS,
        )

    def test_an_unreadable_MEMBER_is_refused_even_when_nothing_is_claimed(self):
        # The struct-level escape above was closed while the MEMBER-level one stayed
        # open: a declaration this parser cannot classify never entered `declared`,
        # and the absent-member arm read that as zero slots -- so a sidecar claiming
        # zero passed against a struct that declares some, which is the state that
        # leaves every declared slot of an uninitialized local unwritten.
        #
        # Both spellings below are ordinary C, and both were ACCEPTED against the
        # real exported scatter_add header before the fix.
        for label, member in (
            ("comma-separated declarator", "int64_t reserved[1], dynamic_strides[1];"),
            (
                "attribute before the name",
                "int64_t __attribute__((aligned(8))) dynamic_strides[1];",
            ),
        ):
            with self.subTest(spelling=label):
                self._refuses(
                    self._mx_header(member),
                    "could not read as a declaration",
                    tensor_args=_MX_NO_SLOTS,
                )

    def test_an_octal_bound_is_not_read_as_decimal(self):
        # C reads [010] as 8; int("010") is 10. Comparing the sidecar's count with
        # the header's is this check's entire job, so a bound whose value differs
        # between the two languages is refused rather than parsed.
        self._refuses(
            self._mx_header("int64_t dynamic_strides[010];"),
            "has a leading zero, which C reads as octal",
            tensor_args=[
                {
                    "name": "mX",
                    "dynamic_sizes": [0],
                    "dynamic_strides": list(range(10)),
                },
                SIDECAR["tensor_args"][1],
            ],
        )

    def test_ordinary_c_spellings_of_the_same_declaration_are_accepted(self):
        # None of these comes out of today's wheel (the generator uses literals),
        # but each is what an upstream refactor would emit, and refusing one fails
        # every build. The type is an allowlist rather than "not int32_t": an
        # unknown spelling is refused loudly, since treating it as 64-bit would
        # restore the silent truncation.
        p = SIDECAR["prefix"]
        for label, decl in (
            ("long long", "long long dynamic_strides[1];"),
            ("signed long long", "signed long long dynamic_strides[1];"),
            ("std::int64_t", "std::int64_t dynamic_strides[1];"),
            ("u-suffixed bound", "int64_t dynamic_strides[1u];"),
        ):
            with self.subTest(spelling=label):
                self._accepts(self._mx_header(decl), tensor_args=_MX_ONE_SLOT)
        # The tag form and an attribute before the name are the same declaration.
        for label, header in (
            (
                "tag form",
                f"typedef struct Tag_mX {{\n  void* data;\n"
                f"  int32_t dynamic_shapes[1];\n  int64_t dynamic_strides[1];\n"
                f"}} {p}_Tensor_mX_t;\n",
            ),
            (
                "attribute",
                "typedef struct {\n  void* data;\n"
                "  int32_t dynamic_shapes[1];\n  int64_t dynamic_strides[1];\n"
                f"}} __attribute__((aligned(16))) {p}_Tensor_mX_t;\n",
            ),
        ):
            with self.subTest(form=label):
                self._accepts(
                    "#pragma once\n#include <stdint.h>\n"
                    + header
                    + self._struct("mOut", shapes=2, strides=1),
                    tensor_args=_MX_ONE_SLOT,
                )

    def test_an_unrelated_pointer_typedef_does_not_disturb_the_parse(self):
        # `typedef struct {...} *Ptr_t;` is ordinary C. A body that could run past
        # its own closing brace merged it into the NEXT struct, which now reads as
        # two declarations of one member and is refused -- so this legitimate
        # header would fail to build. A brace-free body keeps them separate.
        p = SIDECAR["prefix"]
        self._accepts(
            "#pragma once\n#include <stdint.h>\n"
            "typedef struct {\n  void* data;\n  int64_t dynamic_strides[1];\n"
            f"}} *{p}_TensorPtr_scratch_t;\n" + self._header(),
        )

    def test_an_unknown_stride_spelling_is_refused(self):
        # The width check is an ALLOWLIST of 64-bit spellings, not a refusal of
        # int32_t: an unrecognized type could be any width, and treating it as
        # 64-bit restores the silent truncation. The message says how to extend it.
        for spelling in ("int32_t", "int", "short", "cute_i64", "uint64_t", "float"):
            with self.subTest(type=spelling):
                self._refuses(
                    self._mx_header(f"{spelling} dynamic_strides[1];"),
                    "must be declared 64-bit",
                    tensor_args=_MX_ONE_SLOT,
                )

    def test_a_pathological_bound_is_refused_not_a_traceback(self):
        # int(bound) after bound.isdigit() raised ValueError for '\u00b2' (isdigit
        # is True) and for a bound past CPython's 4300-digit limit.
        for label, bound in (
            ("superscript two", "\u00b2"),
            ("5000 digits", "9" * 5000),
        ):
            with self.subTest(bound=label):
                self._refuses(
                    self._mx_header(f"int64_t dynamic_strides[{bound}];"),
                    "not a literal count",
                    tensor_args=_MX_ONE_SLOT,
                )

    def test_a_non_ascii_header_is_read_not_a_decode_error(self):
        # open() used the ambient locale encoding, so a valid UTF-8 header raised
        # UnicodeDecodeError under LC_ALL=C -- neither the documented skip nor a
        # diagnosis.
        tc = toolchains.CuteDslToolchain()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, SIDECAR["prefix"] + ".h")
            # Invalid UTF-8 (a lone 0xFF), not merely non-ASCII: that raises for
            # a plain open() whatever the runner's locale is, where a valid UTF-8
            # header only fails under LC_ALL=C and the test would pass here while
            # the build broke in a container.
            with open(path, "wb") as f:
                f.write(b"/* \xff */\n" + self._header().encode("utf-8"))
            tc.validate_abi(dict(SIDECAR, _dir=d))

    def test_a_header_that_cannot_be_read_is_not_silently_skipped(self):
        # Only ABSENT means "nothing to check". A header that exists but cannot be
        # read used to take the same path, so the ABI went unvalidated and the
        # launcher shipped anyway -- and generation's own check is os.path.exists,
        # which is true for this. A directory stands in for the unreadable file
        # because mode 000 is still readable by root, which CI runs as.
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, SIDECAR["prefix"] + ".h"))
            with self.assertRaises(OSError):
                toolchains.CuteDslToolchain().validate_abi(dict(SIDECAR, _dir=d))

    def test_absent_header_is_not_an_error(self):
        # Unit fixtures have no header; a real generation always does.
        toolchains.CuteDslToolchain().validate_abi(dict(SIDECAR, _dir="/nonexistent"))


class TestMultiCapabilitySelector(unittest.TestCase):
    """One generated .cpp serves every arch an op shipped for, so its selector
    is what keeps each artifact on its own hardware. _by_arch's grouping and
    the absence of dead launchers are covered above; this pins the SHAPE of
    what gen_op emits from those groups, which an inverted or misplaced gate
    passes through unchanged."""

    def _body(self):
        # Deliberately passed newest-first: the emitted order must come from
        # the capability, not from directory or argument order.
        s100 = dict(
            SIDECAR, prefix="fakeop_p__sm100a", arch="sm_100a", spec={"N": 1024, "K": 8}
        )
        s90 = dict(
            SIDECAR, prefix="fakeop_p__sm90a", arch="sm_90a", spec={"N": 1024, "K": 8}
        )
        src = gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [s100, s90],
            "const at::Tensor & self, int64_t k, const at::Tensor & out",
        )
        body = src.split("bool fakeop_cuda_aot_kernel(", 1)[1]
        return body[: body.index("\n}\n")]

    def test_properties_read_once_before_any_gate(self):
        body = self._body()
        self.assertEqual(body.count("= at::cuda::getCurrentDeviceProperties();"), 1)
        self.assertLess(body.index("_naot_props ="), body.index("_naot_props->"))

    def test_early_out_accepts_exactly_the_shipped_capabilities(self):
        body = self._body()
        accept = next(l for l in body.splitlines() if l.startswith("  if (!(("))
        self.assertEqual(accept.count("major =="), 2)
        self.assertIn("major == 9 && _naot_props->minor == 0", accept)
        self.assertIn("major == 10 && _naot_props->minor == 0", accept)

    def test_each_capability_launches_only_its_own_kernels(self):
        # The cross pairing is the failure this whole grouping exists to
        # prevent: loading a module built for other hardware fails inside the
        # launcher instead of declining to aten.
        body = self._body()
        i9 = body.index("_naot_props->major == 9 && _naot_props->minor == 0) {")
        i10 = body.index("_naot_props->major == 10 && _naot_props->minor == 0) {")
        self.assertLess(i9, i10)
        sm90_branch, sm100_branch = body[i9:i10], body[i10:]
        self.assertIn("launch_fakeop_p__sm90a(", sm90_branch)
        self.assertNotIn("launch_fakeop_p__sm100a(", sm90_branch)
        self.assertIn("launch_fakeop_p__sm100a(", sm100_branch)
        self.assertNotIn("launch_fakeop_p__sm90a(", sm100_branch)


class TestReadOnlyInputs(unittest.TestCase):
    # read_only tensor args must go through const_data_ptr in every
    # toolchain's launcher: a mutable data_ptr() materializes
    # copy-on-write inputs on each call.

    def test_cutedsl_launcher(self):
        sc: dict = dict(SIDECAR)
        sc["tensor_args"] = [
            {"name": "mX", "dynamic_sizes": [0], "read_only": True},
            {"name": "mOut", "dynamic_sizes": [0]},
        ]
        src = gen_aot_lib.gen_launcher(sc)
        self.assertIn("mX_s.data = const_cast<void*>(mX.const_data_ptr());", src)
        self.assertIn("mOut_s.data = mOut.mutable_data_ptr();", src)

    def test_closure_covers_shared_declaration_machinery(self):
        # The grid expander and the validating loader decide which spec
        # points exist and what a declaration means, so editing them changes what
        # an artifact MEANS. They live outside the tools/*.py glob and arrive by
        # ordinary import, so only the sys.modules half of the closure catches
        # them -- and it used to filter on "torch._native" alone.
        #
        # By basename: an editable install can resolve torchgen to a different
        # checkout than REPO, where relpath yields a ../.. traversal.
        names = {os.path.basename(p) for p in export.source_closure()}
        for want in (
            "native_aot_spec_grid.py",
            "native_aot_decl.py",
            "toolchains.py",
        ):
            self.assertIn(want, names, f"{want} must invalidate artifacts")

    def test_closure_excludes_the_consumer_tools(self):
        # Neither can change what an artifact MEANS, and hashing them is
        # expensive in a way that is easy to miss: every kernel of every arch
        # re-exports (minutes) for an edit that could not have changed one.
        # gen_aot_lib.py only reads sidecars; build_stage2.py only decides
        # whether stage 2 runs and then relinks -- export.py reads the arch
        # list itself, so the driver passes it nothing kernel-affecting.
        names = {os.path.basename(p) for p in export.source_closure()}
        for unwanted in ("gen_aot_lib.py", "build_stage2.py"):
            self.assertNotIn(unwanted, names)

    def test_closure_survives_sys_modules_mutation(self):
        # source_closure hashes files while walking sys.modules, and
        # hashing imports hashlib lazily -- so on a cold interpreter the
        # walk mutates the dict it is iterating and raises "dictionary
        # changed size during iteration". Force that ordering by having
        # the hash step import a module that is definitely not loaded yet.
        import sys

        real_hash = export._file_hash

        def hash_and_import(path):
            importlib.import_module("wave")  # stdlib, unlikely to be loaded
            return real_hash(path)

        sys.modules.pop("wave", None)
        with mock.patch.object(export, "_file_hash", hash_and_import):
            export.source_closure()

    def test_runtimes_current_detects_a_compiler_upgrade(self):
        # The DSL's version is in no file the closure hashes, so an upgraded
        # wheel changes nothing on disk and without this re-exports nothing,
        # leaving the tree mixing compilers.
        #
        # runtime_versions is PATCHED, not read from this machine: CI's image has
        # no DSL wheels, where the live call is all-"absent" and runtimes_current
        # takes its ignorance arm -- which left the comparison uncovered in CI.
        current = {"nvidia-cutlass-dsl": "4.6.2", "apache-tvm-ffi": "0.1.11"}
        with mock.patch.object(export, "runtime_versions", lambda kind: current):
            self.assertTrue(
                export.runtimes_current({"kind": "cutedsl", "runtimes": current})
            )
            older = dict.fromkeys(current, "0.0.1")
            self.assertFalse(
                export.runtimes_current({"kind": "cutedsl", "runtimes": older})
            )
            # A sidecar predating the record is stale, like one with no closure.
            self.assertFalse(export.runtimes_current({"kind": "cutedsl"}))

    def test_runtimes_current_does_not_call_ignorance_staleness(self):
        # Generation may run where the DSL wheels are absent. Declaring every
        # artifact stale there would fail a build that cannot re-export anyway.
        with mock.patch.object(
            export, "runtime_versions", lambda kind: {"nvidia-cutlass-dsl": "absent"}
        ):
            self.assertTrue(export.runtimes_current({"kind": "cutedsl"}))

    def test_runtime_versions_reads_metadata_not_the_module(self):
        # Distribution names, so the skip path never imports the DSL (importing
        # cutlass pulls MLIR bindings in). The mapping is not derivable: module
        # `cutlass` ships in the nvidia-cutlass-dsl distribution.
        self.assertIn(
            "nvidia-cutlass-dsl", toolchains.get_toolchain("cutedsl").RUNTIME_DISTS
        )
        versions = export.runtime_versions("cutedsl")
        # Sorted keys, so two runs on one machine record byte-identical sidecars
        # (a dict whose order drifts would compare unequal and re-export).
        self.assertEqual(list(versions), sorted(versions))
        self.assertEqual(
            sorted(versions), sorted(toolchains.get_toolchain("cutedsl").RUNTIME_DISTS)
        )
        for dist, v in versions.items():
            self.assertTrue(v, f"{dist} recorded an empty version")

        # The VALUE, against a known one. _RUNTIMES and the sidecar assertions that
        # compare against it are produced by THIS function, so a mutation that
        # recorded a garbage version moved the fixture with it and passed.
        from importlib.metadata import PackageNotFoundError

        dists = toolchains.get_toolchain("cutedsl").RUNTIME_DISTS
        with mock.patch("importlib.metadata.version", return_value="1.2.3"):
            got = export.runtime_versions("cutedsl")
        self.assertEqual(got, dict.fromkeys(dists, "1.2.3"))
        # An uninstalled distribution is recorded as absent rather than omitted, so
        # "compiled where the wheel was missing" differs from "compiled before this
        # was recorded". Previously pinned only on a machine without the wheels.
        with mock.patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            got = export.runtime_versions("cutedsl")
        self.assertEqual(got, dict.fromkeys(dists, "absent"))

    def test_sources_current_roundtrip(self):
        # A sidecar whose recorded closure matches the tree is current;
        # editing any recorded file (or recording none) makes it stale.
        rel = _DECL_REL
        h = export._file_hash(os.path.join(export.REPO, rel))
        good = {"version": export.SIDECAR_VERSION, "sources": {rel: h}}
        self.assertTrue(export.sources_current(good))
        # schema-version mismatch is stale even with current sources
        self.assertFalse(export.sources_current({"version": 0, "sources": {rel: h}}))
        self.assertFalse(export.sources_current({"sources": {rel: "0" * 16}}))
        self.assertFalse(export.sources_current({}))
        self.assertFalse(export.sources_current({"sources": {"no/such/file.py": "aa"}}))

    def test_stale_point_reexports_without_force(self):
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32"}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point)
            with _no_ambient_arch():
                self.assertFalse(export._job_needed(job, force=False))
            _write_sidecar(d, point, sources={_DECL_REL: "0" * 16})
            # Inside the guard like the call above: unguarded, the recorded arch
            # (None) mismatched the DETECTED one and _job_needed answered True on
            # that, never reaching staleness -- and with CUTE_DSL_ARCH set it
            # raised instead.
            with _no_ambient_arch():
                self.assertTrue(export._job_needed(job, force=False))


class TestToolchainRegistry(unittest.TestCase):
    def test_cutedsl_registered(self):
        self.assertIn("cutedsl", toolchains.TOOLCHAINS)

    def test_unknown_kind_raises(self):
        with self.assertRaisesRegex(RuntimeError, "unknown toolchain kind"):
            toolchains.get_toolchain("nvfuser")

    def test_builder_validation_names_missing_keys(self):
        tc = toolchains.get_toolchain("cutedsl")
        with self.assertRaisesRegex(RuntimeError, "missing keys.*fake_args"):
            tc.validate_build_result({"prefix": "x", "fn": object(), "tensor_args": []})


CUBIN_SIDECAR = {
    "prefix": "fakemm_f32",
    "kind": "triton",
    "symbol": "_fakemm_kernel",
    "spec": {"dtype": "float32"},
    "launch": {"grid_x": "B_dim*((M+31)/32)"},
    "shared": 512,
    "block_x": 128,
    "args": [
        {"name": "a", "kind": "tensor"},
        {"name": "B_dim", "kind": "scalar", "ctype": "int32_t"},
        {"name": "M", "kind": "scalar", "ctype": "int32_t"},
    ],
}


class TestEndToEndGeneration(unittest.TestCase):
    _DECL = (
        'ATEN_OP = "fakeop"\n'
        'DISPATCH_KEY = "CUDA"\n'
        'KERNEL_MODULE = "kernel.py"\n'
        "def kernel_precompile_grid():\n"
        '    return [{"N": 1024, "K": 8}]\n'
        "def covered_axes(self, k):\n"
        '    return {"N": 0, "K": k}\n'
        "def cpp_dispatch(spec):\n"
        "    return f\"N == {spec['N']}\"\n"
        "def cpp_launch(spec, launch_fn):\n"
        '    return f"{launch_fn}();"\n'
    )

    @contextlib.contextmanager
    def _generated(self, touch=True, header=None, arch_list=None):
        """Run main() over a one-kernel tree; yield (artifacts_dir, error|None).

        ``touch`` writes the artifacts the sidecar claims; ``header`` overwrites
        the ABI header, for the cases where generation must REFUSE and the
        interesting assertion is that nothing was left behind; ``arch_list`` is
        passed through as --arch-list."""
        with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as ops:
            art_op = os.path.join(art, "sm_100a", "fakeop")
            os.makedirs(art_op)
            if touch:
                _touch_artifacts(art_op, SIDECAR["prefix"])
            if header is not None:
                with open(os.path.join(art_op, SIDECAR["prefix"] + ".h"), "w") as f:
                    f.write(header)
            sidecar = dict(
                SIDECAR,
                spec={"N": 1024, "K": 8},
                sources=_current_sources(),
                runtimes=_RUNTIMES,
                version=export.SIDECAR_VERSION,
            )
            with open(os.path.join(art_op, SIDECAR["prefix"] + ".json"), "w") as f:
                json.dump(sidecar, f)
            os.makedirs(os.path.join(ops, "fakeop"))
            with open(os.path.join(ops, "fakeop", "aot.py"), "w") as f:
                f.write(self._DECL)
            err = None
            with _patched_generation(ops, declarations=None):
                argv = ["--artifacts-dir", art]
                if arch_list is not None:
                    argv += ["--arch-list", arch_list]
                try:
                    gen_aot_lib.main(argv)
                except Exception as e:
                    err = e
            yield art, err

    def test_main_writes_aot_source(self):
        # Artifacts live at <root>/<arch>/<decl_id>/ -- the one layout, whatever
        # the arch count. _generated writes the real artifacts, not just the
        # sidecar: generation lists the objects CMake links and refuses a sidecar
        # describing a file that is not there.
        with self._generated() as (art, err):
            self.assertIsNone(err)
            art_op = os.path.join(art, "sm_100a", "fakeop")
            out = os.path.join(art, "fakeop", "aot_fakeop_cuda.cpp")
            self.assertTrue(os.path.exists(out))
            with open(out) as f:
                src = f.read()
            self.assertIn("fakeop_cuda_aot_kernel", src)
            # The header include reaches from the generated source at
            # <root>/<decl_id>/ into the arch tree. Invert this relpath and
            # every generated file fails to compile, which no other test sees.
            self.assertIn(f'#include "../sm_100a/fakeop/{SIDECAR["prefix"]}.h"', src)
            # The emitted file is everything CMake reads: the sources to compile, the
            # objects to link, and the arch list they were generated for.
            man = _manifest(art)
            self.assertEqual(man["sources"], [out])
            self.assertEqual(
                man["objects"], [os.path.join(art_op, SIDECAR["prefix"] + ".o")]
            )
            for p in man["sources"] + man["objects"]:
                self.assertTrue(os.path.isabs(p), f"{p} is not absolute")
            # No --arch-list was passed, so the file must claim NOTHING rather
            # than "" -- CMake distinguishes the two, and asserting against
            # os.getenv here (as this once did) compares "" with "" whenever the
            # variable is unset, which is a tautology.
            self.assertEqual(man["arch_list"], "absent")
            # Version-script CONTENTS, not just existence: writing it from an
            # empty prefix list localizes nothing, which puts every kernel symbol
            # back into torch_cuda's public ABI.
            with open(os.path.join(art, gen_aot_lib.VERSION_SCRIPT)) as f:
                ver = f.read()
            self.assertIn(f"{SIDECAR['prefix']}_*;", ver)
            self.assertIn(f"_mlir_*{SIDECAR['prefix']}*;", ver)

    def test_main_records_the_arch_list_it_was_given(self):
        # A literal, not os.getenv: the recorded claim is what a reader compares a
        # mismatch against, and it must come from the flag rather than from
        # whatever the ambient environment happens to hold.
        with self._generated(arch_list="9.0a;10.0a") as (art, err):
            self.assertIsNone(err)
            self.assertEqual(_manifest(art)["arch_list"], "9.0a;10.0a")

    def test_main_refuses_a_sidecar_whose_object_is_absent(self):
        # The launcher would be emitted and the object missing at link time. A
        # comment in test_main_writes_aot_source claimed to cover this; it
        # creates the artifacts and never reaches the refusal.
        with self._generated(touch=False) as (art, err):
            self.assertIsNotNone(err, "expected a refusal")
            self.assertIn("not on disk", str(err))
            # ...and nothing was left behind for the CMake glob to compile.
            self.assertFalse(glob.glob(os.path.join(art, "*", "aot_*.cpp")))

    def test_main_refuses_an_int32_stride_abi(self):
        # validate_abi's only production call site. Removing the call left the
        # whole suite green: TestAbiValidation invokes the method directly.
        # A real export's layout, with mX's strides narrowed to int32 -- the
        # silent-truncation shape validate_abi exists for.
        pre = SIDECAR["prefix"]
        bad = "".join(
            f"typedef struct {{ void* data; int32_t dynamic_shapes[2];\n"
            f"  {w} dynamic_strides[1]; }} {pre}_Tensor_{t}_t;\n"
            for t, w in (("mX", "int32_t"), ("mOut", "int64_t"))
        )
        with self._generated(header=bad) as (
            art,
            err,
        ):
            self.assertIsNotNone(err, "expected a refusal")
            self.assertIn("must be declared 64-bit", str(err))
            self.assertFalse(glob.glob(os.path.join(art, "*", "aot_*.cpp")))

    def test_main_tolerates_missing_artifacts_dir(self):
        # Zero declarations: export creates no artifacts dir at all
        # (stage 2 on a repo state with no ops). main() must no-op, not
        # FileNotFoundError (regressed on the sm100 CI build of the
        # commit before any op lands).
        import sys

        with tempfile.TemporaryDirectory() as d:
            argv = sys.argv
            sys.argv = ["gen_aot_lib.py", "--artifacts-dir", os.path.join(d, "absent")]
            try:
                gen_aot_lib.main()
            finally:
                sys.argv = argv


class TestDeclarationArchs(unittest.TestCase):
    def test_a_malformed_archs_entry_is_refused_at_load(self):
        # _SM_RE accepts "sm_9" and "sm_1000", which name no capability. Refused by
        # the LOADER, which is the only place that knows which file to name -- and
        # because export compares ARCHS by string, a typo there silently matched
        # nothing, so the op was absent from the build with no diagnostic.
        for bad in ("sm_9", "sm_1000"):
            with self.subTest(archs=bad):
                with tempfile.TemporaryDirectory() as ops:
                    _write_fake_decl(ops, f"ARCHS = ({bad!r},)\n")
                    path = os.path.join(ops, "fakeop", "aot.py")
                    with self.assertRaisesRegex(RuntimeError, "compute capability"):
                        native_aot_decl.load_declarations(path)
                    # ...and the message names the file, which a refusal from
                    # inside export could not.
                    try:
                        native_aot_decl.load_declarations(path)
                    except RuntimeError as e:
                        self.assertIn("aot.py", str(e))


class TestDeclarationStaleness(unittest.TestCase):
    def test_source_closure_includes_the_declaration(self):
        # Declarations load by file path and never enter sys.modules, so
        # without this a kernel_precompile_grid() edit reuses artifacts
        # built from the old grid.
        with tempfile.TemporaryDirectory() as tmpdir:
            decl_path = os.path.join(tmpdir, "aot.py")
            with open(decl_path, "w") as f:
                f.write("ATEN_OP = 'fake'\n")
            closure = export.source_closure(decl_path)
            rel = os.path.relpath(decl_path, export.REPO)
            self.assertIn(rel, closure)
            self.assertEqual(closure[rel], export._file_hash(decl_path))

    def test_source_closure_omits_a_declaration_not_passed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            decl_path = os.path.join(tmpdir, "aot.py")
            with open(decl_path, "w") as f:
                f.write("ATEN_OP = 'fake'\n")
            # The closure records the declaration it is GIVEN, so one it was not
            # given must be absent -- otherwise editing any aot.py anywhere would
            # invalidate every other op's artifacts.
            self.assertNotIn(
                os.path.relpath(decl_path, export.REPO), export.source_closure()
            )
            self.assertIn(
                os.path.relpath(decl_path, export.REPO),
                export.source_closure(decl_path),
            )


class TestStaleGridPointArtifacts(unittest.TestCase):
    def test_sidecar_for_dropped_grid_point_is_fatal(self):
        # Its .o is still matched by the CMake glob and would link with no
        # launcher referencing it.
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "gone.json"), "w") as f:
                # version, because a real sidecar always has one: the check reads
                # the schema first, so a version-less fixture is "unreadable", not
                # "stale", and would test nothing.
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "gone",
                        "spec": {"N": 4096},
                    },
                    f,
                )
            with self.assertRaisesRegex(RuntimeError, "no longer in the grid"):
                export._check_no_orphan_artifacts(tmpdir, [{"N": 1024}])

    def test_a_sidecar_from_another_schema_is_not_called_stale(self):
        # It cannot be read, so it cannot be judged: the next SIDECAR_VERSION bump
        # would otherwise make the first export in an existing tree demand
        # `rm -rf` instead of re-exporting the points it no longer understands.
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "old.json"), "w") as f:
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION + 1,
                        "prefix": "old",
                        "spec": {"N": 4096},
                    },
                    f,
                )
            export._check_no_orphan_artifacts(tmpdir, [{"N": 1024}])

    def test_sidecar_still_in_grid_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "live.json"), "w") as f:
                # version, for the reason the sibling test above gives: without it
                # this fixture is "unreadable" rather than "in the grid", the spec
                # comparison never runs, and declaring EVERY sidecar stale passed.
                json.dump(
                    {
                        "version": export.SIDECAR_VERSION,
                        "prefix": "live",
                        "spec": {"N": 1024},
                    },
                    f,
                )
            export._check_no_orphan_artifacts(tmpdir, [{"N": 1024}])


class TestInt32GateTypeClassifier(unittest.TestCase):
    def test_plain_and_optional_tensors_are_gated(self):
        gate = gen_aot_lib._int32_size_gate(
            "const at::Tensor & self, const ::std::optional<at::Tensor> & weight"
        )
        self.assertIn("self.sizes()", gate)
        self.assertIn("weight.has_value()", gate)

    def test_the_gate_bounds_dims_only_and_not_numel(self):
        # The contract is "every dim the CALLER passes must fit". Bounding numel() here
        # instead declined any tensor over 2**31 elements even when the extent a
        # prelude derives is tiny -- (2**28, 8) collapses to 8, is served correctly and
        # is bitwise equal to aten, so declining it is pure lost coverage on exactly
        # the large shapes this path exists for. A prelude that DERIVES an extent bounds
        # it itself, a duty _int32_size_gate's docstring records (unenforced: the
        # generator cannot see what a prelude computes).
        gate = gen_aot_lib._int32_size_gate(
            "const at::Tensor & self, const ::std::optional<at::Tensor> & weight"
        )
        self.assertIn("self.sizes()", gate)
        self.assertNotIn("numel()", gate)

    def test_unhandled_tensor_like_types_are_refused(self):
        # torchgen renders Tensor? as at::OptionalTensorRef and Tensor[] as
        # at::ITensorListRef in structured impl signatures; neither fits the
        # accessors the gate emits, and guessing would either leave a dim
        # ungated or emit C++ that does not compile.
        for params in (
            "const at::Tensor & self, at::OptionalTensorRef bias",
            "const at::ITensorListRef & tensors",
            "const c10::List<::std::optional<at::Tensor>> & indices",
        ):
            with self.subTest(params=params):
                with self.assertRaisesRegex(RuntimeError, "unhandled tensor-like"):
                    gen_aot_lib._int32_size_gate(params)

    def test_scalar_only_signature_emits_nothing(self):
        self.assertEqual(gen_aot_lib._int32_size_gate("int64_t k, bool largest"), "")


class TestGeneratedVersionScript(unittest.TestCase):
    def test_patterns_are_anchored_on_the_kernel_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen_aot_lib.write_version_script(tmpdir, ["topk_f32_n1024_k8"])
            with open(path) as f:
                text = f.read()
        # <prefix>_* covers every symbol the DSL emits for the kernel;
        # enumerating known suffixes left _args_spec/_kernel_info/
        # _function_name/_version in torch_cuda's ABI.
        self.assertIn("topk_f32_n1024_k8_*;", text)
        self.assertIn("_mlir_*topk_f32_n1024_k8*;", text)
        # The DIRECTIVE line, not the word: VER_TMPL's own comment block says
        # "`local:` alone restricts", so a substring check passed with the
        # directive changed to `global:` -- which puts all 864 DSL symbols back
        # into torch_cuda's exported ABI, the exact inverse of this file.
        self.assertIn("\n  local:\n", text)
        self.assertNotIn("\n  global:\n", text)
        # EVERY pattern names the prefix. Asserting the absence of one
        # hand-picked unanchored pattern ("*_cuda_init;") proved nothing -- it
        # was never a candidate. This fails for any pattern that would reach
        # past this kernel's symbols and quietly drop something unrelated out
        # of torch_cuda's ABI. Indented and ";"-terminated selects the pattern
        # lines, not the script's own closing "};" or its comment block.
        patterns = [
            l.strip()
            for l in text.splitlines()
            if l.startswith("    ") and l.strip().endswith(";")
        ]
        self.assertTrue(patterns)
        for p in patterns:
            self.assertIn("topk_f32_n1024_k8", p, f"unanchored pattern: {p}")


class TestRegistryConsistency(unittest.TestCase):
    def test_artifact_exts_are_shared_by_both_sweeps(self):
        # One notion of "kernel artifact", exercised through BOTH sweeps with a
        # toolchain neither knew about. Asserting that all_artifact_exts()
        # contains each toolchain's exts proves nothing -- it is defined as
        # their union, and would still pass if a sweep kept a private set.
        class _Novel(toolchains.Toolchain):
            kind = "novel"
            artifact_exts = (".novelobj",)

        with mock.patch.dict(toolchains.TOOLCHAINS, {"novel": _Novel()}, clear=False):
            self.assertIn(".novelobj", toolchains.all_artifact_exts())
            # Sweep 1: export's per-directory orphan check, which REPORTS an
            # artifact no sidecar claims (nothing links one, so refusing it only
            # forced a hand-delete) -- naming it is what proves the shared set.
            with tempfile.TemporaryDirectory() as d:
                open(os.path.join(d, "k.novelobj"), "w").close()
                with contextlib.redirect_stdout(io.StringIO()) as said:
                    export._check_no_orphan_artifacts(d, [])
                self.assertIn("k.novelobj", said.getvalue())

            # Sweep 2: generation's no-declaration check, which refuses to leave
            # undeclared artifacts for the link glob to pick up.
            # "fakeop" has artifacts but no declaration; "other" is declared, so
            # by_id is non-empty (an empty one means "this commit declares
            # nothing", which is deliberately NOT treated as orphaned).
            class _OtherDecl(_FakeDecl):
                ATEN_OP = "other"

            with tempfile.TemporaryDirectory() as tmpdir:
                art = os.path.join(tmpdir, "sm_100a", "fakeop")
                os.makedirs(art)
                open(os.path.join(art, "k.novelobj"), "w").close()
                ops = os.path.join(tmpdir, "_ops")
                os.makedirs(os.path.join(ops, "other"))
                open(os.path.join(ops, "other", "aot.py"), "w").close()
                with (
                    mock.patch.object(gen_aot_lib, "OPS_DIR", ops),
                    mock.patch.object(
                        gen_aot_lib.decl,
                        "load_declarations",
                        return_value=[_OtherDecl],
                    ),
                ):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        gen_aot_lib.main(["--artifacts-dir", tmpdir])
                    # Reported, per arch directory, and NOT fatal: see
                    # TestOrphanArtifactSafety for why.
                    self.assertIn("no declaration", out.getvalue())

    def test_link_exts_must_be_a_subset_of_artifact_exts(self):
        # Generation iterates artifact_exts and links `if ext in link_exts`, so a
        # kind whose link_exts names something artifact_exts does not contributes
        # NO objects: nothing is listed to link, nothing passes
        # --no-undefined, torch_cuda links green, and the first call of the op
        # fails on an undefined symbol.
        class _Bad(toolchains.Toolchain):
            kind = "bad"
            artifact_exts = (".cubin", ".h")
            link_exts = (".o",)

        with self.assertRaisesRegex(RuntimeError, "not a subset"):
            toolchains._assert_link_exts_are_exportable({"bad": _Bad()})

    def test_generation_and_export_look_for_declarations_in_one_place(self):
        # Two modules spell this path independently. Diverged, export writes
        # artifacts for the declarations it found while generation scans a different
        # tree to map artifact dirs back to declarations -- so by_id comes up empty,
        # every artifact looks undeclared, and the sweep refuses a correct build.
        self.assertEqual(gen_aot_lib.OPS_DIR, export.OPS_DIR)

    def test_link_exts_must_be_declared(self):
        # The realistic mistake is forgetting the attribute, and that direction is a
        # subset of everything, so the subset rule alone accepted a kind that links
        # nothing -- which is the silence this check exists to prevent.
        class _Forgot(toolchains.Toolchain):
            kind = "forgot"
            artifact_exts = (".o", ".h")

        with self.assertRaisesRegex(RuntimeError, "link_exts is not declared"):
            toolchains._assert_link_exts_are_exportable({"forgot": _Forgot()})

        # ...while an EXPLICIT empty tuple stays a real answer, for a kind whose
        # launcher embeds the artifact instead of linking it.
        class _Embeds(toolchains.Toolchain):
            kind = "embeds"
            artifact_exts = (".cubin",)
            link_exts = ()

        toolchains._assert_link_exts_are_exportable({"embeds": _Embeds()})

    def test_the_shipped_toolchains_are_consistent(self):
        # The import-time call, re-run explicitly so this is a test rather than a
        # side effect of collection.
        toolchains._assert_link_exts_are_exportable(toolchains.TOOLCHAINS)


class TestSpecRoundTrip(unittest.TestCase):
    def test_tuple_grid_values_reach_the_declaration_as_tuples(self):
        # The grid distinguishes a list (an axis) from a tuple (one compound
        # value), but JSON has only arrays. Unrestored, `spec["dtypes"] ==
        # ("f32", "bf16")` was silently False in a declaration's cpp_dispatch --
        # emitting a cond that never fires, so the kernel shipped, linked, and
        # declined every call -- and `_CTYPE[spec["dtypes"]]` raised "unhashable
        # type: 'list'" inside generation.
        restored = gen_aot_lib._spec_from_json(
            {"dtypes": ["f32", "bf16"], "N": 1024, "pairs": [[1, 2], [3, 4]]}
        )
        self.assertEqual(restored["dtypes"], ("f32", "bf16"))
        self.assertEqual(restored["pairs"], ((1, 2), (3, 4)))
        self.assertEqual(restored["N"], 1024)

    def test_a_tuple_spec_survives_the_sidecar_round_trip(self):
        # The restore is exact rather than a guess: expand_specs consumed every
        # list as an axis, so a sequence still in a recorded point was a tuple.
        (point,) = export.expand_specs([{"dtypes": ("f32", "bf16"), "N": 1024}])
        self.assertEqual(point["dtypes"], ("f32", "bf16"))
        recorded = json.loads(json.dumps(export._json_normal(point)))
        self.assertEqual(recorded["dtypes"], ["f32", "bf16"])
        self.assertEqual(gen_aot_lib._spec_from_json(recorded), point)

    def test_the_generator_hands_cpp_dispatch_the_restored_spec(self):
        # The CALL SITE, not just _spec_from_json: dropping the restore in gen_op
        # left the direct test above green, the same shape as validate_abi's
        # missing call-site test. A declaration comparing a tuple-valued field
        # then sees a list and emits a cond that never fires.
        seen = {}

        class _TupleDecl(_FakeDecl):
            @staticmethod
            def cpp_dispatch(spec):
                seen.update(spec)
                return "true"

            @staticmethod
            def cpp_launch(spec, launch_fn):
                return launch_fn + "();"

        sc = dict(
            SIDECAR,
            spec=json.loads(json.dumps({"dtypes": ["f32", "bf16"], "N": 1, "K": 8})),
            _dir="/art",
        )
        gen_aot_lib.gen_op(
            op="fakeop",
            key="CUDA",
            d=_TupleDecl,
            sidecars=[sc],
            impl_params="const at::Tensor & self, int64_t k",
        )
        self.assertEqual(seen["dtypes"], ("f32", "bf16"))


class TestClaimedSpellingPreference(unittest.TestCase):
    def test_the_conditional_spelling_wins_when_both_are_claimed(self):
        # Alphabetical order would pick sm_100 over sm_100a. The conditional one
        # is what the generator's tie-break keeps for a capability, so exporting
        # the plain build would pay a compile and then lose that tie-break.
        both = ("sm_100", "sm_100a", "sm_90", "sm_90a")
        self.assertEqual(export._claimed_spelling("sm_100", both), "sm_100a")
        self.assertEqual(export._claimed_spelling("sm_100a", both), "sm_100a")
        self.assertEqual(export._claimed_spelling("sm_90", both), "sm_90a")

    def test_a_capability_that_is_not_claimed_has_no_spelling(self):
        self.assertIsNone(export._claimed_spelling("sm_103", ("sm_100a",)))

    def test_the_plain_spelling_is_used_when_it_is_the_only_claim(self):
        # A declaration pinning the plain build must not be handed a conditional
        # target its kernels were not written for.
        self.assertEqual(export._claimed_spelling("sm_100a", ("sm_100",)), "sm_100")


class TestSidecarSchemaIsReadFirst(unittest.TestCase):
    """gen_aot_lib refuses a mismatched sidecar SCHEMA before reading any field,
    because a different version need not mean the same thing by a name. The skip
    check has to do the same: it read sc["kind"] first and raised a bare KeyError,
    naming no file and no remedy, where the version field exists precisely so the
    point re-exports."""

    def _job(self, d, sidecar):
        _touch_artifacts(d, "x")
        with open(os.path.join(d, "x.json"), "w") as f:
            json.dump(sidecar, f)
        return (
            "fakeop",
            "aot_kernel.py",
            {"dtype": "float32", "N": 4096},
            d,
            "sm_100a",
        )

    def _sidecar(self, **over):
        sc = {
            "version": export.SIDECAR_VERSION,
            "prefix": "x",
            "kind": "cutedsl",
            "spec": {"dtype": "float32", "N": 4096},
            "arch": "sm_100a",
            "sources": _current_sources(),
            "runtimes": _RUNTIMES,
        }
        sc.update(over)
        return sc

    def test_a_matching_sidecar_still_skips(self):
        # The control: without it, "re-exports" below could be for any reason.
        with tempfile.TemporaryDirectory() as d:
            job = self._job(d, self._sidecar())
            with _no_ambient_arch():
                self.assertFalse(export._job_needed(job, force=False))

    def test_a_sidecar_from_another_schema_re_exports(self):
        with tempfile.TemporaryDirectory() as d:
            job = self._job(d, self._sidecar(version=export.SIDECAR_VERSION + 1))
            with _no_ambient_arch():
                self.assertTrue(export._job_needed(job, force=False))

    def test_a_sidecar_without_a_kind_re_exports_instead_of_raising(self):
        sc = self._sidecar()
        del sc["kind"]
        with tempfile.TemporaryDirectory() as d:
            job = self._job(d, sc)
            with _no_ambient_arch():
                self.assertTrue(export._job_needed(job, force=False))


class TestSourceClosureCoversVendoredKernels(unittest.TestCase):
    def test_a_loaded_vendored_module_is_hashed(self):
        # torch/_vendor/quack/ holds real CuTeDSL kernel bodies, and a
        # declaration's build() is meant to share code with its JIT wrapper. With
        # the prefix missing, editing a vendored body left every artifact's
        # closure unchanged, sources_current() True, and a relink shipping kernels
        # compiled from the old source -- silently.
        #
        # REPO and _HERE both redirected into a temp tree: the first version of
        # this test put its probe file under tools/native_aot/, which the closure
        # hashes by GLOB whatever the prefixes say, so it passed with
        # torch._vendor removed. The file has to sit where only the prefix reaches.
        with tempfile.TemporaryDirectory() as root:
            here = os.path.join(root, "tools", "native_aot")
            vendored = os.path.join(root, "torch", "_vendor", "quack")
            os.makedirs(here)
            os.makedirs(vendored)
            path = os.path.join(vendored, "rmsnorm.py")
            with open(path, "w") as f:
                f.write("# a stand-in for a vendored kernel body\n")
            mod = types.ModuleType("torch._vendor.quack.rmsnorm")
            mod.__file__ = path
            with (
                mock.patch.object(export, "REPO", root),
                mock.patch.object(export, "_HERE", here),
                mock.patch.dict(
                    sys.modules, {"torch._vendor.quack.rmsnorm": mod}, clear=False
                ),
            ):
                closure = export.source_closure()
            rel = os.path.join("torch", "_vendor", "quack", "rmsnorm.py")
            self.assertIn(rel, closure)
            # ...and the recorded hash is the file's, so an edit invalidates it.
            self.assertEqual(closure[rel], export._file_hash(path))


class TestGeneratedCoversGuards(unittest.TestCase):
    """The covers predicate is registered as a catch-all custom op, so the router
    calls it for ANY dispatch of that op -- CPU tensors included, on hosts with no
    GPU at all."""

    def _gen(self, params="const at::Tensor & self, int64_t k", covers_schema=None):
        # N and K: _FakeDecl.cpp_dispatch reads both.
        sc = dict(
            SIDECAR,
            spec={"N": 1024, "K": 8},
            _dir="/art",
            tensor_args=[{"name": "mX", "dynamic_sizes": [0], "dynamic_strides": [0]}],
        )
        schema = covers_schema or "covers_fakeop(Tensor self, int k) -> bool"
        return gen_aot_lib.gen_op(
            op="fakeop",
            key="CUDA",
            d=_FakeDecl,
            sidecars=[sc],
            impl_params=params,
            covers=(params, schema, "return self.is_cuda();"),
        )

    def test_the_covers_body_checks_is_cuda_before_reading_properties(self):
        # Untested before: the string "is_cuda" appeared nowhere in this file, and
        # deleting the guard left the suite green. getDeviceProperties(-1) falls
        # back to the CURRENT device, so a CPU tensor initializes a CUDA context
        # from inside a predicate the router runs on every call -- and on a
        # GPU-less host it ABORTS (num_gpus=0) where the answer is simply false.
        src = self._gen()
        body = src.split("_covers(", 1)[1]
        guard = body.index("is_cuda()")
        props = body.index("getDeviceProperties")
        self.assertLess(guard, props, "is_cuda must be checked BEFORE the props read")
        self.assertIn("if (!self.is_cuda()) return false;", src)

    def test_covers_reads_the_tensors_device_not_the_current_one(self):
        # The router calls covers before any device guard, so on a mixed-capability
        # host the current device need not be the one the op will run on.
        src = self._gen()
        self.assertIn("getDeviceProperties(self.device().index())", src)

    def test_a_covers_signature_without_a_tensor_is_refused(self):
        # Nothing to ask is_cuda() of, and falling back to current-device
        # properties reintroduces the abort above. Refused at generation, where an
        # author sees it, rather than shipped.
        with self.assertRaisesRegex(RuntimeError, "plain at::Tensor"):
            self._gen(params="const at::ITensorListRef & tensors, int64_t dim")

    def test_a_double_quoted_schema_default_is_escaped(self):
        # 15 in-tree schemas carry one (str mode="constant" on pad, str
        # padding="valid" on conv1d.padding, str UPLO="L" on _linalg_eigh).
        # Unescaped it closed the C++ string literal early: a compile error inside
        # a @generated file, far from the declaration that caused it.
        src = self._gen(
            covers_schema='covers_fakeop(Tensor self, str mode="constant") -> bool'
        )
        self.assertIn(r"str mode=\"constant\"", src)
        # ...and the literal is still one argument: nothing between the escaped
        # quotes ends it.
        line = next(l for l in src.splitlines() if "m.def(" in l)
        self.assertEqual(line.count('", &::'), 1)


class TestSourceCommitOrdering(unittest.TestCase):
    """Generation writes NOTHING until every declaration has passed its refusals,
    and every file it does write is written atomically. Both properties exist
    because the main build compiles whatever is on disk at configure time, and
    stage 2 -- the only writer -- runs after that build."""

    def _tree(self, root, decl_id, prefix, ok=True):
        art = os.path.join(root, "sm_100a", decl_id)
        os.makedirs(art, exist_ok=True)
        _touch_artifacts(art, prefix, exts=(".o", ".h") if ok else (".h",))
        with open(os.path.join(art, prefix + ".json"), "w") as f:
            json.dump(
                dict(
                    SIDECAR,
                    prefix=prefix,
                    spec={"N": 1024, "K": 8},
                    sources=_current_sources(),
                    runtimes=_RUNTIMES,
                    version=export.SIDECAR_VERSION,
                ),
                f,
            )
        return art

    def test_no_source_is_written_when_a_later_declaration_is_refused(self):
        # TWO declarations, the second refusing. With one, both existing refusal
        # tests pass even with the buffering removed, because the refusal happens
        # before that declaration's own source would be written. The state this
        # prevents is declaration 1's FRESH source beside the previous run's
        # objects: undefined symbols in the main build, unrepairable from there.
        with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as ops:
            self._tree(art, "aaa", "aaa_p__sm100a", ok=True)
            self._tree(art, "bbb", "bbb_p__sm100a", ok=False)  # .o missing

            class _Aaa(_FakeDecl):
                ATEN_OP = "aaa"

            class _Bbb(_FakeDecl):
                ATEN_OP = "bbb"

            for name in ("aaa", "bbb"):
                os.makedirs(os.path.join(ops, name))
                open(os.path.join(ops, name, "aot.py"), "w").close()
            decls = {"aaa": [_Aaa], "bbb": [_Bbb]}
            with _patched_generation(
                ops, lambda path: decls[os.path.basename(os.path.dirname(path))]
            ):
                with self.assertRaisesRegex(RuntimeError, "not on"):
                    gen_aot_lib.main(["--artifacts-dir", art])
            self.assertEqual(glob.glob(os.path.join(art, "*", "aot_*.cpp")), [])
            self.assertFalse(
                os.path.exists(os.path.join(art, gen_aot_lib.CMAKE_INCLUDE)),
                "no native_aot.cmake: its presence marks a finished generation",
            )

    def test_the_old_cmake_is_gone_before_any_source_is_written(self):
        # Sources are individually atomic, but their paths are deterministic and
        # the PREVIOUS file already names them: a run that died between two
        # sources left a NEW source paired with the previous run's OBJECT list, and
        # the main build then failed on undefined symbols -- with stage 2, the only
        # writer that could repair it, running after that build. Removing the
        # it first makes that state read as "not generated yet" instead.
        order = []
        real = gen_aot_lib._write_atomic
        state = {}

        with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as ops:
            self._tree(art, "fakeop", SIDECAR["prefix"])
            os.makedirs(os.path.join(ops, "fakeop"))
            open(os.path.join(ops, "fakeop", "aot.py"), "w").close()
            stale = os.path.join(art, gen_aot_lib.CMAKE_INCLUDE)
            with open(stale, "w") as f:
                f.write("ARCH_LIST_ABSENT\nOBJECT=/gone/old.o\n")

            def spy(path, text):
                name = os.path.basename(path)
                order.append(name)
                if name.startswith("aot_"):
                    state.setdefault("cmake_existed", os.path.exists(stale))
                return real(path, text)

            with (
                _patched_generation(ops),
                mock.patch.object(gen_aot_lib, "_write_atomic", spy),
            ):
                gen_aot_lib.main(["--artifacts-dir", art])
        self.assertFalse(
            state["cmake_existed"],
            "the previous native_aot.cmake must be gone before any source is written",
        )
        # ...and the new one is written LAST, after every source.
        self.assertEqual(order[-1], gen_aot_lib.CMAKE_INCLUDE)

    def test_every_written_file_goes_through_the_atomic_writer(self):
        # TestAtomicWrites exercises _write_atomic directly, which cannot see a
        # CALL SITE that stopped using it -- the same shape as validate_abi's
        # missing call-site test. All three writers are checked here.
        calls = []
        real = gen_aot_lib._write_atomic

        def spy(path, text):
            calls.append(os.path.basename(path))
            return real(path, text)

        with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as ops:
            self._tree(art, "fakeop", SIDECAR["prefix"])
            os.makedirs(os.path.join(ops, "fakeop"))
            open(os.path.join(ops, "fakeop", "aot.py"), "w").close()
            with (
                _patched_generation(ops),
                mock.patch.object(gen_aot_lib, "_write_atomic", spy),
            ):
                gen_aot_lib.main(["--artifacts-dir", art])
        self.assertIn(gen_aot_lib.CMAKE_INCLUDE, calls)
        self.assertIn("native_aot_local.ver", calls)
        # The generated source too: its path is deterministic, so the PREVIOUS
        # emitted file already names it, and CMake only checks it exists -- a
        # truncated one is compiled by the main build.
        self.assertTrue(
            any(c.startswith("aot_") and c.endswith(".cpp") for c in calls),
            f"the generated source was not written atomically: {calls}",
        )


class TestSizeGateIsPerKindNotPerFile(unittest.TestCase):
    def test_one_narrowing_kind_among_several_emits_the_gate(self):
        # `narrows = any(...)` read as `all(...)` passed identically, because every
        # existing case builds a single-kind sidecar list. The mixed list is the
        # only shape that tells them apart -- and `all` would drop the gate for a
        # file that DOES carry a narrowing kernel, so a dim past INT32_MAX would
        # be truncated instead of declined.
        class _Wide(toolchains.Toolchain):
            kind = "wide"
            NARROWS_SHAPES_TO_INT32 = False
            artifact_exts = (".o",)
            link_exts = (".o",)

            def gen_launcher(self, sidecar):
                return f"void launch_{sidecar['prefix']}() {{}}"

            def kernel_includes(self, sidecar):
                return []

        spec = {"N": 1024, "K": 8}
        narrowing = dict(SIDECAR, prefix="k_narrow", spec=spec, _dir="/art")
        wide = dict(SIDECAR, prefix="k_wide", kind="wide", spec=spec, _dir="/art")
        with mock.patch.dict(toolchains.TOOLCHAINS, {"wide": _Wide()}, clear=False):
            src = gen_aot_lib.gen_op(
                op="fakeop",
                key="CUDA",
                d=_FakeDecl,
                sidecars=[wide, narrowing],
                impl_params="const at::Tensor & self, int64_t k",
            )
        self.assertIn("_naot_dim_too_big", src)


class TestExportMain(unittest.TestCase):
    """export.main() had no coverage at all, including the TORCH_CUDA_ARCH_LIST
    translation that another comment in the same file depends on: _CLOSURE_EXCLUDED
    justifies keeping build_stage2.py out of the source closure on the grounds
    that "export.py reads the arch list itself, so the driver passes it nothing
    kernel-affecting"."""

    def _main(self, argv, env, jobs_seen):
        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(
                export,
                "_collect_jobs",
                lambda ops, root, archs: jobs_seen.append(archs) or [],
            ),
        ):
            export.main([*argv, "--out-dir", out])

    def test_the_arch_list_is_translated_into_arches(self):
        seen = []
        self._main([], {"TORCH_CUDA_ARCH_LIST": "9.0a;10.0a"}, seen)
        self.assertEqual(seen, [["sm_90a", "sm_100a"]])

    def test_an_explicit_arch_wins_over_the_arch_list(self):
        seen = []
        self._main(["--arch", "sm_90a"], {"TORCH_CUDA_ARCH_LIST": "10.0a"}, seen)
        self.assertEqual(seen, [["sm_90a"]])

    def test_no_arch_list_means_on_device(self):
        # archs [None] is what makes _collect_jobs resolve from the device.
        seen = []
        self._main([], {"TORCH_CUDA_ARCH_LIST": ""}, seen)
        self.assertEqual(seen, [[None]])

    def test_a_pending_export_invalidates_the_previous_generation(self):
        # The EXPORT half of "invalidate the previous generation first". Its gen-side
        # twin is pinned twice, this one by nothing: the fixture above patches
        # _collect_jobs to return [], so `if todo:` never ran in any test. What it
        # prevents is silent -- artifacts are direct link inputs in build.ninja, so an
        # export interrupted part-way plus a plain `cmake --build` relinks torch_cuda
        # from two revisions' objects, described by launchers generated for the older.
        from tools.native_aot.gen_aot_lib import CMAKE_INCLUDE

        with tempfile.TemporaryDirectory() as out:
            stale = os.path.join(out, CMAKE_INCLUDE)
            with open(stale, "w") as f:
                f.write("# the previous generation\n")
            job = (
                "fakeop",
                "aot",
                {"N": 1},
                os.path.join(out, "sm_100a", "fakeop"),
                "sm_100a",
            )
            with (
                mock.patch.object(
                    export, "_collect_jobs", lambda ops, root, archs: [job]
                ),
                mock.patch.object(export, "_job_needed", lambda j, force: True),
                mock.patch.object(export, "_run_job", lambda j: "k__sm100a"),
                mock.patch.dict(
                    os.environ, {"TORCH_CUDA_ARCH_LIST": "10.0a"}, clear=False
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                export.main(["--out-dir", out])
            # OVERWRITTEN, not removed: a deleted include is not registered as a
            # configure dependency, so the next generation would be invisible to
            # `cmake --build` (verified: the kernels stayed out of the library
            # until an explicit reconfigure). The empty variant reads the same to
            # the build and keeps the dependency alive.
            self.assertTrue(os.path.exists(stale), "the include must survive")
            with open(stale) as f:
                self.assertEqual(f.read(), gen_aot_lib.NOTHING_TO_EMBED)

    def test_a_refusal_invalidates_the_generation_it_tells_you_to_break(self):
        # Every refusal in _collect_jobs advises `rm -rf <arch tree>`, and the previous
        # generation names every object in that tree -- so following the advice made the
        # next main build die in CMake on a missing source, inside a @generated file
        # that names no remedy. Verified with real cmake before this was changed.
        from tools.native_aot.gen_aot_lib import CMAKE_INCLUDE

        with tempfile.TemporaryDirectory() as out:
            stale = os.path.join(out, CMAKE_INCLUDE)
            with open(stale, "w") as f:
                f.write("# names objects in the tree the user is about to delete\n")

            def refuse(ops, root, archs):
                raise RuntimeError("kernel artifacts with no sidecar; run `rm -rf ...`")

            with (
                mock.patch.object(export, "_collect_jobs", refuse),
                mock.patch.dict(
                    os.environ, {"TORCH_CUDA_ARCH_LIST": "10.0a"}, clear=False
                ),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "no sidecar"),
            ):
                export.main(["--out-dir", out])
            self.assertTrue(os.path.exists(stale), "the include must survive")
            with open(stale) as f:
                self.assertEqual(f.read(), gen_aot_lib.NOTHING_TO_EMBED)

    def test_an_arch_list_with_no_exportable_arch_exports_nothing(self):
        # Not an error: a CUDA build for Ampere alone simply has no AOT kernels.
        seen = []
        self._main([], {"TORCH_CUDA_ARCH_LIST": "8.0;8.6"}, seen)
        self.assertEqual(seen, [], "_collect_jobs should not even be reached")


class TestCollectJobsRefusals(unittest.TestCase):
    def test_an_unnameable_arch_is_refused(self):
        # There is no unnamed layout: an artifact whose arch nobody can state is
        # one the runtime gate cannot match to hardware. Untested before --
        # replacing the raise with a placeholder left the suite green, because both
        # existing _collect_jobs tests patch _detected_arch to a real sm string.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops)
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device=None),
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot determine the arch"):
                    export._collect_jobs(None, out, [None])

    def test_a_malformed_arch_is_refused(self):
        # The explicit path compares ARCHS by STRING, and routing every arch
        # through _claimed_spelling used to be the only thing that ever validated
        # an sm string. Without cc_of here, `--arch sm100a` (or sm_1000, SM_100)
        # matched no declaration, exported nothing, and exited 0 -- a typo that
        # looked like a successful build. `gen_aot_lib.py`'s own stale-artifact
        # advice tells users to run export with --arch, so this is a live path.
        for bad in ("sm100a", "SM_100", "sm_1000", "sm_9", "90", "sm_100+PTX"):
            with self.subTest(arch=bad):
                with (
                    tempfile.TemporaryDirectory() as ops,
                    tempfile.TemporaryDirectory() as out,
                ):
                    _write_fake_decl(ops)
                    with (
                        mock.patch.object(export, "OPS_DIR", ops),
                        _no_ambient_arch(device=None),
                    ):
                        # The exact refusal: "compute capability" alone also
                        # matches cc_of's range error and any other refusal in this
                        # function that mentions the phrase.
                        with self.assertRaisesRegex(
                            RuntimeError, "cannot read a compute capability"
                        ):
                            export._collect_jobs(None, out, [bad])

    def test_a_declaration_that_ships_nothing_says_so(self):
        # ARCHS spelling is load-bearing on the explicit path: a declaration
        # pinning ('sm_100a',) ships nothing for a release list of plain spellings
        # ("10.0"), and with several declarations that is PARTIAL -- the build
        # embeds the ops that matched, relinks, passes its post-relink check, and
        # the others are simply absent. Nothing downstream can report it, because
        # there is no tree for generation to complain about.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops, "ARCHS = ('sm_100a',)\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device=None),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                jobs = export._collect_jobs(None, out, ["sm_100"])
        self.assertEqual(jobs, [])
        said = printed.getvalue()
        self.assertIn("declares kernels but none for this build", said)
        self.assertIn("sm_100", said)
        # The report is the ONLY thing that surfaces this, so it has to state the
        # declaration's real ARCHS: it used to end with a fixed illustration, which
        # read "an ARCHS of ('sm_100a',)" for declarations claiming something else.
        self.assertIn("ARCHS (sm_100a)", said)

    def test_a_declaration_that_misses_one_requested_arch_says_so(self):
        # The whole-declaration report is suppressed once a declaration ships for
        # ANY requested arch, which hid the case with the worst outcome: the matched
        # arches embed, generation is happy, the post-relink check passes, and every
        # device of the missed capability falls back to aten with nothing in the log.
        # Reported per arch here, because the declaration DOES claim this capability
        # -- under the other spelling.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops, "ARCHS = ('sm_90a', 'sm_100a')\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device=None),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                jobs = export._collect_jobs(None, out, ["sm_90a", "sm_100"])
        self.assertEqual(len(jobs), 1, "the arch that DID match must still export")
        said = printed.getvalue()
        self.assertIn("requested sm_100", said)
        self.assertIn("only as sm_100a", said)

    def test_an_arch_the_declaration_does_not_target_stays_quiet(self):
        # The other half: a capability the declaration claims under NO spelling is
        # not news, or every partial build reports every op. Suppressed only because
        # this declaration ships for the arch that did match.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops, "ARCHS = ('sm_90a',)\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device=None),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                jobs = export._collect_jobs(None, out, ["sm_90a", "sm_100"])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(printed.getvalue(), "")

    def test_an_on_device_export_that_ships_nothing_says_so(self):
        # The explicit path reported this and the automatic path did not, for a
        # request that means the same thing ("export for this machine"), so the user
        # saw only `exported 0 kernels` -- no op named, no ARCHS, nothing to act on.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops, "ARCHS = ('sm_100a',)\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device="sm_90"),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                jobs = export._collect_jobs(None, out, [None])
        self.assertEqual(jobs, [])
        said = printed.getvalue()
        self.assertIn("declares kernels but none for this build", said)
        self.assertIn("sm_90", said)

    def test_a_declaration_that_does_ship_is_not_reported(self):
        # The control: the report must not fire for the ordinary case, or it is
        # noise on every build.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops, "ARCHS = ('sm_100', 'sm_100a')\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device=None),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                jobs = export._collect_jobs(None, out, ["sm_100"])
        self.assertTrue(jobs)
        self.assertNotIn("declares kernels but none", printed.getvalue())

    def test_a_declaration_that_disowns_the_local_capability_is_skipped(self):
        # Skipped, not exported into a tree the declaration disowns: that tree made
        # generation refuse with "delete and re-export", whose remedy rebuilt the
        # identical tree, so `pip install -e .` failed permanently.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops, "ARCHS = ('sm_90a',)\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device="sm_100"),
            ):
                self.assertEqual(export._collect_jobs(None, out, [None]), [])
                self.assertEqual(os.listdir(out), [], "no tree for a disowned arch")

    def test_the_declarations_spelling_is_adopted_for_the_local_capability(self):
        # sm_100 detected, ('sm_100a',) claimed: same capability, so the job is
        # created and BOTH the tree and the compile target use the declaration's
        # spelling. Preferring the conditional one matches the generator's
        # tie-break, which drops the plain build for the same capability anyway.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops, "ARCHS = ('sm_100a',)\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device="sm_100"),
            ):
                (job,) = export._collect_jobs(None, out, [None])
        self.assertEqual(os.path.basename(os.path.dirname(job[3])), "sm_100a")
        self.assertEqual(job[4], "sm_100a")


class TestOrphanCheckIsCalled(unittest.TestCase):
    def test_collect_jobs_runs_the_orphan_check(self):
        # Every orphan test calls _check_no_orphan_artifacts DIRECTLY, so deleting
        # its only production call site left the suite green -- the same
        # missing-call-site shape already fixed for validate_abi and _write_atomic.
        # An undescribed .o would then be linked with no launcher referencing it.
        seen = []
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops)
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                mock.patch.object(
                    export,
                    "_check_no_orphan_artifacts",
                    lambda d, points: seen.append(d),
                ),
                _no_ambient_arch(device="sm_100"),
            ):
                export._collect_jobs(None, out, [None])
        self.assertEqual(len(seen), 1, "the orphan check must run per (decl, arch)")
        self.assertTrue(seen[0].endswith(os.path.join("sm_100a", "fakeop")))

    def test_an_undescribed_artifact_is_reported_through_the_call_site(self):
        # End to end through the call site: the report reaches the build log, and
        # does not stop the export -- nothing links an artifact no sidecar names.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            _write_fake_decl(ops)
            stray = os.path.join(out, "sm_100a", "fakeop")
            os.makedirs(stray)
            open(os.path.join(stray, "leftover.o"), "w").close()
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device="sm_100"),
                contextlib.redirect_stdout(io.StringIO()) as said,
            ):
                jobs = export._collect_jobs(None, out, [None])
        self.assertIn("leftover.o", said.getvalue())
        self.assertTrue(jobs, "the export still runs; the orphan is only disk")


class TestEmittedCMake(unittest.TestCase):
    """caffe2/CMakeLists.txt is one include(); everything it used to derive is
    emitted here. These assert the emitted CALLS, which is what CMake consumes."""

    def _emit(self, tmpdir, dsl=None, arch_list="10.0a", sources=None, objects=None):
        src = os.path.join(tmpdir, "aot_fakeop_cuda.cpp")
        with open(src, "w") as f:
            f.write("// generated\n")
        obj = os.path.join(tmpdir, "k.o")
        with open(obj, "w") as f:
            f.write("")
        ver = os.path.join(tmpdir, "native_aot_local.ver")
        with open(ver, "w") as f:
            f.write("{ local: *; };\n")
        path = gen_aot_lib.write_cmake_include(
            tmpdir,
            [src] if sources is None else sources,
            [obj] if objects is None else objects,
            ver,
            dsl,
            arch_list,
        )
        with open(path) as f:
            return f.read()

    def test_the_linker_options_are_emitted_de_duplication_safe(self):
        # Three hazards, all measured by linking: -Wl, (what LINKER: expands to) is
        # split on COMMAS by the compiler driver; bare -Xlinker tokens are collapsed
        # by target_link_options' de-duplication, which stopped torch_cuda linking
        # at all; and SHELL: splits its own value on SPACES, so the path must be
        # quoted inside it.
        with tempfile.TemporaryDirectory() as d:
            emitted = self._emit(d, dsl=os.path.join(d, "libdsl.a"))
        self.assertNotIn('"LINKER:', emitted)
        self.assertNotIn('"-Xlinker" "--', emitted)
        for opt in ("--version-script", "--exclude-libs"):
            with self.subTest(option=opt):
                self.assertIn(f'"SHELL:-Xlinker {opt} -Xlinker \\"', emitted)

    def test_paths_are_absolute_and_quoted(self):
        # CMake resolves a relative path against a different directory than the
        # generator, and an unquoted one breaks on the first space.
        with tempfile.TemporaryDirectory() as d:
            rel = os.path.relpath(d)
            emitted = self._emit(rel)
        for line in emitted.splitlines():
            if line.strip().startswith('"') and line.strip().endswith('"'):
                self.assertTrue(
                    line.strip()[1:].startswith("/"), f"not absolute: {line!r}"
                )

    def test_the_install_rpath_property_is_set_on_the_target(self):
        # The other half of stage 2's hand copy over site-packages, which is
        # byte-identical to what install writes ONLY because of this property
        # (cmake/Dependencies.cmake sets CMAKE_BUILD_WITH_INSTALL_RPATH FALSE
        # globally). Without it the shipped library carries the builder's build-tree
        # RUNPATH: fine on the machine that built it, broken everywhere else. The copy
        # half is covered; this half was asserted by nothing.
        with tempfile.TemporaryDirectory() as d:
            emitted = self._emit(d)
        prop = "BUILD_WITH_INSTALL_RPATH TRUE"
        self.assertIn(f"set_target_properties(torch_cuda PROPERTIES {prop})", emitted)

    def test_the_status_line_stage_two_greps_for_is_emitted(self):
        # A contract across two files: stage 2 refuses to relink unless the
        # reconfigure echoes this. Only the consumer side was pinned, and by a fake of
        # this output at that -- so a reworded message here would fail every stage-2
        # build with a diagnostic blaming the CMake cache. Shared constant, asserted
        # on the side that emits it.
        with tempfile.TemporaryDirectory() as d:
            emitted = self._emit(d)
        self.assertIn(f'message(STATUS "{gen_aot_lib.EMBED_STATUS} ', emitted)

    def test_a_semicolon_in_a_path_is_refused(self):
        # Escaping it as \; does yield a literal semicolon, but CMake re-splits a
        # source list on it downstream, so the build asks for the prefix before the
        # semicolon ("ninja: error: '/tmp/x/semi' ... missing") -- verified. Refused
        # here rather than emitted as a file that configures and cannot build.
        with tempfile.TemporaryDirectory() as d:
            odd = os.path.join(d, "semi;colon")
            os.makedirs(odd)
            with self.assertRaisesRegex(RuntimeError, "splits source lists"):
                self._emit(odd)

    def test_spaces_and_dollars_survive_escaped(self):
        # Assert the emitted PATH, with the escaping spelled out here rather than
        # taken from _cmake_str: the prose in the emitted file contains `\$` and `\"`
        # of its own, so the old "is there a backslash-dollar anywhere" assertion
        # passed with the escaper deleted. A $ reaches a real link intact (verified by
        # building a shared library from an artifacts dir named `has$dollar`).
        with tempfile.TemporaryDirectory() as d:
            cases = (
                ("has space", "has space"),
                ("has$dollar", "has\\$dollar"),
                # ${...} and $ENV{...}, not just a bare $: the linker options quote
                # their path BY HAND, and unescaped there CMake expanded the reference
                # away -- configure passed, the source compiled, and the link then
                # failed on a version script that never existed. A bare $ survived
                # raw, which is exactly why asserting only target_sources hid it.
                ("has${brace}x", "has\\${brace}x"),
                ("has$ENV{HOME}x", "has\\$ENV{HOME}x"),
            )
            for name, want in cases:
                with self.subTest(path=name):
                    odd = os.path.join(d, name)
                    os.makedirs(odd, exist_ok=True)
                    emitted = self._emit(odd)
                    self.assertIn(f"{d}/{want}/aot_fakeop_cuda.cpp", emitted)
                    # The version-script option, which is the half that was raw.
                    ver = f'\\"{d}/{want}/native_aot_local.ver\\"'
                    self.assertIn(f"-Xlinker {ver}", emitted)

    def test_a_quote_or_backslash_in_a_path_is_refused(self):
        # Not escapable: the version script reaches the linker inside a SHELL:
        # argument, so the path meets a second level of quoting. Both spellings made
        # cmake fail to PARSE the emitted include (verified), which is a syntax error
        # pointing at caffe2/CMakeLists.txt's include() rather than at the path.
        for name in ('has"quote', "has\\back"):
            with self.subTest(path=name), tempfile.TemporaryDirectory() as d:
                odd = os.path.join(d, name)
                os.makedirs(odd)
                with self.assertRaisesRegex(RuntimeError, "cannot embed"):
                    self._emit(odd)

    def test_the_opt_out_is_emitted_not_left_to_cmake(self):
        # Only CMake runs at configure time and a previous run's file can be on
        # disk, so the check has to exist -- but it is generated, which is what
        # keeps caffe2/CMakeLists.txt free of it. CMake truthiness, so OFF/false/no
        # work like 0, matching build_stage2._OPT_OUT_VALUES.
        with tempfile.TemporaryDirectory() as d:
            emitted = self._emit(d)
        # The BLOCK, in order: three independent substrings passed with either arm
        # inverted (embedding when the user opted OUT), with either return()
        # deleted, and with the cache read unconditionally -- the last being
        # verbatim the regression the emitted comment above it describes.
        self.assertIn(
            'if(DEFINED ENV{TORCH_NATIVE_AOT} AND NOT "$ENV{TORCH_NATIVE_AOT}" '
            'STREQUAL "")\n'
            '  if(NOT "$ENV{TORCH_NATIVE_AOT}")\n'
            '    message(STATUS "native-AOT: '
            'TORCH_NATIVE_AOT=$ENV{TORCH_NATIVE_AOT}, not embedding kernels")\n'
            "    return()\n"
            "  endif()\n"
            'elseif(DEFINED TORCH_NATIVE_AOT AND NOT "${TORCH_NATIVE_AOT}")\n'
            '  message(STATUS "native-AOT: TORCH_NATIVE_AOT=${TORCH_NATIVE_AOT}, '
            'not embedding kernels")\n'
            "  return()\n"
            "endif()\n",
            emitted,
        )

    def test_invalidation_keeps_the_include_so_a_later_generation_is_seen(self):
        # CMake keeps a configure dependency on an include()d file only if it
        # existed at configure time, so DELETING it made the next generation
        # invisible to `cmake --build`: it relinked without the kernels, reported
        # success, and stayed that way however often the tree was regenerated.
        # Overwriting reads the same to the build and keeps the dependency.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, gen_aot_lib.CMAKE_INCLUDE)
            with open(path, "w") as f:
                f.write("target_sources(torch_cuda PRIVATE /x/aot_op_cuda.cpp)\n")
            with contextlib.redirect_stdout(io.StringIO()) as said:
                export._invalidate_generation(d)
            self.assertTrue(os.path.exists(path), "the include must survive")
            with open(path) as f:
                text = f.read()
        self.assertEqual(text, gen_aot_lib.NOTHING_TO_EMBED)
        self.assertIn("invalidated", said.getvalue())

    def test_no_prefixes_writes_no_version_script(self):
        # An empty `local:` block is a syntax error to ld, and the path is one an
        # already-configured build.ninja still names through LINK_DEPENDS, so a
        # relink without a reconfigure would fail on a @generated file.
        with tempfile.TemporaryDirectory() as d:
            stale = os.path.join(d, gen_aot_lib.VERSION_SCRIPT)
            with open(stale, "w") as f:
                f.write("{\n  local:\n    old_*;\n};\n")
            path = gen_aot_lib.write_version_script(d, [])
            self.assertEqual(path, stale)
            self.assertFalse(os.path.exists(stale), "a stale script must go too")

    def test_nothing_is_emitted_when_there_are_no_sources(self):
        # No generated source means no launcher references the objects, so there is
        # nothing to embed -- and the file must not name them, or CMake would link
        # objects nothing calls.
        with tempfile.TemporaryDirectory() as d:
            emitted = self._emit(d, sources=[])
        self.assertNotIn("target_sources", emitted)
        self.assertIn("Nothing to embed", emitted)


class TestParamListSplitting(unittest.TestCase):
    """Both the size gate and the covers device read pick parameters out of a
    rendered C++ signature, so the split has to survive a template argument
    list that contains a comma."""

    def test_top_level_split_keeps_template_args_intact(self):
        params = (
            "const at::Tensor & self, std::array<bool, 3> mask, "
            "const std::optional<at::Tensor>& out"
        )
        self.assertEqual(
            gen_aot_lib._split_params(params),
            [
                "const at::Tensor & self",
                "std::array<bool, 3> mask",
                "const std::optional<at::Tensor>& out",
            ],
        )

    def test_gate_sees_a_comma_bearing_type_whole(self):
        # The refusal below exists to stop a guess at an unknown tensor-shaped
        # type. It can only be honest if it is handed the WHOLE type: split on
        # every comma, the classifier judges the fragment "at::Tensor>" and
        # names that in the error, sending the next author after a type that
        # does not appear in their signature.
        with self.assertRaisesRegex(RuntimeError, r"std::pair<at::Tensor, at::Tensor>"):
            gen_aot_lib._int32_size_gate(
                "const at::Tensor & self, std::pair<at::Tensor, at::Tensor> pr"
            )

    def test_non_tensor_comma_bearing_type_is_not_gated(self):
        gate = gen_aot_lib._int32_size_gate(
            "const at::Tensor & self, std::array<bool, 3> mask, "
            "const std::optional<at::Tensor>& out"
        )
        self.assertIn("self.sizes().begin()", gate)
        self.assertIn("out.has_value()", gate)
        self.assertNotIn("mask", gate)


class TestArchScopedGeneration(unittest.TestCase):
    """Nothing prunes arch trees, so an incremental build whose
    TORCH_CUDA_ARCH_LIST changed still has the old one on disk. It must be
    neither generated from nor linked, and must not fail the build."""

    def _tree(self, tmpdir, arch, prefix, stale_sources=False):
        d = os.path.join(tmpdir, arch, "fakeop")
        os.makedirs(d)
        _touch_artifacts(d, prefix)
        rel = _DECL_REL
        digest = (
            "0" * 16
            if stale_sources
            else export._file_hash(os.path.join(export.REPO, rel))
        )
        with open(os.path.join(d, prefix + ".json"), "w") as f:
            json.dump(
                dict(
                    SIDECAR,
                    version=export.SIDECAR_VERSION,
                    prefix=prefix,
                    arch=arch,
                    spec={"N": 1024, "K": 8},
                    sources={rel: digest},
                    runtimes=_RUNTIMES,
                ),
                f,
            )
        return d

    def _generate(self, tmpdir, argv):
        # The ops dir must NOT live under the artifacts dir: discovery reads
        # every top-level directory there as an arch tree, so an "_ops" child
        # would register as one (and its fakeop/ as that arch's declaration).
        #
        # A TemporaryDirectory, not mkdtemp: the mkdtemp version leaked one tree
        # per call, and the rest of this file cleans up after itself.
        with tempfile.TemporaryDirectory() as ops:
            os.makedirs(os.path.join(ops, "fakeop"), exist_ok=True)
            open(os.path.join(ops, "fakeop", "aot.py"), "w").close()
            with _patched_generation(ops):
                gen_aot_lib.main(["--artifacts-dir", tmpdir, *argv])

    def test_a_declaration_with_cpp_covers_emits_the_predicate(self):
        # main()'s covers assembly, which no test reached: every covers test builds the
        # (params, schema, body) tuple by hand and calls gen_op directly, and no main()
        # fixture declares cpp_covers. Losing it is silent -- aot_manifest resolves
        # torch.ops._native_aot.covers_<id> inside try/except and falls back to the
        # Python covered_axes, which carries neither the device gate nor the int32 size
        # gate, so a call the stub will decline is reported covered and loses its JIT
        # route.
        class _CoversDecl(_FakeDecl):
            @staticmethod
            def cpp_covers():
                return "return self.scalar_type() == at::kFloat && k == 8;"

        signature = (
            "const at::Tensor & self, int64_t k",
            "covers_fakeop(Tensor self, int k) -> bool",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            with (
                tempfile.TemporaryDirectory() as ops,
                mock.patch.object(
                    gen_aot_lib, "covers_signature", lambda op: signature
                ),
            ):
                os.makedirs(os.path.join(ops, "fakeop"))
                open(os.path.join(ops, "fakeop", "aot.py"), "w").close()
                with _patched_generation(ops, declarations=(_CoversDecl,)):
                    gen_aot_lib.main(["--artifacts-dir", tmpdir])
            with open(os.path.join(tmpdir, "fakeop", "aot_fakeop_cuda.cpp")) as f:
                src = f.read()
        self.assertIn("bool fakeop_cuda_covers(", src)
        self.assertIn('m.def("covers_fakeop(Tensor self, int k) -> bool"', src)

    def test_the_dsl_runtime_reaches_the_emitted_cmake(self):
        # Through main(), not write_cmake_include directly: the archive is discovered
        # in stage 2, passed as --dsl-runtime and handed on from there, and BOTH hops
        # were unpinned -- every test that had an archive passed it to the emitter by
        # hand. Nothing links torch_cuda with --no-undefined, so dropping the archive
        # links green and the first AOT call fails on an undefined symbol.
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            archive = os.path.join(tmpdir, "libcuda_dialect_runtime_static.a")
            open(archive, "w").close()
            self._generate(tmpdir, ["--dsl-runtime", archive])
            with open(os.path.join(tmpdir, gen_aot_lib.CMAKE_INCLUDE)) as f:
                emitted = f.read()
        self.assertIn(f'target_link_libraries(torch_cuda PRIVATE "{archive}")', emitted)
        self.assertIn("--exclude-libs", emitted)

    def test_tree_outside_the_arch_list_is_ignored_not_generated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            self._tree(tmpdir, "sm_90a", "fakeop_p__sm90a")
            self._generate(tmpdir, ["--archs", "sm_100a"])
            with open(os.path.join(tmpdir, "fakeop", "aot_fakeop_cuda.cpp")) as f:
                src = f.read()
            self.assertIn("launch_fakeop_p__sm100a", src)
            self.assertNotIn("sm90a", src)
            # ...and its objects are not in the link set either.
            listed = " ".join(_manifest(tmpdir)["objects"])
            self.assertIn("sm_100a", listed)
            self.assertNotIn("sm_90a", listed)

    def test_object_list_excludes_the_tie_break_loser(self):
        # The companion to the launcher assertion: gen_op computed the surviving
        # set privately, so main() built the object list from the UNFILTERED
        # sidecars and CMake linked the dropped arch's objects with nothing
        # referencing them -- 3 of 6 in a measured tree. Reachable with no
        # --archs, which is every dev build with no TORCH_CUDA_ARCH_LIST.
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            self._tree(tmpdir, "sm_100", "fakeop_p__sm100")
            self._generate(tmpdir, [])
            with open(os.path.join(tmpdir, "fakeop", "aot_fakeop_cuda.cpp")) as f:
                src = f.read()
            listed = _manifest(tmpdir)["objects"]
            # One launcher, one object, and they are the same kernel.
            self.assertEqual(len(listed), 1, listed)
            self.assertIn("/sm_100a/", listed[0])
            self.assertIn("launch_fakeop_p__sm100a", src)
            for obj in listed:
                prefix = os.path.basename(obj)[: -len(".o")]
                self.assertIn(f"void launch_{prefix}(", src)

    def test_generated_source_is_deleted_when_the_sidecars_are_gone(self):
        # The tree still EXISTS, but holds no sidecar -- a partial `rm`, or the
        # "delete their trees" advice taken literally. Distinct from the whole
        # tree being gone (which the pre-loop covers): this path used to
        # `continue` in silence, leaving a source that compiles, references module
        # entry points whose object nothing listed, links green because nothing
        # passes --no-undefined, and fails at symbol lookup on first use.
        with tempfile.TemporaryDirectory() as tmpdir:
            tree = self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            self._generate(tmpdir, [])
            out = os.path.join(tmpdir, "fakeop", "aot_fakeop_cuda.cpp")
            self.assertTrue(os.path.exists(out))
            for f in os.listdir(tree):
                if f.endswith(".json"):
                    os.remove(os.path.join(tree, f))
            self._generate(tmpdir, [])
            self.assertFalse(os.path.exists(out), "stale source survived")
            self.assertEqual(_manifest(tmpdir)["sources"], [])

    def test_emitted_paths_are_absolute_from_a_relative_artifacts_dir(self):
        # The module's own documented usage is `--artifacts-dir build/native_aot`.
        # CMake resolves a relative path against a different directory than the
        # generator, so it would report the file as naming paths that do not
        # exist and silently embed nothing -- blamed on a deleted arch tree.
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            rel = os.path.relpath(tmpdir, os.getcwd())
            self._generate(rel, [])
            man = _manifest(tmpdir)
            self.assertTrue(man["sources"] and man["objects"])
            for p in man["sources"] + man["objects"]:
                self.assertTrue(os.path.isabs(p), f"{p} is not absolute")

    def test_stale_tree_outside_the_arch_list_does_not_fail_the_build(self):
        # The trap this closes: the ignored tree is ALSO stale, and generation
        # used to stop the build telling you to re-export an arch you dropped.
        with tempfile.TemporaryDirectory() as tmpdir:
            self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            self._tree(tmpdir, "sm_90a", "fakeop_p__sm90a", stale_sources=True)
            self._generate(tmpdir, ["--archs", "sm_100a"])
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir, "fakeop", "aot_fakeop_cuda.cpp"))
            )

    def test_generated_source_is_deleted_when_no_artifacts_remain(self):
        # Deleting an arch tree by hand left the .cpp behind, and the link glob
        # then compiled a file whose #include "../<arch>/..." no longer resolves.
        with tempfile.TemporaryDirectory() as tmpdir:
            tree = self._tree(tmpdir, "sm_100a", "fakeop_p__sm100a")
            self._generate(tmpdir, [])
            out = os.path.join(tmpdir, "fakeop", "aot_fakeop_cuda.cpp")
            self.assertTrue(os.path.exists(out))
            import shutil

            shutil.rmtree(os.path.dirname(tree))
            self._generate(tmpdir, [])
            self.assertFalse(os.path.exists(out))


class TestSidecarFieldValidation(unittest.TestCase):
    """Generation reads these fields straight into emitted C++, so a value it
    cannot use must fail here rather than as a compiler error in a @generated
    file, or as a wrongly-read field."""

    def _run(self, tmpdir, sc, extra_argv=()):
        d = os.path.join(tmpdir, "sm_100a", "fakeop")
        os.makedirs(d)
        for e in (".o", ".h"):
            open(os.path.join(d, "k__sm100a" + e), "w").close()
        with open(os.path.join(d, "k__sm100a.json"), "w") as f:
            json.dump(sc, f)
        opsdir = os.path.join(tmpdir, "_ops")
        os.makedirs(os.path.join(opsdir, "fakeop"))
        open(os.path.join(opsdir, "fakeop", "aot.py"), "w").close()
        with _patched_generation(opsdir):
            gen_aot_lib.main(["--artifacts-dir", tmpdir, *extra_argv])

    def _sc(self, **over):
        # N and K: _FakeDecl.cpp_dispatch reads both grid axes, so a case that
        # gets PAST the refusals can still generate.
        sc = {
            "version": export.SIDECAR_VERSION,
            "prefix": "k__sm100a",
            "kind": "cutedsl",
            "arch": "sm_100a",
            "spec": {"N": 1, "K": 8},
            "tensor_args": [],
        }
        sc.update(over)
        return sc

    def test_prefix_must_be_a_c_identifier(self):
        # The prefix names extern "C" entry points and launch_<prefix>.
        #
        # A NON-ASCII case as well as a hyphen: the pattern is spelled out
        # ([A-Za-z_][A-Za-z0-9_]*) rather than \w precisely because \w is Unicode
        # by default, and "h\u00e9llo" satisfies a \w check whose entire contract is
        # "usable as a C identifier". With only the hyphen fixture, swapping the
        # pattern back to the Unicode-aware one left the suite green.
        for prefix in ("k-sm100a", "h\u00e9llo__sm100a", "1k__sm100a"):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(RuntimeError, "not a C identifier"):
                    self._run(tmpdir, self._sc(prefix=prefix))

    def test_schema_version_mismatch_is_not_waivable(self):
        # --allow-stale exists for artifacts whose SOURCES drifted; those still
        # describe themselves in a shape this generator reads. A schema bump
        # may not, so forcing past it would emit from misread fields.
        for argv in ((), ("--allow-stale",)):
            with tempfile.TemporaryDirectory() as tmpdir:
                sc = self._sc(version=export.SIDECAR_VERSION + 1)
                with self.assertRaisesRegex(RuntimeError, "sidecar schema version"):
                    self._run(tmpdir, sc, argv)

    def test_allow_stale_generates_anyway(self):
        # The permissive half of the flag: with it removed entirely (staleness
        # always fatal) the suite stayed green, so nothing proved the escape
        # hatch works -- only that a schema bump ignores it.
        with tempfile.TemporaryDirectory() as tmpdir:
            sc = self._sc(sources={"tools/native_aot/decl.py": "0" * 16})
            self._run(tmpdir, sc, ("--allow-stale",))
            self.assertTrue(
                glob.glob(os.path.join(tmpdir, "*", "aot_*.cpp")),
                "--allow-stale must still generate",
            )

    def test_stale_error_names_the_arches_and_a_command_that_fixes_them(self):
        # A bare `export.py` re-run maintains only the arch it resolves for, so
        # it leaves other arch trees stale forever: the message has to name the
        # arches and the --arch invocation, not just say "re-run export".
        with tempfile.TemporaryDirectory() as tmpdir:
            sc = self._sc(sources={"tools/native_aot/decl.py": "0" * 16})
            with self.assertRaisesRegex(RuntimeError, r"--arch sm_100a"):
                self._run(tmpdir, sc)


class TestSizeGateIsPerToolchain(unittest.TestCase):
    """The i32 size gate belongs to kinds whose exported ABI takes i32
    extents (CuTeDSL), not to every AOT op."""

    class _Decl:
        ATEN_OP = "fakeop"
        DISPATCH_KEY = "CUDA"
        ARCHS = ("sm_100a",)

        def cpp_dispatch(self, spec):
            return "true"

        def cpp_launch(self, spec, fn):
            return f"{fn}(self, k, at::cuda::getCurrentCUDAStream());"

    _SC = {
        "prefix": "p",
        "spec": {"N": 1},
        "tensor_args": [{"name": "self", "dynamic_sizes": [0], "dynamic_strides": [0]}],
        "arch": "sm_100a",
    }

    def _gen(self, kind):
        return gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            self._Decl(),
            [dict(self._SC, kind=kind)],
            "const at::Tensor & self, int64_t k",
        )

    def test_narrowing_kind_gets_gate_and_helper(self):
        src = self._gen("cutedsl")
        self.assertIn("// Size gate:", src)
        self.assertIn("inline bool _naot_dim_too_big", src)

    def test_non_narrowing_kind_gets_neither(self):
        # An unused inline helper would otherwise sit in every generated
        # file, and the gate would decline dims the kernel could serve.
        class _Wide(toolchains.CuteDslToolchain):
            kind = "wide"
            NARROWS_SHAPES_TO_INT32 = False

        with mock.patch.dict(toolchains.TOOLCHAINS, {"wide": _Wide()}, clear=False):
            src = self._gen("wide")
        self.assertNotIn("// Size gate:", src)
        self.assertNotIn("_naot_dim_too_big", src)


class TestSelectiveIncludes(unittest.TestCase):
    """torch/library.h pulls the dispatcher in (~110 transitive headers), and
    only the cpp_covers registration needs it."""

    _SC = {
        "prefix": "p",
        "spec": {"N": 1024, "K": 8},  # _FakeDecl.cpp_dispatch reads both
        "kind": "cutedsl",
        "arch": "sm_100a",
        "tensor_args": [{"name": "self", "dynamic_sizes": [0], "dynamic_strides": [0]}],
    }
    _COVERS = (
        "const at::Tensor & self",
        "fakeop(Tensor self) -> bool",
        "return true;",
    )

    def _gen(self, covers):
        return gen_aot_lib.gen_op(
            "fakeop",
            "CUDA",
            _FakeDecl,
            [self._SC],
            "const at::Tensor & self",
            covers,
        )

    def test_library_header_omitted_without_covers(self):
        src = self._gen(None)
        self.assertNotIn("#include <torch/library.h>", src)
        self.assertFalse(
            any(l.startswith("TORCH_LIBRARY_FRAGMENT") for l in src.splitlines())
        )

    def test_library_header_present_with_covers(self):
        src = self._gen(self._COVERS)
        self.assertIn("#include <torch/library.h>", src)
        self.assertTrue(
            any(l.startswith("TORCH_LIBRARY_FRAGMENT") for l in src.splitlines())
        )

    def test_always_needed_headers_are_unconditional(self):
        # These are used by every generated file (stub registration, launcher
        # stream narrowing, the arch gate's device query).
        src = self._gen(None)
        for h in (
            "<ATen/core/Tensor.h>",
            "<ATen/NativeAotStubs.h>",
            "<ATen/cuda/CUDAContext.h>",
            "<c10/cuda/CUDAStream.h>",
        ):
            with self.subTest(header=h):
                self.assertIn(f"#include {h}", src)


class TestOrphanArtifactSafety(unittest.TestCase):
    """Generation must never destroy exported kernels: a .cpp costs nothing to
    regenerate, a .o costs a full export."""

    def _art(self, tmpdir, arches=("sm_100a", "sm_90a")):
        """The one layout: kernels at <root>/<arch>/<decl_id>/, and the
        generated source that covers every arch at <root>/<decl_id>/.
        Returns the FIRST arch dir and the source dir, since orphan handling
        touches each differently -- the .cpp is regenerable and gets deleted, the
        kernels never do.

        TWO arch dirs, each holding a differently-named artifact: with one, a
        report that names art_dirs[0] while listing basenames gathered from every
        tree reads exactly like one that names each directory."""
        op = os.path.join(tmpdir, arches[0], "fakeop")
        for arch in arches:
            d = os.path.join(tmpdir, arch, "fakeop")
            os.makedirs(d)
            for ext in (".o", ".h", ".json"):
                with open(os.path.join(d, f"k_{arch}{ext}"), "w") as f:
                    f.write("x")
        src = os.path.join(tmpdir, "fakeop")
        os.makedirs(src)
        with open(os.path.join(src, "aot_fakeop_cuda.cpp"), "w") as f:
            f.write("x")
        return op, src

    def test_same_prefix_from_two_arch_dirs_is_fatal(self):
        # Prefixes carry their arch, so two arch dirs cannot collide by
        # construction -- but a copied or renamed tree (rsynced artifacts, a
        # hand-made sm_100a.bak) puts the same prefix under two arch dirs, and
        # discovery merges by declaration. Caught here rather than as a
        # launch_<prefix> redefinition, or the same .o globbed twice, much
        # later in the build.
        with tempfile.TemporaryDirectory() as tmpdir:
            for rel in (
                os.path.join("sm_100a", "fakeop"),
                os.path.join("sm_100a_copy", "fakeop"),
            ):
                d = os.path.join(tmpdir, rel)
                os.makedirs(d)
                for fn in ("k__sm100a.o", "k__sm100a.h"):
                    with open(os.path.join(d, fn), "w") as f:
                        f.write("x")
                with open(os.path.join(d, "k__sm100a.json"), "w") as f:
                    json.dump(
                        {
                            "version": export.SIDECAR_VERSION,
                            "prefix": "k__sm100a",
                            "arch": "sm_100a",
                        },
                        f,
                    )
            # The duplicate check runs after a declaration is matched, so
            # fakeop has to BE declared or discovery takes the orphan path
            # first. gen_aot_lib imports export inside main(), so patch the
            # module itself rather than an attribute on gen_aot_lib.
            opsdir = os.path.join(tmpdir, "_ops")
            os.makedirs(os.path.join(opsdir, "fakeop"))
            open(os.path.join(opsdir, "fakeop", "aot.py"), "w").close()
            with (
                mock.patch.object(gen_aot_lib, "OPS_DIR", opsdir),
                mock.patch.object(
                    gen_aot_lib.decl, "load_declarations", return_value=[_FakeDecl]
                ),
                mock.patch.object(export, "sources_current", return_value=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "present in both"):
                    gen_aot_lib.main(["--artifacts-dir", tmpdir])

    def test_archs_cannot_be_given_no_values(self):
        # Stage 2 composes `--archs *archs`, so an empty arch list produced a BARE
        # flag; nargs="*" then assigned [], and the filter -- whose purpose is to
        # stop a tree from another TORCH_CUDA_ARCH_LIST being shipped -- silently
        # passed everything. Refused here rather than trusting the caller to check.
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    gen_aot_lib.main(["--artifacts-dir", d, "--archs"])

    def test_no_declarations_at_all_leaves_the_artifacts_alone(self):
        # A commit earlier in the stack (or a bisect) declares nothing; that
        # is not the same as every artifact being orphaned.
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            tempfile.TemporaryDirectory() as opsdir,
        ):
            op, src = self._art(tmpdir)
            # An EMPTY ops dir: "declares nothing" must not depend on which
            # commit of the stack is checked out.
            with (
                mock.patch.object(gen_aot_lib, "OPS_DIR", opsdir),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                gen_aot_lib.main(["--artifacts-dir", tmpdir])
            left = sorted(os.listdir(op))
            src_left = sorted(os.listdir(src))
            said = printed.getvalue()
        # The ARTIFACTS are what must survive a bisect -- they cost a full export.
        self.assertIn("k_sm_100a.o", left)
        self.assertIn("k_sm_100a.h", left)
        # The generated source must NOT: this run emitted a "nothing to embed"
        # include, so a leftover .cpp makes stage 2's glob report that kernels were
        # generated and fail the build blaming the CMake cache. Regenerating a
        # source is free.
        self.assertNotIn("aot_fakeop_cuda.cpp", src_left)
        # ...and nothing is REPORTED as undeclared: these artifacts are not orphans,
        # they predate the declarations, which is the whole point of the arm that
        # stops before the leftover report.
        self.assertNotIn("with no declaration", said)

    def test_orphan_with_other_declarations_is_reported_and_keeps_artifacts(self):
        # With declarations present, an unclaimed dir is a real orphan: drop the
        # generated .cpp so nothing references a vanished stub, keep the objects
        # (they cost a full export), and REPORT.
        #
        # Not fatal: the generated CMake names exact paths, so an unnamed artifact
        # cannot be linked, and the .cpp that could have referenced one is deleted
        # here. As a fatal it failed every later build until a full export was
        # hand-deleted, and fired for directories holding only the ABI header.
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            tempfile.TemporaryDirectory() as opsdir,
        ):
            op, src = self._art(tmpdir)
            # by_id is built by walking OPS_DIR, so declare something there.
            os.makedirs(os.path.join(opsdir, "otherop"))
            open(os.path.join(opsdir, "otherop", "aot.py"), "w").close()
            with (
                mock.patch.object(gen_aot_lib, "OPS_DIR", opsdir),
                mock.patch.object(
                    gen_aot_lib.decl, "load_declarations", return_value=[_OtherDecl]
                ),
            ):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    gen_aot_lib.main(["--artifacts-dir", tmpdir])
            # Names the directory the files are actually IN: art_dirs[0] was
            # named while basenames were gathered from every arch tree, so a
            # multi-arch orphan pointed at files that were not in the directory
            # named, and its "delete the directory" left the others behind.
            said = out.getvalue()
            self.assertIn("no declaration", said)
            # EVERY arch dir named alongside ITS OWN artifacts: naming art_dirs[0]
            # while listing files from all of them pointed the user at files that
            # were not in the directory named, and "delete this directory" then
            # left the others behind.
            for arch in ("sm_100a", "sm_90a"):
                one = os.path.join(tmpdir, arch, "fakeop")
                line = next(l for l in said.splitlines() if l.startswith(one))
                self.assertIn(f"k_{arch}.o", line)
                other = "sm_90a" if arch == "sm_100a" else "sm_100a"
                self.assertNotIn(f"k_{other}.o", line)
            left = sorted(os.listdir(op))
            src_left = sorted(os.listdir(src))
        self.assertIn("k_sm_100a.o", left)
        # The regenerable source is dropped from <root>/<decl_id>/, where it
        # lives now, so the CMake glob cannot compile it against a stub that
        # no longer exists.
        self.assertNotIn("aot_fakeop_cuda.cpp", src_left)


class _OtherDecl(_FakeDecl):
    """A declaration whose decl_id does not match the artifact dir above."""

    ATEN_OP = "otherop"


class TestMissingArtifacts(unittest.TestCase):
    def test_missing_artifacts_reexport_despite_current_sidecar(self):
        # Sidecar matches spec/arch/sources, but its .o is gone: skipping
        # here surfaces as a missing include when torch_cuda compiles.
        point = {"dtype": "float32", "N": 4096}
        with tempfile.TemporaryDirectory() as d:
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point)
            with _no_ambient_arch():
                self.assertFalse(export._job_needed(job, force=False))

                os.remove(os.path.join(d, "x.o"))
                self.assertTrue(export._job_needed(job, force=False))

    def test_missing_header_also_reexports(self):
        point = {"dtype": "float32", "N": 4096}
        with tempfile.TemporaryDirectory() as d:
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point, exts=(".o",))  # .h missing
            # _no_ambient_arch: the job resolves arch=None, so an exported
            # CUTE_DSL_ARCH (this commit's Test Plan sets one) would refuse rather
            # than answer, failing this test for a reason it is not about.
            with _no_ambient_arch():
                self.assertTrue(export._job_needed(job, force=False))


if __name__ == "__main__":
    unittest.main()
