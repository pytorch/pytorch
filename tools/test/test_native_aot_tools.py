from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import unittest.mock as mock

# Ordinary package imports, like every other tools test (CI runs
# `PYTHONPATH=$(pwd) pytest tools/test`). These modules keep their module
# scope torch-free precisely so this works: the Test tools job runs in the
# linter image, which has no built torch.
from tools.native_aot import export, toolchains

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


class TestLauncherCodegen(unittest.TestCase):
    """CuteDslToolchain.gen_launcher is what puts C++ into the build; these
    pin the properties that were argued over in review."""

    def _launcher(self, **over):
        sc = dict(SIDECAR, **over)
        return toolchains.CuteDslToolchain().gen_launcher(sc)

    def test_read_only_arg_uses_const_data_ptr(self):
        # A mutable data_ptr() would materialize a copy-on-write input.
        targs = [dict(SIDECAR["tensor_args"][0], read_only=True)]
        src = self._launcher(tensor_args=targs)
        self.assertIn("const_cast<void*>(mX.const_data_ptr())", src)
        self.assertNotIn("mX.mutable_data_ptr()", src)

    def test_written_arg_uses_mutable_data_ptr(self):
        src = self._launcher()
        self.assertIn("mOut_s.data = mOut.mutable_data_ptr();", src)

    def test_arg_order_is_tensors_then_scalars_then_stream(self):
        # Must match the exported wrapper's signature exactly.
        src = self._launcher(scalar_args=[{"name": "k", "ctype": "int64_t"}])
        self.assertIn("const at::Tensor& mX, const at::Tensor& mOut, int64_t k", src)
        self.assertIn("&mX_s, &mOut_s, k", src)
        self.assertIn("c10::cuda::CUDAStream(stream).stream()", src)

    def test_shape_slots_narrow_and_strides_stay_64bit(self):
        src = self._launcher()
        self.assertIn("mX_s.dynamic_shapes[0] = static_cast<int32_t>(mX.size(0));", src)
        self.assertIn("mX_s.dynamic_strides[0] = mX.stride(0);", src)

    def test_module_load_is_once_per_process(self):
        src = self._launcher()
        self.assertIn("c10::call_once", src)


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


class TestReadOnlyInputs(unittest.TestCase):
    # read_only tensor args must go through const_data_ptr in every
    # toolchain's launcher: a mutable data_ptr() materializes
    # copy-on-write inputs on each call.

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


class TestRegistryConsistency(unittest.TestCase):
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
