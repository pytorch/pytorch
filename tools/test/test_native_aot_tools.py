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

# Ordinary package imports: these modules keep their scope torch-free so the
# Test tools job, which runs in the linter image, can import them without torch.
from tools.native_aot import export, gen_aot_lib, toolchains

from torchgen import native_aot_decl


_TOOLS_FILE = os.path.abspath(toolchains.__file__)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_TOOLS_FILE)))


# The DSL versions this environment would compile with; sidecars must carry
# them or the skip check treats them as built by a different compiler.
_RUNTIMES = export.runtime_versions("cutedsl")

SIDECAR = {
    "prefix": "fakeop_f32_n1024_k8",
    # Generation reads arch and kind rather than defaulting either, so a fixture
    # missing them is not a sidecar export could have written.
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

    ``device`` is what _detected_arch should report: None for "no GPU", or an sm string
    to exercise the device fallback.

    _effective_arch resolves an unspecified arch from the local GPU or from a
    toolchain's ARCH_ENV_VAR, so tests about spec/source matching must patch both or
    depend on the runner's hardware and shell."""
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
    """Create the files a sidecar claims. _job_needed verifies they exist, so a fixture
    writing only the .json would always re-export.

    The .h is derived from the sidecar's own tensor_args -- one struct per tensor, each
    array bound equal to the dims that tensor claims -- because validate_abi requires
    that equality. Hand-picked bounds would describe an ABI no export could produce."""
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
        # A spec match alone re-exports: the source closure must match too.
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
        # Tuple-valued grid fields read back as lists, so both sides must be
        # normalized or such points re-export on every run.
        with tempfile.TemporaryDirectory() as d:
            point = {"aten": "add.Tensor", "in_dtypes": ("float32", "bfloat16")}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point, spec=export._json_normal(point))
            with _no_ambient_arch():
                self.assertFalse(export._job_needed(job, force=False))

    def test_run_job_is_module_level(self):
        # The pool pickles the job function by qualified name, so a closure would
        # break only at --jobs > 1.
        self.assertEqual(export._run_job.__qualname__, "_run_job")
        self.assertEqual(export.export_point.__qualname__, "export_point")

    def test_pool_never_forks_after_cuda_init(self):
        # A fork parent that has initialized CUDA gives workers a dead context,
        # silently. forkserver specifically: main() calls set_forkserver_preload.
        self.assertEqual(export.POOL_START_METHOD, "forkserver")

    def test_cutedsl_export_passes_gpu_arch(self):
        # A --gpu-arch option rather than process state, which is what lets one
        # worker serve several arches, appended to any builder-supplied options.
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
        # Backends are data, not a platform check in the gate, so a future ROCm DSL
        # is a new class rather than an edit to build_stage2.
        for kind, tc in toolchains.TOOLCHAINS.items():
            self.assertTrue(tc.BACKENDS, f"{kind} declares no BACKENDS")
        self.assertEqual(sorted(toolchains.for_backend("rocm")), [])
        self.assertEqual(
            sorted(toolchains.for_backend("cuda")),
            sorted(toolchains.TOOLCHAINS),
        )

    def test_missing_runtime_is_fatal_not_skipped(self):
        # A declaration targeting this backend was asked for, so a missing runtime must
        # fail rather than ship fewer kernels than declared.
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
        # The forkserver's server process is the fork parent, so only modules inert
        # there may be preloaded: cutlass or triton would build state workers inherit.
        self.assertEqual(export.POOL_PRELOAD, ("torch",))

    def test_json_normal_matches_sidecar_round_trip(self):
        # It stands in for a json.dumps/loads pair, so any divergence makes a spec
        # mismatch its own sidecar and re-export forever.
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
    def test_effective_arch_cannot_vary_by_kind(self):
        # Why the resolver takes no toolchain: with no --arch, resolution is device
        # detection, and a per-kind answer lets a tree disagree with its sidecars.
        self.assertNotIn("tc", export._effective_arch.__code__.co_varnames)
        with _no_ambient_arch(device="sm_100"):
            self.assertEqual(export._effective_arch(None), "sm_100")
        # An explicit arch wins, whatever a kind's variable says.
        with mock.patch.dict(os.environ, {"CUTE_DSL_ARCH": "sm_90a"}):
            self.assertEqual(export._effective_arch("sm_100a"), "sm_100a")

    def test_arch_tag_is_short(self):
        # The tag lands in every exported C symbol, so its shape is part of
        # the artifact ABI: one underscore dropped, nothing else.
        self.assertEqual(export._arch_tag("sm_100a"), "sm100a")
        self.assertEqual(export._arch_tag("sm_90"), "sm90")

    def test_an_arch_env_var_without_explicit_arch_is_refused(self):
        # An arch variable is per kind, so honouring it would name a tree after one
        # kind's arch while holding another kind's sidecars.
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
            # ...and through the overload the production callers use, since
            # export_point and _job_needed both pass a toolchain.
            for tc in (toolchains.get_toolchain("cutedsl"), _Other()):
                with self.subTest(kind=tc.kind):
                    with self.assertRaisesRegex(RuntimeError, "CUTE_DSL_ARCH=sm_90a"):
                        export._effective_arch(None)
            # An explicit --arch is the way to say it, for every kind at once.
            self.assertEqual(export._effective_arch("sm_100a"), "sm_100a")

    def test_export_prefix_is_arch_qualified(self):
        # Every exported symbol derives from the prefix, so two arches sharing one
        # are duplicate definitions once both link into libtorch_cuda.
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
                # The pair, not the prefix alone: the prefix names an arch, and this
                # is the arch the kernel was really compiled for.
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
                # The sidecar WRITER against what the readers require; every other
                # test runs on hand-written fixtures.
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

                # The on-device path: recording the raw argument would leave
                # "arch": null behind an unqualified prefix, which generation refuses.
                with _no_ambient_arch(device="sm_100"):
                    export.export_point("fakeop", "aot_kernel.py", {"n": 2}, d, None)
                with open(os.path.join(d, "k__sm100.json")) as f:
                    on_device = json.load(f)
                self.assertEqual(on_device["prefix"], "k__sm100")
                self.assertEqual(on_device["arch"], "sm_100")
                # The compile is still told the RAW arch: None lets the DSL take the
                # local device, which is what was just detected.
                self.assertEqual(seen[-1], ("k__sm100", None))

    def test_job_skip_compares_the_arch(self):
        # Two exports into one --out-dir differing only in arch: comparing spec alone
        # skips the second and leaves the first arch's objects behind its sidecar.
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
        # The compiler appears in no file the closure hashes, so without this an
        # upgraded DSL wheel invalidates nothing. runtime_versions is patched: with no
        # wheels installed every version is "absent", which is not staleness.
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
        # A sidecar recording no arch must not satisfy a run whose CUTE_DSL_ARCH
        # names one, or that run inherits objects built for the builder's own GPU.
        with tempfile.TemporaryDirectory() as d:
            point = {"dtype": "float32", "N": 4096}
            job = ("fakeop", "aot_kernel.py", point, d, None)
            _write_sidecar(d, point, arch=None)
            with _no_ambient_arch():
                named = ("fakeop", "aot_kernel.py", point, d, "sm_100a")
                self.assertTrue(export._job_needed(named, force=False))
                # ...nor an on-device run, which knows its arch where the sidecar
                # names none. _detected_arch is patched, not trusted.
                with mock.patch.object(export, "_detected_arch", return_value="sm_100"):
                    self.assertTrue(export._job_needed(job, force=False))
                # Only where no arch can be resolved at all is it a match.
                self.assertFalse(export._job_needed(job, force=False))

    def test_cc_of_reads_the_capability_both_spellings_name(self):
        # The exporter matches ARCHS by string while the generator groups by
        # capability, so both spellings of one piece of hardware must parse equal.
        self.assertEqual(native_aot_decl.cc_of("sm_90"), (9, 0))
        self.assertEqual(native_aot_decl.cc_of("sm_103a"), (10, 3))
        self.assertEqual(
            native_aot_decl.cc_of("sm_100a"), native_aot_decl.cc_of("sm_100")
        )

    def test_cc_of_refuses_what_it_cannot_read(self):
        # Each would otherwise compute a plausible capability ("sm_9" -> (0, 9)) and
        # emit a gate no device satisfies. The last four are what str.isdigit() accepts.
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
        # Without this the gate falls back to ARCHS and advertises hardware nothing was
        # compiled for. No "a" suffix: the gate compares major.minor.
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

    def test_the_shipped_arches_are_ones_the_tooling_can_target(self):
        # The two sets live beside each other so they cannot drift; the import-time
        # check is what makes that true rather than merely intended.
        self.assertLessEqual(
            set(native_aot_decl.EXPORTABLE_ARCHES), set(native_aot_decl.KNOWN_ARCHES)
        )
        # ...and the exporter's name is the same object, not a copy that could age.
        self.assertIs(export.EXPORTABLE_ARCHES, native_aot_decl.EXPORTABLE_ARCHES)

    def test_archs_from_cuda_arch_list_collapses_one_capability(self):
        # Both spellings are the same hardware and generation uses one, so exporting
        # both compiles a second full set of kernels no launcher references.
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
            # _detected_arch patched: the on-device call below resolves its
            # directory from it, and this suite runs where there is no GPU.
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
        # Every job nests under <out>/<arch>/<decl_id>, one arch or several: there is
        # no second, flat layout, which the single-arch and on-device cases pin too.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops)
            # _detected_arch patched, so the layout claim does not depend on the
            # machine having a GPU.
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
        # sm_100a, not the detected sm_100: an on-device export adopts the spelling the
        # DECLARATION claims, so the tree cannot be one it disowns.
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
        # Both are valid on 10.0 hardware, and the conditional build is what the
        # kernels were written against; otherwise directory order would decide.
        for order in (
            [("sm_100", "p"), ("sm_100a", "c")],
            [("sm_100a", "c"), ("sm_100", "p")],
        ):
            scs = [{"prefix": n, "arch": a} for a, n in order]
            groups = gen_aot_lib._by_arch(scs)
            self.assertEqual(list(groups), [(10, 0)])
            self.assertEqual([s["prefix"] for s in groups[(10, 0)]], ["c"])

    def test_by_arch_rejects_an_arch_less_sidecar(self):
        # Export names the arch of everything it writes, so this is an older tree.
        # Rejected rather than grouped: an unmatchable capability declines silently.
        with self.assertRaisesRegex(RuntimeError, "records no arch"):
            gen_aot_lib._by_arch([{"prefix": "old", "arch": None}])

    def test_dropped_tie_break_candidate_gets_no_launcher(self):
        # The plain build loses the tie-break, and a launcher emitted for it would be
        # defined and never called: -Wunused-function, fatal under CI's WERROR.
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

    def test_device_match_renders_the_full_capability(self):
        # cc_of itself is covered above, against native_aot_decl directly.
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
        # An export that died before writing the sidecar. Generation names artifacts
        # from sidecars, so an undescribed orphan costs disk, not payload.
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "k_f32.o"), "w").close()
            with contextlib.redirect_stdout(io.StringIO()) as out:
                export._check_no_orphan_artifacts(d, [])
        self.assertIn("no sidecar claims", out.getvalue())
        self.assertIn("k_f32.o", out.getvalue())

    def test_an_orphan_beside_a_committed_point_is_reported_not_fatal(self):
        # Where an interrupt lands: among points that already committed. Per directory
        # this goes unseen; per artifact and fatal it needs a hand-delete.
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
        # What a first export looks like when it dies: the DSL writes the .h before the
        # .o, so a failed compile strands one per point with no sidecar yet.
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
        # Silent, not merely non-fatal: claiming per artifact is what stops a healthy
        # directory reporting its own kernels as orphans.
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


class TestLauncherGeneration(unittest.TestCase):
    def test_read_only_inputs_take_const_data_ptr(self):
        # read_only args must go through const_data_ptr: a mutable data_ptr()
        # materializes copy-on-write inputs on every call.
        sc: dict = dict(SIDECAR)
        sc["tensor_args"] = [
            {"name": "mX", "dynamic_sizes": [0], "read_only": True},
            {"name": "mOut", "dynamic_sizes": [0]},
        ]
        src = gen_aot_lib.gen_launcher(sc)
        self.assertIn("mX_s.data = const_cast<void*>(mX.const_data_ptr());", src)
        self.assertIn("mOut_s.data = mOut.mutable_data_ptr();", src)

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
    # Set explicitly, since fixtures bypass the loader that normalizes ARCHS on
    # real declarations. Annotated so subclasses can narrow it.
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


class TestInt32SizeGate(unittest.TestCase):
    # The exported ABI carries int32_t shape slots while aten sizes are int64_t, so
    # the gate must decline oversized dims rather than let the cast truncate them.
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
        # The WHOLE expression: substrings cannot tell it from an inverted sense, probes
        # joined with && , or only the first tensor gated. Two tensors, so the last shows.
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
        # The comparison, not just the signature: a helper that can never fire leaves
        # every oversized dim to truncate through the launcher's static_cast.
        self.assertIn("return d > std::numeric_limits<int32_t>::max();", src)
        # _FakeDecl declares no cpp_covers, so there is exactly one gate site.
        self.assertEqual(src.count("// Size gate:"), 1)
        self.assertIn("self.sizes().begin()", src)

    def test_covers_gets_the_same_gates_as_the_stub(self):
        # Coverage must be no wider than the stub's acceptance, or the router hands a
        # call to a stub that refuses it and it loses its JIT route.
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
        # ...reading the TENSOR's device, not the current one: the router calls
        # covers before any device guard.
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
        # The LAST statement of the kernel body, which routes an unmatched call to
        # op.impl. Positional, because the two gates each emit a "return false;" too.
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
            # The contract's "absent prelude" case; callable -> None trips bad-override.
            cpp_dispatch_prelude = None  # pyrefly: ignore [bad-override]

        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop", "CUDA", NoPrelude, [sidecar], "const at::Tensor & self, int64_t k"
        )
        self.assertNotIn("scalar_type() != at::kFloat", src)
        self.assertIn("if (N == 1024 && k == 8) {", src)

    def test_a_hook_that_returns_nothing_emits_nothing(self):
        # The attribute exists, so the callable default does not fire; `or ""` is what
        # keeps "None" out of the generated C++ when a body forgets to return.
        class NoneHooks(_FakeDecl):
            @staticmethod
            def cpp_dispatch_prelude():
                return None

            @staticmethod
            def cpp_helpers():
                return None

        sidecar = dict(SIDECAR, spec={"N": 1024, "K": 8})
        src = gen_aot_lib.gen_op(
            "fakeop", "CUDA", NoneHooks, [sidecar], "const at::Tensor & self, int64_t k"
        )
        self.assertNotIn("None", src)
        self.assertIn("if (N == 1024 && k == 8) {", src)

    def test_the_branch_launches_inside_its_condition(self):
        # The BLOCK, not its three pieces: separate assertions accept a branch that
        # launches before the if, or returns false after launching.
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
        # The declaration's BODY: the is_cuda() comes from the generated guard, and a
        # covers reduced to those guards claims coverage the stub declines.
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
        # The schema shapes per-argument rendering must handle: a kwarg-only section,
        # a list default ('int[1] dim=[]'), Scalar defaults.
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
        # Why the generated header states this: index_add's dim arrives wrapped and
        # sum.dim_IntList's raw, and a raw negative dim declines every dim=-1 call.
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


class TestShippedVsDeclaredArchs(unittest.TestCase):
    def test_shipping_an_arch_the_declaration_disowns_is_fatal(self):
        # The shared fixture declares every arch, so this needs its own narrow one:
        # otherwise a packaging bug emits a gate for hardware the op disowns.
        class _Narrow(_FakeDecl):
            ARCHS = ("sm_100a",)

        s90 = dict(SIDECAR, prefix="fakeop_p__sm90a", arch="sm_90a")
        with self.assertRaisesRegex(RuntimeError, "declaration supports only"):
            gen_aot_lib.gen_op(
                "fakeop", "CUDA", _Narrow, [s90], "const at::Tensor & self, int64_t k"
            )

    def test_a_disowned_tree_beside_a_claimed_one_is_fatal(self):
        # Same capability, so the plain spelling loses the tie-break: the check must run
        # over every exported tree or a stale sm_100 beside sm_100a passes unnoticed.
        class _Narrow(_FakeDecl):
            ARCHS = ("sm_100a",)

        s100 = dict(SIDECAR, prefix="fakeop_p__sm100", arch="sm_100", _dir="out/sm_100")
        s100a = dict(
            SIDECAR, prefix="fakeop_p__sm100a", arch="sm_100a", _dir="out/sm_100a"
        )
        with self.assertRaisesRegex(RuntimeError, "declaration supports only"):
            gen_aot_lib.gen_op(
                "fakeop",
                "CUDA",
                _Narrow,
                [s100, s100a],
                "const at::Tensor & self, int64_t k",
            )
        # The message must name the tree to delete, not the survivor.
        with self.assertRaisesRegex(RuntimeError, r"Delete out/sm_100 --"):
            gen_aot_lib.gen_op(
                "fakeop",
                "CUDA",
                _Narrow,
                [s100, s100a],
                "const at::Tensor & self, int64_t k",
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
        # Defaults match SIDECAR's claims (mX one shape and one stride, mOut two and
        # one); anything else describes an ABI no export could produce.
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
        # use_32bit_stride is per argument, so a whole-file search for one
        # "int64_t dynamic_strides" accepts a header that narrows another tensor's.
        self._refuses(
            self._header(mout=self._struct("mOut", shapes=2, stride_t="int32_t")),
            "mOut",
        )

    def test_a_longer_argument_name_does_not_shadow_a_shorter_one(self):
        # A find() of "<prefix>_Tensor_mX_t" also matches inside "..._mX_tile_t", so
        # argument order alone would decide which tensor's struct is read.
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
        # Two independent statements about one number. Over-claiming stores past the end
        # of an uninitialized local; under-claiming leaves slots unwritten.
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
                # Each side named as its own: the sidecar's key and the header's
                # member differ (dynamic_sizes against dynamic_shapes).
                self._refuses(
                    header, r"sidecar's dynamic_(sizes|strides) claims \d+ slot"
                )

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
        # `int64_t  dynamic_strides [ 1 ]` is the same declaration, and the DSL's own
        # C-type table stores some spellings with a trailing space.
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
        # Fails closed on a layout this code cannot read, so an upstream header change
        # cannot switch the guard off silently.
        self._refuses(
            "struct x { int64_t dynamic_strides[1]; };\n", "no `typedef struct"
        )
        self._refuses(
            f"/* ABI: {SIDECAR['prefix']}_Tensor_mX_t* */\n" + self._header(mx=""),
            "no `typedef struct",
        )

    def test_a_malformed_sidecar_is_refused_with_a_useful_message(self):
        # Each of these must name the file and the fix, not raise a bare
        # KeyError/TypeError out of a build step.
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
        # A search over the raw body takes the first textual match, so a commented-out
        # declaration above the real one wins.
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
        # `} *Ptr_t;` is ordinary C, and a body regex running past it would check one
        # tensor against another's declarations.
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
        # "Claims nothing, so nothing to check" is exactly the state that leaves every
        # slot the header does declare unwritten in an uninitialized local.
        p = SIDECAR["prefix"]
        self._refuses(
            "#pragma once\n#include <stdint.h>\n"
            f"typedef struct {p}_Tensor_mX_s {{\n  void* data;\n"
            "  int32_t dynamic_shapes[2];\n  int64_t dynamic_strides[2];\n"
            f"}} {p}_Tensor_mX_t;\n" + self._struct("mOut", shapes=2, strides=1),
            # The tag form is readable, so this refuses on the real problem: the
            # header declares slots the sidecar claims none of.
            r"sidecar's dynamic_\w+ claims 0 slot",
            tensor_args=_MX_NO_SLOTS,
        )
        # ...and where the declaration cannot be parsed at all, a zero claim is
        # refused rather than skipped.
        self._refuses(
            "#pragma once\n#include <stdint.h>\n"
            "typedef struct {\n  void* data;\n  int64_t dynamic_strides[2];\n"
            f"}} *{p}_TensorPtr_mX_t;\n" + self._struct("mOut", shapes=2, strides=1),
            "no `typedef struct",
            tensor_args=_MX_NO_SLOTS,
        )

    def test_an_unreadable_MEMBER_is_refused_even_when_nothing_is_claimed(self):
        # A declaration this parser cannot classify never enters `declared`, and the
        # absent-member arm would read that as zero slots. Both spellings are ordinary C.
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
        # C reads [010] as 8 where int("010") is 10, and comparing those counts is this
        # check's whole job, so such a bound is refused rather than parsed.
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
        # Each is what an upstream refactor would emit, and refusing one fails every
        # build. An allowlist, since assuming 64-bit restores the silent truncation.
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
        # `typedef struct {...} *Ptr_t;` is ordinary C, and a body running past its
        # closing brace would merge it into the next struct.
        p = SIDECAR["prefix"]
        self._accepts(
            "#pragma once\n#include <stdint.h>\n"
            "typedef struct {\n  void* data;\n  int64_t dynamic_strides[1];\n"
            f"}} *{p}_TensorPtr_scratch_t;\n" + self._header(),
        )

    def test_an_unknown_stride_spelling_is_refused(self):
        # An allowlist of 64-bit spellings, not a refusal of int32_t: an unrecognized
        # type could be any width. The message says how to extend it.
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
        # A plain open() uses the ambient locale, so a valid UTF-8 header raises
        # UnicodeDecodeError under LC_ALL=C.
        tc = toolchains.CuteDslToolchain()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, SIDECAR["prefix"] + ".h")
            # Invalid UTF-8 (a lone 0xFF), not merely non-ASCII: valid UTF-8 fails
            # only under LC_ALL=C, passing locally and breaking in a container.
            with open(path, "wb") as f:
                f.write(b"/* \xff */\n" + self._header().encode("utf-8"))
            tc.validate_abi(dict(SIDECAR, _dir=d))

    def test_a_header_that_cannot_be_read_is_not_silently_skipped(self):
        # Only ABSENT means "nothing to check", or the ABI goes unvalidated. A directory
        # stands in for the unreadable file, mode 000 being readable by root.
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
        # The failure the grouping exists to prevent: loading a module built for other
        # hardware fails inside the launcher instead of declining to aten.
        body = self._body()
        i9 = body.index("_naot_props->major == 9 && _naot_props->minor == 0) {")
        i10 = body.index("_naot_props->major == 10 && _naot_props->minor == 0) {")
        self.assertLess(i9, i10)
        sm90_branch, sm100_branch = body[i9:i10], body[i10:]
        self.assertIn("launch_fakeop_p__sm90a(", sm90_branch)
        self.assertNotIn("launch_fakeop_p__sm100a(", sm90_branch)
        self.assertIn("launch_fakeop_p__sm100a(", sm100_branch)
        self.assertNotIn("launch_fakeop_p__sm90a(", sm100_branch)


class TestSourceClosureAndRuntimes(unittest.TestCase):
    # What makes an artifact stale: the files whose contents decide what it means,
    # and the compiler that built it. The artifact records neither.

    def test_closure_covers_shared_declaration_machinery(self):
        # The grid expander and the loader decide what a declaration means, and arrive
        # by import, so only the sys.modules half of the closure catches them.
        names = {os.path.basename(p) for p in export.source_closure()}
        for want in (
            "native_aot_spec_grid.py",
            "native_aot_decl.py",
            "toolchains.py",
        ):
            self.assertIn(want, names, f"{want} must invalidate artifacts")

    def test_closure_excludes_the_consumer_tools(self):
        # Neither can change what an artifact means. Against fixtures in a patched
        # _HERE, since asserting absence from the tree would pass with no filter.
        with tempfile.TemporaryDirectory() as d:
            for name in ("export.py", "gen_aot_lib.py", "build_stage2.py"):
                with open(os.path.join(d, name), "w") as f:
                    f.write("# fixture\n")
            with mock.patch.object(export, "_HERE", d):
                names = {os.path.basename(p) for p in export.source_closure()}
        # The exporter's own sources ARE hashed, which is what makes the two
        # absences below mean the filter rather than an empty glob.
        self.assertIn("export.py", names)
        for unwanted in ("gen_aot_lib.py", "build_stage2.py"):
            self.assertNotIn(unwanted, names)

    def test_closure_survives_sys_modules_mutation(self):
        # Hashing imports hashlib lazily, so on a cold interpreter the walk mutates the
        # dict it is iterating. Forced by hashing a module that is not loaded yet.
        import sys

        real_hash = export._file_hash

        def hash_and_import(path):
            importlib.import_module("wave")  # stdlib, unlikely to be loaded
            return real_hash(path)

        sys.modules.pop("wave", None)
        with mock.patch.object(export, "_file_hash", hash_and_import):
            export.source_closure()

    def test_runtimes_current_detects_a_compiler_upgrade(self):
        # The DSL's version is in no file the closure hashes, so an upgraded wheel
        # changes nothing on disk. Patched rather than read: with no wheels installed
        # the live call is all-"absent" and takes the ignorance arm instead.
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
        # Distribution names, so the skip path never imports the DSL: module `cutlass`
        # ships in nvidia-cutlass-dsl, which is not derivable.
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

        # The VALUE, against a known one: _RUNTIMES is produced by this function, so a
        # garbage version would move the fixture with it.
        from importlib.metadata import PackageNotFoundError

        dists = toolchains.get_toolchain("cutedsl").RUNTIME_DISTS
        with mock.patch("importlib.metadata.version", return_value="1.2.3"):
            got = export.runtime_versions("cutedsl")
        self.assertEqual(got, dict.fromkeys(dists, "1.2.3"))
        # Absent rather than omitted, so a sidecar compiled without the wheel differs
        # from one predating this record.
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
            # mismatches the detected one and staleness is never reached.
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
        # _SM_RE accepts "sm_9" and "sm_1000", which name no capability. Refused by the
        # loader, the only place that knows which file to name.
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
        # Declarations load by file path and never enter sys.modules, so without this a
        # grid edit reuses artifacts built from the previous one.
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
            # The closure records the declaration it is given, or editing any aot.py
            # would invalidate every other op's artifacts.
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
                # version, because the check reads the schema first: without one this
                # fixture is "unreadable" rather than "stale".
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
        # It cannot be read, so it cannot be judged: otherwise the next
        # SIDECAR_VERSION bump makes an existing tree demand `rm -rf`.
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
                # version, for the reason above: without it the spec comparison never
                # runs and every sidecar reads as stale.
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
        # The contract is "every dim the CALLER passes must fit": bounding numel() would
        # decline (2**28, 8), whose derived extent is 8 and is served correctly.
        gate = gen_aot_lib._int32_size_gate(
            "const at::Tensor & self, const ::std::optional<at::Tensor> & weight"
        )
        self.assertIn("self.sizes()", gate)
        self.assertNotIn("numel()", gate)

    def test_unhandled_tensor_like_types_are_refused(self):
        # torchgen renders Tensor? as at::OptionalTensorRef and Tensor[] as
        # at::ITensorListRef, neither of which fits the accessors the gate emits.
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


class TestRegistryConsistency(unittest.TestCase):
    def test_link_exts_must_be_a_subset_of_artifact_exts(self):
        # Generation links `if ext in link_exts`, so an ext the kind cannot produce
        # contributes nothing: a green link, then an undefined symbol at first call.
        class _Bad(toolchains.Toolchain):
            kind = "bad"
            artifact_exts = (".cubin", ".h")
            link_exts = (".o",)

        with self.assertRaisesRegex(RuntimeError, "not a subset"):
            toolchains._assert_link_exts_are_exportable({"bad": _Bad()})

    def test_generation_and_export_look_for_declarations_in_one_place(self):
        # Two modules spell this path independently: diverged, every artifact looks
        # undeclared and the sweep refuses a correct build.
        self.assertEqual(gen_aot_lib.OPS_DIR, export.OPS_DIR)

    def test_link_exts_must_be_declared(self):
        # The realistic mistake is forgetting the attribute, which is a subset of
        # everything, so the subset rule alone would accept a kind that links nothing.
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
        # The grid distinguishes a list (an axis) from a tuple (one compound value) where
        # JSON has only arrays, so unrestored a tuple-valued cond never fires.
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
        # The CALL SITE, not just _spec_from_json: dropping the restore in gen_op leaves
        # the direct test above green while every tuple-valued cond stops firing.
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
        # Alphabetical order would pick sm_100 over sm_100a, and the generator's
        # tie-break keeps the conditional one, so the plain build compiles and loses.
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
        # torch/_vendor/ holds kernel bodies a declaration's build() shares with its JIT
        # wrapper, and the probe file sits where only the PREFIX reaches it: under
        # tools/native_aot/ the glob would hash it whatever the prefixes say.
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
        # getDeviceProperties(-1) falls back to the CURRENT device, so without the guard
        # a CPU tensor initializes CUDA inside a predicate the router runs every call --
        # and aborts on a GPU-less host where the answer is simply false.
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
        # Nothing to ask is_cuda() of, and current-device properties reintroduce the
        # abort above. Refused at generation, where an author sees it.
        with self.assertRaisesRegex(RuntimeError, "plain at::Tensor"):
            self._gen(params="const at::ITensorListRef & tensors, int64_t dim")

    def test_a_double_quoted_schema_default_is_escaped(self):
        # 15 in-tree schemas carry one (str mode="constant" on pad). Unescaped it closes
        # the C++ string literal early, far from the declaration that caused it.
        src = self._gen(
            covers_schema='covers_fakeop(Tensor self, str mode="constant") -> bool'
        )
        self.assertIn(r"str mode=\"constant\"", src)
        # ...and the literal is still one argument: nothing between the escaped
        # quotes ends it.
        line = next(l for l in src.splitlines() if "m.def(" in l)
        self.assertEqual(line.count('", &::'), 1)


class TestSizeGateIsPerKindNotPerFile(unittest.TestCase):
    def test_one_narrowing_kind_among_several_emits_the_gate(self):
        # A mixed sidecar list is the only shape that tells `any` from `all` here, and
        # `all` would drop the gate for a file that does carry a narrowing kernel.
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

    def test_an_arch_list_with_no_exportable_arch_exports_nothing(self):
        # Not an error: a CUDA build for Ampere alone simply has no AOT kernels.
        seen = []
        self._main([], {"TORCH_CUDA_ARCH_LIST": "8.0;8.6"}, seen)
        self.assertEqual(seen, [], "_collect_jobs should not even be reached")


class TestCollectJobsRefusals(unittest.TestCase):
    def test_an_unknown_arch_is_refused_before_the_ops_walk(self):
        # The check has to run before the ops walk: with no declaration on disk that
        # walk never runs, so `--arch sm100a` matches nothing and exits 0.
        with (
            tempfile.TemporaryDirectory() as ops,
            tempfile.TemporaryDirectory() as out,
        ):
            with mock.patch.object(export, "OPS_DIR", ops):
                self.assertEqual(os.listdir(ops), [], "no declaration on disk")
                for bad in ("sm100a", "SM_100", "sm_1000", "sm_86", "90"):
                    with self.subTest(arch=bad):
                        with self.assertRaisesRegex(RuntimeError, "not an arch"):
                            export.main(["--arch", bad, "--out-dir", out])
                # ...and a known arch gets past the check (it exports nothing here,
                # which is the honest answer for a tree with no declarations).
                export.main(["--arch", "sm_100a", "--out-dir", out])

    def test_an_unnameable_arch_is_refused(self):
        # There is no unnamed layout: an artifact whose arch nobody can state is one the
        # runtime gate cannot match to hardware.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops)
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device=None),
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot determine the arch"):
                    export._collect_jobs(None, out, [None])

    def test_a_malformed_arch_is_refused(self):
        # The explicit path compares ARCHS by string, so without cc_of here `--arch
        # sm100a` matches no declaration, exports nothing and exits 0 -- a typo that
        # looks like a successful build.
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
                        # matches cc_of's range error.
                        with self.assertRaisesRegex(
                            RuntimeError, "cannot read a compute capability"
                        ):
                            export._collect_jobs(None, out, [bad])

    def test_a_declaration_that_ships_nothing_says_so(self):
        # ARCHS spelling is load-bearing on the explicit path: a declaration pinning
        # ('sm_100a',) ships nothing for a release list of plain spellings, and with
        # several declarations that is partial -- the matched ops embed and pass the
        # post-relink check while the others are absent, with no tree to complain
        # about.
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
        # The report is the only thing that surfaces this, so it has to state the
        # declaration's real ARCHS rather than a fixed illustration.
        self.assertIn("ARCHS (sm_100a)", said)

    def test_a_declaration_that_misses_one_requested_arch_says_so(self):
        # The whole-declaration report is suppressed once a declaration ships for any
        # requested arch, which hides the worst case: the matched arches embed and pass
        # every check while devices of the missed capability fall back to aten
        # silently. Reported per arch, since the declaration claims this capability
        # under the other spelling.
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
        # The other half: a capability the declaration claims under no spelling is not
        # news, or every partial build reports every op.
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
        # Both paths mean the same thing ("export for this machine"), so both must
        # report it: otherwise the user sees `exported 0 kernels` with nothing named.
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
        # Skipped, not exported into a tree the declaration disowns: generation refuses
        # such a tree with a "delete and re-export" remedy that rebuilds it identically.
        with tempfile.TemporaryDirectory() as ops, tempfile.TemporaryDirectory() as out:
            _write_fake_decl(ops, "ARCHS = ('sm_90a',)\n")
            with (
                mock.patch.object(export, "OPS_DIR", ops),
                _no_ambient_arch(device="sm_100"),
            ):
                self.assertEqual(export._collect_jobs(None, out, [None]), [])
                self.assertEqual(os.listdir(out), [], "no tree for a disowned arch")

    def test_the_declarations_spelling_is_adopted_for_the_local_capability(self):
        # sm_100 detected, ('sm_100a',) claimed: same capability, so both the tree and
        # the compile target use the declaration's spelling.
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
        # Every orphan test calls _check_no_orphan_artifacts directly, so this is what
        # pins its only production call site.
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
        # The refusal below can only be honest if handed the WHOLE type: split on every
        # comma, it judges the fragment "at::Tensor>" and names that in the error.
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
            # CUTE_DSL_ARCH would refuse rather than answer.
            with _no_ambient_arch():
                self.assertTrue(export._job_needed(job, force=False))


if __name__ == "__main__":
    unittest.main()
